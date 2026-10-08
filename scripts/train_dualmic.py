import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os, time, argparse, sys
sys.path.insert(0, os.getcwd())
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from dataloaders.dataloader_dualmic import DualMicDataset
from src.stfts import mag_phase_stft, mag_phase_istft
from src.generator_dualmic import xLSTMSENetDualMic
from src.dualmic_utils import load_base_into_dualmic
from src.loss import pesq_score, phase_losses
from src.discriminator import MetricDiscriminator, batch_pesq
from utils.util import save_checkpoint, build_env, load_config, initialize_seed, print_gpu_info, log_model_info

torch.backends.cudnn.benchmark = True


def create_dataset(cfg, args, train=True, device='cuda:0'):
    d = cfg['data_cfg']
    pre = 'train' if train else 'valid'
    prim = getattr(args, f'{pre}_primary_json')
    return DualMicDataset(
        clean_json=d[f'{pre}_clean_json'],
        reference_json=d[f'{pre}_noisy_json'],
        primary_json=prim,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'], hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=True,
        shuffle=(cfg['env_setting']['num_gpus'] <= 1) if train else False)


def create_dataloader(dataset, cfg, train=True):
    if cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        bs = (cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']) if train else 1
    else:
        sampler = None
        bs = cfg['training_cfg']['batch_size'] if train else 1
    return DataLoader(dataset, num_workers=cfg['env_setting']['num_workers'] if train else 1,
                      shuffle=(sampler is None) and train, sampler=sampler, batch_size=bs,
                      pin_memory=True, drop_last=True if train else False)


def to_dev(batch, device):
    return [b.to(device, non_blocking=True) for b in batch]


def train(rank, args, cfg):
    num_gpus = cfg['env_setting']['num_gpus']
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    batch_size = cfg['training_cfg']['batch_size'] // num_gpus

    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f'cuda:{rank}')

    generator = xLSTMSENetDualMic(cfg)
    load_base_into_dualmic(generator, args.base_ckpt)
    generator = generator.to(device)
    discriminator = MetricDiscriminator().to(device)
    if rank == 0:
        log_model_info(rank, generator, args.exp_path)
        if args.train_primary_json is None:
            print("\n*** WARNING: SIMULATED bone channel. Pipeline test only, results NOT reportable. ***\n")

    generator = DistributedDataParallel(generator, device_ids=[rank], broadcast_buffers=True)
    discriminator = DistributedDataParallel(discriminator, device_ids=[rank], broadcast_buffers=True)

    lr = cfg['training_cfg']['learning_rate']
    betas = (cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2'])
    optim_g = optim.AdamW(generator.parameters(), lr=lr, betas=betas)
    optim_d = optim.AdamW(discriminator.parameters(), lr=lr, betas=betas)
    sched_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=cfg['training_cfg']['lr_decay'])
    sched_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=cfg['training_cfg']['lr_decay'])

    # backbone params for the freeze phase (grads are dropped, so AdamW skips them)
    gm = generator.module
    frozen_params = (list(gm.TSxLSTM.parameters()) + list(gm.mask_decoder.parameters())
                     + list(gm.phase_decoder.parameters()))

    trainset = create_dataset(cfg, args, train=True, device=device)
    train_loader = create_dataloader(trainset, cfg, train=True)
    if rank == 0:
        validset = create_dataset(cfg, args, train=False, device=device)
        validset = torch.utils.data.Subset(validset, list(range(0, len(validset), 8)))
        validation_loader = create_dataloader(validset, cfg, train=False)
        sw = SummaryWriter(os.path.join(args.exp_path, 'logs'))

    generator.train(); discriminator.train()
    steps, best_pesq, best_pesq_step = 0, 0.0, 0
    L = cfg['training_cfg']['loss']

    for epoch in range(cfg['training_cfg']['training_epochs']):
        if rank == 0:
            start = time.time(); print("Epoch: {}".format(epoch + 1))
        for batch in train_loader:
            if rank == 0:
                start_b = time.time()
            (clean_audio, clean_mag, clean_pha, clean_com,
             prim_mag, prim_pha, ref_mag, ref_pha) = to_dev(batch, device)
            one_labels = torch.ones(batch_size).to(device, non_blocking=True)

            mag_g, pha_g, com_g, fusion_w = generator(prim_mag, prim_pha, ref_mag, ref_pha)

            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            batch_pesq_score = batch_pesq(list(clean_audio.cpu().numpy()),
                                          list(audio_g.detach().cpu().numpy()), cfg)

            # Discriminator
            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                loss_disc_g = 0
            loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()
            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2
            loss_time = F.l1_loss(clean_audio, audio_g)
            metric_g = discriminator(clean_mag, mag_g)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
            _, _, rec_com = mag_phase_stft(audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True)
            loss_con = F.mse_loss(com_g, rec_com) * 2

            loss_gen_all = (loss_metric * L['metric'] + loss_mag * L['magnitude'] +
                            loss_pha * L['phase'] + loss_com * L['complex'] +
                            loss_time * L['time'] + loss_con * L['consistancy'])
            loss_gen_all.backward()
            if steps < args.freeze_steps:
                for p in frozen_params:
                    p.grad = None
            optim_g.step()

            if rank == 0:
                if steps % cfg['env_setting']['stdout_interval'] == 0:
                    print('Steps : {:d}, Gen: {:4.3f}, Disc: {:4.3f}, Metric: {:4.3f}, Mag: {:4.3f}, '
                          'Pha: {:4.3f}, Com: {:4.3f}, Time: {:4.3f}, Cons: {:4.3f}, s/b: {:4.3f}{}'.format(
                              steps, loss_gen_all.item(), float(loss_disc_all), loss_metric.item(),
                              loss_mag.item(), loss_pha.item(), loss_com.item() / 2, loss_time.item(),
                              loss_con.item() / 2, time.time() - start_b,
                              '  [backbone frozen]' if steps < args.freeze_steps else ''))

                if steps % cfg['env_setting']['summary_interval'] == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all.item(), steps)
                    sw.add_scalar("Training/Discriminator Loss", float(loss_disc_all), steps)
                    sw.add_scalar("Training/Magnitude Loss", loss_mag.item(), steps)
                    sw.add_scalar("Training/Phase Loss", loss_pha.item(), steps)
                    sw.add_scalar("Training/Time Loss", loss_time.item(), steps)
                    # assumed stream order: 0=primary(bone), 1=reference(ambient), 2=spatial
                    w = fusion_w.detach().float().mean(dim=(0, 2, 3)).cpu()
                    for i, name in enumerate(['primary_bone', 'reference_ambient', 'spatial']):
                        sw.add_scalar(f"FusionWeight/{name}", w[i].item(), steps)

                if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                    save_checkpoint(f"{args.exp_path}/g_{steps:08d}.pth",
                                    {'generator': generator.module.state_dict()})
                    save_checkpoint(f"{args.exp_path}/do_{steps:08d}.pth",
                                    {'discriminator': discriminator.module.state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                     'steps': steps, 'epoch': epoch})

                if torch.isnan(loss_gen_all).any():
                    raise ValueError("NaN values found in loss_gen_all")

                if steps % cfg['env_setting']['validation_interval'] == 0 and steps != 0:
                    generator.eval(); torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    mag_t = pha_t = com_t = 0
                    with torch.no_grad():
                        for j, vb in enumerate(validation_loader):
                            (v_audio, v_mag, v_pha, v_com, v_pm, v_pp, v_rm, v_rp) = to_dev(vb, device)
                            vmag_g, vpha_g, vcom_g, _ = generator(v_pm, v_pp, v_rm, v_rp)
                            v_audio_g = mag_phase_istft(vmag_g, vpha_g, n_fft, hop_size, win_size, compress_factor)
                            audios_r += torch.split(v_audio, 1, dim=0)
                            audios_g += torch.split(v_audio_g, 1, dim=0)
                            mag_t += F.mse_loss(v_mag, vmag_g).item()
                            a, b, c = phase_losses(v_pha, vpha_g, cfg)
                            pha_t += (a + b + c).item()
                            com_t += F.mse_loss(v_com, vcom_g).item()
                        val_pesq = pesq_score(audios_r, audios_g, cfg).item()
                        sw.add_scalar("Validation/PESQ Score", val_pesq, steps)
                        sw.add_scalar("Validation/Magnitude Loss", mag_t / (j + 1), steps)
                        sw.add_scalar("Validation/Phase Loss", pha_t / (j + 1), steps)
                        sw.add_scalar("Validation/Complex Loss", com_t / (j + 1), steps)
                    generator.train()
                    if val_pesq >= best_pesq:
                        best_pesq, best_pesq_step = val_pesq, steps
                    print(f"valid: PESQ {val_pesq:.3f}, Mag {mag_t/(j+1):.4f}, Pha {pha_t/(j+1):.4f}. "
                          f"Best PESQ {best_pesq:.3f} at step {best_pesq_step}")
            steps += 1

        sched_g.step(); sched_d.step()
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_folder', default='results')
    p.add_argument('--exp_name', default='dualmic')
    p.add_argument('--config', default='configs/finetune_a100.yaml')
    p.add_argument('--base_ckpt', default='checkpoints/g_00017000.pth')
    p.add_argument('--freeze_steps', type=int, default=1000,
                   help='steps to keep TF-xLSTM + decoders frozen (0 disables)')
    p.add_argument('--train_primary_json', default=None, help='bone-conduction wav list; None = simulated')
    p.add_argument('--valid_primary_json', default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    initialize_seed(cfg['env_setting']['seed'])
    avail = torch.cuda.device_count()
    if cfg['env_setting']['num_gpus'] > avail:
        warnings.warn(f"num_gpus in config > available ({avail}); resetting.")
        cfg['env_setting']['num_gpus'] = avail
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, 'config.yaml', args.exp_path)
    if torch.cuda.is_available():
        print_gpu_info(torch.cuda.device_count(), cfg)
    train(int(os.environ['LOCAL_RANK']), args, cfg)


if __name__ == '__main__':
    main()
