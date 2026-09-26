"""
Benchmark peak GPU memory + step time for one xLSTM-SENet training step,
at a chosen batch size and segment (clip) length.
"""
import argparse, time, sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from src.generator import xLSTMSENet
from src.discriminator import MetricDiscriminator
from src.stfts import mag_phase_stft, mag_phase_istft
from utils.util import load_config  # matches train.py's import; adjust if it errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--batch_size', type=int, required=True)
    ap.add_argument('--segment_size', type=int, required=True)
    ap.add_argument('--steps', type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda:0')

    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_size = cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']

    generator = xLSTMSENet(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)

    n_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator params: {n_params/1e6:.2f}M")

    optim_g = torch.optim.AdamW(generator.parameters(), lr=cfg['training_cfg']['learning_rate'],
                                 betas=(cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2']))
    optim_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg['training_cfg']['learning_rate'],
                                 betas=(cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2']))

    B, L = args.batch_size, args.segment_size

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    times = []
    for step in range(args.steps):
        clean = (torch.randn(B, L, device=device) * 0.1)
        noisy = clean + torch.randn(B, L, device=device) * 0.1

        t0 = time.time()

        clean_amp, clean_pha, clean_com = mag_phase_stft(clean, n_fft, hop_size, win_size, compress_factor)
        noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy, n_fft, hop_size, win_size, compress_factor)

        mag_g, pha_g, com_g = generator(noisy_amp, noisy_pha)
        audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)

        optim_d.zero_grad()
        real_out = discriminator(clean_amp, clean_amp)
        fake_out = discriminator(clean_amp, mag_g.detach())
        loss_d = F.mse_loss(real_out, torch.ones_like(real_out)) + F.mse_loss(fake_out, torch.zeros_like(fake_out))
        loss_d.backward()
        optim_d.step()

        optim_g.zero_grad()
        fake_out_g = discriminator(clean_amp, mag_g)
        loss_metric = F.mse_loss(fake_out_g, torch.ones_like(fake_out_g))
        loss_mag = F.mse_loss(clean_amp, mag_g)
        n = min(audio_g.shape[-1], clean.shape[-1])
        loss_time = F.l1_loss(audio_g[..., :n], clean[..., :n])
        loss_g = loss_metric * 0.05 + loss_mag * 0.9 + loss_time * 0.2
        loss_g.backward()
        optim_g.step()

        torch.cuda.synchronize()
        times.append(time.time() - t0)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2
    avg_step = sum(times[1:]) / max(1, len(times) - 1) if len(times) > 1 else times[0]

    print(f"\nbatch_size={B}  segment_size={L} ({L/16000:.2f}s)")
    print(f"peak allocated: {peak_mb:.0f} MB   peak reserved: {peak_reserved_mb:.0f} MB")
    print(f"avg step time (post-warmup): {avg_step:.3f} s")
    print(f"est. epoch time @ 11572 clips: {11572/B*avg_step/3600:.2f} h")


if __name__ == '__main__':
    main()
