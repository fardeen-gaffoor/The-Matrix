from chunked_enhance import chunked_enhance
import glob
import argparse
import sys
import os
sys.path.insert(0, os.getcwd())
import json
import torch
import librosa
from src.stfts import mag_phase_stft, mag_phase_istft
from src.generator import xLSTMSENet
import soundfile as sf

from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed, 
    print_gpu_info, log_model_info, initialize_process_group,
)

h = None
device = None

def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = xLSTMSENet(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in state_dict['generator'].items()})

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    with torch.no_grad():
        # You can use data.json instead of input_folder with:
        # ---------------------------------------------------- #
        # with open("data/test_noisy.json", 'r') as json_file:
        #     test_files = json.load(json_file)
        # for i, fname in enumerate( test_files ): 
        #     folder_path = os.path.dirname(fname)
        #     fname = os.path.basename(fname)
        #     noisy_wav, _ = librosa.load(os.path.join( folder_path, fname ), sr=sampling_rate)
        #     noisy_wav = torch.FloatTensor(noisy_wav).to(device)
        # ---------------------------------------------------- #
        for i, fname in enumerate(os.listdir( args.input_folder )):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            audio_g = chunked_enhance(model, noisy_wav, mag_phase_stft, mag_phase_istft, n_fft, hop_size, win_size, compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(args.output_folder, fname)
            
            sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='vctk16/noisy_testset_wav_16k')
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--config', default='configs/xLSTM-SENet_4N.yaml')
    parser.add_argument('--checkpoint_file', required=True)
    args = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        #device = torch.device('cpu')
        raise RuntimeError("Currently, CPU mode is not supported.")
        

    inference(args, device)


if __name__ == '__main__':
    print("Initializing inference...")
    main()

