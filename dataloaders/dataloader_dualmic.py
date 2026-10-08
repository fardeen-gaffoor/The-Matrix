import random
import numpy as np
import torch
import torch.utils.data
import librosa
from scipy.signal import butter, sosfiltfilt

from src.stfts import mag_phase_stft
from dataloaders.dataloader_vctk import load_json_file, extract_identifier


class DualMicDataset(torch.utils.data.Dataset):
    """
    Returns:
      clean_audio, clean_mag, clean_pha, clean_com,
      primary_mag, primary_pha, reference_mag, reference_pha

    clean_json     : clean target
    reference_json : noisy ambient mic
    primary_json   : bone-conduction mic. If None, a SIMULATED bone channel is
                     built (low-passed clean + heavily attenuated noise).
                     PIPELINE TESTING ONLY - not valid for reported results.
    """
    def __init__(self, clean_json, reference_json, primary_json=None,
                 sampling_rate=16000, segment_size=32000,
                 n_fft=400, hop_size=100, win_size=400, compress_factor=1.0,
                 split=True, shuffle=True,
                 sim_bone_cutoff=2000, sim_bone_leak_db=25):
        self.clean_dict = {extract_identifier(p): p for p in load_json_file(clean_json)}
        self.ref_paths = load_json_file(reference_json)
        self.primary_dict = None
        if primary_json is not None:
            self.primary_dict = {extract_identifier(p): p for p in load_json_file(primary_json)}
        random.seed(1234)
        if shuffle:
            random.shuffle(self.ref_paths)

        self.sr = sampling_rate
        self.segment_size = segment_size
        self.n_fft, self.hop_size, self.win_size = n_fft, hop_size, win_size
        self.compress_factor = compress_factor
        self.split = split
        self.leak = 10 ** (-sim_bone_leak_db / 20)
        self.sos = butter(4, sim_bone_cutoff, btype='low', fs=sampling_rate, output='sos')

    @staticmethod
    def _lookup(d, ident, name):
        if ident not in d:
            raise KeyError(f"{ident} not found in {name} list")
        return d[ident]

    def __getitem__(self, index):
        ref_path = self.ref_paths[index]
        ident = extract_identifier(ref_path)
        clean_path = self._lookup(self.clean_dict, ident, 'clean')

        ref, _ = librosa.load(ref_path, sr=self.sr)
        clean, _ = librosa.load(clean_path, sr=self.sr)

        if self.primary_dict is not None:
            prim, _ = librosa.load(self._lookup(self.primary_dict, ident, 'primary'), sr=self.sr)
        else:
            n = min(len(clean), len(ref))
            noise = ref[:n] - clean[:n]
            prim = (sosfiltfilt(self.sos, clean[:n])
                    + self.leak * sosfiltfilt(self.sos, noise)).astype(np.float32)

        # align lengths across all three signals
        n = min(len(clean), len(ref), len(prim))
        clean, ref, prim = clean[:n], ref[:n], prim[:n]

        clean = torch.FloatTensor(clean)
        ref = torch.FloatTensor(ref)
        prim = torch.FloatTensor(prim)

        # ONE norm factor (from the reference mic) for all channels,
        # so the inter-channel level difference (ILD) is preserved
        norm = torch.sqrt(len(ref) / (torch.sum(ref ** 2.0) + 1e-12))
        clean = (clean * norm).unsqueeze(0)
        ref = (ref * norm).unsqueeze(0)
        prim = (prim * norm).unsqueeze(0)

        if self.split:
            L = clean.size(1)
            if L >= self.segment_size:
                s = random.randint(0, L - self.segment_size)
                clean = clean[:, s:s + self.segment_size]
                ref = ref[:, s:s + self.segment_size]
                prim = prim[:, s:s + self.segment_size]
            else:
                pad = (0, self.segment_size - L)
                clean = torch.nn.functional.pad(clean, pad, 'constant')
                ref = torch.nn.functional.pad(ref, pad, 'constant')
                prim = torch.nn.functional.pad(prim, pad, 'constant')

        args = (self.n_fft, self.hop_size, self.win_size, self.compress_factor)
        clean_mag, clean_pha, clean_com = mag_phase_stft(clean, *args)
        prim_mag, prim_pha, _ = mag_phase_stft(prim, *args)
        ref_mag, ref_pha, _ = mag_phase_stft(ref, *args)

        return (clean.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze(),
                prim_mag.squeeze(), prim_pha.squeeze(), ref_mag.squeeze(), ref_pha.squeeze())

    def __len__(self):
        return len(self.ref_paths)
