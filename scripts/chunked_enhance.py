import os, torch
import torch.nn.functional as F

def chunked_enhance(model, wav, stft, istft, n_fft, hop, win, cf, overlap=3200):
    chunk = int(os.environ.get('CHUNK', 32000))
    dev = wav.device
    N = wav.shape[1]
    out = torch.zeros(1, N, device=dev)
    wsum = torch.zeros(1, N, device=dev)
    step = chunk - overlap
    with torch.no_grad():
        for s0 in range(0, max(N - overlap, 1), step):
            seg = wav[:, s0:s0 + chunk]
            L = seg.shape[1]
            if L < n_fft:
                seg = F.pad(seg, (0, n_fft - L))
            a, p, _ = stft(seg, n_fft, hop, win, cf)
            ag, pg, _ = model(a, p)
            y = istft(ag, pg, n_fft, hop, win, cf).reshape(1, -1)
            y = y[:, :L] if y.shape[1] >= L else F.pad(y, (0, L - y.shape[1]))
            w = torch.ones(L, device=dev)
            r = min(overlap, L)
            if s0 > 0:
                w[:r] = torch.linspace(0, 1, r, device=dev)
            if s0 + chunk < N:
                w[L - r:] = torch.minimum(w[L - r:], torch.linspace(1, 0, r, device=dev))
            out[:, s0:s0 + L] += y * w
            wsum[:, s0:s0 + L] += w
    torch.cuda.empty_cache()
    return out / wsum.clamp_min(1e-8)
