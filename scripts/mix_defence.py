import os, glob, csv, random, argparse, functools
import numpy as np, soundfile as sf, torch, torchaudio
p = argparse.ArgumentParser()
p.add_argument('--noise_dir', required=True)
p.add_argument('--clean_root', default='data/vctk16')
p.add_argument('--out_root', default='data/defence16')
p.add_argument('--train_snr', nargs=2, type=float, default=[-5, 15])
p.add_argument('--test_snr', nargs=2, type=float, default=[0, 15])
p.add_argument('--train_impulsive_shift', type=float, default=0)
p.add_argument('--splits', default='train,test')
a = p.parse_args()
random.seed(0); np.random.seed(0)
SR = 16000
IMPULSIVE = {'gunshot', 'gunshot_real', 'mad_gunshot', 'mad_shelling', 'explosion', 'fireworks'}

@functools.lru_cache(maxsize=32)
def load(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(1)
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x

cats = {}
for ext in ('wav', 'flac', 'ogg'):
    for f in glob.glob(os.path.join(a.noise_dir, '**', '*.' + ext), recursive=True):
        cats.setdefault(os.path.relpath(f, a.noise_dir).split(os.sep)[0], []).append(f)
pools = {'train': {}, 'test': {}}
for c, fl in cats.items():
    units = {}
    for f in sorted(fl):
        parts = os.path.relpath(f, a.noise_dir).split(os.sep)
        units.setdefault(parts[1] if len(parts) > 2 else f, []).append(f)
    keys = sorted(units); random.shuffle(keys)
    if len(keys) >= 5:
        k = max(1, round(0.1 * len(keys)))
        pools['test'][c] = [(f, 0, 1) for u in keys[:k] for f in units[u]]
        pools['train'][c] = [(f, 0, 1) for u in keys[k:] for f in units[u]]
    else:
        pools['test'][c] = [(f, 0.9, 1) for f in fl]
        pools['train'][c] = [(f, 0, 0.9) for f in fl]
print({c: len(v) for c, v in cats.items()})

def rms(x):
    return np.sqrt((x ** 2).mean() + 1e-12)

def draw(pool, L):
    c = random.choice(list(pool))
    f, lo, hi = random.choice(pool[c])
    x = load(f); x = x[int(lo * len(x)):int(hi * len(x))]
    if len(x) < L:
        x = np.tile(x, int(np.ceil(L / len(x))))
    s = random.randint(0, len(x) - L)
    return x[s:s + L], c

def mix(c, pool, lo, hi, shift=0):
    L = len(c)
    n, cat = draw(pool, L)
    if random.random() < 0.5:
        n2, cat2 = draw(pool, L)
        n = n + n2 * (rms(n) / rms(n2)) * 10 ** (-random.uniform(0, 6) / 20)
        cat += '+' + cat2
    snr = random.uniform(lo, hi) + (shift if any(k in IMPULSIVE for k in cat.split('+')) else 0)
    n = n * rms(c) / (rms(n) * 10 ** (snr / 20))
    y = c + n
    m = np.abs(y).max()
    if m > 0.99:
        g = 0.99 / m; c, y = c * g, y * g
    return c, y, snr, cat

for split, (lo, hi) in [('train', a.train_snr), ('test', a.test_snr)]:
    if split not in a.splits.split(','):
        continue
    src = os.path.join(a.clean_root, f'clean_{split}')
    for d in ('clean', 'noisy'):
        os.makedirs(os.path.join(a.out_root, f'{d}_{split}'), exist_ok=True)
    with open(os.path.join(a.out_root, f'meta_{split}.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['file', 'snr_db', 'noise'])
        for f in sorted(os.listdir(src)):
            c, _ = sf.read(os.path.join(src, f), dtype='float32')
            c, y, snr, cat = mix(c, pools[split], lo, hi, a.train_impulsive_shift if split == 'train' else 0)
            sf.write(os.path.join(a.out_root, f'clean_{split}', f), c, SR)
            sf.write(os.path.join(a.out_root, f'noisy_{split}', f), y, SR)
            w.writerow([f, round(snr, 2), cat])
    print(split, 'done')
