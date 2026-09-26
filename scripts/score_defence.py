import os, csv, argparse, collections
import numpy as np, soundfile as sf
from pesq import pesq
from pystoi import stoi
ap = argparse.ArgumentParser()
ap.add_argument('--enh', required=True)
ap.add_argument('--noisy', default='data/defence16/noisy_test')
ap.add_argument('--clean', default='data/defence16/clean_test')
ap.add_argument('--meta', default='data/defence16/meta_test.csv')
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()

def metrics(c, e):
    n = min(len(c), len(e)); c, e = c[:n], e[:n]
    try: p = pesq(16000, c, e, 'wb')
    except Exception: p = np.nan
    s = 10 * np.log10((c ** 2).sum() / (((c - e) ** 2).sum() + 1e-12) + 1e-12)
    return p, stoi(c, e, 16000), s

rows = list(csv.DictReader(open(a.meta)))
if a.limit: rows = rows[:a.limit]
res = []
for r in rows:
    f = r['file']
    c, _ = sf.read(os.path.join(a.clean, f))
    y, _ = sf.read(os.path.join(a.noisy, f))
    e, _ = sf.read(os.path.join(a.enh, f))
    res.append((float(r['snr_db']), r['noise'], metrics(c, y), metrics(c, e)))

def show(title, groups):
    print('\n' + title)
    print(f'{"group":<24}{"n":>5} | {"PESQ in":>8}{"out":>6} | {"STOI in":>8}{"out":>7} | {"SNR in":>7}{"out":>6}')
    for k, items in groups:
        m = lambda i, j: np.nanmean([x[i][j] for x in items])
        print(f'{k:<24}{len(items):>5} | {m(2,0):>8.2f}{m(3,0):>6.2f} | {m(2,1):>8.3f}{m(3,1):>7.3f} | {m(2,2):>7.1f}{m(3,2):>6.1f}')

show('OVERALL (targets: PESQ>2.5, STOI>0.85, SNR>15 dB)', [('all', res)])
b = collections.defaultdict(list)
for x in res:
    lo = int(min(x[0], 14.99) // 5 * 5); b[f'input SNR {lo}-{lo+5} dB'].append(x)
show('BY INPUT SNR', sorted(b.items()))
c = collections.defaultdict(list)
for x in res:
    for k in set(x[1].split('+')): c[k].append(x)
show('BY NOISE CATEGORY (stacked clips count in each)', sorted(c.items()))
