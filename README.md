# Defence ANC — xLSTM-SENet

**Clear speech through gunfire, rotors and sirens. AI noise cancellation built for mission-critical communication.**

[![License: MIT](https://img.shields.io/badge/License-MIT-2f6f4f.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-1f6feb.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-1f6feb.svg)](https://pytorch.org)
[![Model](https://img.shields.io/badge/Model-xLSTM--SENet-6f42c1.svg)](docs/ARCHITECTURE.md)
[![Edge](https://img.shields.io/badge/Target-Jetson%20AGX%20Orin-6f42c1.svg)](docs/DEPLOYMENT.md)

[Architecture](docs/ARCHITECTURE.md) · [Installation](docs/INSTALLATION.md) · [User Guide](docs/USER_GUIDE.md) · [Dataset](docs/DATASET.md) · [Results](docs/RESULTS.md) · [License](LICENSE)

> Built for **Smart India Hackathon 2026**.

---

## The problem

In defence and mission-critical communication, the moments when clear speech matters most are the moments the battlefield is loudest. Gunshots, artillery, helicopter rotors, armoured vehicles and sirens bury the voice on the other end of the radio.

Classical noise suppression (spectral subtraction, Wiener filtering, LMS-based ANC) assumes the noise is roughly stationary. Impulsive and rapidly changing defence noise breaks that assumption, so these methods leave residual noise or distort the speech they are trying to save.

## What this project does

Defence ANC is a speech-enhancement pipeline that fine-tunes a pretrained **xLSTM-SENet** model on defence-specific noise, then evaluates it per noise category so weak spots are visible instead of averaged away.

- **Handles stationary, non-stationary and impulsive noise.** Trained on VoiceBank+DEMAND, then fine-tuned on a purpose-built defence-noise mixture.
- **Preserves phase.** Operates on complex spectra with full-band and sub-band modelling, so speech stays natural and intelligible.
- **Measures what matters.** PESQ, STOI and SNR, scored per noise category (`scripts/score_defence.py`), with a separate *hard* test set.
- **Built for the edge.** The deployment path targets NVIDIA Jetson AGX Orin via ONNX / TensorRT, with an optional lightweight LMS stage for residual noise.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the model, loss functions and data flow.

## Pipeline at a glance

```
 Clean speech ──┐
 (VoiceBank)    ├─► Mixer (random SNR, reverb, clipping) ─► Noisy / clean pairs
 Defence noise ─┘
 (gunshot, drone, artillery, vehicle, wind, siren)
                                   │
                                   ▼
              STFT (complex) ─► xLSTM-SENet ─► Enhanced speech
                                   │
                    Loss: SI-SNR · L1/L2 · perceptual
                                   │
                                   ▼
                score_defence.py ─► PESQ · STOI · SNR per category
                                   │
                                   ▼
           ONNX ─► TensorRT ─► Jetson AGX Orin (mic + reference in, headset out)
```

| Stage            | Purpose                                    | Tooling                          |
| ---------------- | ------------------------------------------ | -------------------------------- |
| **Dataset**      | Build noisy/clean pairs at varied SNR      | Python · torchaudio              |
| **Training**     | Pretrain, then fine-tune on defence noise  | PyTorch · xLSTM · Mamba (SENet)  |
| **Inference**    | Enhance noisy audio files                  | PyTorch                          |
| **Scoring**      | Per-category PESQ / STOI / SNR             | `scripts/score_defence.py`       |
| **Deployment**   | Real-time edge inference                   | ONNX · TensorRT · Jetson         |

## Targets and results

| Metric | Target  | Normal test set | Hard test set  |
| ------ | ------- | --------------- | ---------------|
| SNR    | > 15 dB | ✅ met (18.8)   | ✅ met (16db) |
| STOI   | > 0.85  | ✅ met (0.943)  | ✅ met (0.90) |
| PESQ   | > 2.5   | ✅ met (3.08)   | ✅ met (2.55) |

The fine-tuned checkpoint `ft_test2` meets all targets on the normal test set. On the hard set, **impulsive military noise (`mad_*` categories)** remains the main open problem. Full per-category tables are in [docs/RESULTS.md](docs/RESULTS.md).

## Repository structure

```text
.
├── configs/          # training and fine-tuning configs
├── data/             # dataset scripts (no raw audio committed)
├── src/              # model, dataset, losses, training loop
├── scripts/          # inference, scoring, export, utilities
├── checkpoints/      # weights (Git LFS / Releases, see below)
├── docs/             # ARCHITECTURE, INSTALLATION, USER_GUIDE, DATASET, RESULTS, DEPLOYMENT
├── LICENSE
└── README.md
```

## Getting started

```bash
git clone https://github.com/fardeen-gaffoor/Defence-ANC.git
cd Defence-ANC
pip install -r requirements.txt
```

Pretrained and fine-tuned weights are hosted outside the git history. Download `ft_test2` from the [Releases](../../releases) page and place it in `checkpoints/`.

Full setup, including the `mamba_ssm` CUDA build, WSL2 notes and troubleshooting, is in [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Using it

**Enhance a file**

```bash
python scripts/inference.py --checkpoint_file checkpoints/ft_test2.pth --input_folder data/noisy/ --output_folder out/
```

**Score a folder of enhanced audio per noise category**

```bash
python scripts/score_defence.py --enh out/ --noisy data/noisy/ --clean data/clean/ --meta data/meta.csv
```

Inference and scoring are two separate steps: run inference first, then score the output folder. [docs/USER_GUIDE.md](docs/USER_GUIDE.md) walks through both, plus fine-tuning on your own noise.

## Datasets

Speech: **VoiceBank+DEMAND**. Defence noise is assembled from multiple sources: Cadre Forensics, the MAD dataset, Al-Emadi drone recordings, ESC-50, FSD50K and DEMAND. Raw audio is **not** redistributed here; see [docs/DATASET.md](docs/DATASET.md) for sources, licences and the build script.

## Known limitations and roadmap

- [ ] Improve suppression of impulsive `mad_*` noise on the hard test set
- [ ] Larger-batch training on Kaggle
- [ ] Export to ONNX and TensorRT, and benchmark latency on Jetson AGX Orin
- [ ] Optional LMS residual-noise stage with primary + reference microphones
- [ ] Live headset demo

## Built with

- [xLSTM-SENet](https://github.com/NikolaiKyhne/xLSTM-SENet) — pretrained speech-enhancement model by Nikolai Lund Kühne, Jan Østergaard, Jesper Jensen, and Zheng-Hua Tan ([Interspeech 2025 paper](https://www.isca-archive.org/interspeech_2025/kuhne25_interspeech.html)).
- PyTorch, torchaudio, `mamba_ssm`
- PESQ, STOI for evaluation
- ONNX, TensorRT for edge deployment

## Team
Team Leader: 
Naveen Srinath S
Team Members:
Kamaleshwaran S
Syed samuideen 
D R Netaranjan Chowdary 
Fardeen Gaffoor S
## License

MIT, see [LICENSE](LICENSE). The upstream xLSTM-SENet code, pretrained weights and every dataset listed above carry their own licences; check them before redistributing weights or data.
