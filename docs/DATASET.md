# Dataset

This project utilizes a mixture of clean speech and various noise sources to train and evaluate the Defence ANC system. 

> **Important**: Raw audio files are **not** committed to this repository due to size constraints and licensing. You must download the sources independently.

## Speech Source
- **VoiceBank+DEMAND (Clean Speech)**: The clean speech utterances used for both training and testing are derived from the standard VoiceBank corpus.

## Noise Sources (The "Defence Mixture")
To simulate a battlefield and mission-critical environment, noise was sourced from the following datasets:
1. **Cadre Forensics** (Gunshots)
2. **MAD Dataset** (Military Aircraft, Fighter Jets, Artillery)
3. **Al-Emadi Drone Recordings** (UAV and Drone noise)
4. **ESC-50** (Sirens, Wind, Engine noise)
5. **FSD50K** (General explosive and mechanical impacts)
6. **DEMAND** (Traffic, Field, Background)

## Mixture Generation (`scripts/mix_defence.py`)

The `mix_defence.py` script automatically mixes the clean speech with random cuts from the noise directories. 

- **SNR Range (Training)**: -5 dB to +15 dB.
- **SNR Range (Testing)**: 0 dB to +15 dB.
- **Impulsive Shift**: Impulsive noises (e.g., `gunshot`, `explosion`) are intentionally mixed at lower relative SNRs during training to force the model to handle extreme instantaneous peaks.
- **Augmentations**: Random scaling, clipping prevention, and concurrent noise mixing (50% chance to layer two different noise categories together).

## Licenses
Each dataset carries its own license. If you are reproducing this work, ensure you comply with the respective Creative Commons or Academic licenses of VoiceBank, ESC-50, FSD50K, and the MAD dataset.
