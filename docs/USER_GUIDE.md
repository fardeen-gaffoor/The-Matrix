# User Guide

This guide covers how to run inference to enhance audio, how to score the outputs, and how to fine-tune the model on your own dataset.

## 1. Enhancing Audio (Inference)

Use `scripts/inference.py` to denoise a folder of `.wav` files. 

```bash
python scripts/inference.py \
    --checkpoint_file checkpoints/ft_test2.pth \
    --input_folder data/defence16/noisy_test/ \
    --output_folder out/ \
    --config configs/xLSTM-SENet_4N.yaml
```

**Parameters:**
- `--checkpoint_file`: Path to the `.pth` weights.
- `--input_folder`: Directory containing 16kHz noisy `.wav` files.
- `--output_folder`: Directory where enhanced audio will be saved.
- `--config`: The YAML config matching the checkpoint architecture.

*Note: Large audio files are automatically processed in overlapping chunks to prevent Out-Of-Memory errors.*

## 2. Scoring

Once you have generated the enhanced audio, use `scripts/score_defence.py` to evaluate the PESQ, STOI, and SNR metrics.

```bash
python scripts/score_defence.py \
    --enh out/ \
    --noisy data/defence16/noisy_test/ \
    --clean data/defence16/clean_test/ \
    --meta data/defence16/meta_test.csv
```

The script outputs three tables:
1. **Overall average** across all files.
2. **By Input SNR**: Grouped in 5dB brackets (0-5 dB, 5-10 dB, etc.).
3. **By Noise Category**: Scores separated by gunshot, drone, helicopter, siren, etc.

## 3. Fine-tuning on Custom Noise

### Step A: Dataset Preparation
1. Place your clean speech (16kHz) in `data/clean_train/`.
2. Place your target noise (16kHz) in a directory (e.g., `data/my_noises/`).
3. Mix them using `scripts/mix_defence.py`:
   ```bash
   python scripts/mix_defence.py \
       --noise_dir data/my_noises/ \
       --clean_root data/ \
       --out_root data/custom_mix/ \
       --train_snr -5 15 \
       --test_snr 0 15
   ```
4. Generate the JSON list files:
   ```bash
   python scripts/make_dataset_json.py --prefix_path data/custom_mix/
   ```

### Step B: Training
Update `configs/finetune_8N.yaml` to point `train_clean_json` and `train_noisy_json` to your new lists.

Run the training script via `torchrun`:
```bash
NCCL_P2P_DISABLE=1 torchrun --nnodes=1 --nproc-per-node=1 scripts/train.py \
    --exp_name=my_finetune \
    --exp_folder=results/ \
    --config=configs/finetune_8N.yaml \
    --init_generator=checkpoints/xLSTM_checkpoint.pth
```
