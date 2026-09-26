# Project Handoff

This document describes the contents of the packaged handoff bundle for collaboration.

## Bundle Contents

The bundle you received (e.g., via shared drive or zip file) should contain:
1. **The codebase**: A clone of this repository.
2. **Model Weights**: Inside the `checkpoints/` directory, you should have `ft_test2.pth` (the fine-tuned defence model).
3. **Dataset (Optional)**: If you received the data bundle, the `data/` folder will contain the `clean_test`, `noisy_test`, `clean_train`, and `noisy_train` directories with `.wav` files. If not, only the JSON file lists are included.

## Getting Started

1. Set up a Python environment (Python 3.10+, CUDA 12.0+).
2. Install standard dependencies: `pip install -r requirements.txt`.
3. Build the `mamba_ssm` extension:
   ```bash
   cd mamba_install
   pip install .
   ```
4. Test inference using the provided checkpoint to ensure your environment is working:
   ```bash
   python scripts/inference.py \
       --checkpoint_file checkpoints/ft_test2.pth \
       --input_folder data/defence16/noisy_test/ \
       --output_folder out/ \
       --config configs/xLSTM-SENet_4N.yaml
   ```

Please see the `docs/` folder for architectural details, training instructions, and deployment plans.
