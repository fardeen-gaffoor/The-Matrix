# Installation & Setup

## Prerequisites
- **OS**: Linux / Windows via WSL2
- **Hardware**: NVIDIA GPU (Required: CUDA architecture >= 8.0, ideally 8.6+). Currently, CPU mode is not supported.
- **CUDA**: 12.0+
- **Python**: 3.10+

## 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate
# Or on Windows WSL2:
# source venv/Scripts/activate

pip install -r requirements.txt
```

## 2. Compiling Mamba (mamba_ssm)

The `mamba_ssm` module must be built from source for your specific GPU architecture.

```bash
cd mamba_install
pip install .
```

*Note: Ensure your `nvcc` compiler matches the PyTorch CUDA version. You can check this with `nvcc --version` and `python -c "import torch; print(torch.version.cuda)"`.*

## Troubleshooting

### WSL2 Disk Filling Up
If building `mamba_ssm` runs out of space, WSL2 may have expanded its virtual disk (`ext4.vhdx`). 
- **Fix**: Run `wsl --shutdown`, then compact the disk using `diskpart` in Windows.
- Make sure to clear your `pip` cache: `pip cache purge`.

### Validation OOM (Out Of Memory)
If validation crashes due to OOM:
- **Fix**: Reduce validation batch size in `scripts/train.py`, or ensure `torch.cuda.empty_cache()` is fully completing. The current configs use a validation batch size of 1 for safety.

### mamba_ssm Rebuild Fails for GPU Architecture
If the `mamba_ssm` wheel fails to build complaining about architectures (e.g. `sm_86` not found or not matching):
- **Fix**: Explicitly set the target architecture before building:
  ```bash
  export TORCH_CUDA_ARCH_LIST="8.6" # Set to your GPU's compute capability (e.g., 8.9 for Ada, 8.6 for Ampere)
  pip install .
  ```

### Kaggle Notes
When training on Kaggle, the environment comes pre-packaged with PyTorch. You will still need to run `pip install .` in `mamba_install/`. Because Kaggle T4 GPUs have compute capability 7.5, you may need to set `export TORCH_CUDA_ARCH_LIST="7.5"`.
