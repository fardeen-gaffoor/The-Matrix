# Checkpoints

Model weights are not stored in the git history due to their size.

## Download Instructions

1. Download the `ft_test2.pth` fine-tuned checkpoint from the [GitHub Releases](../../releases) page.
2. Place the file in this directory.

```bash
# Expected structure:
checkpoints/
├── README.md
├── ft_test2.pth
└── xLSTM_checkpoint.pth (optional, for starting from upstream baseline)
```

## Upstream Checkpoints
If you wish to fine-tune from the original xLSTM-SENet pretrained baseline, download `xLSTM_checkpoint.pth` from the upstream repository or the provided release assets.
