"""
generator_dualmic.py
---------------------
Dual-mic version of src/generator.py, built from your ACTUAL repo classes
(DenseEncoder, TFxLSTMBlock, MagDecoder, PhaseDecoder) -- not placeholders.

Key fact confirmed from your real code: xLSTMSENet.forward() takes
noisy_mag and noisy_pha as SEPARATE pre-computed tensors [B, F, T] --
STFT already happened before this class is called. That means your
Primary/Reference/Spatial encoders need matching mag+phase pairs as
input, computed the same way your existing training pipeline already
computes noisy_mag/noisy_pha for the single-channel case.

Drop this file into src/ alongside generator.py.
"""

import torch
import torch.nn as nn
from einops import rearrange
from .xlstm_block import TFxLSTMBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder


class SpatialFeatureExtractor(nn.Module):
    """
    Computes IPD (phase difference) and ILD (level difference) between
    primary and reference mag/phase, then encodes them with the SAME
    DenseEncoder architecture as the other two streams -- so all three
    streams arrive at the fusion stage with matching shapes.
    """
    def __init__(self, cfg):
        super().__init__()
        # IPD + ILD = 2 "channels", matching the shape DenseEncoder expects
        # (it currently expects cfg['model_cfg']['input_channel'] channels,
        # which is 2 for mag+phase -- reuse that same expectation here)
        self.encoder = DenseEncoder(cfg)

    def forward(self, primary_mag, primary_pha, reference_mag, reference_pha):
        """
        All inputs: [B, F, T]
        """
        ipd = reference_pha - primary_pha
        ipd = torch.remainder(ipd + torch.pi, 2 * torch.pi) - torch.pi  # wrap to [-pi, pi]

        ild = torch.log(reference_mag.clamp(min=1e-8)) - \
              torch.log(primary_mag.clamp(min=1e-8))

        ipd = rearrange(ipd, 'b f t -> b t f').unsqueeze(1)   # [B, 1, T, F]
        ild = rearrange(ild, 'b f t -> b t f').unsqueeze(1)   # [B, 1, T, F]
        x = torch.cat((ipd, ild), dim=1)                       # [B, 2, T, F]

        return self.encoder(x)   # [B, hid_feature, T, F//2]


class GatedFusion(nn.Module):
    """
    Stream Normalization -> Softmax Weight Generation -> Gated Fusion,
    matching your slide's Multi-Encoder Gated Fusion box exactly.
    """
    def __init__(self, hid_feature):
        super().__init__()
        self.hid_feature = hid_feature
        # LayerNorm over the channel dimension for each stream individually
        self.norm_primary = nn.GroupNorm(1, hid_feature)
        self.norm_reference = nn.GroupNorm(1, hid_feature)
        self.norm_spatial = nn.GroupNorm(1, hid_feature)

        # Produces 3 weights per (batch, time, freq) position -- one per stream
        self.weight_gen = nn.Conv2d(hid_feature * 3, 3, kernel_size=1)

    def forward(self, primary_feat, reference_feat, spatial_feat):
        # --- Stream Normalization ---
        p = self.norm_primary(primary_feat)
        r = self.norm_reference(reference_feat)
        s = self.norm_spatial(spatial_feat)

        # --- Softmax Weight Generation ---
        stacked = torch.cat([p, r, s], dim=1)          # [B, 3*hid_feature, T, F]
        weights = self.weight_gen(stacked)               # [B, 3, T, F]
        weights = torch.softmax(weights, dim=1)           # sums to 1 across the 3 streams

        w_p = weights[:, 0:1, :, :]
        w_r = weights[:, 1:2, :, :]
        w_s = weights[:, 2:3, :, :]

        # --- Gated Fusion (weighted sum of three streams) ---
        fused = w_p * p + w_r * r + w_s * s
        return fused, weights  # return weights too -- useful to inspect/log later


class xLSTMSENetDualMic(nn.Module):
    """
    Dual-mic xLSTM-SENet: Primary Encoder + Reference Encoder + Spatial
    Encoder -> Gated Fusion -> existing (unmodified) TFxLSTMBlock stack
    -> existing (unmodified) MagDecoder + PhaseDecoder.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg']['num_tfxlstm'] if cfg['model_cfg']['num_tfxlstm'] is not None else 4

        # Three encoders -- Primary and Reference reuse DenseEncoder as-is,
        # Spatial uses the new SpatialFeatureExtractor above
        self.primary_encoder = DenseEncoder(cfg)
        self.reference_encoder = DenseEncoder(cfg)
        self.spatial_encoder = SpatialFeatureExtractor(cfg)

        self.fusion = GatedFusion(hid_feature=cfg['model_cfg']['hid_feature'])

        # UNCHANGED from the base model
        self.TSxLSTM = nn.ModuleList([TFxLSTMBlock(cfg) for _ in range(self.num_tscblocks)])
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, primary_mag, primary_pha, reference_mag, reference_pha):
        """
        primary_mag, primary_pha, reference_mag, reference_pha: [B, F, T] each

        Returns same outputs as the base model:
        denoised_mag, denoised_pha, denoised_com, fusion_weights
        """
        # Reshape primary + reference into [B, 2, T, F] the same way the
        # base model already does for its single noisy input
        p_mag = rearrange(primary_mag, 'b f t -> b t f').unsqueeze(1)
        p_pha = rearrange(primary_pha, 'b f t -> b t f').unsqueeze(1)
        primary_x = torch.cat((p_mag, p_pha), dim=1)        # [B, 2, T, F]

        r_mag = rearrange(reference_mag, 'b f t -> b t f').unsqueeze(1)
        r_pha = rearrange(reference_pha, 'b f t -> b t f').unsqueeze(1)
        reference_x = torch.cat((r_mag, r_pha), dim=1)        # [B, 2, T, F]

        primary_feat = self.primary_encoder(primary_x)
        reference_feat = self.reference_encoder(reference_x)
        spatial_feat = self.spatial_encoder(primary_mag, primary_pha, reference_mag, reference_pha)

        x, fusion_weights = self.fusion(primary_feat, reference_feat, spatial_feat)

        # UNCHANGED from the base model from this point on
        for block in self.TSxLSTM:
            x = block(x)

        denoised_mag = rearrange(self.mask_decoder(x) * reference_mag.unsqueeze(1).transpose(2, 3),
                                   'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com, fusion_weights
