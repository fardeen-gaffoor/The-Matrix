# Architecture: xLSTM-SENet

This project builds upon the **xLSTM-SENet** model for single-channel speech enhancement. The original model operates on complex spectra, utilizing a Dual-Path topology with full-band and sub-band processing to handle non-stationary noise without distorting phase.

## xLSTM-SENet core model

The architecture translates the noisy audio into a complex-domain STFT representation and processes it with:
- **DenseEncoder**: Projects the concatenated magnitude and phase into the feature dimension.
- **TFxLSTM Blocks**: Time-Frequency xLSTM (extended LSTM) blocks that process features both across time frames and frequency bins. Each block uses the mLSTM (matrix LSTM) cell from Vision-LSTM.
- **Decoders**: Separate `MagDecoder` and `PhaseDecoder` modules reconstruct the denoised magnitude and phase, which are recombined via inverse STFT.

### Loss Functions

The model is trained to minimize a combination of multiple losses:
1. **Magnitude Loss (L2)**: Error between the predicted and clean magnitude spectrogram.
2. **Phase Losses (L1)**: In-phase loss, gradient delay loss, and integrated absolute frequency loss.
3. **Complex Loss (L2)**: Error in the combined complex representation.
4. **Time Loss (L1)**: Waveform domain L1 loss.
5. **Metric Loss**: A learned MetricDiscriminator that acts as an adversarial discriminator predicting PESQ scores.
6. **Consistency Loss**: Consistency between magnitude and phase.

*Note: In the original codebase SI-SNR is tracked but the actual loss function primarily uses the L1/L2 and Metric components defined above.*

## Training Pipeline

1. **Pretraining**: 
   - 4-layer TFxLSTM model pre-trained on VoiceBank+DEMAND to establish baseline speech enhancement.
   - Initial configs use batch size 1/2 depending on VRAM.
2. **Fine-tuning on Defence Noise**:
   - The model depth is expanded (xLSTM-SENet2 architecture specifies up to 8 blocks, expansion factor 2).
   - Fine-tuned using `scripts/train.py` on the custom defence mixture (gunshots, artillery, helicopters, drones).
