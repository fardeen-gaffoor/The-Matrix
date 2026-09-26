# Architecture

This document describes **TM-DC-xLSTM-SENet2**, the dual-microphone speech-enhancement model in this repository. It is built on the published **xLSTM-SENet2** backbone (a time-frequency xLSTM speech enhancer) and extends it with a reference microphone, spatial cues, gated stream fusion, and auxiliary multi-task heads for defence noise.

[← Back to README](../README.md) · [Installation](INSTALLATION.md) · [User Guide](USER_GUIDE.md) · [Dataset](DATASET.md) · [Results](RESULTS.md)

## Contents

1. [Design goals](#design-goals)
2. [Pipeline at a glance](#pipeline-at-a-glance)
3. [Stage by stage](#stage-by-stage)
4. [Loss functions](#loss-functions)
5. [Training recipe](#training-recipe)
6. [Configuration summary](#configuration-summary)
7. [Real-time considerations](#real-time-considerations)
8. [What is original and what is extended](#what-is-original-and-what-is-extended)

---

## Design goals

| Goal | How the architecture addresses it |
| --- | --- |
| Suppress impulsive noise (gunshots, artillery) | Transient-detection auxiliary head, exponential-gated mLSTM that can react quickly to sudden changes |
| Suppress non-stationary noise (rotors, vehicles, sirens) | Long-range time and frequency modelling with matrix-memory recurrence |
| Preserve speech intelligibility | Complex-domain processing with an explicit phase decoder, plus perceptual (PESQ) loss |
| Exploit a second microphone | Reference-mic encoder plus spatial cues (IPD, ILD, coherence), fused by a learned gate |
| Run on edge hardware | Frequency halving before the recurrent core, parallelisable mLSTM cells |

---

## Pipeline at a glance

```
Primary mic ──┐
              ├─► STFT + magnitude compression ─► CNN Encoder (E_p) ──┐
Reference mic ┘                                                       │
     │                                                                ├─► Gated Fusion
     └─► IPD / ILD / Coherence ─► CNN Encoder (E_s) ─────────────────┤        │
                                  Reference encoder (E_r) ───────────┘        ▼
                                                              8 × [ Time-mLSTM + Freq-mLSTM ]
                                                              (bidirectional, exponential gates,
                                                               matrix memory)
                                                                              │
                                        ┌─────────────────────────────────────┼──────────────────────┐
                                        ▼                                     ▼                      ▼
                                 Magnitude-mask decoder                Phase decoder          5 auxiliary heads
                                        │                                     │               (side losses only)
                                        └──────────────┬──────────────────────┘
                                                       ▼
                                        Reconstruct complex spectrum ─► iSTFT ─► Enhanced audio
```

The five auxiliary heads only shape the shared representation during training. They do not feed the enhanced output.

---

## Stage by stage

### 1. Input

Two synchronised signals at 16 kHz:

- **Primary mic** $y_p(t)$: speaker voice plus noise
- **Reference mic** $y_r(t)$: mostly ambient noise

Training uses 4.0 s segments (64,000 samples per channel). Real-time inference uses a rolling buffer of shorter chunks (see [Real-time considerations](#real-time-considerations)).

### 2. STFT and magnitude compression

Each channel is transformed independently:

$$Y_p, Y_r \in \mathbb{C}^{T \times F}$$

| Parameter | Value |
| --- | --- |
| FFT size | 400 |
| Hop size | 100 samples (6.25 ms) |
| Window | Hann |
| Frequency bins | $F = 400/2 + 1 = 201$ |

Each spectrum is split into magnitude and wrapped phase, $Y = Y_m \, e^{j Y_\phi}$. Power-law compression is applied to the magnitude:

$$(Y_m)^c, \quad c = 0.3$$

This reduces the dynamic range between quiet and loud frames, so the network is not dominated by large amplitude swings. It is applied separately to both channels.

### 3. Spatial cue extraction (dual-mic addition)

Because the two microphones are physically separated, their difference carries directional information a single-mic model never sees. Three cues are computed per time-frequency bin:

**Inter-microphone phase difference (IPD)**

$$\text{IPD}(t,f) = \angle Y_p(t,f) - \angle Y_r(t,f)$$

**Inter-microphone level difference (ILD)**

$$\text{ILD}(t,f) = 20\log_{10}\left(\frac{|Y_p(t,f)|}{|Y_r(t,f)| + \epsilon}\right)$$

**Coherence** (noise from many directions decorrelates faster than a direct speech path)

$$\Gamma(t,f) = \frac{|E[Y_p(t,f)\, Y_r^*(t,f)]|}{\sqrt{E[|Y_p(t,f)|^2]\,E[|Y_r(t,f)|^2]}}$$

The three maps $\text{IPD}, \text{ILD}, \Gamma \in \mathbb{R}^{T\times F}$ are stacked and passed through a small CNN, the **spatial cue extractor**, producing $S \in \mathbb{R}^{T\times F\times C}$.

> The expectation $E[\cdot]$ in the coherence is estimated with recursive or windowed averaging over neighbouring frames. Document the smoothing constant used in your config.

### 4. Three parallel CNN encoders

Following the original feature-encoder design, there are three encoder branches:

$$E_p = \text{Encoder}_p\big([(Y_{p,m})^c,\; Y_{p,\phi}]\big)$$
$$E_r = \text{Encoder}_r\big([(Y_{r,m})^c,\; Y_{r,\phi}]\big)$$
$$E_s = \text{Encoder}_s(S)$$

Each output has shape $\mathbb{R}^{T\times F'\times C}$. Every encoder is:

```
Conv2D → InstanceNorm → PReLU → Dilated DenseNet (dilations 1, 2, 4, 8) → Conv2D (F → F' = F/2)
```

The dilated DenseNet widens the temporal receptive field. Halving frequency resolution reduces the cost of the recurrent stage that follows. Channel width is **C = 96** (widened from 64 in the original).

### 5. Gated fusion

A learned softmax gate decides, per position, how much to trust each stream:

$$[\alpha_p, \alpha_r, \alpha_s] = \text{softmax}\big(W_g \cdot [E_p; E_r; E_s] + b_g\big)$$

$$E_{fused} = \alpha_p \odot E_p + \alpha_r \odot E_r + \alpha_s \odot E_s$$

This is conceptually similar to Squeeze-and-Excitation channel gating, but it reweights whole encoder streams instead of individual channels. For example, when the reference mic is dominated by noise, the gate can lean on $E_r$ to help cancel it. $E_{fused} \in \mathbb{R}^{T\times F'\times C}$ feeds the recurrent core.

### 6. Recurrent core: 8 stacked TF-xLSTM blocks

Each block runs a **time-axis pass** and then a **frequency-axis pass**, each bidirectional, with residual connections:

$$\hat{E} = \text{LayerNorm}(E)$$
$$\hat{E} = \hat{E} + \text{mLSTM}_{time}(\hat{E})$$
$$\hat{E} = \text{LayerNorm}(\hat{E})$$
$$\hat{E} = \hat{E} + \text{mLSTM}_{freq}(\hat{E})$$

Each axis call uses the bidirectional wrapper from the original paper: run the sequence forward through one mLSTM, run the flipped sequence through a second mLSTM, concatenate, and merge back to the original channel width with a convolution:

$$\upsilon = \text{Conv1d}\Big(\text{mLSTM}(\varepsilon) \oplus \text{flip}\big(\text{mLSTM}(\text{flip}(\varepsilon))\big)\Big)$$

#### The mLSTM cell

At every step $t$ the cell operates on hidden dimension $d = E_f \cdot D$ (expansion factor $E_f$, input dimension $D$).

**Query, key, value projections**

$$q_t = W_q x_t + b_q, \qquad k_t = \tfrac{1}{\sqrt{d}} W_k x_t + b_k, \qquad v_t = W_v x_t + b_v$$

**Gates** (exponential gating instead of sigmoid is the key change from vanilla LSTM)

$$i_t = \exp(w_i^\top x_t + b_i), \qquad f_t = \exp(w_f^\top x_t + b_f), \qquad o_t = \sigma(W_o x_t + b_o)$$

**Matrix memory update** (a full $d\times d$ matrix replaces the scalar cell state, greatly increasing storage capacity)

$$C_t = f_t\, C_{t-1} + i_t\, v_t k_t^\top$$

**Normaliser state** (keeps the readout numerically stable)

$$n_t = f_t\, n_{t-1} + i_t\, k_t$$

**Hidden-state readout**

$$h_t = o_t \odot \left(\frac{C_t q_t}{\max\{|n_t^\top q_t|,\, 1\}}\right)$$

Because there is no memory mixing between hidden units (unlike sLSTM), all heads and cells can be computed in parallel. This is part of why xLSTM scales better than a vanilla LSTM despite being recurrent.

**Configuration:** expansion factor $E_f = 2$, $N = 8$ blocks. This deeper-but-narrower setup is the **xLSTM-SENet2** configuration from the original paper.

### 7. Decoding: two parallel heads

After the 8 blocks, the representation $E_{out}$ splits into two decoders. Each is a dilated DenseNet followed by a 2D transposed convolution that restores $F' \to F$.

**Magnitude-mask decoder** predicts a compressed mask using a learnable sigmoid with steepness $\beta = 2$ (as in MP-SENet):

$$\hat{M}^c = \text{sigmoid}_{\beta=2}\big(\text{Decoder}_{mag}(E_{out})\big)$$

**Phase decoder** outputs pseudo-real and pseudo-imaginary components and recovers wrapped phase:

$$\hat{X}_\phi = \text{Arctan2}(\text{pseudo-imag},\ \text{pseudo-real})$$

### 8. Reconstruction

The masked, compressed magnitude is un-compressed:

$$\hat{X}_m = \left((Y_m)^c \odot \hat{M}^c\right)^{1/c}$$

It is combined with the predicted phase and inverted to a waveform:

$$\hat{x}(t) = \text{iSTFT}\left(\hat{X}_m \, e^{j\hat{X}_\phi}\right)$$

$\hat{x}(t)$ is the enhanced audio sent to the headset.

### 9. Auxiliary heads (multi-task)

Five small heads branch off the shared fused features $E_{fused}$ and are trained alongside the main task to improve the shared representation.

| Head | Type | Purpose |
| --- | --- | --- |
| **VAD** | Binary, per frame | Is speech present? |
| **Noise class** | Categorical | What kind of noise is present? |
| **SNR regression** | Continuous | Current SNR estimate |
| **Transient detection** | Binary | Flags sudden impulsive events (gunshot / artillery detector) |
| **Confidence** | Continuous | The model's own reliability estimate for this frame |

These heads act as training-time side losses only. They do not change the enhanced output.

---

## Loss functions

The generator loss follows the MP-SENet / SEMamba recipe.

| Loss | Definition | What it enforces |
| --- | --- | --- |
| Time | $\mathcal{L}_{time} = \lVert \hat{x} - x \rVert_1$ | Waveform fidelity |
| Magnitude | $\mathcal{L}_{mag} = \lVert \hat{X}_m - X_m \rVert_2^2$ | Spectral magnitude accuracy |
| Complex | $\mathcal{L}_{complex} = \lVert \hat{X}_m e^{j\hat{X}_\phi} - X_m e^{jX_\phi} \rVert_2^2$ | Real and imaginary parts jointly |
| Phase | $\mathcal{L}_{phase} = \mathcal{L}_{aw}(\hat{X}_\phi - X_\phi)$ | Phase accuracy, with anti-wrapping to handle discontinuities |
| Consistency | $\lVert \text{STFT}(\text{iSTFT}(\hat{X})) - \hat{X} \rVert_2^2$ | Predicted spectrum is a valid STFT |
| PESQ (adversarial) | $\mathbb{E}\big[(D(\hat{x}) - \text{target PESQ})^2\big]$ | A metric discriminator pushes output toward higher perceptual quality |

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{time} + \lambda_2 \mathcal{L}_{mag} + \lambda_3 \mathcal{L}_{complex} + \lambda_4 \mathcal{L}_{phase} + \lambda_5 \mathcal{L}_{consistency} + \lambda_6 \mathcal{L}_{PESQ}$$

Project-specific auxiliary losses are added on top:

$$\mathcal{L}_{aux} = \lambda_{vad}\mathcal{L}_{VAD} + \lambda_{noise}\mathcal{L}_{noise} + \lambda_{snr}\mathcal{L}_{SNR} + \lambda_{trans}\mathcal{L}_{transient} + \lambda_{conf}\mathcal{L}_{confidence}$$

$$\mathcal{L}_{final} = \mathcal{L}_{total} + \mathcal{L}_{aux}$$

> TODO: list the actual $\lambda$ values used in `configs/`.

---

## Training recipe

| Setting | Value |
| --- | --- |
| Optimiser | AdamW |
| Gradient clipping | Global norm 5.0 |
| LR schedule | 10,000-step warmup, then cosine decay |
| Segment length | 4.0 s (64,000 samples) |

The long warmup is deliberate. The softmax fusion gate and the exponential mLSTM gates are unstable under large early gradients, and an exploding exponential gate early in training can blow up the whole forward pass.

Training stages: (1) pretrain on VoiceBank+DEMAND, (2) fine-tune on the defence-noise mixture. See [DATASET.md](DATASET.md).

---

## Configuration summary

| Component | Setting |
| --- | --- |
| Sample rate | 16 kHz |
| STFT | FFT 400, hop 100, Hann, 201 bins |
| Magnitude compression | $c = 0.3$ |
| Encoder channels | $C = 96$ |
| Dense-block dilations | 1, 2, 4, 8 |
| Frequency after encoder | $F' = F/2$ |
| Recurrent core | 8 × (time-mLSTM + freq-mLSTM), bidirectional |
| mLSTM expansion factor | $E_f = 2$ |
| Mask activation | Learnable sigmoid, $\beta = 2$ |
| Auxiliary heads | VAD, noise class, SNR, transient, confidence |

---

## Real-time considerations

- **Frame rate.** A new frame is produced every 6.25 ms (hop 100 at 16 kHz).
- **Bidirectional processing.** The backward mLSTM pass along the time axis needs future frames. The model is therefore non-causal, and streaming inference works on chunks: end-to-end latency is roughly the chunk length plus compute time, not one hop. Choose the chunk size as a trade-off between latency and context.
- **Edge deployment.** The target is NVIDIA Jetson AGX Orin through ONNX and TensorRT. See [DEPLOYMENT.md](DEPLOYMENT.md). TODO: add measured latency and RTF once benchmarked.

---

## What is original and what is extended

| Part | Source |
| --- | --- |
| STFT with power-law compression, CNN encoder/decoder design, TF-xLSTM blocks, bidirectional mLSTM wrapper, mask + phase decoders | Published xLSTM-SENet / xLSTM-SENet2 (TODO: cite paper and repo) |
| Loss recipe (time, magnitude, complex, phase, consistency, metric discriminator) | MP-SENet / SEMamba (TODO: cite) |
| Learnable sigmoid mask, anti-wrapping phase loss | MP-SENet |
| Reference-mic encoder, IPD / ILD / coherence spatial branch, gated three-stream fusion, channel width 96, five auxiliary heads and losses | **This project** |

## References

- TODO: xLSTM-SENet paper and repository
- TODO: xLSTM (Beck et al.)
- TODO: MP-SENet, SEMamba
