# Deployment

The ultimate goal for the Defence ANC pipeline is real-time, on-device inference using low-power edge hardware suitable for tactical scenarios.

## Target Hardware
- **NVIDIA Jetson AGX Orin**: Capable of running heavy deep-learning pipelines locally.
- **Headset Integration**: A primary microphone for the noisy speech and an optional reference microphone for background noise sampling.

## Planned Pipeline
*(Note: These steps are currently marked as **planned, not done**.)*

1. **Export to ONNX**: The PyTorch `xLSTMSENet` model will be traced and exported to the ONNX format. Special care will be taken to ensure that the custom `mamba_ssm` and complex STFT operations are supported or rewritten with ONNX-compatible equivalents.
2. **TensorRT Optimization**: The ONNX graph will be compiled into a TensorRT engine on the Jetson AGX Orin device to minimize latency and maximize throughput via FP16 quantization.
3. **C++ / DeepStream Integration**: A low-latency C++ audio pipeline will manage the streaming audio I/O (handling the overlapping chunk mechanism internally).
4. **Optional LMS Stage**: A lightweight Least Mean Squares (LMS) adaptive filter could be layered to handle residual linear noise components picked up by the reference microphone.
