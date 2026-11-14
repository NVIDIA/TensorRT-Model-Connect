# GLM-ASR

This family builds GLM-ASR (`zai-org/GLM-ASR-Nano-2512` in the manifests) for
the `transcription` task. GLM-ASR pairs a Whisper-style audio encoder with a
Llama decoder: a two-layer projector maps stacked encoder frames into the
decoder embedding space, and those embeddings replace the audio token positions
in the prompt. The bundle holds an audio encoder engine and a decoder engine.
The manifests build fp16 with a maximum sequence length of 1024.

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#1850). Decoder q/k/v/o/gate/up/down (28 layers) and encoder q/k/v/fc1/fc2
(32 layers) get per-tensor FP8 weights (amax / 448) and static activation scales
from `fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, 192 LibriSpeech
validation clips). Dequantize and the matmul run in fp32 and are cast back. The
encoder self-attention output projections, the convolution stem, embedding,
norms, projector and output projection stay fp16. On 80 LibriSpeech test clips
the WER stays at 3.23%, the same as fp16, and the bundle is 42% smaller.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. No ModelOpt NVFP4 checkpoint
exists for GLM-ASR, and FP8 itself gives only about 1.1x on both engines
(numbers from #1850, B300, batch 1; the decoder step is engine only with a CUDA
graph, the encoder runs one 30 s window):

| TensorRT | Engine | fp16 | FP8 | FP8 vs fp16 |
|---|---|---|---|---|
| 11.1 | Encoder (30 s) | 3.64 ms | 3.29 ms | 1.11x |
| 11.1 | Decoder step | 2.17 ms | 1.96 ms | 1.11x |
| 11.4 | Encoder (30 s) | 3.64 ms | 3.37 ms | 1.08x |
| 11.4 | Decoder step | 2.15 ms | 1.96 ms | 1.10x |
| both | Bundle | 4.22 GiB | 2.43 GiB | 42% smaller |

The FP8 gain is small because the attention matmuls, the layer norms, the Q/DQ
element-wise ops, the encoder attention output projections and the output
projection stay fp16. The decoder (hidden size 2048, 28 layers) is a small
decoder, and its one-token step is not much faster with 8-bit weights.
