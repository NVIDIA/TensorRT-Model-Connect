# DeepSeek-OCR

This family builds DeepSeek-OCR-2 (`deepseek-ai/DeepSeek-OCR-2`) for
`vision_language_generation`. A single-GPU build writes a vision engine
(`vision.plan`), a batched prefill engine (`prefill.plan`) and a single-token
decode engine (`engine.plan`). The language decoder is DeepSeek-V2 style: 12
layers, a dense MLP in layer 0, and 64 routed experts (top-6) plus two shared
experts in layers 1 to 11.

## Quantization

`quantization="fp8"` self-quantizes the language decoder at build time (see
`quantization.py`). The dense projections (attention q/k/v, the layer-0 MLP and
the shared experts) are W8A8, with static input scales from
`fp8_activation_scales.json`. The routed experts (about 93% of the decoder-layer
weights) keep FP8 weights with 16-bit inputs (W8A16). The vision engine stays
fp16.

### FP8 inputs for the routed experts (studied, not shipped)

The routed experts were also run as real FP8 GEMMs (FP8 weights and FP8 inputs,
one static input scale per layer and projection shared by all 64 experts).
TensorRT cannot gather FP8 tensors, so all experts run every step and
unselected experts get a zero routing weight. Two layouts of the prefill down
projection were measured: one flat GEMM over all experts, and one GEMM per
expert (routing weight applied after the GEMM).

Prefill speedup over the shipped FP8 build (E2E manifest config, 155-token E2E
prompt, batch 1, TensorRT 11.4):

| Prefill down projection | B200 | B300 |
|---|---|---|
| Flat, all experts | 1.54x | 1.51x |
| Per expert | 1.47x | 1.44x |

Decode was about the same (0.99x to 1.09x). Teacher-forced quality against
the Hugging Face model was within build-to-build noise of the shipped build.

Not shipped because the output is not stable. In some builds the greedy
output of the E2E test image changes: the model drops all spaces and skips two
lines of the text, so the E2E required-text check fails. This was seen on B200
and GB300 with the flat layout, and on B200 with the per-expert layout; the
shipped FP8 build did not show it. Revisit with a more accurate recipe for the
expert inputs, for example per-expert input scales or 16-bit inputs for the
down projection of layers 1 and 11, which have the largest input range.

### NVFP4 (studied, not shipped)

NVFP4 (W4A4) routed experts were tried and not shipped. The study
self-quantized the routed experts from bf16 (one `weight_scale_2` per layer and
projection, an e4m3 block scale per 16 inputs, dynamic activation quantization
with a calibrated input scale) and ran them as batched block-scaled FP4 GEMMs;
attention, the dense MLP and the shared experts stayed FP8. Results on B200,
TensorRT 11.4, E2E manifest config, against the shipped FP8 build:

| TensorRT 11.4, B200 | fp16 | FP8 (shipped) | NVFP4 experts |
|---|---|---|---|
| Bundle (GiB) | 11.82 | 7.00 | 5.03 |
| Prefill, 155 tokens (ms) | 5.28 | 6.82 | 4.64 |
| Decode, CUDA graph (ms/token) | 1.90 | 1.91 | 1.91 |
| Top-1 agreement vs fp16 (teacher-forced, 25 images) | | 98.18% | 96.44% |
| KL vs fp16 (nats) | | 0.0059 | 0.0210 |
| Mean text edit distance vs fp16 output (free run, 25 images) | | 0.096 | 0.206 |

- Decode does not get faster, and KL is 3.6x that of the FP8 build.
- The size saving over FP8 is about 2 GiB.
- A flat all-experts layout (`[T,H] x [H,64*I]` with dynamic NVFP4
  quantization of the down-projection input) gives wrong output with no error
  on TensorRT 11.1 and 11.4; the batched per-expert layout above does not have
  this problem.
