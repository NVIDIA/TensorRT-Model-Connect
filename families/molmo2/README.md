# Molmo2

This family builds `allenai/Molmo2-8B` into a native TensorRT bundle for
single-image, greedy vision-language generation. The default build uses BF16,
one GPU, and a 2,048-token total sequence limit. The checkpoint revision is
`e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b`.

The family owns the SigLIP vision tower, the ordered `[-3, -9]` feature
concatenation, masked 2×2 attention pooling, SwiGLU image projection, and the
36-layer Qwen3-derived text decoder. Image features are added to their token
embeddings. Prefill allows bidirectional attention between image tokens while
retaining causal attention for text. Native preprocessing retains both the
overview and up to eight overlapping crops, including padding in pooling
indices and the checkpoint's image framing and BOS token.

Prepare the source-build environment described in
[Build from Source](../../website/docs/getting-started/source-build.md).
No additional family packages are needed to build or run the model. Build the
native targets `trtmc`, `trtmc_backend_trt`, and `trtmc_model_molmo2`.

```bash
PYTHONPATH=core/builder:. python -m tensorrt_model_connect build \
  allenai/Molmo2-8B \
  --revision e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b \
  --precision bf16 --max-sequence-length 2048 \
  -o molmo2-8b-bf16.bundle

build/trtmc run molmo2-8b-bf16.bundle --runtime-root build \
  --image families/molmo2/tests/data/test_img.jpeg \
  --prompt 'What is the main object in this image? Answer with one word.' \
  --max-new-tokens 16 --top-k 1
```

The bundle contains `vision.plan`, `engine.plan`, the FP32 host embedding
table, tokenizer, and the family runtime contract. A masked sentinel cache row
lets prefill and decode share one dynamic decoder engine. Request state is
reset before every generation; TensorRT contexts and weights remain loaded.
The native image API accepts RGB HWC floats in `[0,1]` and applies the
checkpoint's uint8 resize/normalization semantics.

The unmodified upstream reference needs a separate environment using
[tests/requirements.txt](tests/requirements.txt), because its Transformers
4.57.1 API differs from the builder's 5.2.0 pin. The test manifest pins both
the selected checkpoint revision as metadata; it does not authenticate supplied
local files. Native Unicode tokenization requires ICU's development package.
The native tokenizer applies the checkpoint's NFC normalizer and uses ICU's
Unicode letter, number, and whitespace properties for Qwen3 pre-tokenization.

To compare token IDs without downloading the model weights, build
`test_molmo2_preprocess`, which also builds `test_molmo2_tokenizer` and
`molmo2_qualification`. Set `TRTMC_NATIVE_BUILD_DIR` to that build directory and run
`python -m pytest families/molmo2/tests/test_tokenizer.py -q`.
The tests resolve the manifest's immutable checkpoint revision from the offline
Hub cache by default. A tokenizer-only cached snapshot needs `config.json` and
`tokenizer.json`; full E2E also needs the checkpoint weights and processor files.
`TRTMC_MOLMO2_TOKENIZER_DIR` can select an explicit tokenizer-only directory.
Invalid explicit paths fail instead of selecting another snapshot.

For direct E2E:

```bash
# With the matching native wheel under dist/, prepare the opted-in reference
# dependencies while network access is available.
MODEL=molmo2 python -m tools.ci pipeline setup

export TRTMC_NATIVE_BUILD_DIR=/path/to/native/build
export TRTMC_RUNTIME_ROOT=/path/to/installed/tensorrt_model_connect/bin
export TRTMC_BINARY="$TRTMC_RUNTIME_ROOT/trtmc"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/molmo2/tests --e2e-model molmo2 -q
```

Provision the pinned checkpoint in `HF_HUB_CACHE` before offline execution, or
set `TRTMC_MOLMO2_MODEL_DIR` to an explicit local checkpoint. Native test helpers
come from `TRTMC_NATIVE_BUILD_DIR`; the CLI and model libraries come from
`TRTMC_RUNTIME_ROOT`. A combined local build can use the same path for both.
The manifest opts into isolated reference preparation. Trusted CI forwards its
selected `MODEL` to the container; the existing setup stage prepares only that
family's reference requirements before the workflow disconnects networking.
The environment lives under `.ci/reference/molmo2` (or `TRTMC_CI_STATE_DIR`),
with reference overrides installed ahead of the builder's shared packages so
the existing CUDA/PyTorch stack can be reused. The builder retains Transformers
5.2.0 while the reference imports 4.57.1. E2E finds this prepared interpreter
automatically and rejects missing or stale preparation before engine build.
For a manually prepared reference environment, set
`TRTMC_MOLMO2_REFERENCE_PYTHON` inside the test container; the interpreter path
must preserve its virtual environment.
By default E2E builds a fresh bundle and runs the independent reference. For
an explicitly prepared bundle/reference run, the test accepts
`TRTMC_MOLMO2_BUNDLE` and `TRTMC_MOLMO2_REFERENCE_RESULTS`; select the cases of
one manifest with them.
Before generation, qualification checks the builder-recorded Hub identity and
all recorded build settings against the selected manifest, including the
2,048-token sequence limit. Use a checkpoint path resolving to the pinned Hub
snapshot so the builder can record its identity. Bundles with missing or
mismatched provenance are rejected; the recorded identity remains metadata
and does not authenticate the contents of supplied local files.

Video, multiple images, batching, parallel execution, NVFP4 quantization,
FP16, sampling, system prompts, and LoRA are outside this initial path. FP32 graph
construction is implemented but has not been qualified on the full checkpoint.

## Quantization

`--quantization fp8` (BF16 builds only; the default stays `none`) quantizes
the text decoder of `allenai/Molmo2-8B` at build time. No ModelOpt FP8 or NVFP4
checkpoint of Molmo2 exists, so the original checkpoint is quantized one
projection at a time and FP8 weights are never stored:

| Part | Precision |
| --- | --- |
| `att_proj` (q, k, v), `attn_out`, `ff_proj` (up, gate), `ff_out` of all 36 decoder layers | FP8 W8A8: weight scale `amax / 448`, static per-tensor input scale |
| `ff_out` of layers 6, 16 and 35 | BF16 (input scale 126, 65 and 36 times the median of its kind) |
| SigLIP tower, pooling attention, image projector, embeddings, `lm_head`, norms, attention core | BF16 (unchanged) |

The input scales in `fp8_activation_scales.json` were calibrated once with
ModelOpt `FP8_DEFAULT_CFG` (max) on 384 COCO images with questions and
teacher-forced news / Wikipedia answers (`calibrate_scales.py`, run in the
reference environment). They are tied to the pinned checkpoint: another
decoder config or another checkpoint of the same config raises and points to
the calibration tool. The attention is explicit (FP32 scores), so `attn_out`
can be FP8. FP8 builds turn the fused gate/up GEMM off: TensorRT 11.1 can fail
to compile it in front of an FP8 Quantize, and on TensorRT 11.4 the build
without it is faster on B200 and about 6% slower on B300.

```bash
PYTHONPATH=core/builder:. python -m tensorrt_model_connect build \
  allenai/Molmo2-8B \
  --revision e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b \
  --precision bf16 --quantization fp8 --max-sequence-length 2048 \
  -o molmo2-8b-fp8.bundle
```

Measured with TensorRT 11.4 (decoder engines driven like the native runtime;
a 768-row prompt with a five-crop COCO image; request = vision engine, prompt
and 64 decode steps; medians of 9 interleaved rounds):

| GPU | Engine | Decoder plan (MiB) | Decode step at row 808 / 1900 (ms) | Prompt (ms) | Request (ms) |
| --- | --- | --- | --- | --- | --- |
| B200 | BF16 | 14454 | 11.26 / 14.74 | 30.6 | 775 |
| B200 | FP8 | 7986 | 7.29 / 11.03 (1.54x / 1.34x) | 25.8 | 517 (1.50x) |
| B300 | BF16 | 14454 | 10.80 / 14.20 | 29.1 | 744 |
| B300 | FP8 | 7984 | 7.40 / 10.96 (1.46x / 1.30x) | 25.1 | 523 (1.42x) |

Quality against the upstream BF16 model (transformers, teacher-forced) on 26
held-out COCO images with 6776 answer tokens (up to about 1470 cache rows): KL mean
0.0081 (p99 0.052) and top-1 agreement 0.949 for FP8, against KL 0.0003 and
top-1 0.990 for the BF16 engine; the same on B200 and B300. No non-finite
logit in the teacher-forced runs, in 14 greedy captions, or in noise, black
and white images decoded until the 2048-row cache is full. The vision engine
is 3% to 4% of a request and stays BF16. The `molmo2-8b-fp8` E2E manifest
runs the three BF16 requests on an FP8 bundle and requires the same tokens as
the upstream BF16 model.

NVFP4 (W4A4, block 16, `add_dynamic_quantize`) was measured and is not
shipped. All layouts build and run without non-finite values on TensorRT 11.4,
and decode is 1.02x to 1.14x of FP8, but the loss is 6 to 11 times that of FP8
(the limit is about 2 times):

| Layout (TensorRT 11.4) | Decoder plan (MiB) | Decode B200 / B300 (ms) | KL mean | Top-1 |
| --- | --- | --- | --- | --- |
| FP8, all projections (fused gate/up on) | 7983 | 8.17 / 7.68 | 0.0081 | 0.950 |
| NVFP4 all projections | 5148 | 7.40 / 6.96 | 0.090 | 0.845 |
| NVFP4 MLP, FP8 attention | 5778 | 7.14 / 7.15 | 0.064 | 0.862 |
| NVFP4 MLP except layers 0, 1, 34, 35 (FP8), FP8 attention | 6010 | 7.16 / 7.26 | 0.051 | 0.882 |

