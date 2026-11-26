# MusicGen medium

The native tokenizer supports the reconstructed T5 backend's WhitespaceSplit
followed by Metaspace with `▁`, `prepend_scheme=always`, and `split=true`.
Unsupported normalizers, pre-tokenizer layouts/options, and added-token matching
flags are rejected when loading a bundle. Checkpoint loading requires PyTorch
2.10 or newer.

This family implements `facebook/musicgen-medium` as the native `text_to_audio`
Task. The bundle contains three FP32 TensorRT engines: the T5 text encoder and
projection, MusicGen's four-codebook causal decoder, and the non-causal 32 kHz
EnCodec decoder. Both codec LSTMs execute inside TensorRT loops. Native inference
does not invoke Python or PyTorch.

The initial path uses one GPU, one prompt, mono PCM, and a bounded sequence that
is recomputed at each generation step. It does not implement KV caching, tensor
parallelism, quantization, audio continuation, stereo, or melody conditioning.
Other MusicGen model sizes are not claimed by this owner.

## Build and run

Prepare the repository's documented development environment, install this
family's `requirements.txt`, then download the checkpoint:

```python
from huggingface_hub import snapshot_download

checkpoint = snapshot_download(
    "facebook/musicgen-medium",
    revision="d3bd7b00761b78ad7a8a05145ee31e7832e9916c",
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "*token*.json",
        "special_tokens_map.json",
        "spiece.model",
        "pytorch_model.bin",
    ],
)
```

Build the one-second qualification case from the repository root:

```python
from pathlib import Path
from tensorrt_model_connect import BuildRequest, build

build(
    BuildRequest(
        model_dir=Path(checkpoint),
        output_path=Path("musicgen-medium.bundle"),
        family="musicgen",
        task="text_to_audio",
        precision="fp32",
        max_sequence_length=54,
    )
)
```

The optional sequence bound includes the four delayed BOS/EOS positions. A
bound of 54 produces 50 codec frames, or exactly one second of mono 32 kHz audio.
The default bound is 256 (5.04 seconds); accepted bounds are 11 through 1504.
The checked-in qualification covers bound 54. Prompts must fit in 64 T5 tokens,
including EOS; longer prompts fail without silent truncation. The bundle stores
the tokenizer backend produced by the repository-pinned Transformers version,
including its actual normalization and special-token framing.

Build the `trtmc`, `trtmc_backend_trt`, `trtmc_model_musicgen`, and
`test_musicgen_tokenizer` CMake targets. Enable `TRTMC_BUILD_TESTS` for the last
target and the public SDK consumer. Then run:

```bash
build-musicgen/trtmc generate-audio musicgen-medium.bundle \
  --runtime-root build-musicgen \
  --prompt "A mellow jazz piano with soft drums." \
  --set do_sample=false --set guidance_scale=3.0 --output musicgen.wav
```

The Task defaults to sampling with `top_k=50`, `temperature=1.0`,
`guidance_scale=3.0`, and `seed=0`. Each request owns its RNG and sequence state.
Equal sampling seeds reproduce native output; native sampled tokens are not
claimed to match PyTorch's RNG bit for bit. Set `do_sample=false` for the
official-reference greedy comparison.

## Quantization

FP8 and NVFP4 builds of the decoder were measured and are not shipped. The
bundle stays FP32 and the build accepts only `quantization=none`.

Setup:

- No ModelOpt FP8 or NVFP4 checkpoint of MusicGen exists, so both formats were
  self-quantized from the FP32 checkpoint: per-tensor FP8 weights
  (`amax / 448`); NVFP4 W4A4 with per-16 e4m3 block scales,
  `weight_scale_2 = amax / (6 * 448)` and a run-time dynamic FP4 quantize of
  the input. Static activation scales came from ModelOpt `FP8_DEFAULT_CFG`
  (max) over 48 calibration prompts at 256 positions, with the guidance batch
  (no projection had a scale above 20x the median of its kind). NVFP4 input
  scale = FP8 scale / 6.
- Quantized: the 480 decoder projections (`self_attn` and `encoder_attn`
  q/k/v/out, `fc1`, `fc2`). The T5 encoder, EnCodec, embeddings, norms,
  attention scores and softmax, and the codebook heads stay FP32. fp16
  Dequantize (fp32 Dequantize was not better).
- Reference: the FP32 PyTorch model (Transformers 5.2.0, TF32 off), greedy,
  guidance 3.0, on held-out prompts (32 at bound 54, 16 at bound 256; 6400 and
  16128 scored tokens). Quality is per scored token under teacher forcing:
  KL(reference || engine) and top-1 agreement of the guided scores that the
  runtime samples from. "Mel dB" is the mean log-mel distance of the free-running
  output audio to the reference audio (two different prompts are 27 to 30 dB
  apart).
- Time: full request (T5 + all decoder steps + EnCodec), greedy, prompt one,
  median of 5 interleaved runs, all variants in one job per GPU, TensorRT
  11.4.0.106, builder level 1 as in the family. "fp16" and "bf16" rows are
  the same decoder with its linears and attention in 16-bit (residual stream,
  norms and logits FP32); they were measured as a reference, not as a
  shipped option.

| GPU | Bound | Decoder | Plan MiB | Full request ms | vs FP32 | KL mean | KL p99 | Top-1 | Mel dB |
|---|---|---|---|---|---|---|---|---|---|
| B200 | 54 | FP32 (default) | 7015 | 989 | 1.00x | 0.0000 | 0.0000 | 0.999 | 0.0 |
| B200 | 54 | fp16 | 3537 | 187 | 5.28x | 0.0016 | 0.021 | 0.989 | 4.1 |
| B200 | 54 | bf16 | 3537 | 171 | 5.77x | 0.031 | 0.27 | 0.947 | 7.9 |
| B200 | 54 | FP8 | 1832 | 237 | 4.17x | 0.297 | 3.69 | 0.853 | 10.8 |
| B200 | 54 | NVFP4 | 1079 | 257 | 3.86x | 12.48 | 59.4 | 0.143 | 25.6 |
| B300 | 54 | FP32 (default) | 7016 | 986 | 1.00x | 0.0000 | 0.0000 | 0.999 | 0.0 |
| B300 | 54 | fp16 | 3537 | 170 | 5.81x | 0.030 | 0.29 | 0.948 | 8.1 |
| B300 | 54 | bf16 | 3537 | 161 | 6.13x | 0.031 | 0.26 | 0.946 | 7.8 |
| B300 | 54 | FP8 | 1833 | 219 | 4.50x | 0.286 | 3.44 | 0.853 | 10.6 |
| B300 | 54 | NVFP4 | 1079 | 240 | 4.14x | 12.51 | 59.3 | 0.142 | 25.6 |
| B200 | 256 | FP32 (default) | 7017 | 12535 | 1.00x | 0.0000 | 0.0000 | 1.000 | 0.0 |
| B200 | 256 | fp16 | 3538 | 1157 | 10.8x | 0.014 | 0.071 | 0.994 | 3.6 |
| B200 | 256 | bf16 | 3538 | 1090 | 11.5x | 0.124 | 2.30 | 0.965 | 9.1 |
| B200 | 256 | FP8 | 1834 | 1649 | 7.60x | 0.697 | 14.5 | 0.861 | 12.0 |
| B200 | 256 | NVFP4 | 1079 | 1800 | 6.99x | 12.34 | 59.2 | 0.136 | 32.7 |
| B300 | 256 | FP32 (default) | 7017 | 11216 | 1.00x | 0.0000 | 0.0000 | 1.000 | 0.0 |
| B300 | 256 | fp16 | 3538 | 1147 | 9.78x | 0.082 | 1.34 | 0.970 | 8.9 |
| B300 | 256 | bf16 | 3538 | 1052 | 10.7x | 0.127 | 2.45 | 0.964 | 6.9 |
| B300 | 256 | FP8 | 1834 | 1557 | 7.20x | 0.592 | 12.8 | 0.873 | 11.3 |
| B300 | 256 | NVFP4 | 1079 | 1687 | 6.67x | 12.41 | 59.2 | 0.135 | 32.8 |

NVFP4 times are from the first measurement job (same setup); NVFP4 is 0.93x
of the FP8 speed. Other NVFP4 layouts (MLP only with FP8 attention,
self-attention and MLP with FP8 cross-attention, all linears with an fp16
decoder) gave KL 4.1 to 18 and top-1 0.05 to 0.26. No layout produced
non-finite logits or audio (real prompts, random token ids and a 64-token
prompt at three positions per engine).

Sensitivity (B200, bound 54, FP8 on one projection kind at a time, KL mean):
`fc2` 0.100, `fc1` 0.033, `self_attn.k_proj` 0.038, `self_attn.q_proj` 0.019,
`self_attn.out_proj` 0.018, `self_attn.v_proj` 0.0055, each `encoder_attn`
projection 0.0001 to 0.0003. Keeping layer 0 FP32 lowers the full FP8 KL from
0.297 to 0.123. The best FP8 subsets kept 2 to 6 edge layers and `fc2` in FP32:

| GPU | Bound | FP8 layout | Full request ms | vs FP32 | vs fp16 | KL mean | Top-1 |
|---|---|---|---|---|---|---|---|
| B200 | 54 | all, layers 0 and 47 FP32 | 270 | 3.66x | 0.69x | 0.089 | 0.907 |
| B200 | 54 | all but `fc2`, layers 0, 1, 46, 47 FP32 | 471 | 2.10x | 0.40x | 0.054 | 0.937 |
| B200 | 256 | all, layers 0 and 47 FP32 | 2117 | 5.92x | 0.55x | 0.570 | 0.910 |
| B200 | 256 | all but `fc2`, layers 0, 1, 46, 47 FP32 | 5466 | 2.29x | 0.21x | 0.329 | 0.931 |
| B300 | 54 | all, layers 0 and 47 FP32 | 254 | 3.89x | 0.67x | 0.195 | 0.906 |
| B300 | 54 | all but `fc2`, layers 0, 1, 46, 47 FP32 | 448 | 2.20x | 0.38x | 0.097 | 0.937 |
| B300 | 256 | all, layers 0 and 47 FP32 | 1968 | 5.70x | 0.58x | 0.429 | 0.917 |
| B300 | 256 | all but `fc2`, layers 0, 1, 46, 47 FP32 | 4771 | 2.35x | 0.24x | 0.520 | 0.922 |

Reasons:

- FP8 quality: with all 480 projections, KL against the FP32 reference is
  0.29 to 0.70 and top-1 agreement 0.85 to 0.87, about 5x to 10x the loss of
  a bf16 decoder; greedy output follows the reference for about 20% of the
  tokens (bf16: 34% to 44%). Classifier-free guidance multiplies the logit
  error (the guided score is 3x the conditional minus 2x the unconditional
  logits): the conditional head alone has KL 0.05 to 0.08. FP8 layouts that
  keep the sensitive layers in FP32 get closer at bound 54 but stay 2.7x to
  4.6x the bf16 KL at bound 256.
- FP8 speed: FP8 is 4.2x to 7.6x faster than the FP32 default, but every FP8
  layout is slower than the same decoder in fp16 or bf16 (FP8 0.70x to 0.79x
  of fp16), because the FP32 attention, norms and casts around the FP8 GEMMs
  remain. FP8 on an fp16 decoder was 1.04x to 1.11x of fp16, with KL 0.17 to
  0.18 at bound 54. FP8 therefore is not shipped: an fp16 or bf16 decoder is faster
  and more accurate.
- NVFP4: unusable quality (top-1 0.14, the output is different music) and
  0.93x of the FP8 speed.
- Precision finding (not changed here): an fp16 or bf16 decoder is 5.3x to
  6.1x (bound 54) and 9.8x to 11.5x (bound 256) faster than the FP32 default.
  It is a separate precision decision: the loss is small but measurable
  (bf16 KL 0.03 to 0.13; fp16 KL 0.0016 to 0.014 on B200 but 0.03 to 0.08 on
  B300 with the same network, where TensorRT picked other kernels), and greedy
  output then differs from the FP32 reference after the first changed token.
  The remaining FP32 cost is the full-sequence recompute per step (no KV
  cache): the cross-attention keys and values of the fixed text are computed
  again at each of the `bound - 1` steps.

TensorRT 11.1.0.106 (B200, bound 54) builds every variant and gives the same
picture: FP32 980 ms, fp16 185 ms, FP8 240 ms (KL 0.271, top-1 0.856),
NVFP4 261 ms (KL 12.5, top-1 0.146), no non-finite output.

## Validation

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRTMC_MUSICGEN_MODEL_DIR TRTMC_MUSICGEN_BUNDLE
export TRTMC_BINARY="$PWD/build-musicgen/trtmc"
export TRTMC_RUNTIME_ROOT="$PWD/build-musicgen"
export TRTMC_NATIVE_BUILD_DIR="$PWD/build-musicgen"
export TRTMC_E2E_ARTIFACT_DIR="$PWD/artifacts/musicgen"
cmake --build build-musicgen --target test_musicgen_tokenizer_config
ctest --test-dir build-musicgen -R '^musicgen_tokenizer_config$' --output-on-failure
PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/musicgen/tests --e2e-model musicgen -q
build-musicgen/test_musicgen_sdk_cpp musicgen-medium.bundle build-musicgen
git diff --check
```

The tests resolve the manifest's immutable revision from the offline Hub cache.
`TRTMC_MUSICGEN_MODEL_DIR` can instead name an explicit local checkpoint;
an invalid override fails rather than falling back to another model. The native
tokenizer probe comes from `TRTMC_NATIVE_BUILD_DIR`, with `TRTMC_RUNTIME_ROOT`
as a fallback for direct local builds. Building the tokenizer configuration
CTest also builds this probe, without requiring a GPU bundle.

The CPU bundle regression serializes the pinned tokenizer through the production
writer into a fresh tokenizer-only bundle and compares the complete backend.
The E2E test also checks that backend in the full inference bundle.

The E2E test builds a fresh bundle unless `TRTMC_MUSICGEN_BUNDLE` explicitly names
an existing one. It compares two distinct prompts with official Transformers
5.2.0 FP32 greedy generation at guidance scale 3.0. Both outputs must have the
exact sample count, finite PCM, waveform cosine at least 0.999, and relative
RMSE at most 0.01. A separate check requires prompt-dependent audio. Native
tokenizer fixtures check Unicode and whitespace against the official backend.
The C++ SDK consumer tests seeded sampling, repeat calls on one loaded model,
invalid configuration, prompt bounds, and recovery after invalid requests.

Checkpoint identity and upstream implementation:
[model card](https://huggingface.co/facebook/musicgen-medium),
[MusicGen reference](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/musicgen/modeling_musicgen.py),
[EnCodec reference](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/encodec/modeling_encodec.py).
The upstream model weights use CC-BY-NC-4.0.
