# Qwen3.8 Flash-Next

This family is for the **ordinary** prepared `nvidia/Qwen3.8-Flash-Next-NVFP4`
checkpoint (`qwen4_exp`). It does not select a draft, calibrate a new checkpoint,
or claim other Qwen checkpoints. The target accepts text and image/video input and
returns text. The prepared NVFP4 experts, FP8 lexical table, and both generation
EOS IDs are preserved; `fp16` denotes runtime activation precision.

```bash
trtmc qwen3_8_flash_next build /path/to/snapshot \
  --task vision_language_generation --precision fp16 --max-sequence-length 8192 \
  --edge-builder-python /path/to/edge-builder-env/bin/python -o model.bundle
```

The Edge path needs a Model Connect build linked to the matching Edge-LLM 0.12
C++ SDK and an isolated 0.12 Python builder environment. It uses the official
LLM and visual builders and preserves the provider's chat template and checkpoint
identity checks when relocating their outputs. The C++ runtime receives a
checkpoint directory extracted from the bundle, not the original model directory.

The optional `--edge-ple-residency cached` stores the Edge lexical table on the host
and stages requested rows. The Edge default remains `resident`; this flag does not
change native storage.

If Edge preparation fails, the family warns and selects its native TensorRT
builder. Native execution does not use Edge, PyTorch inference, or ONNX. The
native bundle holds the trained language and visual stages and the selected
experts' original weight encodings. Its lexical lookup streams the FP8 table
once into temporary storage for bounded random reads. Allow space for the
self-contained bundle and for temporary files; this is a large model.

With `top_k=0`, native sampling has no top-k cap, regardless of whether a
seed is supplied. Edge 0.12 accepts top-k up to 1024 and rejects requests
the SDK would silently turn greedy (for example `temperature=1, top_k=0,
top_p=1`). The default top-k-one, unfiltered request remains greedy.

The native vision profile has a separate budget of `min(1024, max_input_length)`
merged visual tokens per request. The budget includes every image and every temporal
group in all videos. It does not replace the language prompt limit. An oversized
provider-resized input is rejected, not silently downsampled.

See [NOTICE](NOTICE) for checkpoint terms. Exposure of a build path is not
validation of a configuration. Standalone Edge tests, Model Connect offload,
and native fallback results must be reported separately.

## Validation scope

The full checkpoint case uses the installed family runtime and the public
text, image-plus-text, and ordered-video-plus-text tasks. It checks the first ten
primes, the species in a still image, and the first and last species in a video.
Each response must match the reference after case folding and trimming outer
whitespace. The evidence records which route was actually built.

An earlier resident Edge run matched all three references. In the latest native
run, text and image matched, but the ordered-video answer was “red panda, panda”
instead of “red panda, giant panda”. An earlier native run matched all three.
Row-cached Edge also missed the second species name; the shared symptom does
not establish a common cause. Native and row-cached Edge remain executable,
but neither is qualified for ordered video.

The native lexical-stage check exercises the production packed-table graph
against an independent E4M3FN reference with one and three token rows.
It does not build the language model or test generation.
