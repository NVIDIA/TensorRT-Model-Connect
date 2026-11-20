---
title: Gemma4 with Edge-LLM
---

Gemma uses the Edge-LLM SDK selected when Model Connect is built. The default
is 0.11.0; opt in to 0.12.0 with `-DTRTMC_EDGELLM_RELEASE=0.12.0`. Standalone
builds can include the checkpoint's image or audio towers. Speculative builds
are text-only and require the matching local draft checkpoint.

```sh
trtmc gemma build /path/to/gemma-4-E2B-it --precision fp16 -o gemma.bundle
trtmc gemma build /path/to/gemma-4-12B-it --precision fp16 \
  --execution-variant mtp --companion draft=/path/to/assistant -o gemma-mtp.bundle
```

Provision the optional native Edge SDK with ONNX tools and all native kernels.
Python installation alone does not provide the native SDK. All selection and
media mapping stay inside Gemma; earlier generations retain native behavior.

### Using Edge-LLM 0.12

Build Model Connect on the execution platform with
`TRTMC_ENABLE_EDGELLM=ON`, `TRTMC_EDGELLM_ONNX=ON` and
`TRTMC_EDGELLM_ALL_KERNELS=ON`. Install the matching Python tools from
the 0.12.0 source release or the `tensorrt-edgellm[onnx]==0.12.0` wheel in a
separate environment. Pass its interpreter to Gemma; its TensorRT version must
match the native SDK. For example, to build the prepared 26B JetSpec pair:

```sh
trtmc gemma build /path/to/Gemma-4-26B-A4B-NVFP4 --precision fp16 \
  --task text_generation --max-sequence-length 1024 \
  --execution-variant jetspec --companion draft=/path/to/jetspec-gemma4-26B-A4B0-it \
  --edge-builder-python /path/to/edge-env/bin/python -o gemma-jetspec.bundle
```

Here `--precision fp16` selects compute precision; it does not unpack the
checkpoint into an unquantized model. This pair uses calibrated NVFP4 experts,
an FP8 target cache, and an FP16 draft cache. Its Edge bundle matched the
prepared-target FP32 reference on the tested text prompt. The native bundle
built on a 32 GB GPU, but TensorRT could not allocate its execution context;
that build is not evidence of native inference or accuracy.

For the documented Bolt-E4B DFlash pair, 0.12 uses the checkpoint-direct
builder rather than ONNX. Use `--spec-draft-top-k 1` for chain verification or
`--spec-draft-top-k 4` for DDTree. These are Gemma family options, not global
runtime switches.

If you installed a Model Connect wheel built with the 0.12 SDK, use its
`trtmc` for both commands. It uses the SDK packaged with its runner; you do
not need to set `CMAKE_PREFIX_PATH` or `--runtime-root`. The separate Edge
Python environment is needed for the build, not for inference. For example,
after building a text bundle:

```sh
trtmc run gemma-jetspec.bundle --prompt "What is the capital of France?" --top-k 1
```

### Native fallback

When Edge is unavailable or its builder fails, the family-owned native fallback
constructs TensorRT graphs directly in Python and executes them through the
Gemma C++ runtime, without ONNX. It preserves the requested checkpoint format
and companion rather than silently substituting ordinary decoding.
Native FP8 attention keeps calibrated projection storage and uses explicit
FP32 score, mask and softmax math, without a GPU-specific dispatch rule.

Previously recorded 0.11 native cases include E2B/E4B/12B and the original or published prepared
26B/31B variants. Sampled 12B DSpark, EAGLE3 and both DFlash draft architectures
have actual draft-execution and same-native-target parity evidence. This is
separate from Edge's sampled EAGLE3/DFlash vanilla-fallback limitation and does
not claim a speculative speedup or cross-framework random-number equivalence.

Command-successful accuracy failures remain executable. Consult the
[family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md)
for exact checkpoints, tested profiles, historical versus current-source MTP
evidence, accuracy failures and installation requirements. These results do
not qualify every modality, capacity, generation policy or platform.
