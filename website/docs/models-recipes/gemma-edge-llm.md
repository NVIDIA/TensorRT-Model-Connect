---
title: Gemma4 with Edge-LLM
---

Gemma uses the Edge-LLM SDK selected when Model Connect is built. Enabling Edge
selects 0.12.0 by default. Standalone
builds can include the checkpoint's image or audio towers. Speculative builds
are text-only and require the matching local draft checkpoint.

```sh
trtmc gemma build /path/to/gemma-4-E2B-it --precision fp16 -o gemma.bundle
trtmc gemma build /path/to/gemma-4-12B-it --precision fp16 \
  --execution-variant mtp --companion draft=/path/to/assistant -o gemma-mtp.bundle
```

Provision the optional native Edge SDK with ONNX tools and all native kernels.
The full kernel set still targets the GPU architectures chosen when the SDK was
built. If that SDK does not include the local GPU, Gemma tries native TensorRT.
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
an FP8 target cache, and an FP16 draft cache. On the tested greedy text prompt, Edge on SM120 and
native TensorRT on SM103 both matched the prepared-target FP32 tokens. The
native draft proposed 127 nodes; one token was accepted in one verification
round. On a 32 GiB SM120 GPU the native engine built, but its execution
context did not fit. This is a placement limit, not a model-wide restriction.

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

Build success alone does not establish that a bundle fits the GPU or matches
the reference. See the [0.12 standalone results for QAT, Bolt and prepared
NVFP4](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md#standalone-validation-on-012-qat-bolt-and-prepared-nvfp4)
for the tested devices, inputs and outcomes.

The [original FP16 31B standalone](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md#original-31b-standalone-on-012)
matched FP32 on the Edge text request. Its image description missed the 0.15
limit (NED 0.198), with identical output from the released Edge CLI. Native TRT
matched FP32 tokens on both requests on SM100.

Original Gemma4 targets that store floating-point experts in fused tensors use
native TRT on 0.12 because the exporter cannot load those weights faithfully.
This does not affect prepared NVFP4 targets. See the
[26B-A4B exporter diagnosis](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md#original-floating-point-26b-a4b-experts-on-edge-012).

The original 31B [DFlash](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md#native-original-31b-dflash)
and [MTP](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md#gemma4-mtp-on-012)
pairs matched the CPU FP32 tokens on both Edge and native TRT for the tested
greedy request. Each exercised its draft; this is not a speedup measurement.

Previously recorded 0.11 native cases include E2B/E4B/12B and the original or published prepared
26B/31B variants. Sampled 12B DSpark, EAGLE3 and both DFlash draft architectures
have actual draft-execution and same-native-target parity evidence. This is
separate from Edge 0.11's sampled EAGLE3/DFlash vanilla-fallback limitation and does
not claim a speculative speedup or cross-framework random-number equivalence.

Command-successful accuracy failures remain executable. Consult the
[family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md)
for exact checkpoints, tested profiles, historical versus current-source MTP
evidence, accuracy failures and installation requirements. These results do
not qualify every modality, capacity, generation policy or platform.
