---
title: Nemotron Omni
---

# Nemotron Omni

The `nemotron_omni` family owns the officially documented prepared
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` checkpoint. It implements
text, image, acoustic and temporal-video reasoning using native TensorRT Python
graph construction and the Model Connect C++ runtime, without ONNX.

Optional Edge-LLM 0.11.0 offload is family-owned. Compatible source or wheel
Python tools use the existing CMake native SDK discovery contract. Text/image
tasks may use Edge; audio/video tasks select native directly, before Edge
preparation. Failed Edge preparation for eligible tasks warns and retries native
once. Publication and inference failures do not switch backends.

```sh
python -m tensorrt_model_connect build /path/to/Nemotron-Omni \
  --family nemotron_omni --task vision_language_generation \
  --precision fp16 --quantization nvfp4 --max-sequence-length 4096 \
  -o nemotron-omni.bundle
```

For an installation built with the 0.12.0 SDK, the family-specific
`trtmc nemotron_omni build` command accepts `--edge-builder-python`.
Use the interpreter containing the matching wheel- or source-installed Edge
builder; a Python wheel alone does not provide Model Connect's linked C++ SDK.
Text and image use the direct builder. An ONNX-enabled SDK can additionally
build audio and image/audio bundles: the family exports only the acoustic
component and invokes the installed `audio_build`. Video and aligned audio/video
stay native. See the family recipe below for the full command.

## Evidence boundaries

The native build and all six public task modes execute. A replay with the
corrected native decoder and retained media plans passed all six questions in
the earlier fixture. That replay is not a fresh build or a result for a later
prompt. The native premerge case builds and executes the complete model. Original free-form image, video and image/audio accuracy
failures remain recorded and runnable, not reclassified as captioning passes.

The 0.11 public Model Connect Edge build and original text/image executions also
succeeded: text passed quality, while the free-form image case failed the gate.
That 0.11 run did not exercise the revised factual image question. Direct
Edge audio and typed-video execution errors remain separate known limitations.
With Edge 0.12, both the text case and the factual image case matched
the independent reference exactly on SM103. The direct acoustic engine failed
at TensorRT enqueue; a released-runner control changing only the acoustic
engine to an ONNX-built one matched the audio and image/audio references
exactly. The public Model Connect hybrid bundle then matched the independent
reference tokens for all four tested modes: text, image, audio, and
image/audio. Typed video failed Edge placeholder validation and stays native. A
fresh native SM103 build ran all six modes. Five passed; aligned audio/video
missed the gate on the old free-form prompt (NED 0.1887 against 0.15). The
[current native fixture](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/2633)
asks for the road direction and verbatim speech with the same threshold. Component tests and scoped
factual answers are not catalog-wide qualification.

Small native FP8 and NVFP4 projections execute correctly on Ada; lack of FP4
acceleration alone is not a correctness restriction. TensorRT rejects these
FP8-containing graphs on A30. These component observations are not full-model
qualification or a family-owned GPU allowlist.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/nemotron_omni/README.md)
for pinned checkpoint details, public task bindings, capacity/timing contracts,
installation behavior and reproducible owning validation.
