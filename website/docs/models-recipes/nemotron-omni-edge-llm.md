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

## Evidence boundaries

The native build and all six public task modes execute. The current owning
fixture uses independently verified factual grounding/transcription questions;
all six pass the unchanged 0.15 edit-distance gate in a replay with the corrected
native decoder and retained media plans. This replay is not a fresh full-builder
run of every source change; the declared premerge case still builds and executes
the complete model. Original free-form image, video and image/audio accuracy
failures remain recorded and runnable, not reclassified as captioning passes.

The public Model Connect Edge build and original text/image executions also
succeeded: text passed quality, while the free-form image case failed the gate.
The revised factual image question has not been rerun through Edge. Direct
Edge audio and typed-video execution errors remain separate known limitations.
Component tests and scoped factual answers are not catalog-wide qualification.

Small native FP8 and NVFP4 projections execute correctly on Ada; lack of FP4
acceleration alone is not a correctness restriction. TensorRT rejects these
FP8-containing graphs on A30. These component observations are not full-model
qualification or a family-owned GPU allowlist.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/nemotron_omni/README.md)
for pinned checkpoint details, public task bindings, capacity/timing contracts,
installation behavior and reproducible owning validation.
