---
title: Cosmos3-Edge reasoning
---

# Cosmos3-Edge reasoning

An installed Model Connect wheel uses its own native Edge SDK, not a
different SDK on `CMAKE_PREFIX_PATH`. To switch releases, use a matching
wheel or a standalone Model Connect builder and runner together.

The `cosmos3_edge` family implements text, image, video and mixed image/video
reasoning for `nvidia/Cosmos3-Edge`. It is distinct from Cosmos3 Diffusers
image/video generation and does not produce trajectories or policy actions.

The family owns optional Edge-LLM offload (default 0.11; build-time opt-in 0.12) and a native TensorRT Python-graph /
Model Connect C++ fallback without ONNX. Compatible wheel-installed Python tools
and source-installed tools use the same native SDK discovery contract.

For 0.12, configure MC with `TRTMC_ENABLE_EDGELLM=ON` and
`TRTMC_EDGELLM_RELEASE=0.12.0`. Build on the target GPU:

```sh
trtmc cosmos3_edge build "$MODEL_DIR" --task image_video_text_to_text \
  --precision fp16 --max-sequence-length 1024 \
  --edge-builder-python "$EDGE_PYTHON" -o cosmos3-edge.bundle
```

A compatible optional Edge SDK enables FP16 offload. Without one, or if Edge
preparation fails, the original request falls back to native construction with
a warning for a failed preparation. FP32 selects the pure TensorRT native graph.

## 0.12 opt-in: tested scope

On the original checkpoint (FP16, batch 1, capacity 1024, SM80), Edge and native
both matched the independent HF FP32 token IDs on the two chat prompts and one
image. Both ran timestamped video and mixed image/video, but missed the 0.15
edit-distance gate: Edge scored 0.7726 and 0.9058; native scored 0.6893 and
0.8363. Those modes are runnable, not quality-qualified.

## Historical 0.11 evidence boundaries

Native FP32 matches the independent HF FP32 reference exactly on the tested
text, image, video and mixed cases. Native FP16 and Edge FP16 pass the tested
chat text/image cases but have recorded video and raw-continuation accuracy failures under the unchanged
0.15 normalized edit-distance gate. These successful build/inference paths
remain runnable; they are not claimed as quality-qualified video configurations.
Identical-engine direct Edge runs reproduce Model Connect's Edge outputs.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/cosmos3_edge/README.md)
for the exact checkpoint, modalities, capacities, public task APIs, installation
contract and validation limitations.
