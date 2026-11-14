---
title: Cosmos3-Edge reasoning
---

# Cosmos3-Edge reasoning

The `cosmos3_edge` family implements text, image, video and mixed image/video
reasoning for `nvidia/Cosmos3-Edge`. It is distinct from Cosmos3 Diffusers
image/video generation and does not produce trajectories or policy actions.

The family owns optional Edge-LLM 0.11.0 offload and a native TensorRT Python-graph /
Model Connect C++ fallback without ONNX. Compatible wheel-installed Python tools
and source-installed tools use the same native SDK discovery contract.

Build on the target GPU:

```sh
python -m tensorrt_model_connect build /path/to/Cosmos3-Edge \
  --family cosmos3_edge --task image_video_text_to_text \
  --precision fp16 --max-sequence-length 1024 -o cosmos3-edge.bundle
```

A compatible optional Edge SDK enables FP16 offload. Without one, or if Edge
preparation fails, the original request falls back to native construction with
a warning for a failed preparation. FP32 selects the pure TensorRT native graph.

## Evidence boundaries

Native FP32 matches the independent HF FP32 reference exactly on the tested
text, image, video and mixed cases. Native FP16 and Edge FP16 pass the tested
chat text/image cases but have recorded video and raw-continuation accuracy failures under the unchanged
0.15 normalized edit-distance gate. These successful build/inference paths
remain runnable; they are not claimed as quality-qualified video configurations.
Identical-engine direct Edge runs reproduce Model Connect's Edge outputs.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/cosmos3_edge/README.md)
for the exact checkpoint, modalities, capacities, public task APIs, installation
contract and validation limitations.
