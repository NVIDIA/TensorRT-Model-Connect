# CLIP

This family builds OpenAI CLIP checkpoints (`openai/clip-vit-base-patch32` in
the manifest; ViT-B/16 and ViT-L/14 build from the same code path) as two
TensorRT engines, the vision tower and the text tower, because the towers meet
only at a dot product. The bundle serves image embedding, text embedding and
image-text relevance (`text_image_to_relevance` in the manifest). The engines
return the projected embeddings unnormalized; L2 normalization and the logit
scale are applied in the runtime. Batch is 1; the manifest builds fp16.

## Quantization

`--quantization fp8` quantizes the original checkpoint at build time (#1963).
q/k/v/out_proj, fc1 and fc2 of every encoder layer of both towers get
per-tensor FP8 weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`); outlier projections
(text layer 0 `mlp.fc2`) stay unquantized. The patch convolution, embeddings,
layer norms, attention matmuls, QuickGELU and the two final projections stay
in the work precision. B200, batch 1: FP8 is 1.07x to 1.20x faster than fp16
(TensorRT 11.4 and 11.1), the plans are 27% to 49% smaller, and ImageNet
zero-shot top-1 is within 0.6 point of the fp16 engine.

### NVFP4

NVFP4 was not built (#1963). The engines are short and FP8 with half-size
weights already gives a small gain, so weight reads are not the limit (B200,
batch 1, CUDA graph):

| Model | Tower | fp16 ms | FP8 vs fp16, TensorRT 11.4 / 11.1 |
|---|---|---|---|
| ViT-B/32 | vision | 0.388 | 1.075x / 1.092x |
| ViT-B/32 | text | 0.334 | 1.125x / 1.121x |
| ViT-B/16 | vision | 0.414 | 1.071x / 1.120x |
| ViT-B/16 | text | 0.353 | 1.129x / 1.201x |
| ViT-L/14 | vision | 0.923 | 1.099x / 1.146x |
| ViT-L/14 | text | 0.353 | 1.073x / 1.102x |
