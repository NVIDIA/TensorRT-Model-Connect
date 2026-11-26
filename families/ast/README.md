# AST

This family builds Audio Spectrogram Transformer classifiers
(`MIT/ast-finetuned-audioset-10-10-0.4593` in the manifest, the AudioSet
527-label model) as one TensorRT engine for the `audio_to_class_scores` task.
AST is a ViT whose patches are cut from a mel spectrogram: the engine takes the
normalized `[1024, 128]` spectrogram at batch 1 and returns every class logit.
The Kaldi filter-bank front end is computed outside the engine; its analysis
window and mel filters ship beside the plan. The manifest builds fp32; fp16 is
also supported.

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#1907). The query, key, value, attention output, fc1 and fc2 projections
of all 12 layers (72 projections) get per-tensor FP8 weights (amax / 448) and
static activation scales from `fp8_activation_scales.json` (ModelOpt
`FP8_DEFAULT_CFG`, 100 AudioSet clips). The patch projection, layer norms,
positions and classifier stay fp16. B300, batch 1: the FP8 engine is 1.17x
(TensorRT 11.1) and 1.25x (TensorRT 11.4) faster than fp16, the bundle drops
from 169 to 89 MiB, and top-1 matches the fp16 engine on 282 of 292 inputs
(AudioSet mAP -0.009); no non-finite logits.

### NVFP4

NVFP4 was not tried. It was a non-goal of the FP8 change, for these reasons:

| Item | Value |
|---|---|
| Model size | about 87M parameters (fp16 bundle 169 MiB, FP8 bundle 89 MiB) |
| Batch | 1 (the family builds batch 1 only) |
| FP8 vs fp16, B300, batch 1 | 1.17x (TensorRT 11.1), 1.25x (TensorRT 11.4) |
| Bound | not bound by weight reads: halving the weight bytes again saves little |
