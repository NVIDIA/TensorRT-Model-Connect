# SigLIP 2 NaFlex image and text embeddings

This family builds
[`google/siglip2-base-patch16-naflex`](https://huggingface.co/google/siglip2-base-patch16-naflex/tree/b53b807d3a2d5e2b3911292f2d69e5341cdc064c)
(`Siglip2Model`, `model_type: siglip2`) into one bundle that serves three existing
Tasks: `image_to_embedding`, `text_to_embedding` and `text_image_to_relevance` (the
default). Both towers return 768-dimensional L2-normalised vectors in the `siglip2`
embedding space; relevance is the checkpoint's per-pair sigmoid,
`sigmoid(exp(logit_scale) * cos(image, text) + logit_bias)`, returned as a
probability.

Scope: float32 (the checkpoint's own precision), one image or one text per request,
TP1 / CP1, the TensorRT backend. Reduced precision, quantization, batching,
`Siglip2ForImageClassification` and other image processors are refused at build
time. The fixed-resolution SigLIP 2 checkpoints publish `model_type: siglip` and the
`SiglipModel` graph; they belong to the `siglip` family, not to this one. Only the
checkpoint above is qualified: other `siglip2` checkpoints resolve to this family and
build by the same rules, but have not been validated.

## Bundle and runtime

The bundle carries a vision engine, a text engine, `tokenizer.json` and
`runtime.json`. The checkpoint's `config.json` states almost nothing, so the
geometry is read from the tensor shapes and `preprocessor_config.json`, and any
stated value is checked against them.

- **Image.** NaFlex resizes each image, keeping its aspect ratio, to the largest
  size whose sides are whole 16-pixel patches and whose patch count fits the
  processor's budget of 256, using the reference processor's own binary search.
  The runtime then reproduces the reference processor value for value: Pillow 13's
  8-bit bilinear resize (fixed-point coefficients and Pillow 13's choice of which
  pass runs first), numpy's float64 rescale and float32 normalisation, and the
  patch cut in (row, column, channel) order. The vision engine takes exactly the
  image's patches, with the patch count dynamic from 1 to 256, so no padding or
  attention mask is needed: the reference pads to the budget and masks the padding,
  which computes the same function. The 16 x 16 position table stays in the engine;
  the runtime passes the antialiased bilinear taps that resample it to the image's
  grid (`[patches, 256]`, built in float64), and the engine multiplies them in.
  Pooling is the checkpoint's attention head (one learned query over all patches).
- **Text.** A byte-fallback BPE tokenizer reproduces the reference tokenizer: the
  only normalisation is the space marker, there is no pre-tokenizer, and `<eos>` is
  appended. Like the reference, it does **not** lowercase: the `do_lower_case` entry
  in `tokenizer_config.json` is not applied by the tokenizer that transformers loads
  for this checkpoint. SigLIP 2 was trained on lowercased text (transformers'
  `Siglip2Tokenizer` class, which is not the one loaded here, adds a lowercase
  normaliser for that reason), so callers should lowercase their labels. The prompt is
  right-padded with `<pad>` to the 64-token window, and a longer prompt keeps its
  first 63 tokens and `<eos>`, as the reference truncation does. The pooled row is
  the last row of the window, so the window length is a correctness rule, not a
  capacity. The tokenizer source is the repository's family-local BPE
  implementation, copied from `families/embeddinggemma2/runtime/bpe_tokenizer.cpp`
  with its factory renamed.

## Reference behaviours this family takes a position on

- **Pillow version.** Pillow 13.0.0 resamples vertically first when the height
  shrinks by more than twice as many pixels as the width; Pillow 12 always went
  horizontally first. Each pass rounds to 8 bits, so the two versions disagree by one
  step on such images. The family reproduces Pillow 13.0.0, pinned in
  `requirements.txt`.
- **Slow and fast processors.** The checkpoint names `Siglip2ImageProcessorFast`,
  which transformers selects when torchvision is installed. It resizes with
  torchvision instead of Pillow and lands one 8-bit step away on about 45 % of the
  pixel values of the test images, which moves the image embeddings by up to
  2.6e-4 per component. The family reproduces the Pillow processor
  (`use_fast=False`), and the E2E compares against it.
- **A grid one patch wide.** PyTorch's CPU kernel for antialiased bilinear
  interpolation returns wrong values when the width is 1 (torch 2.12.0 and 2.15.0a0
  on CPU: up to 7.1 on this checkpoint's position table and 0.8 to 1.1 on random
  tables), while its CUDA kernel matches the
  separable definition to 4e-7. The reference resizes the position table with that
  call, so on CPU it disagrees with itself on GPU for images whose NaFlex grid is a
  single column (for example 3000 x 20 pixels). The runtime implements the
  definition, which is what the reference computes on GPU. Such images are not in
  the E2E, whose reference runs on CPU.

## Validation

CPU tests (no GPU, no checkpoint):

```sh
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest families/siglip2/tests -q
```

`test_model.py` covers the geometry resolution, every refused build option, the
processor settings and tokenizer contracts the runtime does not reproduce, and the
prompt window. The native CTests (`test_siglip2_naflex_preprocess`,
`test_siglip2_pipeline`, `test_siglip2_tokenizer`) check the target-size search,
Pillow's resize and the position taps against values produced by the reference
implementations, the patch layout, the padded text window, normalisation, the
sigmoid relevance and the runtime's refusals.

GPU E2E: cache the checkpoint at the revision above (or set
`TRTMC_SIGLIP2_MODEL_DIR`), install `requirements.txt`, configure `TRTMC_BINARY`,
`TRTMC_RUNTIME_ROOT` and `TRTMC_NATIVE_BUILD_DIR` for the native build, then:

```sh
python -m pytest families/siglip2/tests/test_e2e.py --e2e-model siglip2 -q
```

The case first runs the runtime's host preprocessing through
`siglip2_preprocess_dump` on three images with different grids (the test
photograph, a portrait crop and a small crop the processor upscales) and requires
every pixel value to equal the reference processor's, and the resampled position
table to match the reference's to 1e-4. It then builds the bundle and compares the
native embeddings of the three images and of six texts (English, mixed case,
French, Chinese and a 78-token prompt that is truncated), and the relevance logit
of a matching and a non-matching pair, with the reference model in float32 on the
CPU.

Measured on one B300 PCIe (TensorRT 11.4.0.106, CUDA 13.4, torch 2.15.0a0 from the
NGC PyTorch 26.10 image with transformers 5.2.0 and Pillow 13.0.0): host
preprocessing is exact on all three images (grids 12 x 21, 22 x 11 and 14 x 18);
the embeddings are within 4.2e-07 (images) and 6.4e-07 (texts) of the reference,
the logits within 1.3e-05. The gates (1e-04 per embedding component, 2e-03 on the
logit) were set between that noise and deliberate breaks of the graph or the host
inputs, built and measured on the same case (photograph, two texts):

| Variant | Image max \|diff\| | Text max \|diff\| | Logit \|diff\| |
| --- | --- | --- | --- |
| Correct float32 build | 2.1e-07 | 1.2e-07 | 1.3e-05 |
| Exact GELU instead of the tanh approximation | 1.1e-03 | 4.5e-04 | 2.3e-02 |
| Patches padded to the budget, no mask | 5.5e-03 | - | 1.3e-01 |
| Position table not resampled (first rows) | 1.4e-01 | - | 2.9e+00 |
| Position table resampled transposed | 2.0e-01 | - | 3.7e+00 |
| Patch flattened in (channel, row, column) order | 2.9e-01 | - | 7.7e+00 |
| Pooling-head residual taken after its norm | 3.3e-01 | - | 2.6e+01 |
| Text pooled at the first row of the window | - | 3.2e-01 | 2.9e+01 |

Outside the E2E, the host preprocessing was also compared value for value with
the reference processor on 17 images from 1 x 1 to 3000 x 20 and 2 x 4000 pixels
(identical grids and pixels), and the runtime tokenizer with the reference
tokenizer on 2,614 strings (identical ids), including all 256 single-byte code
points, nine scripts, whitespace runs, added tokens and over-long prompts.

## Assets

`tests/data/test_img.png` is the repository's shared vehicle photograph (see
`ASSET_LICENSES.md`). The checkpoint and its tokenizer are Apache-2.0 and are not
redistributed by this family. The NaFlex preprocessing adapts code from transformers
(Apache-2.0), Pillow (MIT-CMU) and PyTorch (BSD-3-Clause); `NOTICE` gives the sources
and their license texts.
