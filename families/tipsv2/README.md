# TIPSv2

[TIPSv2](https://huggingface.co/google/tipsv2-so400m14) (Text-Image Pre-training with Spatial
awareness) is a contrastive vision-language model: a DINOv2-shaped ViT and a 27-layer text
transformer that meet at a dot product in one 1152-wide space.

The family serves three Tasks from one bundle — `image_to_embedding`, `text_to_embedding` and
`text_image_to_relevance` (the default) — and builds at `fp32` or `fp16`.

```bash
trtmc build google/tipsv2-so400m14 --task text_image_to_relevance --precision fp32 -o tipsv2.bundle
trtmc rerank tipsv2.bundle --query "a photo of a car" --image photo.png
```

## Two engines, one space

The bundle carries a `vision.plan` and a `text.plan`. Fusing them would make every call run both
towers, which is the wrong shape for three Tasks of which two need only one tower. Neither engine
normalises its output and neither applies the temperature: the runtime owns the L2 normalisation
and the division, so what leaves an engine stays comparable with what the reference returns.

## What distinguishes this checkpoint

Each of these produces a same-shaped wrong answer if a neighbouring family's habit is carried
over.

- **No ImageNet normalisation.** The towers are trained on pixels in `[0, 1]`: resize to 448x448
  with Pillow's bilinear filter, divide by 255, stop. Subtracting a mean here moves every pixel
  and nothing loudly.
- **The register token is inserted after the position table is added**, between the class row and
  the patch rows, so it carries no position of its own. Concatenating it first reads the whole
  table off by one patch.
- **The vision MLP is the error-function GELU** (`nn.GELU()` with its default
  `approximate="none"`), not the tanh approximation that CLIP- and SigLIP-shaped towers use. The
  two agree to about 1e-3.
- **Each residual branch is multiplied by a learned LayerScale vector** before it is added. It is
  exactly one on the released so400m checkpoint, so it is a no-op there, but it is applied from
  the weights rather than assumed away.
- **The text tower has no learned position table**: the position signal is a computed sinusoid,
  sines then cosines, not interleaved.
- **The text tower's padding mask is used three times** — the attention keys, both MLP maskings,
  and the pooled average. Because of all three, the embedding of a prompt that fits does not
  depend on how much padding follows it. `tests/test_model.py` runs the tower over two window
  lengths and compares, then does it again with each masking disabled in turn: the embedding
  then moves by 0.26 to 7.6, against 2.7e-07 when all three are in place.
- **The vision head count comes from the reference's factory table**, keyed on
  `(hidden_size, num_hidden_layers)`, not from `vision_config.num_attention_heads`, which the
  reference ignores. `config.py` refuses a file that disagrees with the factory rather than
  believing either one silently.

## The tokenizer

The checkpoint ships a SentencePiece **BPE** model (`tokenizer.model`, byte fallback, the
`nmt_nfkc` normalizer) and no `tokenizer.json`. `tokenizer_export.py` flattens it at build time
into one record the runtime reads without protobuf or SentencePiece: the pieces with their merge
ranks, the precompiled charsmap verbatim, and a Unicode lowercase table — the checkpoint's
tokenizer lower-cases with Python's full `str.lower()` before SentencePiece normalises, which is
not an ASCII fold. `runtime.json` records which Unicode release the table came from.

One character does not fit a table and is the reason the `Cased` and `Case_Ignorable` sets travel
with it: a word-final capital sigma lowercases to U+03C2 and every other one to U+03C3, so
`"ΟΣ".lower()` is `"ος"`. Getting it wrong changes the token ids of every Greek word that ends in
a sigma.

`tipsv2_tokenizer_check` diffs the bundle's own tokenizer against the reference SentencePiece
processor; `tests/test_e2e.py` runs it on prompts chosen for the parts an embedding comparison
cannot reach on its own (the lowercase table, NFKC, byte fallback, whitespace collapsing,
truncation, and text that spells a special token).

## Precision

`fp32` is the default and matches the stored checkpoint. `fp16` halves the constants; the
projections and the MLPs run at half width and these five places do not, each because of a named
failure rather than as a precaution:

- the **pooled average** of the text tower. The reference divides by `count + 1e-8`, and 1e-8 is
  below FP16's smallest normal, so in FP16 the epsilon would vanish and an all-padding prompt
  would divide zero by zero;
- the whole **attention** — scores, mask, softmax and context. A score is an unbounded sum over
  the head width, and an attention without a fused kernel accumulates it in half precision, which
  both overflows on large rows and drops the small terms of a softmax over 1,026 keys. The mask
  is also a large *finite* negative number rather than the reference's `-inf`, so a masked score
  against a zero weight is a zero and not a NaN;
- the **GELU**, because `1 + erf(x)` cancels: at x = −4 the erf rounds to exactly −1 in FP16 and
  the activation returns 0 where the answer is −1.27e−4, a value FP16 represents perfectly well;
- the **layer-norm statistics**, a sum over the full 1152-wide row;
- the **engine outputs**, which are FP32 whatever the engine computed in, so the runtime's L2
  normalisation starts from the widest value the engine has.

The text tower's MLP masks **select** their rows rather than multiplying by a 0/1 factor. For a
finite value the two are the same; for an overflowed one the product is a NaN that would spread
through the residual stream into every later attention.

A prompt that produces no token is refused rather than scored: the reference averages over zero
rows and returns the zero vector, which has no direction to compare. A prompt far longer than the
window can hold is refused too, rather than normalised and tokenised in full before being cut.

## Validation

`tests/test_e2e.py` compares, for each manifest case, the image embedding, both text embeddings,
the relevance logit and the zero-shot top class against the checkpoint's own Transformers code,
and runs the tokenizer diff. `tests/cpp/test_runtime.cpp` pins Pillow's bilinear resize against
values recorded from Pillow itself and the tokenizer's rules on a miniature model.

The thresholds come from measurement, not from habit. Three structural errors were built on
purpose and run through the same fp32 case, and what they cost the image embedding is what the
gate is set against:

| quantity | correct | tanh GELU | register given a position | q/v/k split |
|---|---|---|---|---|
| image embedding max abs | 1.08e-07 | 1.61e-04 | 1.52e-02 | 3.66e-01 |
| image embedding cosine | 1.0000000 | 0.9999990 | 0.9861881 | 0.2587784 |
| zero-shot top class | agrees | agrees | agrees | agrees |

So the gate is the maximum absolute difference, at 1e-5. Neither of the other two quantities
could carry it: no cosine floor one would dare set separates 0.9999990 from 1.0, and the top
class agrees even for a tower that reads its own attention projection in the wrong order.

The fp32 case is the structural gate and the fp16 case is not. A correct fp16 build already
differs from the reference by 3.70e-04, ten times the weakest break, and the same three breaks
measured at fp16 confirm it: the GELU one lands at 3.71e-04 and the case passes with the break in
place. The fp16 tolerances gate the lowering, not the graph.

`families/tipsv2/requirements.txt` pins Pillow deliberately. The runtime reproduces one resampler
coefficient for coefficient, including which axis Pillow resamples first — observable because its
intermediate is 8-bit, and a rule that changed in Pillow 13. `tests/test_model.py` recomputes the
recorded values with the installed Pillow so a drift in that pin fails there rather than in an
embedding, and the recorded cases cover both branches of the rule.

## Not implemented

- The spatial patch features that give TIPSv2 its name. The vision engine computes them — the
  final layer norm runs over the whole sequence — but only the class row is marked as an output,
  because no Task here consumes them. A family that adds `image_to_spatial_features` would mark
  the patch rows as a second output of the same engine.
- `google/tipsv2-*-dpt`, the dense-prediction siblings. They reuse these towers under depth,
  normal and segmentation heads, declare `model_type: tipsv2_dpt`, and are refused by name here.
- The B/14, L/14 and g/14 variants. Their head counts and MLP ratios come from other entries of
  the reference's factory table; `config.py` refuses a geometry it has not been measured on.
- Quantization. The checkpoint is float32 throughout and no quantized variant is published.
