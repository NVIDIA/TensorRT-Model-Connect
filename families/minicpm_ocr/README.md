# MiniCPM-OCR

This family builds `openbmb/MiniCPM-OCR-1.0` (`MiniCPMOcrForConditionalGeneration`) for `images_text_to_text`.
A bundle holds a vision engine that encodes one image piece per call (the 27-layer SigLIP-style tower, then
`post_layernorm` and the 2 x 2 merger MLP of the checkpoint's `4x` mode), a one-token step engine of the Qwen3.5
hybrid text decoder (18 Gated DeltaNet layers and 6 gated full-attention layers with a tied LM head), the checkpoint
tokenizer and `runtime.json`. The native runtime slices, resizes and normalizes images, expands the image
placeholders and renders the chat template as the Hugging Face processor does. Batch is 1. The default precision is
FP16; FP32 is also built and validated.

## Precision

Activations are stored in the build precision. Every matrix product accumulates in FP32, and the normalizations,
softmax inputs and the Gated DeltaNet recurrence run in FP32, as the reference does. FP16 bundles therefore keep their
FP16-rounded weights as FP32 constants (about 3.8 GB; FP32 bundles about 4.8 GB). The decoder's attention is written
out as explicit operations (FP32 scores, additive mask and softmax), so FP32 and FP16 engines apply the mask the same
way.

## Validation

FP32 bundles must reproduce the greedy tokens and the text of the Transformers model run in FP32 with TF32 disabled.
FP16 bundles are checked with a teacher-forced near-tie oracle: fed the prompt and the native tokens, the FP32
reference must rank every native token within 2^-6 logits of its top logit (the official model run in FP16 departs
from its own FP32 argmax once in 2,638 tokens over 25 image / prompt pairs, by 0.00069). In every case the native
text must equal the official decoding of the native tokens.

## Not supported

- Video input, the `16x` downsample mode, BF16 builds, quantized derivatives of the checkpoint (AWQ, GPTQ), batch
  sizes above 1, tensor or context parallelism; the builder refuses them.
- A `tokenizer.json` that declares another normalizer, pre-tokenizer or model type than the checkpoint's; the builder
  refuses it. Tokenization follows the installed ICU's Unicode tables, so characters added in recent Unicode versions
  may tokenize differently from the reference tokenizer.
- The chat template is reproduced for this checkpoint; a derivative checkpoint with another chat template is not
  detected.
- Prefill runs the step engine once per prompt token.
