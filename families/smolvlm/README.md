# SmolVLM2 images and text to text

The `smolvlm` family runs SmolVLM2 (`model_type: smolvlm`,
`SmolVLMForConditionalGeneration`) as the `images_text_to_text` Task: a
conversation of system, user and assistant messages with text and image parts
in, generated text out.

Validated checkpoint: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467` (507.5 M parameters, stored as FP32).

## Build and run

```sh
python -m tensorrt_model_connect build HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --precision fp32 --max-sequence-length 4096 -o smolvlm2-500m.bundle
trtmc run smolvlm2-500m.bundle --image photo.jpg --prompt "Can you describe this image?"
```

`--precision` accepts `fp32` (default, the stored precision), `fp16` and `bf16`;
norms, attention scores and softmax run in FP32 in every precision. FP32 and
FP16 are validated (below). BF16 builds and runs but is not qualified: over the
five E2E and nine more conversations the official BF16 run perturbs the FP32
reference's logit differences about six times as much as the official FP16 run
(up to 0.50) and, in one conversation, picks a token the reference disfavors by
0.149. On the five E2E conversations its departures stay within 0.1 logit, so
the FP16 check below would not by itself reject BF16-level noise.
`--max-sequence-length` sizes the decoder cache (prompt plus generated tokens,
default 4096, at most the checkpoint's 8192 positions). An image split into
4 x 4 tiles takes 1,127 prompt tokens. Every other build option is rejected.

Task configuration: `max_new_tokens` (default 128), `temperature`, `top_k`,
`top_p`, `seed`. The defaults (`top_k` 1) give greedy decoding, the
checkpoint's generation default; `temperature` 0 is greedy as well.

## What the family implements

- **Engines.** `vision.plan` runs the SigLIP-style encoder (patch embedding,
  learned positions, 12 layers), the pixel-shuffle connector (scale factor 4)
  and the modality projection on a batch of up to 17 tiles of 512 x 512, giving
  64 text-space vectors per tile. `decoder.plan` runs the Llama decoder (32
  layers, grouped-query attention, RoPE, untied `lm_head`) over a span of input
  embeddings against fixed per-layer caches, returning the new key/value rows
  and the logits of the last row. Prompts longer than 2,048 tokens prefill in
  chunks. Token embeddings are gathered on the host.
- **Image processing**, ported from the checkpoint's `SmolVLMImageProcessor`:
  resize so the longer side is 2,048 pixels, resize to whole 512-pixel tiles,
  split into rows x columns tiles plus one global 512 x 512 view, rescale and
  normalize. Resampling is Pillow 13's fixed-point Lanczos, including its rule
  of running the vertical pass first when the height shrinks more than twice
  the width; Pillow 12 and earlier always start horizontally, so tall images
  can differ from those releases in the last bit of a pixel. Inputs are
  3-channel RGB, `UInt8` or `Float32` in [0, 1] (rounded to 8 bits, the
  processor's domain).
- **Prompt.** The checkpoint chat template (`<|im_start|>`, `Role:` before an
  image or `Role: ` otherwise, `<end_of_utterance>\n` per message, then
  `Assistant:`), each image expanded to `<fake_token_around_image><row_r_col_c>`
  plus 64 `<image>` tokens per tile, a newline per tile row, and the global
  view. Text may not contain the literal `<image>` token.
- **Tokenizer.** The checkpoint's byte-level BPE: literal special tokens,
  digits split one by one, the GPT-2 split expression, rank-ordered merges.
  Like the reference, bytes without a vocabulary unit (21 control and invalid
  UTF-8 bytes) are dropped.

The shared `images_text_to_text` contract requires at least one image and one
text part per request; text-only conversations are rejected before they reach
the family.

Not implemented: video input (the checkpoint also accepts videos; this family
exposes images only), tool definitions and tool messages, reasoning parts,
batches of conversations, `--kv-cache-size` and every other build option.

## Validation

`tests/test_e2e.py` builds the bundle from the pinned checkpoint, runs five
conversations on the repository's test photograph, its mirror image and a
portrait strip of it (one image, two images, a system message plus an image, a
three-turn exchange with an assistant answer, and three images whose 2,300-token
prompt prefills in two chunks) through the public SDK
(`tests/cpp/run_model.cpp`) and compares with the
Transformers reference (FP32, eager attention, TF32 off) on the same GPU:

- the native image processor reproduces the reference `pixel_values` bit for
  bit, and the native chat rendering and tokenizer reproduce `input_ids`
  exactly (`tests/cpp/runtime_probe.cpp`);
- FP32 bundles must reproduce the reference's greedy token ids exactly;
- FP16 bundles are checked token by token with teacher forcing: every
  generated token must be the FP32 reference's greedy choice given the bundle's
  own prefix, or trail it by at most 0.1 logit (`tests/thresholds/`); a token
  the reference disfavors by more fails. The tolerance comes from the official
  PyTorch FP16 run, taken as a stand-in for TensorRT FP16: over 746 steps of 14
  conversations it perturbs the logit differences among the reference's top
  five tokens by at most 0.089 (p99 0.048). Free-running texts are not
  compared: the reference has near-ties in several conversations (top-2 margins
  of 0.010 to 0.18 within the first 16 tokens, 0.001 at token 59 of the first),
  after which greedy texts diverge without any error.

```sh
TRTMC_BINARY=<build>/trtmc TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  python -m pytest families/smolvlm/tests --e2e-model smolvlm -vv
```

The reference turns TF32 off and checks that FP32 matrix products run at full
precision; NGC PyTorch images enable TF32 by default.

CPU tests: `test_support.py`, `test_config.py`, `test_oracle.py` and
`test_fixtures.py` (the C++ unit fixtures recomputed with Pillow 13, the
Transformers processor and `tokenizers`), plus the CTest targets
`test_smolvlm_media` and `test_smolvlm_tokenizer`. `requirements.txt` lists what the reference
processor needs beyond the base environment.
