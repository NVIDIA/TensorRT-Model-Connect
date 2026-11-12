# Gemma 3n

This family builds `Gemma3nForConditionalGeneration` checkpoints (`model_type: gemma3n`) for the
`images_text_to_text` Task: one or more RGB images and text in a chat conversation, greedy or
sampled text out, through the CLI (`trtmc run BUNDLE --image IMAGE --prompt TEXT`) and the C/C++ SDK.

Validated checkpoint: [`unsloth/gemma-3n-E4B-it`](https://huggingface.co/unsloth/gemma-3n-E4B-it)
at revision `45e9fb1dd0e34db5ff9db1f43a49ac5d8e8b8778` (7.85 B parameters stored as BF16).

## What the bundle contains

- `vision.plan`: the MobileNetV5-300M image encoder (768 x 768 input, multi-scale fusion to a 16 x 16
  grid) and the multimodal embedder that projects its 256 soft tokens into the text space.
- `decoder.plan`: the text decoder with one prefill and one decode optimization profile. It keeps the
  four AltUp streams, the LAuReL blocks, the per-layer embeddings (one token table per layer), the
  Gaussian top-k activation sparsity of the first layers, sliding-window and global attention, and
  the KV sharing of the last layers (they attend to the keys and values of the last earlier layer of
  the same attention type, so only those owners have a cache). Token ids of the vision and audio
  vocabularies use the embedders' hard embeddings, as in the reference model.
- `tokenizer.json`, `runtime.json` and the checkpoint's configuration files.

The runtime renders the chat template, tokenizes, resizes images (antialiased bilinear on 8-bit
pixels, as the reference image processor's torchvision path does) and runs the decode loop. The
build refuses tokenizer pipelines and image-processor settings the runtime does not implement; chat
templates that render a set of probe conversations differently from the runtime (rendered with
Jinja2 as Transformers does); a tokenizer BOS token other than `<bos>`, with which Transformers adds
special tokens to the rendered prompt; and a language-model head that is not tied to the token
embeddings.

## Precision and limits

- `bf16` (default, the stored precision) and `fp32`. `fp16` is refused: decoder activations exceed
  its range.
- `max_sequence_length` (default 2048) bounds the prompt plus generated tokens; one image takes
  260 tokens of it.
- Requests carry text and image parts in user/assistant turns with an optional leading system
  message; tool definitions, tool calls and reasoning parts are rejected. Images are RGB, as 8-bit
  or float32 `[0, 1]` pixels (float32 pixels are quantized to 8 bits first).
- Generation settings: `max_new_tokens`, `temperature` (0 is greedy), `top_k`, `top_p`, `seed`.
  Generation stops at `<end_of_turn>` or `<eos>`.
- The audio tower is not built; audio inputs are not part of this Task.

## Validation

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest families/gemma3n/tests/test_support.py
TRTMC_BINARY=<build>/trtmc TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest families/gemma3n/tests \
  --e2e-model gemma3n -vv
```

The E2E builds each manifest's bundle and runs five conversations (one image, a system prompt, two
images, a multi-turn follow-up and float32 pixels) through the C++ SDK consumer, one of them through
the C SDK consumer and the CLI, which must return the same tokens and text. Against the Transformers
model in FP32, the native prompt token ids of every conversation (on the checkpoint's tokenizer) must
be the reference processor's, and the native text the reference tokenizer's decoding of the native
tokens; FP32 bundles must reproduce the reference's greedy tokens exactly; BF16 bundles are checked teacher-forced against
the FP32 reference within the logit margin in `tests/thresholds/`, which is the reference model's own
BF16 error on the gap between its two best tokens along these conversations. `test_gemma3n_processing`
(CTest) checks the tokenizer, the image resize and the chat rendering against goldens recorded from
the reference stack, and invariants of the sampler (greedy first maximum, top-k 1, seeded
reproducibility, a tiny nucleus).

The image resize is adapted from PyTorch (BSD-3-Clause), which derives it from Pillow-SIMD
(MIT-CMU); see `NOTICE`.
