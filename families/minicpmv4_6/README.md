# MiniCPM-V 4.6

This family builds [`openbmb/MiniCPM-V-4.6`](https://huggingface.co/openbmb/MiniCPM-V-4.6)
(`MiniCPMV4_6ForConditionalGeneration`, 1.3 B parameters, BF16 checkpoint) for
`images_text_to_text`. The model pairs a SigLIP-style vision tower, whose
encoder merges every 2 x 2 window of patches into one token after layer
`insert_layer_id`, and an MLP projector with a Qwen3.5 hybrid decoder (Gated
DeltaNet and gated full-attention layers).

## Bundle

| Section | Content |
|---|---|
| `engine.plan` | one-token decoder: the KV cache of the attention layers and the FP32 DeltaNet convolution and recurrent state in, present tensors and logits out |
| `prefill.plan` | chunked prefill of 1 to 64 prompt rows per launch; a row takes its token embedding or a vision feature row |
| `vision.plan` | one image slice per launch (16 to 4096 patches): vision tower, windowed merger and projector |
| `runtime.json`, `vision.json` | decoder geometry and stop ids; image-processor settings, markup token ids and the chat-template recipes |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `preprocessor_config.json` | copied from the checkpoint |

Build with `--precision fp16` (the default) or `--precision fp32`. The DeltaNet
recurrence and its state stay FP32 at both precisions, and FP32 math never uses
TF32. `max_sequence_length` (default 4096) bounds the prompt, including the
image tokens, plus the generated tokens.

## Tasks

`images_text_to_text` takes system, user and assistant messages with ordered
text and image parts (several images per request are supported). The runtime
reproduces the checkpoint's processor: each image becomes a source thumbnail
plus up to `max_slice_nums` (9) slices of about 448 x 448 pixels, every slice
is resized with antialiased bicubic resampling, encoded separately and
represented by `<image_id>`, `<image>` and `<slice>` markup around its image
tokens (16x token compression, 64 tokens for a 448 x 448 slice). The chat
template is applied as the checkpoint defines it, with thinking disabled
unless `enable_thinking` is set.

The loaded bundle also offers `text_continuation` for text-only prompts (the
CLI selects it when no image is given); `use_chat_template` formats the prefix
as one user message.

Config fields of both tasks:

| Field | Default | Meaning |
|---|---|---|
| `max_new_tokens` | 128 | generated token budget |
| `temperature` | 1.0 | sampling temperature; 0 is greedy |
| `top_k` | 1 | 1 is greedy, 0 keeps the whole vocabulary |
| `top_p` | 1.0 | nucleus sampling mass |
| `seed` | -1 | sampling seed; -1 uses a fixed default |
| `enable_thinking` | false | chat-template thinking mode |
| `use_chat_template` | false | `text_continuation` only |

```bash
trtmc build openbmb/MiniCPM-V-4.6 -o minicpm-v-4.6.bundle
trtmc run minicpm-v-4.6.bundle --image car.jpg --prompt "What color is the car? Answer in one word."
```

## Not supported

- `video_text_to_text` and video inputs.
- The processor's `downsample_mode="4x"` (finer images with four times the
  image tokens); the bundle implements the default 16x compression.
- Tool definitions, tool calls and reasoning parts in requests.
- Images whose source thumbnail or slices exceed 4096 patches (aspect ratios
  beyond about 1000:1, for example 14 x 20000 pixels).
- A `max_slice_nums` above 4096 (the checkpoint sets 9).
- Images that are not RGB (grayscale or RGBA buffers), batches larger than one,
  quantized builds and tensor parallelism.

## Validation

`tests/test_e2e.py` builds each manifest's bundle once from the pinned
checkpoint, resolved offline from the local Hugging Face cache as CI resolves it
(download it first, as below), runs the native CLI and compares the generated tokens with the
Transformers reference (`requirements.txt`) run in FP32 on the same decoded
pixels: the FP32 bundle must reproduce the reference's greedy tokens exactly;
for the FP16 bundle the reference re-scores the native tokens and each must be
its argmax or within the calibrated logit tolerance. The case marked `sdk`
also runs the public C and C++ SDK consumers (`tests/sdk_consumer.c`,
`tests/sdk_consumer.cpp`) and requires the CLI's tokens.
`tests/cpp/test_minicpmv4_6_host.cpp` checks image slicing, resampling and
grid indices, placeholder text, conversation formatting and tokenization on the
host, including `tokenizers`' Unicode tables: version 9.0 for NFC and 16.0 for
the character classes of the split pattern.

```bash
hf download openbmb/MiniCPM-V-4.6 --revision 97328f3c2d70c4645b6a35594807eb6932068635
HF_HUB_OFFLINE=1 TRTMC_BINARY=<build>/trtmc TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  python -m pytest families/minicpmv4_6/tests --e2e-model minicpmv4_6 -vv
```
