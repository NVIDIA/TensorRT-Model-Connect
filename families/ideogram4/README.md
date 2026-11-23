# Ideogram4 FP8

This family serves the `ideogram-ai/ideogram-4-fp8` foundation checkpoint through
the native `text_to_image` task. It builds the Qwen3-VL text encoder, conditional
and unconditional transformers, and Flux2 VAE decoder into TensorRT plans.
The runtime runs both guidance branches and the publisher's `V4_DEFAULT_20`
flow-matching sampler in C++.

The build precision is BF16. The publisher's FP8 weights use per-row scales;
dequantization preserves the reference's BF16 rounding before matrix
multiplication. Graph export uses the pinned official implementation declared
in `requirements.txt`; inference runs the native plans without Python.
Normalization and attention accumulate in FP32 with explicit BF16 boundaries.

The prompt is a structured JSON caption string, as required by the foundation
model. The native tokenizer applies NFC normalization, byte-level BPE, and the
published user/assistant chat framing. Prompt enhancement belongs to the
application. Overlong framed captions are rejected without truncation.

Execution supports one image on one GPU. Build dimensions default to 1024 by
1024 and must be multiples of 16, at least 256. The text profile defaults to
256 tokens and can be set up to 2048. Image dimensions and the 20-step sampler
are fixed by the bundle. Engines use the actual caption length followed by
masked image slots, matching the publisher's packed sequence; the text capacity
bounds the profile without adding padding tokens. Sampling seeds select native
initial noise;
`initial_latents` allows exact replay in `[1, H/16 * W/16, 128]` order across
backends. The native random generator does not reproduce PyTorch's generator.

The E2E manifest pins the actual checkpoint and compares 512 by 512 image
generation against the original FP8 PyTorch components and sampler with
identical explicit initial latents. It also checks the native tokenizer against
the checkpoint tokenizer. Reports and generated images are runtime artifacts.
