# SigLIP

The family serves `image_to_embedding`, `text_to_embedding`, and
`text_image_to_relevance` from one bundle containing separate text and vision
engines. The FP32 manifests cover Google's SigLIP base and SigLIP 2 base
patch16 checkpoints at 224 pixels.

SigLIP uses Unigram tokenization and Pillow bicubic resize. The fixed-resolution
SigLIP 2 checkpoint uses Gemma BPE with byte fallback and Pillow bilinear resize.
The BPE implementation accepts the checkpoint's declared space replacement,
split, decoder, added-token options, and EOS template. Unsupported processing
is rejected. Padding and truncation remain owned by the runtime's fixed text
window. This does not add NaFlex or a different model architecture.

## Validation

The standard E2E test builds the selected manifest and compares all three Task
outputs with the official reference, using the existing FP32 tolerances:

```sh
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest families/siglip/tests \
  --e2e-model siglip2-base-patch16-224 -q
```

This requires the native runtime, TensorRT, CUDA, and model weights. CPU tokenizer
and preprocessing tests do not establish engine, embedding, score, or performance
parity. SigLIP 2 GPU validation is still required; its new manifest selects FP32
without quantization. BPE checkpoints reject FP8 because the existing SigLIP
activation scales are not calibrated for them.

The native `siglip_bpe_tokenizer` CTest runs the small self-contained BPE fixture
and a Unigram regression. `siglip_image_preprocess_seam` compares both resize
filters against Pillow pixels, including upsampling, downsampling, mixed axes,
and single-pixel dimensions.

For the pinned official tokenizer comparison, download only its tokenizer asset:

```sh
hf download google/siglip2-base-patch16-224 tokenizer.json \
  --revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 --local-dir /tmp/siglip2-tokenizer
build/families/siglip/test_siglip_bpe_tokenizer \
  families/siglip/tests/data/siglip2-tokenizer-golden.json \
  /tmp/siglip2-tokenizer/tokenizer.json
```

The golden IDs and decoded strings were generated with `tokenizers==0.22.2`,
after `Tokenizer.from_file(...)`, `no_padding()`, and `no_truncation()`, using
`encode(text, add_special_tokens=False)`, `encode(text)`, and `decode(ids)`.
The checkpoint-shape fixture records selected safetensors header shapes, not
tensor values. Official assets are from Google's Apache-2.0 model repository
at the revision above; the full tokenizer and weights are not vendored.
