# BLIP image captioning

This family implements `Salesforce/blip-image-captioning-base` at revision
`82a37760796d32b1411fe092ab5d4e227313294b` as the `images_text_to_text` task.
It accepts one RGB image and an optional caption prefix in one user message.
The result contains the generated continuation and token IDs, including the
terminating SEP token. It excludes the conditioning prefix from the result.

The builder produces an FP32 vision encoder and an FP32 causal text decoder.
The decoder recomputes the complete prefix at each step; it does not maintain
a KV cache. The image encoder runs once per request. All topology,
preprocessing, WordPiece tokenization, and generation behavior belong to this
family. TensorRT owns GPU lowering.

## Build and run

Use the repository's [source build instructions](../../website/docs/getting-started/source-build.md)
to prepare a matching CUDA/TensorRT environment. Install this family's
`requirements.txt`, then build the selected native targets:

```bash
cmake --build build --parallel 8 --target trtmc trtmc_backend_trt trtmc_model_blip
PYTHONPATH=core/builder:. python -m tensorrt_model_connect build \
  Salesforce/blip-image-captioning-base \
  --revision 82a37760796d32b1411fe092ab5d4e227313294b \
  --precision fp32 --max-sequence-length 64 -o blip-base.bundle
build/trtmc run blip-base.bundle --runtime-root build \
  --image /path/to/image.jpg --prompt "" --set max_new_tokens=32
build/trtmc run blip-base.bundle --runtime-root build \
  --image /path/to/image.jpg --prompt "a photography of" --set max_new_tokens=32
```

The checkpoint publishes a PyTorch state dictionary, loaded with
`torch.load(weights_only=True)`. BLIP VQA, image-text retrieval, and BLIP-2 are
separate architectures and are not claimed by this family.

Only greedy generation is exposed. `max_new_tokens` defaults to 20, and the
prefix plus output budget must fit the bundle's sequence capacity (64 by
default, at most the checkpoint's 512-position table). FP16, quantization,
batching, tensor/context parallelism, videos, tools, and conversation histories
are rejected. Image resizing uses Pillow-compatible bicubic filtering to
384 by 384; RGB input is normalized with the checkpoint's channel statistics.

## Validation

The family manifest pins the checkpoint revision and the immutable storage
generation of the upstream example image. The tests compare an explicitly
supplied input image byte for byte with that reference copy. The image is
downloaded for selected E2E tests; it is not redistributed in this repository.
For offline runs, set `TRTMC_BLIP_REFERENCE_IMAGE` to an existing verified copy
of the pinned image and `TRTMC_BLIP_MODEL_DIR` to the pinned local checkpoint.
`TRTMC_BLIP_IMAGE` optionally supplies a separate input copy. Otherwise the
test requires the pinned checkpoint in the Hugging Face cache and downloads
the example image.

```bash
cmake -S . -B build -DTRTMC_BUILD_TESTS=ON
cmake --build build --parallel 8 --target trtmc trtmc_backend_trt \
  trtmc_model_blip test_blip_native
TRTMC_BINARY="$PWD/build/trtmc" TRTMC_RUNTIME_ROOT="$PWD/build" \
  PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/blip/tests --e2e-model blip -q -s
```

The four cases cover the original and mirrored image, each with and without
a caption prefix. Gates require exact greedy continuation tokens, native
preprocessing parity, tokenizer parity, repeatable requests after intervening
valid and invalid requests, and numerical comparison of the vision features
and teacher-forced decoder logits. The image-feature tolerance is
`atol=5e-4, rtol=1e-4`; the decoder tolerance is `atol=1e-3, rtol=1e-4`.
No performance or deployment qualification is implied by a correctness run.
