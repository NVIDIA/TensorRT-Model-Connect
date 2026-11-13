# LLaVA 1.5

This family builds `llava-hf/llava-1.5-7b-hf` into native TensorRT engines for
`vision_language_generation`. The checked testcase pins Hugging Face revision
`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`.

The supported path uses unquantized FP16, one GPU, one image per request, and
the checkpoint's 336×336 CLIP input. CLIP's penultimate patch features pass
through the GELU projector and replace 576 image tokens in the Llama prompt.
Image preprocessing reproduces Pillow bicubic resize and center crop. Prompts
are plain user questions; the runtime adds the checkpoint's conversation
template and image tokens.

After preparing the documented source-build environment, build the native
targets `trtmc`, `trtmc_backend_trt`, and `trtmc_model_llava`, then run:

```bash
PYTHONPATH=core/builder:. python3 -m tensorrt_model_connect build \
  llava-hf/llava-1.5-7b-hf \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --precision fp16 --max-sequence-length 1024 --output llava.bundle

build/trtmc run llava.bundle --runtime-root build \
  --image families/llava/tests/data/test_img.jpeg \
  --prompt "What color is the car? Answer with one word." \
  --max-new-tokens 4 --temperature 0
```

The bundle contains `vision.plan`, `prefill.plan`, `engine.plan`, the native
tokenizer, and `runtime.json`. Prefill and decode use separate engine weights;
the FP16 bundle is approximately 25.7 GiB. Prompt tokens, the 576 image tokens,
and requested answer tokens must fit the selected context length. The maximum
checkpoint context is 4,096 tokens.
Cache exhaustion raises an error at the selected context limit. Native cache
unit tests cover zero initialization, allocation failures, and exhaustion:

```bash
cmake --build build --target test_llava_kv_cache
ctest --test-dir build -R '^test_llava_kv_cache_' --output-on-failure
```

For the official E2E, provide `TRTMC_LLAVA_MODEL_DIR`, `TRTMC_BINARY`, and
`TRTMC_RUNTIME_ROOT`. Set `TRTMC_REFERENCE_PYTHON` to an independent CUDA
PyTorch interpreter with Transformers 5.2.0, Pillow, and sentencepiece:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest \
  families/llava/tests --e2e-model llava -q
```

Four image cases require exact greedy token equality with Hugging Face and
the expected semantic answer: white car, outdoors, red, and blue. Two native
input checks separately compare portrait/landscape image processing and
tokenization with Hugging Face. This does not establish performance or
deployment qualification, or support for other LLaVA variants and precisions.
