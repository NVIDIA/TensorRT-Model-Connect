# Qwen

This family builds text-only Qwen decoders (Qwen, Qwen2, Qwen3, QwQ) for
`text_continuation` and Qwen3 embedding models for `embedding`
(`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B-Instruct-2507` and
`Qwen/Qwen3-Embedding-0.6B` in the manifests, fp16 or bf16, tensor parallel
size 1 and 4). Unquantized Qwen3 decoders use the family's native KV-cache
engine; FP8 builds use the family's quantized decoder graph.

## Text continuation Task

This family implements `IModel` and `ITextContinuation` with its own binding
and Config declaration. The existing shared C ABI and header-only C++
`TextContinuation` wrapper discover that binding. No sibling family is imported.
Single-process bundles also retain the upstream streaming Task, cancellation and
exclusive generation lease. Tensor-parallel bundles do not advertise streaming.

The input is UTF-8 text or checkpoint token IDs. Text uses the existing family
tokenizer and optional chat template. Token IDs are passed directly to the
decoder, without decoding/re-tokenizing or applying a chat template. Results own
only the newly generated text and token IDs, plus setup/prefill/decode timings.
The original graph, weights, sampler, native-KV cache and tensor-parallel
implementation remain. Family-local logits tracing still uses the same pipeline.

Build a fresh bundle; the old `text_generation` primary mode is not retained by
this family runtime. The usual CLI remains:

```sh
trtmc build Qwen/Qwen3-0.6B -o model.bundle
trtmc run model.bundle --prompt "Hello" --max-new-tokens 20
```

## Config and defaults

`runtime/task_config.h` owns the complete declaration and conversion. Defaults
remain 128 new tokens, temperature 1, top-k 1, top-p 1, min-p 0, seed -1, checkpoint
EOS, autoregressive mode, chat template off, thinking on, answer-stop off and
stop-check interval 16. An explicit token limit of zero returns an empty
continuation. Unknown or mistyped options are errors, not silently ignored.
`repetition_penalty` accepts only the neutral value 1: the existing sampler does
not implement a non-neutral penalty. No new sampling algorithm is added.
`system_prompt` retains the upstream ChatML behavior and requires
`use_chat_template=true`.

## Validation

The existing Qwen3 and Qwen2.5 checkpoint cases, including FP8, native-KV and
tensor-parallel cases, and their reference criteria remain.
The single-device HF-reference cases also call the direct public C11 and C++17
consumers with both input representations. Their complete token IDs and text
must agree; text input must also agree with the existing CLI. Consumers read
the owned result after releasing the model handle. Sampling and contract-only
cases retain their existing CLI checks. The added checks do not replace the
original reference oracle or certify unexecuted distributed profiles.

```sh
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_qwen test_qwen_task_config
ctest --test-dir build --output-on-failure -R '^qwen_task_config$'
build/test_qwen_sdk_c model.bundle build text "Hello" max_new_tokens=20
build/test_qwen_sdk_cpp model.bundle build text "Hello" max_new_tokens=20
```

The family and Config CTest targets also build both SDK consumers. Existing E2Es use
`TRTMC_NATIVE_BUILD_DIR` to locate them. The server consumer must support semantic
Tasks before serving a migrated family; legacy-interface inheritance is not kept
in this family to work around an unmigrated application.

## Quantization

`quantization=fp8` is supported for dense Qwen3 on two routes: single GPU with
fp16 (dual-profile engine) and tensor parallel size 4 with bf16. The MLP
`up_proj` of every layer runs as an FP8 GEMM (per-tensor weight and activation
scales from ModelOpt `FP8_DEFAULT_CFG`, calibrated at build time on a fixed
prompt set). All other projections, embedding, norms and `lm_head` stay 16 bit.
The manifests cover Qwen3-0.6B with FP8.

### NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family.

Setup: Qwen3-4B-Instruct-2507 and Qwen3-0.6B, fp16, single GPU, one job per
GPU (B200 and B300), TensorRT 11.4. NVFP4 self-quantized from the BF16
checkpoint (block size 16, activations quantized at run time, input scale from
the FP8 calibration). "NVFP4 `up_proj`" quantizes the same projection as the
shipped FP8; "NVFP4 MLP" quantizes gate, up and down. The fp16 engine row uses
the same graph with nothing quantized. Decode is one token with a CUDA graph,
in a 2048-token and a 256-token build. KL is against the PyTorch bf16 model on
771 wikitext-2 tokens. In this measurement the fp16 engine itself already has a
KL of 0.078 to 0.086 (4B) and 0.369 (0.6B), so the loss of a quantized engine is
counted as the KL it adds on top of the fp16 engine.

Qwen3-4B-Instruct-2507:

| GPU | Variant | Plan GiB (2048 build) | Decode ms, 2048 build (vs FP8) | Decode ms, 256 build (vs FP8) | KL vs PyTorch | KL added over fp16 engine | WikiText ppl | Non-finite |
|---|---|---|---|---|---|---|---|---|
| B200 | fp16 engine | 7.52 | 3.813 | 3.105 | 0.0861 | - | 19.91 | 0 |
| B200 | FP8 (shipped) | 6.68 | 3.510 (1.00x) | 2.802 (1.00x) | 0.0764 | -0.0097 | 19.82 | 0 |
| B200 | NVFP4 `up_proj` | 6.32 | 3.520 (1.00x) | 2.950 (0.95x) | 0.0992 | +0.0131 | 19.79 | 0 |
| B200 | NVFP4 MLP | 3.92 | 3.123 (1.12x) | 2.413 (1.16x) | 0.1487 | +0.0626 | 20.50 | 0 |
| B300 | fp16 engine | 8.24 | 3.641 | 2.941 | 0.0777 | - | 19.81 | 0 |
| B300 | FP8 (shipped) | 7.41 | 3.511 (1.00x) | 2.816 (1.00x) | 0.0770 | -0.0007 | 19.76 | 0 |
| B300 | NVFP4 `up_proj` | 6.31 | 3.550 (0.99x) | 2.822 (1.00x) | 0.0961 | +0.0184 | 19.87 | 0 |
| B300 | NVFP4 MLP | 4.64 | 3.162 (1.11x) | 2.551 (1.10x) | 0.1487 | +0.0710 | 20.44 | 0 |

The PyTorch bf16 perplexity is 18.70 (B200 run) and 18.76 (B300 run).

Qwen3-0.6B (same setup): NVFP4 `up_proj` is 0.96x to 0.98x of FP8 at decode,
NVFP4 MLP 1.04x to 1.05x on B200 and 0.98x on B300. Over the fp16 engine
(KL 0.369), FP8 adds 0.005 to 0.009 KL, NVFP4 `up_proj` 0.087 and NVFP4 MLP
0.216 to 0.230.

Reasons:

- Speed: on Qwen3-4B, NVFP4 MLP is 1.10x to 1.16x faster than FP8 at decode
  and the plan is 37% to 41% smaller. NVFP4 on the shipped `up_proj` alone is
  not faster. On Qwen3-0.6B there is no clear gain.
- Quality: FP8 adds no KL over the fp16 engine (-0.010 to -0.001), while NVFP4
  MLP adds 0.063 to 0.071 on Qwen3-4B (KL against PyTorch 1.9x the FP8 KL) and
  perplexity rises by about 0.6 over the fp16 engine. On Qwen3-0.6B NVFP4 MLP
  adds 0.22 KL.
- No non-finite logits or cache values (quality runs and 3 x 1056 stress tokens
  per engine).
