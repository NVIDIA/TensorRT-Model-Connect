# Qwen3.5: optional native Edge-LLM execution

This family owns checkpoint admission, configuration mapping, bundle composition,
paired execution, runtime orchestration, and validation. Edge owns the complete
network. The optional CMake package is pinned to official GitHub Edge-LLM
**0.10.1 / e8b29522938901f6df19ebeedd4b69bc8edbcd97**. Build on the executing GPU;
there is no cross-compilation or use of an alternate Edge checkout.

## Scope and fallback

The route admits the recorded dense 0.8B/2B/4B/9B text configurations, original
source weights, FP16 compute, batch one, TP/CP one, on native Linux x86 SM80.
It does not admit Qwen3 or older, Qwen3.5 MoE, 27B, quantized sources, additional
platforms, or image/video requests. Unmapped ordinary requests keep the existing
native builder. Edge preparation failures warn and retry that unchanged native
request once; publication failures never switch backends.

An explicit DFlash companion is admitted only for the recorded 4B/9B base
topologies and linear block16 profile. The family checks draft architecture,
hidden/vocabulary widths, target layers, mask token, context capacity, and
unquantized weights. If Edge fails, native reports that the requested DFlash
variant is unsupported; it must never silently return an ordinary base bundle.

## Build and runtime

Enable `TRTMC_ENABLE_EDGELLM=ON` and expose its native installation using
`CMAKE_PREFIX_PATH`. The family checks the Edge pin, SM, CUDA, and TensorRT
identity. Existing ordinary public build requests remain the entrypoint;
paired requests use the existing `execution_variant="dflash"` and explicit
draft-checkpoint input contract.

`edge_llm.py` maps those requests into the pinned Python direct builder
(`experimental.builder.cli`). These recorded FP16 profiles already passed that
flow; they are not additional ONNX conversions. The family packages all required
engine, tokenizer, chat-template, configuration, and external-weight files.
DFlash maps both speculative engines and keeps the draft checkpoint's tensor
payloads while normalizing its safetensors header for the upstream reader.

Ordinary builds preserve the requested context. DFlash caps the input profile
at min(context, 1024), keeps the requested KV capacity, and uses block16 with
one draft step and top-k one. The C++ family adapter invokes the pinned Edge
runtime and preserves the explicit execution variant. Native model math and
its recurrent-state initialization remain unchanged.

## Historical full-model qualification

These are saved local build/inference receipts, **not fresh publication-head
inference**. Every row passed its own local Model Connect build, public/direct
comparison, independent HF comparison under unchanged family criteria, and the
original Edge basic/context-reuse workloads. All eight ordinary runs used
native SM80 FP16, batch one, context4096. The original 9B owning E2E also passed
at its context256 profile with strict CLI JSON parsing.

| Qwen checkpoint | Exact revision | Edge basic ROUGE-1 / ROUGE-L |
| --- | --- | --- |
| Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | 0.3955 / 0.2599 |
| Qwen3.5-0.8B-Base | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 0.3567 / 0.2166 |
| Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 0.3929 / 0.2500 |
| Qwen3.5-2B-Base | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | 0.5455 / 0.3102 |
| Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 0.4485 / 0.2667 |
| Qwen3.5-4B-Base | `1001bb4d826a52d1f399e183466143f4da7b741b` | 0.4246 / 0.2011 |
| Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 0.5116 / 0.2558 |
| Qwen3.5-9B-Base | `68c46c4b3498877f3ef123c856ecfde50c39f404` | 0.4379 / 0.2130 |

The paired receipts use the 4B/9B Instruct revisions above and these z-lab drafts:

| Draft | Exact revision | Speculative iterations | Edge basic ROUGE-1 / ROUGE-L |
| --- | --- | --- | --- |
| Qwen3.5-4B-DFlash | `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf` | 14 | 0.4485 / 0.2667 |
| Qwen3.5-9B-DFlash | `5fc3b3d474760f18c516db87d84c37edbfd3ede6` | 10 | 0.5116 / 0.2558 |

Both used native SM80 FP16, context2048/input1024, block16, top-k one/one
draft step. Public and direct outputs agreed; HF comparisons passed the
unchanged maximum NED0.15 gate. The 9B pair reused the source/reference-verified
CPU-FP32 HF baseline rather than claiming a newly generated reference.

## Existing tests and replay gaps

Only the existing `tests/test_e2e.py` diagnostic handling changes: save native
stdout/stderr, keep current evidence instrumentation, then enforce the return
code and strict JSON parsing. No quality threshold, fixture, timeout, or
comparison is weakened. No new test driver or test framework is published.

The current owning manifests cover 0.8B/2B/4B/9B, not their Base variants or
DFlash pairs. Manifest presence is not a fresh-head model pass. The existing
recurrent-output-initializer C++ test remains registered.

Successful full-model payloads were retired after preserving compact evidence.
Fresh full inference requires restoring these exact source revisions and
rebuilding locally. CPU/source checks, native compilation, and generic CLI
checks do not replace that work or qualify unlisted models. No independent
logit-bias coverage is claimed from an upstream fixture that repeats the
basic workload.
