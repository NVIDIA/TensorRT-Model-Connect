<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3-Next

**Status: experimental native execution; numerical qualification is incomplete.**
The complete pinned BF16 checkpoint builds and generates the expected coding
completion through the native Task, including repeated-request replay. The
strict end-to-end oracle currently fails: maximum relative logit error is
9.91% against the unchanged 5% limit, with minimum cosine similarity 0.99543
against 0.999. Top-token agreement is 100% on the four-step testcase. The
premerge numerical test remains enabled and blocking; this is not a qualified
production support claim.

This family owns `Qwen/Qwen3-Coder-Next`, revision
`a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb`, through the
`Qwen3NextForCausalLM` / `qwen3_next` metadata contract.

The BF16 build consumes every checkpoint tensor and produces one engine per
hybrid decoder layer plus the final normalization and vocabulary projection.
TensorRT lowers all graph operations, including routed experts, gated attention,
and the Gated DeltaNet recurrence. The native Task owns tokenization, request
state, sampling, and the upstream single-user chat template.

Dense projections accumulate FP32 products before rounding their outputs to
BF16. Routed experts preserve the upstream grouped execution's FP32 weighted
sum and single BF16 output rounding. The runtime resolves ordered TopK ties
using the reference CPU selection policy; if GPU selection differs, it repeats
the layer with forced expert IDs and commits only the final forward's state.
Bundles built before this routing contract must be rebuilt.

The implemented execution contract is text generation, batch one, one GPU,
BF16 weights and activations with explicit upstream FP32 boundaries. The default
built capacity is 1,024 tokens; the requested prompt plus output budget must fit.
Prefill advances the same token recurrence as decode. This first path prioritizes
numerical validation; it does not implement batched prefill, tensor parallelism,
quantization, LoRA, or tool-call rendering.

Build using the family-owned public command:

```bash
trtmc qwen3_next build Qwen/Qwen3-Coder-Next \
  --revision a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb \
  --max-sequence-length 1024 --output qwen3-coder-next-bf16.bundle
```

Configure the source build with `TRTMC_BUILD_TESTS=ON` and build `trtmc`,
`trtmc_backend_trt`, `trtmc_model_qwen3_next`, `qwen3_next_logits_trace`, and
`qwen3_next_tokenizer_probe`, and `qwen3_next_routing_probe`. The tokenizer
requires ICU uc and i18n.

The family tests use the official Transformers 5.2.0 implementation as the
independent BF16 oracle. The layer tests compare actual checkpoint DeltaNet and
attention/MoE layers; the end-to-end test compares full-vocabulary logits under
teacher forcing and verifies native Task request replay and tokenizer parity.
The checked-in threshold sidecar fixes numerical acceptance independently of
run results.

The latest full family run has 29 passing tests and one failing full-model
accuracy test. Diagnostic runs of the unchanged official BF16 model differ by
up to 8.76% in relative logits when only the CPU matrix backend changes; the
default-backend control exactly reproduces the frozen oracle. This sensitivity
does not establish native parity or change the blocking acceptance criteria.

```bash
export TRTMC_QWEN3_NEXT_CHECKPOINT=/path/to/pinned/checkpoint
export TRTMC_RUNTIME_ROOT=$PWD/build
export TRTMC_BINARY=$TRTMC_RUNTIME_ROOT/trtmc
# Optional: reuse an existing public-build bundle; provenance is checked.
export TRTMC_QWEN3_NEXT_BUNDLE=/path/to/qwen3-coder-next-bf16.bundle
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/qwen3_next/tests --e2e-model qwen3_next -q
```

Large checkpoints require enough CPU memory for the official full-model oracle
and enough GPU memory for all BF16 native engines. Run reports and large
artifacts belong outside the source tree.
