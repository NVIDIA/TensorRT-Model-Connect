<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FastH3 VSA kernels

`primitives.cuh`, `block_sparse_kernel_sm100a.cuh`, and
`block_sparse_launch_sm100a.cuh` are vendored from the Apache-2.0 FastVideo
repository at commit `48a047c05ff4138f20cfa33351499c6ec5945f5d`. They are the
forward-only Blackwell kernel used by the public FastH3 VSA checkpoint.

`vsa.cu` is the MiniMax-H3-owned TVM-FFI binding. It owns the complete
trained VSA operation: FP32 tile pooling and routing, per-query Top-K block-map
construction, the vendored sparse-attention kernel, and the learned FP32
compression branch with its BF16 gate. Keeping those coupled operations behind
one BYOK boundary avoids a TensorRT 11.2 Myelin format-selection failure when
the dynamic gather/pooling/GEMM producer is fused through a native TopK layer.

The binding always includes a portable SM80+ CUDA implementation of the
block-sparse attention step. A build can additionally include optimized
Blackwell cubins. Runtime selection is automatic: the optimized kernel is used
on `sm100a` and `sm103a`, and every other supported CUDA GPU uses the generic
implementation. Set `TRTMC_MINIMAX_H3_VSA_BACKEND` to
`generic` or `blackwell` to force a route for validation; the default is `auto`.
An explicitly requested unavailable route fails with a diagnostic instead of
silently changing the model semantics.

The TensorRT engine registration remains `minimax_h3.vsa_sm100a` for
compatibility with the official checkpoint contract and existing serialized
engines. The name identifies the trained VSA operation; it no longer means
that every runtime invocation must execute the architecture-specific cubin.

The official recipe calls the optimized kernel `sm100a`, but architecture-specific
cubins do not carry across Blackwell variants. Enabling
`TRTMC_BUILD_MINIMAX_H3_VSA_SM100A` embeds both `sm_100a` (GB200/B200) and
`sm_103a` (GB300/B300) cubins. The vendored device-body architecture guard has
a local compatibility patch for CUDA 13.0's `sm_103a` pass. Every VSA build
also embeds portable compute-80 PTX, so the same library can JIT the generic
fallback on newer SM80+ CUDA GPUs.

## Memory profiles

MiniMax-H3 executes the text encoder, denoiser, audio VAE, and video VAE
sequentially. The runtime releases each TensorRT module after its last use, so
their weights and execution contexts do not overlap in device memory. This
changes module residency only; schedules, tensor values, precision, resolution,
and frame count are unchanged.

TensorRT weight streaming is an optional, build-time memory/latency trade-off.
Choose one of the two user-facing profiles:

- `--weight-streaming-mode full-residency` keeps the latency-oriented default.
  Omitting the option has the same behavior.
- `--weight-streaming-mode min-residency` makes the text encoder and monolithic
  denoiser streamable and selects the lowest supported GPU weight residency.

The profile is fixed when the bundle is built, so deployments that target both
latency-oriented and memory-constrained devices should build one bundle for
each profile. Advanced users can instead pass
`--weight-streaming-budget-bytes BYTES` to cap each streamable component at a
custom residency budget. The profile and explicit budget options are mutually
exclusive. Weight streaming does not apply to split FirstBlockCache plans.
