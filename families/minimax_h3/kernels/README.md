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
