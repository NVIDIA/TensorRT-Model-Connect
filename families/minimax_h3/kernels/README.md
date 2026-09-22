<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FastH3 VSA sm100a kernel

`primitives.cuh`, `block_sparse_kernel_sm100a.cuh`, and
`block_sparse_launch_sm100a.cuh` are vendored from the Apache-2.0 FastVideo
repository at commit `48a047c05ff4138f20cfa33351499c6ec5945f5d`. They are the
forward-only Blackwell kernel used by the public FastH3 VSA checkpoint.

`vsa_sm100a.cu` is the MiniMax-H3-owned TVM-FFI binding. It owns the complete
trained VSA operation: FP32 tile pooling and routing, per-query Top-K block-map
construction, the vendored sparse-attention kernel, and the learned FP32
compression branch with its BF16 gate. Keeping those coupled operations behind
one BYOK boundary avoids a TensorRT 11.2 Myelin format-selection failure when
the dynamic gather/pooling/GEMM producer is fused through a native TopK layer.

The official recipe calls this kernel `sm100a`, but architecture-specific
cubins do not carry across Blackwell variants. The CMake target defaults to
`103a` for GB300/B300 and accepts `100a` for GB200/B200 through
`TRTMC_MINIMAX_H3_VSA_GPU_ARCH`. The vendored device-body architecture guard
has a local compatibility patch for CUDA 13.0's `sm_103a` pass. TRTMC builds
also enable a fail-closed guard so selecting an architecture without a device
implementation produces a compile error instead of a launchable no-op kernel.
