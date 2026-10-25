<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniRef VSA TVM-FFI binding

`omniref_vsa.cu` implements the eight-step OmniRef student's 128-token VSA
operation behind the model-agnostic TVM-FFI bridge. It accepts tiled BF16 Q/K/V
and learned gates, INT32 valid sizes, and one `[begin,end,keep]` descriptor per
video region. Dense queries retain all keys; video queries retain the dense
prefix and an independent Top-K from each video region. Routing, compression
and attention retain the same 128-token geometry on every backend.

The compression epilogue rounds the compressed vector to BF16, then performs
the multiply/add before one final BF16 rounding, matching the released
`addcmul_` boundary. This differs from the old four-step T2V binding.

The binding contains a generic SM80+ CUDA implementation. Enable
`TRTMC_BUILD_MINIMAX_H3_OMNIREF_VSA_SM100A` to also embed optimized sm100a and
sm103a cubins; runtime `auto` selects the optimized route when available.
`TRTMC_MINIMAX_H3_VSA_BACKEND=generic` or `blackwell` forces a route for
validation. An unavailable explicitly requested route fails.

`primitives.cuh`, `block_sparse_kernel_sm100a.cuh` and
`block_sparse_launch_sm100a.cuh` originate from Apache-2.0 FastVideo commit
`48a047c05ff4138f20cfa33351499c6ec5945f5d`, preserving the existing sm103a
compatibility patch. The new binding selects their 128-token instantiation.
Its optimized 128-token PV operation stores the softmax probability as two
BF16 limbs: the rounded value and its rounded residual. It accumulates both
products with V into the existing FP32 output tile. The residual uses the
unused half of the QK tensor-memory tile, so the external workspace and BF16
input/output contract stay unchanged. This adds PV computation; numerical
agreement and its runtime cost require separate GPU measurements.
The 128-token softmax uses the hardware exponential rather than the released
cubic emulation, whose approximation error can surface after gated cancellation.
The old four-step kernel and its serialized registration are unchanged.

The family runtime must validate transport geometry before uploading it.
Every dense segment is tile-pure, video spans are positive and contiguous,
each region has at most 1024 tiles, and the total block count is even. The
only internal extra block has zero valid rows and is excluded from logical
selection. See `reference_vsa.py` for the CPU geometry contract.

This implementation is experimental. Source presence and successful
compilation do not establish GPU numerical agreement, trained-model quality,
or realtime performance.
