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
Compression softmax retains FP32 exponentials, the warp reduction for at most
2048 key tiles and the full-block reduction above that bound, followed by an
explicit FP32 division. INT32 valid sizes mask empty key tiles directly, so
this shares the family-owned fine-attention reductions without another mask
allocation. Replacing a reduction tree or multiplying by a rounded reciprocal
can move a compressed value across a BF16 midpoint and amplify cancellation
at the final gated addition. The external workspace and BF16 boundaries do
not change.

The binding contains a generic SM80+ CUDA implementation. Enable
`TRTMC_BUILD_MINIMAX_H3_OMNIREF_VSA_SM100A` to also embed optimized sm100a and
sm103a cubins; runtime `auto` selects the optimized route when available.
`TRTMC_MINIMAX_H3_VSA_BACKEND=generic` or `blackwell` forces a route for
validation. An unavailable explicitly requested route fails.

The generic route scales FP32 Q and K separately before the QK product,
retains FP32 scores/probabilities and PV, then rounds fine attention to BF16.
It processes one head at a time while retaining the full original key order
and regional mask. Up to 32 independent query tiles share each QK/PV batch,
bounded by the caller's existing BYOK scratch capacity. The compressed output
stays at the front and routing indices/counts at the aligned tail; only dead
pool/score storage holds the one-head K/V and per-query Q, scores, outputs and
masks. Small allocations can retain one query tile. The current 2048-tile
56-head profile can hold 32 queries at 708 tiles or 24 at 1026 tiles without
increasing its allocation. It uses caller-provided BYOK scratch;
there are no hidden allocations or stream synchronizations. Compression
finishes before fine attention, allowing its dead K/V pools and routing scores
to hold the linear fine-attention arena. Routing indices, counts and compressed
output remain live and disjoint. New small profiles reserve additional scratch;
the existing 56-head, 708/1026-tile workspace sizes remain unchanged. An old
bundle without sufficient generic scratch fails explicitly and needs rebuilding.
When only one query tile is active, PV retains two independent copies instead
of singleton GEMM, whose FP32 reduction can differ at BF16 rounding boundaries.
The dead Q tile holds the first output and the fine tile holds the second.
Larger batches contain independent original query tiles, sharing read-only
K/V while retaining disjoint scores, output tiles and masks. QK's pointer table
uses the dead PV-output buffer; PV's table uses the completed softmax masks.
These tables are rewritten in stream order before use, and none overlaps its
operation's live inputs or outputs. Compression continues to use its own
single validity mask across query rows.
These execution changes require complete GPU comparison and timing evidence;
host arena checks alone do not qualify numerical equivalence or performance.

`primitives.cuh`, `block_sparse_kernel_sm100a.cuh` and
`block_sparse_launch_sm100a.cuh` originate from Apache-2.0 FastVideo commit
`48a047c05ff4138f20cfa33351499c6ec5945f5d`, preserving the existing sm103a
compatibility patch. The new binding selects their 128-token instantiation.
Its optimized 128-token PV operation stores the softmax probability as three
BF16 limbs: the rounded value, its rounded residual, and the remaining rounded
tail. All three products with V accumulate into the existing FP32 output tile.
The second limb uses the unused half of the QK tensor-memory tile. The third
has a dedicated shared tile for each query tile, using the same probability
handoff as the other limbs. It does not borrow the output epilogue's TMA
buffer or wait on that pipeline. Two KV ring stages are reassigned to these
two probability tiles, leaving one KV stage. The external workspace, total
shared-memory allocation, arithmetic and BF16 input/output contract stay
unchanged. When the ring cannot hold both initial K loads, Q is prefetched
first so MMA can consume the first K tile and release its slot. The four-slot
64-token startup order stays unchanged. Less KV prefetching and three PV
products have a runtime cost;
saved-output parity, repeated-call stability and full-model progress require
separate GPU measurements. Compilation alone does not qualify the change.
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

`TRTMC_MINIMAX_H3_VSA_TRACE=1` enables host-side diagnostic messages for each
call's geometry and pooling, routing, fine-attention, compression and gated-add
boundaries, including a host Unix timestamp to distinguish a stalled operation
from setup time before the first call. It is off by default and does not change
device computations or insert synchronization. An `enqueued` message does not establish GPU completion.
For a separate bounded failure-localization run, `CUDA_LAUNCH_BLOCKING=1` can
help identify the last entered operation; that setting changes execution
scheduling and its host elapsed times are not benchmark evidence. Neither
option changes the original correctness or full-model completion gates.

For immutable saved operands from the interleaved-guide oracle case,
`families.minimax_h3.tests.reference_vsa_rounding` observes only its existing
failed coordinates on CPU. It compares FP64 attention with approximate FP32
operation orders and rounds directly from FP64 to BF16 to avoid an extra
FP32 midpoint rounding. Current native fine values remain conditionally
inferred from the gated output and unchanged compression until directly
recorded. This diagnostic never substitutes its output for the GPU reference
or changes an acceptance threshold.
