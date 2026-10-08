/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host launch/export glue only. hstu_fwd.h and every device function are the
// pinned, unchanged original NVIDIA/FBGEMM implementation.
// Upstream: pytorch/FBGEMM@43791a0ade113a0ad5530c2a4948870dd0f7e417.
// Upstream copyright and BSD terms are reproduced in third_party/ and in each
// exported attention_native.NOTICE alongside the compiled implementation.
#include "hstu_fwd.h"
#include "native_attention_kernel.h"

#ifndef HSTU_DENSE_ATTENTION
#error "The family builder must specify the attention storage mode"
#endif

namespace trtmc_hstu_kernel {
constexpr bool kPaged = !HSTU_DENSE_ATTENTION;
using OriginalTraits = Hstu_fwd_kernel_traits<64, 64, 128, 4, true, true, false, false, false, 0,
                                              false, kPaged, true, true, cutlass::bfloat16_t>;
static_assert(OriginalTraits::kSmemSize == 32768);
static_assert(OriginalTraits::kNThreads == 128);

bool input_pointers(const HstuAttentionArgs& args) {
    return args.q && args.k && args.v && args.output && args.q_offsets && args.k_offsets &&
           args.targets;
}

bool input_extents(const HstuAttentionArgs& args) {
    return args.batch > 0 && args.max_q > 0 && args.max_q <= 1024 && args.max_k > 0 &&
           args.max_k <= 1024 && args.q_row_stride >= 256 && args.k_row_stride >= 256 &&
           args.v_row_stride >= 256;
}

bool valid(const HstuAttentionArgs* args) {
    if (!args || !input_pointers(*args) || !input_extents(*args))
        return false;
    if constexpr (kPaged)
        return args->pages && args->page_indptrs && args->page_ids && args->last_page_lens &&
               args->page_count > 0;
    return true;
}

void page_parameters(Hstu_fwd_params& params, const HstuAttentionArgs& args) {
    if constexpr (kPaged) {
        params.kv_cache_ptr = args.pages;
        // Original parameter names differ from their effective tensor axes:
        // page, K/V, token, head, dimension. Strides are BF16 element counts.
        params.kv_cache_kvtensor_stride = 2 * 128 * 256;
        params.kv_cache_page_stride = 128 * 256;
        params.kv_cache_head_stride = 256;
        params.kv_cache_row_stride = 64;
        params.page_size = 128;
        params.total_pages = args.page_count;
        params.page_offsets = args.page_indptrs;
        params.page_ids = args.page_ids;
        params.last_page_lens = args.last_page_lens;
    }
}
} // namespace trtmc_hstu_kernel

extern "C" int hstu_attention_forward(const HstuAttentionArgs* args, cudaStream_t stream) noexcept {
    if (!trtmc_hstu_kernel::valid(args))
        return static_cast<int>(cudaErrorInvalidValue);
    Hstu_fwd_params params{};
    params.q_ptr = args->q;
    params.k_ptr = args->k;
    params.v_ptr = args->v;
    params.o_ptr = args->output;
    params.q_row_stride = args->q_row_stride;
    params.k_row_stride = args->k_row_stride;
    params.v_row_stride = args->v_row_stride;
    params.q_head_stride = params.k_head_stride = params.v_head_stride = 64;
    params.o_row_stride = 256;
    params.o_head_stride = 64;
    params.h = params.h_k = 4;
    params.h_h_k_ratio = params.h_rab = 1;
    params.arch = 80;
    params.cu_seqlens_q = args->q_offsets;
    params.cu_seqlens_k = args->k_offsets;
    params.num_targets = args->targets;
    trtmc_hstu_kernel::page_parameters(params, *args);
    params.b = args->batch;
    params.seqlen_q = args->max_q;
    params.seqlen_k = args->max_k;
    params.seqlen_q_rounded = ((args->max_q + 127) / 128) * 128;
    params.seqlen_k_rounded = ((args->max_k + 127) / 128) * 128;
    params.d = 64;
    params.scaling_seqlen = 1024;
    params.alpha = 0.125F;
    params.target_group_size = 1;
    params.window_size_left = args->max_k;
    params.window_size_right = 0;
    params.is_bf16 = params.is_causal = params.is_target = true;
    params.is_paged_kv = trtmc_hstu_kernel::kPaged;
    const dim3 grid((args->max_q + 63) / 64, 4, args->batch);
    flash::hstu_fwd_kernel<trtmc_hstu_kernel::OriginalTraits, Hstu_fwd_params>
        <<<grid, trtmc_hstu_kernel::OriginalTraits::kNThreads,
           trtmc_hstu_kernel::OriginalTraits::kSmemSize, stream>>>(params);
    return static_cast<int>(cudaGetLastError());
}
