/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Model-owned ATen-free launcher for the pinned FlashAttention 2 forward kernel.
// The vendored upstream closure retains its BSD-3-Clause notices and manifest.

#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>

#define FLASH_NAMESPACE sam2_hoi_flash_v2
#include "flash.h"
#include "flash_fwd_kernel.h"

namespace sam2_hoi_flash_v2 {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define SAM2_HOI_KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define SAM2_HOI_KERNEL_PARAM_MODIFIER
#endif

template <bool IsEvenMN>
__global__ void
sam2_hoi_flash_attn96_kernel(SAM2_HOI_KERNEL_PARAM_MODIFIER const Flash_fwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using Traits = Flash_fwd_kernel_traits<96, 128, 64, 4, false, false, cutlass::bfloat16_t>;
    compute_attn<Traits, false, false, false, false, IsEvenMN, true, false, false>(params);
#endif
}

namespace {

constexpr int kHeadDim = 96;

bool allowed_shape(int b, int h, int sq, int sk) {
    return (b == 1024 && h == 1 && sq == 64 && sk == 64) ||
           (b == 1024 && h == 2 && sq == 16 && sk == 64) ||
           (b == 1024 && h == 2 && sq == 16 && sk == 16) ||
           (b == 1024 && h == 4 && sq == 4 && sk == 16) ||
           (b == 25 && h == 4 && sq == 196 && sk == 196) ||
           (b == 1 && h == 4 && sq == 4096 && sk == 4096) ||
           (b == 25 && h == 8 && sq == 49 && sk == 196) ||
           (b == 25 && h == 8 && sq == 49 && sk == 49);
}

Flash_fwd_params make_params(const void* q, const void* k, const void* v, void* output,
                             void* softmax_lse, int b, int h, int sq, int sk) {
    Flash_fwd_params params{};
    params.q_ptr = const_cast<void*>(q);
    params.k_ptr = const_cast<void*>(k);
    params.v_ptr = const_cast<void*>(v);
    params.o_ptr = output;

    params.q_batch_stride = static_cast<int64_t>(h) * sq * kHeadDim;
    params.k_batch_stride = static_cast<int64_t>(h) * sk * kHeadDim;
    params.v_batch_stride = static_cast<int64_t>(h) * sk * kHeadDim;
    params.o_batch_stride = static_cast<int64_t>(h) * sq * kHeadDim;
    params.q_row_stride = kHeadDim;
    params.k_row_stride = kHeadDim;
    params.v_row_stride = kHeadDim;
    params.o_row_stride = kHeadDim;
    params.q_head_stride = static_cast<int64_t>(sq) * kHeadDim;
    params.k_head_stride = static_cast<int64_t>(sk) * kHeadDim;
    params.v_head_stride = static_cast<int64_t>(sk) * kHeadDim;
    params.o_head_stride = static_cast<int64_t>(sq) * kHeadDim;

    params.h = h;
    params.h_k = h;
    params.h_h_k_ratio = 1;
    params.b = b;
    params.seqlen_q = sq;
    params.seqlen_k = sk;
    params.seqlen_q_rounded = ((sq + 127) / 128) * 128;
    params.seqlen_k_rounded = ((sk + 127) / 128) * 128;
    params.d = kHeadDim;
    params.d_rounded = 96;
    params.total_q = b * sq;

    params.softmax_lse_ptr = softmax_lse;
    params.scale_softmax = 0x1.a20bd8p-4F;
    params.scale_softmax_log2 = 0x1.2d8e80p-3F;
    params.p_dropout = 1.0F;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0F;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.0F;
    params.is_bf16 = true;
    params.is_causal = false;
    params.is_seqlens_k_cumulative = true;
    params.num_splits = 0;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
    return params;
}

} // namespace

} // namespace sam2_hoi_flash_v2

extern "C" int sam2_hoi_hiera_flash_attention96_shape_allowed(int b, int h, int sq, int sk) {
    return sam2_hoi_flash_v2::allowed_shape(b, h, sq, sk) ? 1 : 0;
}

extern "C" int sam2_hoi_hiera_flash_attention96_launch(const void* q, const void* k, const void* v,
                                                       void* output, void* softmax_lse, int b,
                                                       int h, int sq, int sk, cudaStream_t stream) {
    using namespace sam2_hoi_flash_v2;
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
        softmax_lse == nullptr || !allowed_shape(b, h, sq, sk)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    using Traits = Flash_fwd_kernel_traits<96, 128, 64, 4, false, false, cutlass::bfloat16_t>;
    constexpr std::size_t smem = Traits::kSmemSize;
    const int m_blocks = (sq + Traits::kBlockM - 1) / Traits::kBlockM;
    const dim3 grid(m_blocks, b, h);
    auto params = make_params(q, k, v, output, softmax_lse, b, h, sq, sk);
    const bool even_mn = (sq % Traits::kBlockM == 0) && (sk % Traits::kBlockN == 0);
    cudaError_t status;
    if (even_mn) {
        auto kernel = &sam2_hoi_flash_attn96_kernel<true>;
        status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem));
        if (status != cudaSuccess)
            return static_cast<int>(status);
        kernel<<<grid, Traits::kNThreads, smem, stream>>>(params);
    } else {
        auto kernel = &sam2_hoi_flash_attn96_kernel<false>;
        status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem));
        if (status != cudaSuccess)
            return static_cast<int>(status);
        kernel<<<grid, Traits::kNThreads, smem, stream>>>(params);
    }
    return static_cast<int>(cudaPeekAtLastError());
}
