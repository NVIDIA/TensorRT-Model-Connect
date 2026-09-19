/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// Private family ABI. The original vendor kernel owns all device mathematics.
// Q/K/V are BF16 [T,4,64] views; output is contiguous BF16 [T,4,64].
struct HstuAttentionArgs {
    void *q, *k, *v, *output, *pages;
    std::int32_t *q_offsets, *k_offsets, *targets, *page_indptrs, *page_ids, *last_page_lens;
    std::int32_t batch, max_q, max_k, page_count;
    std::int64_t q_row_stride, k_row_stride, v_row_stride;
};
static_assert(sizeof(HstuAttentionArgs) == 128);

// Dense ignores every page field. Paged consumes existing [P,2,128,4,64]
// storage after the native KV update, including its zero-tail guarantee.
extern "C" int hstu_attention_forward(const HstuAttentionArgs* args, cudaStream_t stream) noexcept;
