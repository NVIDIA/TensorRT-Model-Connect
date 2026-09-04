/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gpt2/runtime/plugin_helpers.h"

#include <iostream>
#include <stdexcept>

namespace {

bool rejects(int num_key_value_heads, int head_dim, int tensor_parallel_size) {
    try {
        (void)trtmc::gpt2::rank_local_kv_dim(num_key_value_heads, head_dim, tensor_parallel_size);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    using trtmc::gpt2::rank_local_kv_dim;

    if (rank_local_kv_dim(12, 64, 1) != 768)
        return 1;
    if (rank_local_kv_dim(12, 64, 4) != 192)
        return 2;
    if (!rejects(12, 64, 8))
        return 3;
    if (!rejects(12, 64, 0))
        return 4;

    std::cout << "GPT-2 parallel KV contract tests passed\n";
    return 0;
}
