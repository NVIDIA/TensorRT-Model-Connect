/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/cosmos3/runtime/vae_cache_storage.h"

#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        const auto layout = trtmc::cosmos3::make_vae_cache_layout({1, 257, 513});
        if (layout.offsets.size() != 3 || layout.offsets[0] != 0 ||
            layout.offsets[1] % trtmc::cosmos3::kVaeCacheAlignment != 0 ||
            layout.offsets[2] % trtmc::cosmos3::kVaeCacheAlignment != 0 ||
            layout.total_bytes % trtmc::cosmos3::kVaeCacheAlignment != 0) {
            throw std::runtime_error("Cosmos3 VAE cache layout is not aligned");
        }
        if (trtmc::cosmos3::select_vae_cache_memory_kind(false, false, 10, 0) !=
                trtmc::cosmos3::VaeCacheMemoryKind::kDevice ||
            trtmc::cosmos3::select_vae_cache_memory_kind(true, true, 10, 0) !=
                trtmc::cosmos3::VaeCacheMemoryKind::kMappedHost ||
            trtmc::cosmos3::select_vae_cache_memory_kind(true, true, 11, 0) !=
                trtmc::cosmos3::VaeCacheMemoryKind::kDevice) {
            throw std::runtime_error("Cosmos3 VAE cache memory selection changed");
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
