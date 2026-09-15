/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc::server {

struct NativeWorkerOptions {
    std::string bundle_path;
    std::string runtime_root;
    std::uint64_t kv_cache_size_bytes{0};
    std::string runtime_cache_path;
    bool cuda_graphs{false};
};

// Private native data-plane entry points used by the `trtmc serve` facade.
int run_server_frontend(int argc, char** argv);
int run_native_worker(const NativeWorkerOptions& options);
int run_native_worker(int argc, char** argv);

} // namespace trtmc::server
