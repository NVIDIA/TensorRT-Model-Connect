/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

extern "C" const char* trtmc_vitmatte_register_kernels() noexcept;

namespace {
std::atomic<int> projections{0}, activations{0};
std::atomic<bool> reject_activation{true};
} // namespace

// Interpose only the loader boundary; exercise the family's real registration.
extern "C" const char* trtmc_load_byok_kernel(const char*, const char* function,
                                              const char*) noexcept {
    if (std::strcmp(function, "vitmatte_fp32_mlp_output_v1") == 0)
        ++projections;
    else if (std::strcmp(function, "vitmatte_fp32_gelu_v1") == 0) {
        ++activations;
        if (reject_activation)
            return "injected second registration failure";
    } else
        return "unexpected kernel";
    return nullptr;
}

int main() {
    for (int attempt = 0; attempt < 3; ++attempt) {
        const char* error = trtmc_vitmatte_register_kernels();
        if (!error || std::strcmp(error, "injected second registration failure") != 0)
            return 1;
    }
    if (projections != 1 || activations != 3) {
        std::cerr << "retry reloaded an already registered kernel\n";
        return 1;
    }
    reject_activation = false;
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int index = 0; index < 8; ++index)
        threads.emplace_back([&] {
            if (trtmc_vitmatte_register_kernels())
                ++errors;
        });
    for (auto& thread : threads)
        thread.join();
    if (errors || projections != 1 || activations != 4 || trtmc_vitmatte_register_kernels())
        return 1;
    std::cout << "registration retries and concurrent callers reuse completed bindings\n";
}
