/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <thread>
#include <vector>

extern "C" const char* trtmc_vitmatte_register_kernels() noexcept;

namespace {
std::atomic<int> projections{0}, activations{0};
std::atomic<bool> reject_activation{true};
std::atomic<bool> reject_error_copy{false};
char long_error[258];
} // namespace

void* operator new(std::size_t size) {
    // libstdc++'s exception object has separate storage; only reject the later
    // std::string error copy that the family's noexcept handler used to make.
    if (reject_error_copy && size == sizeof(long_error))
        throw std::bad_alloc();
    if (auto* memory = std::malloc(size ? size : 1))
        return memory;
    throw std::bad_alloc();
}
void operator delete(void* memory) noexcept {
    std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

// Interpose only the loader boundary; exercise the family's real registration.
extern "C" const char* trtmc_load_byok_kernel(const char*, const char* function,
                                              const char*) noexcept {
    if (reject_error_copy)
        return long_error;
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

int main(int argc, char**) {
    if (argc > 1) {
        std::memset(long_error, 'e', sizeof(long_error) - 1);
        reject_error_copy = true;
        const char* error = trtmc_vitmatte_register_kernels();
        reject_error_copy = false;
        if (!error || std::strcmp(error, long_error) != 0)
            return 1;
        std::cout << "registration errors are reported without allocating a message copy\n";
        return 0;
    }
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
