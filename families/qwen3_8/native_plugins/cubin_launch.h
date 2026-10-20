/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared helpers for plugins that run ahead-of-time compiled cuTile cubins with the CUDA
// driver API.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <NvInferRuntime.h>

namespace trtmc {

constexpr int32_t kMaxRank = 4;
constexpr int32_t kMaxArrays = 12;
constexpr std::size_t kAlign = 256;
// The tile compiler pins every kernel to a thread count (REQNTID in the cubin); other block sizes
// fail to launch. The builder reads it from each cubin and passes it to the plugin.

struct KernelKey {
    CUcontext ctx;
    std::size_t hash;
    bool operator==(KernelKey const& o) const noexcept { return ctx == o.ctx && hash == o.hash; }
};
struct KernelKeyHash {
    std::size_t operator()(KernelKey const& k) const noexcept {
        return std::hash<void*>()(k.ctx) ^ (k.hash * 0x9E3779B97F4A7C15ULL);
    }
};
struct LoadedKernel {
    CUfunction function{nullptr};
};

// One CUmodule per (context, cubin); shared by all layers of one engine.
inline bool get_kernel(std::vector<char> const& cubin, std::string const& symbol,
                       LoadedKernel& out) {
    static std::mutex mutex;
    static std::unordered_map<KernelKey, LoadedKernel, KernelKeyHash> cache;
    CUcontext ctx{nullptr};
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr)
        return false;
    KernelKey const key{ctx, std::hash<std::string_view>()({cubin.data(), cubin.size()})};
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        out = it->second;
        return true;
    }
    CUmodule module{nullptr};
    if (cuModuleLoadData(&module, cubin.data()) != CUDA_SUCCESS)
        return false;
    LoadedKernel kernel;
    if (cuModuleGetFunction(&kernel.function, module, symbol.c_str()) != CUDA_SUCCESS)
        return false;
    cache.emplace(key, kernel);
    out = kernel;
    return true;
}

// One kernel array argument. Shape and strides are baked into the cubin but still passed.
// An all-zero `stride` means row-major contiguous.
struct Arr {
    void* ptr;
    int32_t rank;
    std::array<int32_t, kMaxRank> shape;
    std::array<int32_t, kMaxRank> stride{};
};

// cutile_python_v2 ABI: per array a pointer, then int32 shape[rank], then int32 stride[rank].
inline bool launch(LoadedKernel const& kernel, std::array<unsigned, 3> grid, unsigned threads,
                   Arr const* arrays, int32_t count, cudaStream_t stream) {
    std::array<uint64_t, kMaxArrays> ptrs{};
    std::array<std::array<int32_t, 2 * kMaxRank>, kMaxArrays> ints{};
    std::array<void*, kMaxArrays * (1 + 2 * kMaxRank)> params{};
    int32_t n = 0;
    for (int32_t i = 0; i < count; ++i) {
        auto const& a = arrays[i];
        ptrs[i] = reinterpret_cast<uint64_t>(a.ptr);
        std::array<int32_t, kMaxRank> strides = a.stride;
        if (strides[0] == 0) {
            int32_t stride = 1;
            for (int32_t d = a.rank - 1; d >= 0; --d) {
                strides[d] = stride;
                stride *= a.shape[d];
            }
        }
        for (int32_t d = 0; d < a.rank; ++d) {
            ints[i][d] = a.shape[d];
            ints[i][a.rank + d] = strides[d];
        }
        params[n++] = &ptrs[i];
        for (int32_t d = 0; d < 2 * a.rank; ++d)
            params[n++] = &ints[i][d];
    }
    return cuLaunchKernel(kernel.function, grid[0], grid[1], grid[2], threads, 1, 1, 0,
                          reinterpret_cast<CUstream>(stream), params.data(),
                          nullptr) == CUDA_SUCCESS;
}

inline bool is_half_type(nvinfer1::DataType t) noexcept {
    return t == nvinfer1::DataType::kHALF || t == nvinfer1::DataType::kBF16;
}

inline std::size_t align_up(std::size_t v) noexcept { return (v + kAlign - 1) / kAlign * kAlign; }

} // namespace trtmc
