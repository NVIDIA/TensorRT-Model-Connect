/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Tensor types for the HF-style ITrtModule interface.
// These serve the same role as torch.Tensor and TensorMap in PyTorch.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc {

enum class DType {
    kFloat32,
    kFloat16,
    kBFloat16,
    kInt32,
    kInt8,
};

inline std::size_t dtype_size(DType dt) {
    switch (dt) {
    case DType::kFloat32:
        return 4;
    case DType::kFloat16:
        return 2;
    case DType::kBFloat16:
        return 2;
    case DType::kInt32:
        return 4;
    case DType::kInt8:
        return 1;
    }
    return 0;
}

// CPU-side tensor reference (non-owning). Like a numpy array view.
struct Tensor {
    void* data{nullptr};
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};

    std::size_t numel() const {
        if (shape.empty())
            return 0;
        std::size_t n = 1;
        for (auto s : shape)
            n *= static_cast<std::size_t>(s);
        return n;
    }

    std::size_t nbytes() const { return numel() * dtype_size(dtype); }
};

using TensorMap = std::unordered_map<std::string, Tensor>;

// Metadata about an engine tensor (returned by ITrtModule introspection).
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};
    bool is_input{true};
};

} // namespace trtmc
