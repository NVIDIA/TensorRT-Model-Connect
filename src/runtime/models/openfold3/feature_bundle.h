/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace trtmc::openfold3 {

struct FeatureTensor {
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};
    std::vector<std::byte> data;
};

class FeatureBundle {
  public:
    static FeatureBundle parse(const void* data, std::size_t size);

    const FeatureTensor& require(std::string_view name) const;
    std::size_t size() const { return tensors_.size(); }

  private:
    std::unordered_map<std::string, FeatureTensor> tensors_;
};

} // namespace trtmc::openfold3
