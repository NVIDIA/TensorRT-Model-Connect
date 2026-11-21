/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/d3/runtime/media.h"
#include "families/d3/runtime/record.h"

#include <cstdint>
#include <vector>
namespace trtmc::d3 {
struct BatchRow {
    std::size_t owner;
    Question question;
    std::vector<int32_t> ids;
    std::vector<float> angles;
};
struct PackedRows {
    int64_t batch, sequence;
    std::vector<int32_t> ids;
    std::vector<float> angles, mask;
    std::vector<int64_t> lengths;
};
PackedRows pack_rows(const std::vector<BatchRow>&, std::size_t begin, std::size_t end);
VisualInputs pack_visual(const std::vector<VisualInputs>&, std::size_t begin, std::size_t end);
} // namespace trtmc::d3
