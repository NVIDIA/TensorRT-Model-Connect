/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::openfold3 {

struct RandomSamples {
    int32_t seed{0};
    int32_t sampling_steps{0};
    int32_t padded_atom_count{0};
    std::vector<float> initial;
    std::vector<std::array<float, 9>> rotations;
    std::vector<std::array<float, 3>> translations;
    std::vector<float> noise;

    static RandomSamples parse(const void* data, std::size_t size);
};

} // namespace trtmc::openfold3
