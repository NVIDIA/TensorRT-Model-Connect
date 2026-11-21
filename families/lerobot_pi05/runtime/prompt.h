/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::lerobot_pi05 {

// LeRobot's `Pi05PrepareStateTokenizerProcessorStep`: the state (already normalized)
// is bucketed into 256 bins over [-1, 1] with numpy.digitize semantics, so values
// below -1 give -1 and values at or above the last edge give 255.
int32_t discretize_state_value(float value);
std::vector<int32_t> discretize_state(const std::vector<float>& state);

// Python ``str.strip()`` then ``replace("_", " ").replace("\n", " ")``.
std::string clean_prompt(std::string_view prompt);

// "Task: <cleaned>, State: <bins separated by spaces>;\nAction: "
std::string format_prompt(std::string_view prompt, const std::vector<float>& state);

} // namespace trtmc::lerobot_pi05
