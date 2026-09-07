/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc {

inline std::string k2_horizon_expand_layer_name(const std::string& pattern, int32_t layer) {
    std::string result = pattern;
    const auto doubled_layer = static_cast<int64_t>(layer) * 2;
    auto replace_all = [&](const std::string& token, const std::string& value) {
        std::size_t pos = 0;
        while ((pos = result.find(token, pos)) != std::string::npos) {
            result.replace(pos, token.size(), value);
            pos += value.size();
        }
    };

    replace_all("{2i+2}", std::to_string(doubled_layer + 2));
    replace_all("{2i+1}", std::to_string(doubled_layer + 1));
    replace_all("{2i}", std::to_string(doubled_layer));
    replace_all("{i}", std::to_string(layer));
    return result;
}

inline std::string k2_horizon_layer_tensor_name(const char* stem, int32_t layer) {
    return std::string(stem) + "_" + std::to_string(layer);
}

} // namespace trtmc
