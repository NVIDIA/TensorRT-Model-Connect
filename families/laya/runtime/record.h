/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/laya/runtime/tokenizer.h"

#include <nlohmann/json.hpp>

namespace trtmc::laya {
using Json = nlohmann::ordered_json;

struct Question {
    std::string id;
    int type;
    std::vector<std::string> options;
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> markers;
};

std::vector<Question> encode_record(const ITokenizer& tokenizer, const Json& record,
                                    const Json& config);
Json format_answer(const Json& question, const std::vector<std::string>& options,
                   const std::vector<float>& probabilities, float act_probability);
double temperature(const Json& config, int type, std::size_t options);
} // namespace trtmc::laya
