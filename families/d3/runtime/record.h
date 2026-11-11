/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
namespace trtmc::d3 {
using Record = nlohmann::ordered_json;
struct Question {
    std::string id, kind, instructions;
    std::vector<std::string> keys, texts;
    Record levels;
};
std::vector<Question> questions(const Record&);
std::string prompt(const Record& state, const Question&, const std::vector<std::string>& codes,
                   const std::string& media = "");
Record answer(const Question&, const std::vector<float>& probabilities);
} // namespace trtmc::d3
