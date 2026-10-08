/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/clef/runtime/tokenizer.h"

#include <nlohmann/json.hpp>
#include <utility>

namespace trtmc::clef {
using Json = nlohmann::ordered_json;
using Span = std::pair<std::int32_t, std::int32_t>;

struct Question {
    std::string id;
    std::int32_t type;
    Span span;
    std::vector<Span> option_spans;
    std::vector<std::string> option_ids;
};

struct Record {
    std::vector<std::int32_t> input_ids;
    std::vector<Question> questions;
};

Record encode_record(const ITokenizer& tokenizer, const Json& document,
                     std::int32_t max_length = 16384, std::int32_t max_state_tokens = -1,
                     const std::vector<std::int32_t>& media_ids = {});
Json systemone_answer(const Json& question, const std::vector<std::string>& option_ids,
                      const std::vector<float>& probabilities);
} // namespace trtmc::clef
