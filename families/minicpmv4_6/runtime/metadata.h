/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trtmc::minicpmv4_6 {

// Strict readers of the bundle's JSON metadata. Each requires the JSON type
// (a boolean is not a number, a fraction is not an integer) and checks the
// range before narrowing; a missing key or an explicit null is an error.

// An integer field in [minimum, INT32_MAX].
std::int32_t json_int32(const nlohmann::json& object, const char* name, std::int32_t minimum);

// Exactly three finite numbers that FP32 represents.
std::array<float, 3> json_float3(const nlohmann::json& object, const char* name);

// A nonempty list of distinct token ids in [0, vocab_size).
std::vector<std::int32_t> json_token_ids(const nlohmann::json& object, const char* name,
                                         std::int32_t vocab_size);

// a * b, or an error when the product leaves int32.
std::int32_t checked_product(std::int32_t a, std::int32_t b, const char* what);

// Parse JSON text; an object that repeats a key is an error (nlohmann would keep the last value).
nlohmann::json parse_json_unique(const char* data, std::size_t size, const std::string& what);

// The tokenizer's ids must address rows of the embedding table; a padded table may have more rows.
void check_vocabulary(std::int32_t tokenizer_size, std::int32_t embedding_rows);

} // namespace trtmc::minicpmv4_6
