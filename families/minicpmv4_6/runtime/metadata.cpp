/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpmv4_6/runtime/metadata.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace trtmc::minicpmv4_6 {
namespace {

using Json = nlohmann::json;

[[noreturn]] void fail(const std::string& what, const char* rule) {
    throw std::runtime_error("MiniCPM-V 4.6 metadata " + what + " " + rule);
}

const Json& field(const Json& object, const char* name) {
    if (!object.is_object() || !object.contains(name) || object.at(name).is_null())
        fail(name, "is missing");
    return object.at(name);
}

std::int32_t int32_value(const Json& value, const std::string& what, std::int64_t minimum) {
    constexpr auto kMax = std::numeric_limits<std::int32_t>::max();
    if (!value.is_number_integer())
        fail(what, "must be an integer");
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        if (number > static_cast<std::uint64_t>(kMax) ||
            static_cast<std::int64_t>(number) < minimum)
            fail(what, "is out of range");
        return static_cast<std::int32_t>(number);
    }
    const auto number = value.get<std::int64_t>();
    if (number < minimum || number > kMax)
        fail(what, "is out of range");
    return static_cast<std::int32_t>(number);
}

} // namespace

std::int32_t json_int32(const Json& object, const char* name, std::int32_t minimum) {
    return int32_value(field(object, name), name, minimum);
}

std::array<float, 3> json_float3(const Json& object, const char* name) {
    const auto& value = field(object, name);
    if (!value.is_array() || value.size() != 3)
        fail(name, "must hold exactly three numbers");
    std::array<float, 3> result{};
    for (std::size_t i = 0; i < 3; ++i) {
        if (!value[i].is_number())
            fail(name, "must hold exactly three numbers");
        const auto number = value[i].get<double>();
        if (!std::isfinite(number) || std::fabs(number) > std::numeric_limits<float>::max())
            fail(name, "must hold finite FP32 values");
        result[i] = static_cast<float>(number);
    }
    return result;
}

std::vector<std::int32_t> json_token_ids(const Json& object, const char* name,
                                         std::int32_t vocab_size) {
    const auto& value = field(object, name);
    if (!value.is_array() || value.empty())
        fail(name, "must be a nonempty list of token ids");
    std::vector<std::int32_t> ids;
    std::unordered_set<std::int32_t> seen;
    for (const auto& item : value) {
        const auto id = int32_value(item, name, 0);
        if (id >= vocab_size)
            fail(name, "addresses a token outside the vocabulary");
        if (!seen.insert(id).second)
            fail(name, "repeats a token id");
        ids.push_back(id);
    }
    return ids;
}

std::int32_t checked_product(std::int32_t a, std::int32_t b, const char* what) {
    const auto product = static_cast<std::int64_t>(a) * b;
    if (a <= 0 || b <= 0 || product > std::numeric_limits<std::int32_t>::max())
        fail(what, "is out of range");
    return static_cast<std::int32_t>(product);
}

Json parse_json_unique(const char* data, std::size_t size, const std::string& what) {
    std::vector<std::unordered_set<std::string>> keys;
    const Json::parser_callback_t unique = [&](int, Json::parse_event_t event, Json& parsed) {
        if (event == Json::parse_event_t::object_start) {
            keys.emplace_back();
        } else if (event == Json::parse_event_t::object_end) {
            keys.pop_back();
        } else if (event == Json::parse_event_t::key &&
                   !keys.back().insert(parsed.get<std::string>()).second) {
            throw std::runtime_error("MiniCPM-V 4.6 " + what + " repeats the key " + parsed.dump());
        }
        return true;
    };
    return Json::parse(data, data + size, unique);
}

void check_vocabulary(std::int32_t tokenizer_size, std::int32_t embedding_rows) {
    if (tokenizer_size <= 0 || tokenizer_size > embedding_rows)
        throw std::runtime_error("MiniCPM-V 4.6 tokenizer ids exceed the decoder's embedding rows");
}

} // namespace trtmc::minicpmv4_6
