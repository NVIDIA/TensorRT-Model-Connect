/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/record.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace trtmc::hstu {
namespace {

std::int64_t positive(const Json& json, const char* key) {
    const auto value = json.at(key).get<std::int64_t>();
    if (value < 1)
        throw std::invalid_argument(std::string("HSTU runtime.json ") + key + " must be positive");
    return value;
}

Feature feature(const Json& json) {
    Feature result{json.at("name").get<std::string>(), json.at("vocab_size").get<std::int64_t>()};
    if (result.name.empty() || result.vocab_size < 1 ||
        result.vocab_size > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("HSTU runtime.json feature needs a name and an int32 vocab");
    return result;
}

std::int32_t table_id(const Json& value, const Feature& table, const char* where) {
    if (!value.is_number_integer())
        throw std::invalid_argument(std::string("HSTU record ") + where + " " + table.name +
                                    " ids must be integers");
    const auto id = value.get<std::int64_t>();
    if (id < 0 || id >= table.vocab_size)
        throw std::invalid_argument(std::string("HSTU record ") + where + " " + table.name +
                                    " id " + std::to_string(id) + " is outside its table of " +
                                    std::to_string(table.vocab_size));
    return static_cast<std::int32_t>(id);
}

std::vector<std::int32_t> table_ids(const Json& object, const Feature& table, std::int64_t count,
                                    const char* where) {
    if (!object.is_object() || !object.contains(table.name) || !object.at(table.name).is_array())
        throw std::invalid_argument(std::string("HSTU record ") + where + " must hold a " +
                                    table.name + " array");
    const auto& values = object.at(table.name);
    if (static_cast<std::int64_t>(values.size()) != count)
        throw std::invalid_argument(std::string("HSTU record ") + where + " " + table.name +
                                    " holds " + std::to_string(values.size()) +
                                    " ids, but the bundle's static geometry takes exactly " +
                                    std::to_string(count));
    std::vector<std::int32_t> ids;
    ids.reserve(values.size());
    for (const auto& value : values)
        ids.push_back(table_id(value, table, where));
    return ids;
}

} // namespace

Geometry parse_geometry(const Json& runtime) {
    Geometry geometry;
    geometry.precision = runtime.at("precision").get<std::string>();
    if (geometry.precision != "bf16" && geometry.precision != "fp32")
        throw std::invalid_argument("HSTU runtime.json precision must be bf16 or fp32");
    geometry.history_length = positive(runtime, "history_length");
    geometry.num_candidates = positive(runtime, "num_candidates");
    geometry.tokens = positive(runtime, "tokens");
    geometry.num_tasks = positive(runtime, "num_tasks");
    for (const auto& value : runtime.at("contextual_features"))
        geometry.contextual.push_back(feature(value));
    geometry.item = feature(runtime.at("item_feature"));
    geometry.action = feature(runtime.at("action_feature"));
    const auto contextual = static_cast<std::int64_t>(geometry.contextual.size());
    if (contextual < 1 ||
        geometry.tokens != contextual + 2 * geometry.history_length + geometry.num_candidates)
        throw std::invalid_argument("HSTU runtime.json tokens do not match its geometry");
    return geometry;
}

Request parse_request(const Json& document, const Geometry& geometry) {
    if (!document.is_object())
        throw std::invalid_argument("HSTU record must be a JSON object");
    for (const auto& [key, value] : document.items()) {
        (void)value;
        if (key != "user" && key != "history" && key != "candidates" && key != "model")
            throw std::invalid_argument("HSTU record has an unknown field: " + key);
    }
    Request request;
    if (document.contains("model")) {
        if (!document.at("model").is_string())
            throw std::invalid_argument("HSTU record model must be a string when supplied");
        request.model = document.at("model").get<std::string>();
    }
    const auto& user = document.at("user");
    if (!user.is_object() || user.size() != geometry.contextual.size())
        throw std::invalid_argument("HSTU record user must hold exactly the contextual features");
    for (const auto& table : geometry.contextual) {
        if (!user.contains(table.name))
            throw std::invalid_argument("HSTU record user lacks " + table.name);
        request.contextual_ids.push_back(table_id(user.at(table.name), table, "user"));
    }
    const auto& history = document.at("history");
    if (!history.is_object() || history.size() != 2)
        throw std::invalid_argument("HSTU record history must hold its item and action arrays");
    request.history_item_ids =
        table_ids(history, geometry.item, geometry.history_length, "history");
    request.history_action_ids =
        table_ids(history, geometry.action, geometry.history_length, "history");
    const auto& candidates = document.at("candidates");
    if (!candidates.is_object() || candidates.size() != 1)
        throw std::invalid_argument("HSTU record candidates must hold one item array");
    request.candidate_item_ids =
        table_ids(candidates, geometry.item, geometry.num_candidates, "candidates");
    auto sorted = request.candidate_item_ids;
    std::sort(sorted.begin(), sorted.end());
    if (const auto repeat = std::adjacent_find(sorted.begin(), sorted.end());
        repeat != sorted.end())
        throw std::invalid_argument("HSTU record candidates repeat " + geometry.item.name + " " +
                                    std::to_string(*repeat) + ": each candidate is one option");
    return request;
}

} // namespace trtmc::hstu
