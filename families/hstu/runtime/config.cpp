/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/pipeline.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace trtmc::hstu {
namespace {

std::int32_t integer(const nlohmann::json& json, const char* name, std::int32_t minimum) {
    if (!json.contains(name) || !json.at(name).is_number_integer())
        throw std::invalid_argument(std::string("hstu runtime.json requires integer ") + name);
    if (json.at(name).is_number_unsigned() &&
        json.at(name).get<std::uint64_t>() >
            static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument(std::string("hstu runtime.json invalid ") + name);
    const auto value = json.at(name).get<std::int64_t>();
    if (value < minimum || value > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument(std::string("hstu runtime.json invalid ") + name);
    return static_cast<std::int32_t>(value);
}

bool boolean(const nlohmann::json& json, const char* name) {
    if (!json.contains(name) || !json.at(name).is_boolean())
        throw std::invalid_argument(std::string("hstu runtime.json requires boolean ") + name);
    return json.at(name).get<bool>();
}

void load_keys(EmbeddingTable& table, const nlohmann::json& json, const std::vector<char>& keys) {
    if (!json.contains("keys_offset"))
        return;
    const auto offset = static_cast<std::size_t>(integer(json, "keys_offset", 0));
    const auto count = static_cast<std::size_t>(table.num_embeddings);
    if (keys.size() % sizeof(std::int64_t) != 0 || offset > keys.size() / sizeof(std::int64_t) ||
        count > keys.size() / sizeof(std::int64_t) - offset)
        throw std::invalid_argument(
            "hstu sparse embedding key range is outside embedding_keys.bin");
    table.keys.resize(count);
    // Bundle key encoding is little-endian signed int64, independent of host alignment.
    for (std::size_t index = 0; index < count; ++index) {
        std::uint64_t value = 0;
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            const auto raw = static_cast<unsigned char>(keys[(offset + index) * 8U + byte]);
            value |= static_cast<std::uint64_t>(raw) << (byte * 8U);
        }
        std::memcpy(&table.keys[index], &value, sizeof(value));
    }
    if (!std::is_sorted(table.keys.begin(), table.keys.end()) ||
        std::adjacent_find(table.keys.begin(), table.keys.end()) != table.keys.end())
        throw std::invalid_argument("hstu embedding keys must be strictly increasing");
}

EmbeddingTable parse_table(const nlohmann::json& source, const std::vector<char>& keys) {
    EmbeddingTable table;
    table.name = source.at("name").get<std::string>();
    table.role = source.at("role").get<std::string>();
    table.num_embeddings = integer(source, "num_embeddings", 1);
    table.offset = integer(source, "offset", 0);
    if (table.role != "item" && table.role != "action" && table.role != "context")
        throw std::invalid_argument("hstu embedding table role must be item, action, or context");
    load_keys(table, source, keys);
    return table;
}

std::vector<EmbeddingTable> parse_tables(const nlohmann::json& json,
                                         const std::vector<char>& keys) {
    if (!json.contains("embedding_tables") || !json.at("embedding_tables").is_array())
        throw std::invalid_argument("hstu runtime.json requires embedding_tables array");
    std::vector<EmbeddingTable> tables;
    std::set<std::string> names;
    std::int64_t end = 0;
    for (const auto& source : json.at("embedding_tables")) {
        auto table = parse_table(source, keys);
        if (table.name.empty() || !names.insert(table.name).second)
            throw std::invalid_argument("hstu embedding table names must be nonempty and unique");
        if (table.offset != end)
            throw std::invalid_argument("hstu embedding table offsets must be contiguous");
        end += table.num_embeddings;
        if (end > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument("hstu embedding table indices exceed int32");
        tables.push_back(std::move(table));
    }
    return tables;
}

void parse_position_config(const nlohmann::json& json, RuntimeConfig& config) {
    config.position_buckets = integer(json, "position_buckets", 0);
    config.time_buckets = integer(json, "time_buckets", 0);
    if (config.time_buckets != 0 && config.time_buckets != 2048)
        throw std::invalid_argument("hstu timestamp encoding requires 2048 buckets");
    if (config.time_buckets != 0 && config.position_buckets == 0)
        throw std::invalid_argument("hstu timestamp encoding requires position embeddings");
}

std::int32_t prediction_width(const nlohmann::json& json, const std::string& mode) {
    if (mode == "retrieval")
        return 1;
    const auto& head = json.at("prediction_head");
    if (!head.is_array() || head.empty() || !head.back().is_number_integer())
        throw std::invalid_argument("hstu ranking requires a prediction_head dimension array");
    const nlohmann::json output = {{"output_dim", head.back()}};
    return integer(output, "output_dim", 1);
}

} // namespace

RuntimeConfig parse_runtime_config(const std::vector<char>& data,
                                   const std::vector<char>& embedding_keys) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    if (!json.is_object() || integer(json, "schema_version", 1) != 1)
        throw std::invalid_argument("hstu runtime.json requires schema_version=1");
    RuntimeConfig config;
    config.mode = json.at("mode").get<std::string>();
    if (config.mode != "ranking" && config.mode != "retrieval")
        throw std::invalid_argument("hstu mode must be ranking or retrieval");
    config.hidden_size = integer(json, "hidden_size", 1);
    config.max_sequence_length = integer(json, "max_sequence_length", 1);
    config.max_batch_size = integer(json, "max_batch_size", 1);
    parse_position_config(json, config);
    config.target_group_size = integer(json, "target_group_size", 1);
    config.scaling_seqlen = integer(json, "scaling_seqlen", -1);
    if (config.scaling_seqlen == 0)
        throw std::invalid_argument("hstu scaling_seqlen must be -1 or a positive integer");
    config.is_causal = boolean(json, "is_causal");
    config.disable_contextual_mask = boolean(json, "disable_contextual_mask");
    config.embedding_tables = parse_tables(json, embedding_keys);
    config.output_dim = prediction_width(json, config.mode);
    return config;
}

} // namespace trtmc::hstu
