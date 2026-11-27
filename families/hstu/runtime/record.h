/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trtmc::hstu {

using Json = nlohmann::ordered_json;

struct Feature {
    std::string name;
    std::int64_t vocab_size{0};
};

// The bundle's request geometry (runtime.json): every shape is static, so a
// record must hold exactly this many history events and candidates.
struct Geometry {
    std::string precision;
    std::int64_t history_length{0};
    std::int64_t num_candidates{0};
    std::int64_t tokens{0};
    std::int64_t num_tasks{0};
    std::vector<Feature> contextual;
    Feature item;
    Feature action;
};

Geometry parse_geometry(const Json& runtime);

// One ranking request: the engine inputs, every id inside its table.
struct Request {
    std::vector<std::int32_t> contextual_ids;
    std::vector<std::int32_t> history_item_ids;
    std::vector<std::int32_t> history_action_ids;
    std::vector<std::int32_t> candidate_item_ids;
    std::string model{"hstu"};
};

// {"user": {<contextual feature>: id, ...},
//  "history": {<item feature>: [H ids], <action feature>: [H ids]},
//  "candidates": {<item feature>: [C ids]}, "model": optional string}
Request parse_request(const Json& document, const Geometry& geometry);

} // namespace trtmc::hstu
