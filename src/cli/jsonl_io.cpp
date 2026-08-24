/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/jsonl_io.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::cli {

DatasetSample parse_dataset_line(const std::string& line, std::size_t line_no) {
    nlohmann::json obj;
    try {
        obj = nlohmann::json::parse(line);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Malformed JSON at line " + std::to_string(line_no) + ": " +
                                 e.what());
    }

    if (!obj.is_object()) {
        throw std::runtime_error("Expected JSON object at line " + std::to_string(line_no));
    }

    auto require_string = [&](const char* key) -> std::string {
        auto it = obj.find(key);
        if (it == obj.end()) {
            throw std::runtime_error("Dataset line missing required field \"" + std::string(key) +
                                     "\" at line " + std::to_string(line_no));
        }
        if (!it->is_string()) {
            throw std::runtime_error("Field \"" + std::string(key) +
                                     "\" must be a string at line " + std::to_string(line_no));
        }
        return it->get<std::string>();
    };

    DatasetSample sample;
    sample.sample_id = require_string("sample_id");
    sample.answer = require_string("answer");
    sample.prompt = require_string("prompt");

    auto seed_it = obj.find("seed_index");
    if (seed_it != obj.end()) {
        std::int64_t seed_index = 0;
        if (seed_it->is_number_unsigned()) {
            const auto value = seed_it->get<std::uint64_t>();
            if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::runtime_error(
                    "Field \"seed_index\" is outside the int32 range at line " +
                    std::to_string(line_no));
            }
            seed_index = static_cast<std::int64_t>(value);
        } else if (seed_it->is_number_integer()) {
            seed_index = seed_it->get<std::int64_t>();
        } else {
            throw std::runtime_error("Field \"seed_index\" must be an integer at line " +
                                     std::to_string(line_no));
        }
        if (seed_index < std::numeric_limits<std::int32_t>::min() ||
            seed_index > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error("Field \"seed_index\" is outside the int32 range at line " +
                                     std::to_string(line_no));
        }
        sample.seed_index = static_cast<std::int32_t>(seed_index);
    }

    return sample;
}

nlohmann::json build_text_sample_record(int32_t id, const std::string& prompt,
                                        const trtmc::TextResult& result) {
    nlohmann::json record;
    record["id"] = id;
    record["prompt"] = prompt;
    record["generated"] = result.text;
    record["token_ids"] = result.token_ids;
    return record;
}

nlohmann::json build_classify_record(const trtmc::ClassificationResult& result) {
    nlohmann::json record;
    record["top_class"] = result.top_class;
    record["top_score"] = result.top_score;
    record["num_classes"] = result.logits.size();
    return record;
}

nlohmann::json build_tensor_record(const std::vector<int64_t>& shape,
                                   const std::vector<float>& data) {
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i])) {
            throw std::runtime_error("Image feature tensor contains a non-finite value");
        }
    }
    nlohmann::json record;
    record["shape"] = shape;
    record["data"] = data;
    return record;
}

nlohmann::json build_image_features_record(const trtmc::ImageFeaturesResult& result) {
    nlohmann::json record;
    record["last_hidden_state"] =
        build_tensor_record(result.last_hidden_state_shape, result.last_hidden_state);
    record["pooler_output"] = build_tensor_record(result.pooler_output_shape, result.pooler_output);
    return record;
}

} // namespace trtmc::cli
