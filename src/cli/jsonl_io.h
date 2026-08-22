/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/image_features.h"
#include "trtmc/pipeline.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace trtmc::cli {

/// A single record from a JSONL dataset file (e.g. AIME evaluation sets).
struct DatasetSample {
    std::string sample_id;
    std::string answer;
    std::string prompt;
    std::optional<int32_t> seed_index;
};

/// Parse one non-empty JSONL line into a DatasetSample.
/// Required string fields: "sample_id", "answer", "prompt".
/// Optional integer field: "seed_index".
/// Throws std::runtime_error on parse failure, missing fields, or wrong types.
DatasetSample parse_dataset_line(const std::string& line, std::size_t line_no);

/// Build a JSONL record for a single text-generation sample.
/// Fields: id, prompt, generated, token_ids.
nlohmann::json build_text_sample_record(int32_t id, const std::string& prompt,
                                        const trtmc::TextResult& result);

/// Build a JSON record for a classification result.
/// Fields: top_class, top_score, num_classes.
nlohmann::json build_classify_record(const trtmc::ClassificationResult& result);

/// Build a JSON object for a float tensor with shape and data.
/// Throws std::runtime_error if any element is non-finite.
nlohmann::json build_tensor_record(const std::vector<int64_t>& shape,
                                   const std::vector<float>& data);

/// Build a JSON record for image feature extraction results.
/// Fields: last_hidden_state, pooler_output (each with shape + data).
nlohmann::json build_image_features_record(const trtmc::ImageFeaturesResult& result);

} // namespace trtmc::cli
