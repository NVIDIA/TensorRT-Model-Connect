/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct BundleSectionInfo {
    std::string name;
    std::uint64_t offset{0};
    std::uint64_t size{0};
};

// Per-component batch-size envelope baked into a diffusion bundle.
// Absent in the bundle JSON => all caps default to 1 (today's behavior).
struct MaxBatchSize {
    int32_t dit{1};
    int32_t text_encoder{1};
    int32_t vae{1};
};

struct BundleInfo {
    std::string model_id;
    std::string model_type;
    std::string family;
    std::string precision;
    std::string trt_version;
    std::string trt_abi;
    std::string gpu_name;
    std::string created_at;
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t num_layers{0};
    int32_t num_attention_heads{1};
    int32_t num_key_value_heads{1};
    int32_t max_cache_length{32};
    std::string runtime_strategy;
    bool tokenizer_add_special_tokens{false};
    bool tokenizer_add_special_tokens_present{false};
    std::vector<BundleSectionInfo> sections;
    MaxBatchSize max_batch_size{};
};

// Read metadata without loading the engine.
BundleInfo InspectBundle(const std::string& bundle_path);

// Check if path is a .trtfb file (valid magic bytes).
bool IsBundle(const std::string& path);

} // namespace trtmc
