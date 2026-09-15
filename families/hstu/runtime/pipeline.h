/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::hstu {

struct EmbeddingTable {
    std::string name;
    std::string role;
    std::int32_t num_embeddings{0};
    std::int32_t offset{0};
    std::vector<std::int64_t> keys;
};

struct RuntimeConfig {
    std::string mode{"ranking"};
    std::int32_t hidden_size{0};
    std::int32_t output_dim{0};
    std::int32_t max_sequence_length{0};
    std::int32_t max_batch_size{0};
    std::int32_t position_buckets{0};
    std::int32_t time_buckets{0};
    std::int32_t target_group_size{1};
    std::int32_t scaling_seqlen{-1};
    bool is_causal{true};
    bool disable_contextual_mask{false};
    std::vector<EmbeddingTable> embedding_tables;
};

RuntimeConfig parse_runtime_config(const std::vector<char>& json,
                                   const std::vector<char>& embedding_keys);

class Pipeline final : public IRecommendation {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, RuntimeConfig config);
    RecommendationResult recommend(const RecommendationRequest& request) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    RuntimeConfig config_;
};

} // namespace trtmc::hstu
