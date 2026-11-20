/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/lighton_ocr/runtime/image.h"
#include "families/lighton_ocr/runtime/tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <vector>

namespace trtmc::lighton_ocr {

struct RuntimeConfig {
    std::int32_t max_sequence_length{0};
    std::int32_t max_new_tokens{0};
    std::int32_t hidden_size{0};
    std::int32_t vocab_size{0};
    std::int32_t num_hidden_layers{0};
    std::int32_t num_key_value_heads{0};
    std::int32_t head_dim{0};
    std::int32_t image_token_id{0};
    std::vector<std::int32_t> eos_token_ids;
    std::vector<std::int32_t> prompt_prefix_ids;
    std::vector<std::int32_t> prompt_suffix_ids;
    ImageConfig image;
};

struct Observations {
    PreparedImage image;
    std::vector<float> image_features;
    std::vector<std::int32_t> prompt_ids;
    std::vector<std::vector<float>> logits;
    bool cache_prefix_unchanged{true};
};

class Pipeline final : public internal::IModel, public internal::IImagesTextToText {
  public:
    Pipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> decoder,
             std::vector<float> embedding, Tokenizer tokenizer, RuntimeConfig config);
    const char* task() const noexcept override { return internal::IImagesTextToText::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::ImagesTextToTextRequest&, internal::ConfigView) override;

    // Family-owned test observations use the same preprocessing, prompt and
    // engine execution as run(); this is not an additional public Task.
    Observations observe(const internal::ImageView&, const std::vector<std::int32_t>& forced_ids);

  private:
    struct State {
        std::vector<float> key;
        std::vector<float> value;
        std::int32_t length{0};
    };
    struct Prompt {
        PreparedImage image;
        std::vector<float> features;
        std::vector<std::int32_t> ids;
        std::vector<float> embeddings;
    };
    Prompt prompt(const internal::ImageView&, std::int32_t output_budget);
    std::vector<float> decode(const std::vector<float>& embedding, State& state,
                              bool* prefix_unchanged = nullptr);
    std::vector<float> embed(std::int32_t id) const;
    bool is_eos(std::int32_t id) const;

    std::unique_ptr<ITrtModule> vision_;
    std::unique_ptr<ITrtModule> decoder_;
    std::vector<float> embedding_;
    Tokenizer tokenizer_;
    RuntimeConfig config_;
};

} // namespace trtmc::lighton_ocr
