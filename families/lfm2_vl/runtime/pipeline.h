/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lfm2_vl/runtime/image_processor.h"
#include "families/lfm2_vl/runtime/inference_state.h"
#include "families/lfm2_vl/runtime/sampler.h"
#include "families/lfm2_vl/runtime/tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct Lfm2VlRequest {
    std::string prompt;
    std::vector<ImageResult> images;
    std::vector<bool> integer_pixels;
    TextGenerationConfig generation;
    std::int32_t min_image_tokens{-1};
    std::int32_t max_image_tokens{-1};
    std::int32_t max_tiles{-1};
    std::int32_t split_images{-1};
};

Lfm2VlRequest lfm2_vl_prepare_request(const internal::ImagesTextToTextRequest&,
                                      internal::ConfigView);

struct Lfm2VlTextGenConfig {
    std::string primary_task{"vision_language_generation"};
    std::string precision{"bf16"};
    bool clean_up_tokenization_spaces{false};
    int32_t vocab_size{0};
    int32_t id_bos{-1};
    int32_t id_eos{-1};
    std::vector<int32_t> id_eos_ids;
    std::string token_id_name{"token_id"};
    std::string logits_output_name{"logits"};
    std::string chat_template_format;
};

class Lfm2VlTextGenerationPipeline final : public IVisionLanguageGeneration,
                                           public internal::IModel,
                                           public internal::IImagesTextToText,
                                           public internal::ITextConversation {
  public:
    Lfm2VlTextGenerationPipeline(std::unique_ptr<ITrtModule> decoder,
                                 std::unique_ptr<Lfm2VlInferenceState> state,
                                 Lfm2VlTextGenConfig config, cudaStream_t stream,
                                 std::shared_ptr<ITokenizer> tokenizer = nullptr,
                                 std::string model_id = "",
                                 std::unique_ptr<ITrtModule> vision = nullptr,
                                 std::vector<float> positions = {});

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg = {});
    TextResult generate(const std::string& prompt, const float* pixels, std::int32_t height,
                        std::int32_t width, const TextGenerationConfig& cfg = {}) override;
    TextResult generate(const Lfm2VlRequest& request);
    const char* task() const noexcept override { return config_.primary_task.c_str(); }
    TextResult run(const internal::ImagesTextToTextRequest&, internal::ConfigView) override;
    internal::ConversationResult run(const internal::TextConversationRequest&,
                                     internal::ConfigView) override;
    std::vector<internal::TaskInstance> task_bindings() override;
    int32_t default_max_new_tokens() const { return 128; }

    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids,
                                  const TextGenerationConfig& cfg);

  private:
    struct TimedGenerationResult {
        std::vector<int32_t> token_ids;
        double setup_ms{0.0};
        double prefill_ms{0.0};
        double decode_ms{0.0};
    };

    TimedGenerationResult generate_from_ids(const std::vector<int32_t>& input_ids,
                                            int32_t max_new_tokens,
                                            const Lfm2VlSamplingParams& params);
    void run_step(int32_t token_id, std::vector<float>& logits, bool decode = false);
    std::vector<int32_t> encode_prompt(const std::string& prompt,
                                       const TextGenerationConfig& cfg) const;

    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<Lfm2VlInferenceState> state_;
    Lfm2VlTextGenConfig config_;
    cudaStream_t stream_{nullptr};
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    const float* logits_device_ptr_{nullptr};
    std::unique_ptr<ITrtModule> vision_;
    std::vector<float> position_table_;
    std::vector<float> image_embeddings_;
    std::size_t image_offset_{0};
    std::vector<float> zero_embedding_;
};

} // namespace trtmc
