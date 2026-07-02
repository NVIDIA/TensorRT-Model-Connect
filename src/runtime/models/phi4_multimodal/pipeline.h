/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Phi4MultimodalPipeline: vision-language generation (Qwen2.5-VL, Qwen3-VL, InternVL3, Phi4).
// Composes: vision_encoder TrtModule + text_decoder TrtModule + Phi4MultimodalKvCache.

#include "runtime/models/phi4_multimodal/image_preprocessor.h"
#include "runtime/models/phi4_multimodal/inference_state.h"
#include "runtime/models/phi4_multimodal/kv_cache.h"
#include "runtime/models/phi4_multimodal/sampler.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

struct Phi4MultimodalConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    int32_t image_token_id{-1};
    int32_t vision_output_dim{0};
    bool has_position_input{true};
};

class Phi4MultimodalPipeline final : public IPipeline {
  public:
    Phi4MultimodalPipeline(std::unique_ptr<TrtModule> text_decoder,
                           std::unique_ptr<TrtModule> vision_encoder,
                           std::unique_ptr<Phi4MultimodalInferenceState> state,
                           Phi4MultimodalConfig config,
                           Phi4MultimodalPreprocessConfig vl_preprocess, cudaStream_t stream,
                           std::shared_ptr<ITokenizer> tokenizer = nullptr,
                           std::string model_id_str = "",
                           std::unique_ptr<Phi4MultimodalISampler> sampler = nullptr);

    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    TextResult generate(const std::string& prompt, const float* image_pixels, int32_t image_height,
                        int32_t image_width, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Phi4MultimodalPipeline"; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids, const GenerateConfig& cfg);

    // VL preprocessing config (public for testing).
    const Phi4MultimodalPreprocessConfig& vl_preprocess_config() const { return vl_preprocess_; }
    bool has_vision_encoder() const { return vision_encoder_ != nullptr; }

    static int32_t argmax(const std::vector<float>& logits);

  private:
    std::unique_ptr<TrtModule> text_decoder_;
    std::unique_ptr<TrtModule> vision_encoder_;
    std::unique_ptr<Phi4MultimodalInferenceState> state_;
    Phi4MultimodalConfig config_;
    Phi4MultimodalPreprocessConfig vl_preprocess_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<Phi4MultimodalISampler> sampler_;

    std::vector<int32_t> generate_from_ids(const std::vector<int32_t>& input_ids,
                                           int32_t max_new_tokens,
                                           const Phi4MultimodalSamplingParams& params);

    // VL generation with vision features injected at image token positions.
    std::vector<int32_t> generate_vl_from_ids(
        const std::vector<int32_t>& input_ids, const std::vector<float>& image_features,
        const std::vector<std::vector<float>>& deepstack_features, int32_t num_features,
        int32_t feature_dim, int32_t max_new_tokens, const Phi4MultimodalSamplingParams& params);

    std::pair<int32_t, int32_t> resolve_gen_limits(const GenerateConfig& cfg) const;

    void run_vl_prefill_token(int32_t token_id, const std::vector<float>& image_features,
                              const std::vector<std::vector<float>>& deepstack_features,
                              int32_t num_features, int32_t feature_dim, int32_t& feature_index,
                              std::vector<float>& logits);

    void run_vl_decode_loop(Phi4MultimodalISampler* sampler,
                            const Phi4MultimodalSamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens);

    void run_text_step(int32_t token_id, std::vector<float>& logits);

    // Run a text step with optional vision embedding override.
    void run_text_step_with_embed(int32_t token_id, const float* input_embed, float use_input_embed,
                                  const std::vector<const float*>& deepstack_embeds,
                                  float deepstack_active, std::vector<float>& logits);

    // Run vision encoder on preprocessed image inputs.
    bool run_vision_encoder(const Phi4MultimodalPreprocessedImage& preprocessed,
                            std::vector<float>& image_features,
                            std::vector<std::vector<float>>* deepstack_features = nullptr);
};

} // namespace trtmc
