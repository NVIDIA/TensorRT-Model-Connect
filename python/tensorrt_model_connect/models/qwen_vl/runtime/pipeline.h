/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// QwenVlPipeline: vision-language generation (Qwen2.5-VL, Qwen3-VL, InternVL3, Phi4).
// Composes: vision_encoder TrtModule + text_decoder TrtModule + QwenVlKvCache.

#include "image_preprocessor.h"
#include "inference_state.h"
#include "kv_cache.h"
#include "lora_runtime.h"
#include "sampler.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

struct QwenVlConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    std::vector<int32_t> id_eos_ids;
    int32_t image_token_id{-1};
    int32_t vision_output_dim{0};
    bool has_position_input{true};
    int32_t num_layers{0};
    int32_t prefill_max_length{0};
    std::string present_k_pattern{"present_k_{i}"};
    std::string present_v_pattern{"present_v_{i}"};
};

class QwenVlPipeline final : public IPipeline {
  public:
    QwenVlPipeline(std::unique_ptr<TrtModule> text_decoder,
                   std::unique_ptr<TrtModule> vision_encoder,
                   std::unique_ptr<QwenVlInferenceState> state, QwenVlConfig config,
                   QwenVlPreprocessConfig vl_preprocess, cudaStream_t stream,
                   std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "",
                   std::unique_ptr<QwenVlISampler> sampler = nullptr,
                   std::unique_ptr<TrtModule> prefill = nullptr,
                   std::shared_ptr<qwen_vl::LoraAdapterCache> adapter_cache = nullptr);

    QwenVlPipeline(std::unique_ptr<TrtModule> text_decoder,
                   std::unique_ptr<TrtModule> vision_encoder,
                   std::unique_ptr<QwenVlInferenceState> state, QwenVlConfig config,
                   QwenVlPreprocessConfig vl_preprocess, cudaStream_t stream,
                   std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
                   std::unique_ptr<QwenVlISampler> sampler,
                   std::shared_ptr<qwen_vl::LoraAdapterCache> adapter_cache);

    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    TextResult generate(const std::string& prompt, const float* image_pixels, int32_t image_height,
                        int32_t image_width, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "QwenVlPipeline"; }
    bool supports_lora_adapters() const override { return has_dynamic_lora(); }
    void load_lora_adapter(const std::string& adapter_id, const std::string& adapter_path) override;
    void unload_lora_adapter(const std::string& adapter_id) override;
    std::vector<std::string> loaded_lora_adapters() const override;

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids, const GenerateConfig& cfg);

    // VL preprocessing config (public for testing).
    const QwenVlPreprocessConfig& vl_preprocess_config() const { return vl_preprocess_; }
    bool has_vision_encoder() const { return vision_encoder_ != nullptr; }
    bool has_dynamic_lora() const { return lora_bindings_ && lora_bindings_->enabled(); }
    std::vector<std::string> lora_input_names() const;
    void clear_lora_adapter();
    void register_lora_adapter(const std::string& adapter_id, const TensorMap& host_tensors);
    void select_lora_adapter(const std::string& adapter_id);

    static int32_t argmax(const std::vector<float>& logits);

  private:
    std::unique_ptr<TrtModule> text_decoder_;
    std::unique_ptr<TrtModule> prefill_;
    std::unique_ptr<TrtModule> vision_encoder_;
    std::unique_ptr<QwenVlInferenceState> state_;
    double last_setup_ms_{0.0};
    QwenVlConfig config_;
    QwenVlPreprocessConfig vl_preprocess_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<QwenVlISampler> sampler_;
    std::unique_ptr<qwen_vl::LoraInputBindings> lora_bindings_;
    std::unique_ptr<qwen_vl::LoraInputBindings> prefill_lora_bindings_;
    std::shared_ptr<qwen_vl::LoraAdapterCache> lora_adapter_cache_;
    std::unique_ptr<qwen_vl::LoraBindingContext> lora_binding_context_;
    std::unique_ptr<qwen_vl::LoraBindingContext> prefill_lora_binding_context_;

    std::vector<int32_t> generate_from_ids(const std::vector<int32_t>& input_ids,
                                           int32_t max_new_tokens,
                                           const QwenVlSamplingParams& params);

    // VL generation with vision features injected at image token positions.
    std::vector<int32_t> generate_vl_from_ids(
        const std::vector<int32_t>& input_ids, const std::vector<float>& image_features,
        const std::vector<std::vector<float>>& deepstack_features, int32_t num_features,
        int32_t feature_dim, int32_t merged_grid_height, int32_t merged_grid_width,
        int32_t max_new_tokens, const QwenVlSamplingParams& params);

    int32_t resolve_max_new_tokens(const GenerateConfig& cfg) const;

    void reset_generation_context(int32_t prompt_length);

    QwenVlISampler* prepare_sampler(const QwenVlSamplingParams& params,
                                    std::unique_ptr<QwenVlISampler>& local_sampler);

    void run_vl_prefill_token(int32_t token_id, const std::vector<float>& image_features,
                              const std::vector<std::vector<float>>& deepstack_features,
                              int32_t num_features, int32_t feature_dim, int32_t& feature_index,
                              const std::array<int32_t, 3>* mrope_position,
                              std::vector<float>& logits);

    bool run_vl_prefill_batched(const std::vector<int32_t>& input_ids,
                                const std::vector<float>& image_features,
                                const std::vector<std::vector<float>>& deepstack_features,
                                int32_t num_features, int32_t feature_dim,
                                const QwenVlMropePositions* mrope, std::vector<float>& logits);

    void run_vl_decode_loop(QwenVlISampler* sampler, const QwenVlSamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens, int32_t mrope_position = -1);

    void run_text_step(int32_t token_id, std::vector<float>& logits, int32_t mrope_position = -1);

    // Run a text step with optional vision embedding override.
    void run_text_step_with_embed(int32_t token_id, const float* input_embed, float use_input_embed,
                                  const std::vector<const float*>& deepstack_embeds,
                                  float deepstack_active,
                                  const std::array<int32_t, 3>* mrope_position,
                                  std::vector<float>& logits);

    // Run vision encoder on preprocessed image inputs.
    bool run_vision_encoder(const QwenVlPreprocessedImage& preprocessed,
                            std::vector<float>& image_features,
                            std::vector<std::vector<float>>* deepstack_features = nullptr);
};

} // namespace trtmc
