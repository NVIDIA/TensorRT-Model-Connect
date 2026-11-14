/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/bart/runtime/device_buffer.h"
#include "families/bart/runtime/inference_state.h"
#include "families/bart/runtime/tokenizer.h"
#include "trtmc/internal/config.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {

inline constexpr internal::ConfigField bart_config_fields[] = {
    {"max_new_tokens", internal::ConfigKind::I64, std::int64_t{128}, "Generation token cap"},
    {"temperature", internal::ConfigKind::F64, 1.0, "Sampling temperature"},
    {"top_k", internal::ConfigKind::I64, std::int64_t{1}, "Top-k sampling cutoff"},
};

class BartPipeline final : public internal::IModel,
                           public internal::IConditionalTextGeneration,
                           public ITextGeneration {
  public:
    BartPipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> decoder,
                 std::unique_ptr<BartInferenceState> state, int32_t hidden_size,
                 int32_t num_decoder_layers, int32_t max_source_length,
                 int32_t decoder_start_token_id, int32_t eos_token_id, int32_t bos_token_id,
                 int32_t pad_token_id, int32_t vocab_size, cudaStream_t stream,
                 std::shared_ptr<ITokenizer> tokenizer,
                 std::string task = std::string(internal::IConditionalTextGeneration::kTask),
                 std::string model_id = "");

    const char* task() const noexcept override { return task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    // Semantic Task SDK
    internal::TextResult run(const internal::ConditionalTextGenerationRequest& request,
                             internal::ConfigView config) override;

    // Legacy ITextGeneration
    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg) override;
    int32_t default_max_new_tokens() const override { return 128; }

  private:
    std::pair<std::vector<int32_t>, int32_t> prepare_encoder_input(const std::string& prompt);
    std::pair<std::vector<int32_t>, int32_t>
    prepare_encoder_input_semantic(const internal::TextSource& source);
    void run_encoder(const std::vector<int32_t>& padded_ids, int32_t actual_len);
    void setup_cross_attention();
    std::vector<int32_t> run_decoder(int32_t max_new_tokens);
    void run_decoder_step(int32_t token_id, std::vector<float>& logits);

    std::unique_ptr<ITrtModule> encoder_;
    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<BartInferenceState> state_;
    int32_t hidden_size_;
    int32_t num_decoder_layers_;
    int32_t max_source_length_;
    int32_t decoder_start_token_id_;
    int32_t eos_token_id_;
    int32_t bos_token_id_;
    int32_t pad_token_id_;
    int32_t vocab_size_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string task_;
    std::string model_id_;
    std::vector<bart::DeviceBuffer> cross_k_ptrs_;
    std::vector<bart::DeviceBuffer> cross_v_ptrs_;
    std::vector<float> encoder_attention_mask_;
    std::size_t cross_kv_bytes_{0};
    int32_t actual_enc_len_{0};
};

} // namespace trtmc
