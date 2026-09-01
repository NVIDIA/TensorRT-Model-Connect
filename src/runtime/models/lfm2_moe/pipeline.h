/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/lfm2_moe/inference_state.h"
#include "runtime/models/lfm2_moe/sampler.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct Lfm2MoeTextGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{-1};
    int32_t id_eos{-1};
    std::vector<int32_t> id_eos_ids;
    std::string token_id_name{"token_id"};
    std::string logits_output_name{"logits"};
    std::string chat_template_format;
};

class Lfm2MoeTextGenerationPipeline final : public IPipeline {
  public:
    Lfm2MoeTextGenerationPipeline(std::unique_ptr<TrtModule> decoder,
                                  std::unique_ptr<Lfm2MoeInferenceState> state,
                                  Lfm2MoeTextGenConfig config, cudaStream_t stream,
                                  std::shared_ptr<ITokenizer> tokenizer = nullptr,
                                  std::string model_id = "",
                                  std::unique_ptr<Lfm2MoeISampler> sampler = nullptr);

    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;
    int32_t default_max_new_tokens() const override { return 128; }
    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Lfm2MoeTextGenerationPipeline"; }

    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids, const GenerateConfig& cfg);

  private:
    struct TimedGenerationResult {
        std::vector<int32_t> token_ids;
        double setup_ms{0.0};
        double prefill_ms{0.0};
        double decode_ms{0.0};
    };

    TimedGenerationResult generate_from_ids(const std::vector<int32_t>& input_ids,
                                            int32_t max_new_tokens,
                                            const Lfm2MoeSamplingParams& params);
    void run_step(int32_t token_id, std::vector<float>& logits);
    std::vector<int32_t> encode_prompt(const std::string& prompt, const GenerateConfig& cfg) const;

    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<Lfm2MoeInferenceState> state_;
    Lfm2MoeTextGenConfig config_;
    cudaStream_t stream_{nullptr};
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<Lfm2MoeISampler> sampler_;
    const float* logits_device_ptr_{nullptr};
};

} // namespace trtmc
