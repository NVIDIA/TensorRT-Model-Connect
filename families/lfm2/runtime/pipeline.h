/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lfm2/runtime/inference_state.h"
#include "families/lfm2/runtime/sampler.h"
#include "families/lfm2/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct Lfm2TextGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{-1};
    int32_t id_eos{-1};
    std::vector<int32_t> id_eos_ids;
    std::string token_id_name{"token_id"};
    std::string logits_output_name{"logits"};
    std::string chat_template_format;
};

class Lfm2TextGenerationPipeline final : public ITextGeneration {
  public:
    Lfm2TextGenerationPipeline(std::unique_ptr<ITrtModule> decoder,
                               std::unique_ptr<Lfm2InferenceState> state, Lfm2TextGenConfig config,
                               cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer = nullptr,
                               std::string model_id = "");

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg = {}) override;
    int32_t default_max_new_tokens() const override { return 128; }

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
                                            const Lfm2SamplingParams& params);
    void run_step(int32_t token_id, std::vector<float>& logits);
    std::vector<int32_t> encode_prompt(const std::string& prompt,
                                       const TextGenerationConfig& cfg) const;

    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<Lfm2InferenceState> state_;
    Lfm2TextGenConfig config_;
    cudaStream_t stream_{nullptr};
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    const float* logits_device_ptr_{nullptr};
};

} // namespace trtmc
