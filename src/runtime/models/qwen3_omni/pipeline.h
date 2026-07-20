/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// OmniPipeline: omni multimodal pipeline with thinker + talker + code2wav.
// Uses a TensorRT Thinker, the checkpoint's official model-owned Talker bridge,
// and a TensorRT Code2Wav decoder.

#include "runtime/models/qwen3_omni/inference_state.h"
#include "runtime/models/qwen3_omni/kv_cache.h"
#include "runtime/models/qwen3_omni/omni_config.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class Qwen3OmniTalkerRuntime;

struct OmniThinkerRunStats {
    int32_t prompt_tokens{0};
    int32_t prefill_launches{0};
    int32_t decode_launches{0};
    int32_t full_logits_d2h{0};
};

class OmniPipeline final : public IPipeline {
  public:
    OmniPipeline(std::unique_ptr<TrtModule> thinker,
                 std::unique_ptr<Qwen3OmniInferenceState> thinker_state,
                 std::unique_ptr<TrtModule> code2wav, OmniConfig config, cudaStream_t stream,
                 std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "",
                 std::unique_ptr<TrtModule> thinker_prefill = nullptr);

    ~OmniPipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "OmniPipeline"; }

    // Token-ID entry point and launch counters for deterministic runtime tests.
    std::vector<int32_t> generate_thinker_ids(const std::vector<int32_t>& input_ids,
                                              int32_t max_tokens);
    const OmniThinkerRunStats& thinker_run_stats() const { return thinker_stats_; }

  private:
    int32_t run_thinker_step(int32_t token_id);
    bool run_thinker_prefill(const std::vector<int32_t>& input_ids, int32_t& next_token);
    std::vector<int32_t> run_thinker(const std::vector<int32_t>& input_ids, int32_t max_tokens);
    std::vector<float> run_code2wav(const std::vector<int32_t>& codec_tokens, int32_t n_codebooks,
                                    int32_t n_frames, double& code2wav_and_transfer_ms,
                                    double& output_materialization_ms);

    std::unique_ptr<TrtModule> thinker_;
    std::unique_ptr<TrtModule> thinker_prefill_;
    std::unique_ptr<Qwen3OmniInferenceState> thinker_state_;
    std::unique_ptr<TrtModule> code2wav_;
    std::unique_ptr<OmniConfig> config_;
    std::unique_ptr<Qwen3OmniTalkerRuntime> talker_runtime_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    DeviceTensor thinker_token_id_;
    int32_t thinker_token_host_{0};
    OmniThinkerRunStats thinker_stats_;
};

} // namespace trtmc
