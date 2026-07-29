/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/glm/kv_cache.h"
#include "runtime/models/glm/sampler.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct GlmTextGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    std::string chat_template_format;
    std::string token_id_name{"token_id"};
    std::string logits_output_name{"logits"};
    bool disable_cuda_graph{false};
    bool prefer_gpu_greedy{false};
    bool log_runtime_stats{false};
    std::string present_k_pattern{"present_k_{i}"};
    std::string present_v_pattern{"present_v_{i}"};
    int32_t prefill_max_length{0};
    std::string prefill_log_label;
    int32_t num_layers{0};
};

// A non-empty path enables family-owned JSONL logits tracing. This is used by
// E2E parity without loading a second Python TensorRT runtime.
void apply_text_trace_config_from_registry(const std::string& path, std::int32_t start_position,
                                           std::int32_t end_position, std::int32_t top_k);

class GlmTextGenerationPipeline final : public IPipeline {
  public:
    GlmTextGenerationPipeline(std::unique_ptr<TrtModule> decoder,
                              std::unique_ptr<TrtModule> prefill, std::unique_ptr<GlmKvCache> state,
                              GlmTextGenConfig config, cudaStream_t stream,
                              std::shared_ptr<ITokenizer> tokenizer, std::string model_id = "",
                              std::unique_ptr<GlmISampler> sampler = nullptr);

    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;
    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "GlmTextGenerationPipeline"; }

    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids, const GenerateConfig& cfg);
    static int32_t argmax(const std::vector<float>& logits);

  private:
    struct TimedGenResult {
        std::vector<int32_t> token_ids;
        double prefill_ms{0.0};
        double decode_ms{0.0};
    };

    TimedGenResult generate_from_ids(const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
                                     const GlmSamplingParams& params, const GenerateConfig& cfg);
    void reset_generation_context();
    std::unique_ptr<GlmISampler> make_step_sampler(const GlmSamplingParams& params);
    void run_prefill(const std::vector<int32_t>& input_ids, std::vector<float>& logits,
                     bool retain_device_logits);
    void run_prefill_chunk(const int32_t* token_ids, int32_t chunk_size, GlmKvCache& cache,
                           const std::vector<const void*>& present_k,
                           const std::vector<const void*>& present_v, std::vector<float>& logits,
                           bool retain_device_logits);
    void prime_decoder_after_prefill(const std::vector<int32_t>& input_ids);
    TrtModule& bind_decoder();
    void run_step(int32_t token_id, std::vector<float>& logits, const char* trace_phase = "decode");
    void run_step_device(int32_t token_id);
    int32_t run_decode_loop(GlmISampler* sampler, const GlmSamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens, bool gpu_sampling, const GenerateConfig& cfg,
                            int32_t prompt_token_count);
    bool should_stop_on_answer(const std::vector<int32_t>& output, int32_t prompt_token_count,
                               const GenerateConfig& cfg, int32_t steps, int32_t stop_interval,
                               bool is_eos) const;
    void log_prefill(int32_t token_count, int32_t chunk_count) const;
    void log_decode_summary(int32_t steps, double milliseconds) const;

    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<TrtModule> prefill_;
    std::unique_ptr<GlmKvCache> state_;
    GlmTextGenConfig config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<GlmISampler> sampler_;
    bool prefer_gpu_greedy_{false};
    bool decoder_bound_{false};
    const float* device_logits_{nullptr};
    double last_setup_ms_{0.0};
};

} // namespace trtmc
