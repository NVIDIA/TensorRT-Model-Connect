/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Model-owned decoder text pipeline.
//
// Composes: TrtModule (decoder) + GraniteKvCache + ITokenizer for this runtime
// plugin. Architecture-specific behavior remains in this model directory and
// in the TRT engine emitted by the matching family builder.

#include "runtime/models/granite/kv_cache.h"
#include "runtime/models/granite/sampler.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct GraniteTextGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    bool has_position_input{true};
    std::string chat_template_format{};
    std::string token_id_name{"token_id"};
    std::string logits_output_name{"logits"};
    // runtime.* namespace (replaces TRTMC_DISABLE_CUDA_GRAPH, TRTMC_GPU_ARGMAX).
    // decoder_plugin::create() populates these from ctx.runtime_config.
    bool disable_cuda_graph{false};
    bool prefer_gpu_greedy{false};
    bool log_runtime_stats{false};

    // Batched-prefill plumbing for the required split prefill engine.
    std::string present_k_pattern{"present_k_{i}"};
    std::string present_v_pattern{"present_v_{i}"};
    int32_t prefill_max_length{0};
    int32_t prefill_profile_index{-1};
    std::string prefill_log_label;
    int32_t num_layers{0};
    int32_t kv_dim{0};
};

// Populate the process-wide step-trace state from the resolved ConfigBundle.
// Called by decoder_plugin::create() before constructing the pipeline.
// Replaces the TRTMC_TEXT_STEP_TRACE_* env vars (deleted). Empty `path`
// keeps tracing disabled; a non-empty path truncates the target file.
void apply_text_trace_config_from_registry(const std::string& path, std::int32_t start_position,
                                           std::int32_t end_position, std::int32_t top_k);

class GraniteTextGenerationPipeline final : public IPipeline {
  public:
    struct DecoderContext {
        int32_t kv_rows{0};
        std::unique_ptr<TrtModule> module;
    };

    GraniteTextGenerationPipeline(std::vector<DecoderContext> decoders,
                                  std::unique_ptr<GraniteKvCache> state,
                                  GraniteTextGenConfig config, cudaStream_t stream,
                                  std::shared_ptr<ITokenizer> tokenizer = nullptr,
                                  std::string model_id_str = "",
                                  std::unique_ptr<GraniteISampler> sampler = nullptr,
                                  std::unique_ptr<TrtModule> prefill = nullptr,
                                  std::shared_ptr<void> distributed_owner = nullptr);

    // Public API: takes raw text, returns typed result.
    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "GraniteTextGenerationPipeline"; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids, const GenerateConfig& cfg);

    // Argmax over logits (public for testing).
    static int32_t argmax(const std::vector<float>& logits);

  private:
    // Kept before TRT modules so TP communicators outlive contexts/engines.
    std::shared_ptr<void> distributed_owner_;
    std::vector<DecoderContext> decoders_;
    std::unique_ptr<TrtModule> prefill_;
    std::unique_ptr<GraniteKvCache> state_;
    GraniteTextGenConfig config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<GraniteISampler> sampler_;
    bool prefer_gpu_greedy_{false};
    const float* d_logits_ptr_{nullptr}; // device logits pointer (for GPU sampling)
    std::string logits_output_name_;
    bool state_bound_{false};
    bool trace_prefill_phase_{false};
    double last_setup_ms_{0.0};

    // Internal: generate from token IDs with sampling parameters and timing.
    struct TimedGenResult {
        std::vector<int32_t> token_ids;
        double prefill_ms{0.0};
        double decode_ms{0.0};
    };
    TimedGenResult generate_from_ids(const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
                                     const GraniteSamplingParams& params,
                                     const GenerateConfig& cfg);
    void reset_generation_context();

    // Run one decoder step: token_id → logits (D2H to host). Updates cache.
    void run_step(int32_t token_id, std::vector<float>& logits);

    // Run one decoder step: logits stay on device (d_logits_ptr_ updated).
    void run_step_device(int32_t token_id);

    // Decode loop (extracted for CCN).
    int32_t run_decode_loop(GraniteISampler* sampler, const GraniteSamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens, bool gpu_sampling, const GenerateConfig& cfg,
                            int32_t prompt_token_count);
    TrtModule& bind_decoder_for_step();

    std::unique_ptr<GraniteISampler> make_step_sampler(const GraniteSamplingParams& params);
    void run_prefill(const std::vector<int32_t>& input_ids, std::vector<float>& logits,
                     bool gpu_sampling);
    void run_prefill_batched(const std::vector<int32_t>& input_ids, std::vector<float>& logits,
                             bool retain_device_logits);
    void run_prefill_chunk(const int32_t* token_ids, int32_t chunk_size,
                           const std::vector<const void*>& present_k,
                           const std::vector<const void*>& present_v, GraniteKvCache& kv,
                           std::vector<float>& logits, bool retain_device_logits);
    void log_batched_prefill(int32_t token_count, int32_t chunk_count, int32_t chunk_limit) const;
    void prime_decoder_after_batched_prefill(const std::vector<int32_t>& input_ids);
    bool should_stop_on_answer(const std::vector<int32_t>& output, int32_t prompt_token_count,
                               const GenerateConfig& cfg, int32_t steps, int32_t stop_interval,
                               bool is_eos) const;
    void log_decode_summary(int32_t steps, double ms) const;
};

} // namespace trtmc
