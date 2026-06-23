#pragma once

// RecurrentPipeline: Mamba, RWKV, and Hybrid models.
// Uses IStateManager to abstract recurrent vs KV cache state.

#include "runtime/core/chat_template.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/inference_state.h"
#include "trtmc/runtime/recurrent_state.h"
#include "trtmc/runtime/sampler.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct RecurrentGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    bool has_position_input{false};
    ChatTemplateFormat chat_template_format{};
};

class RecurrentPipeline final : public IPipeline {
  public:
    RecurrentPipeline(std::unique_ptr<TrtModule> decoder, std::unique_ptr<IInferenceState> state,
                      RecurrentGenConfig config, cudaStream_t stream, const char* name,
                      std::shared_ptr<ITokenizer> tokenizer = nullptr,
                      std::string model_id_str = "", std::unique_ptr<ISampler> sampler = nullptr);

    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return name_; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids, const GenerateConfig& cfg);

    static int32_t argmax(const std::vector<float>& logits);

  private:
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<IInferenceState> state_;
    RecurrentGenConfig config_;
    cudaStream_t stream_;
    const char* name_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<ISampler> sampler_;

    std::vector<int32_t> generate_from_ids(const std::vector<int32_t>& input_ids,
                                           int32_t max_new_tokens, const SamplingParams& params);

    void run_step(int32_t token_id, std::vector<float>& logits);

    using SteadyClock = std::chrono::steady_clock;
    void report_timing(SteadyClock::time_point t_prefill_start,
                       SteadyClock::time_point t_prefill_end,
                       SteadyClock::time_point t_decode_start, SteadyClock::time_point t_decode_end,
                       int prefill_tokens, int decode_steps);

    // Cached logits output metadata (resolved once, reused every step)
    void* logits_device_ptr_{nullptr};
    std::size_t logits_numel_{0};

    // Per-step profiling accumulators
    double prof_prepare_ms_{0};
    double prof_forward_ms_{0};
    double prof_logits_copy_ms_{0};
    double prof_advance_ms_{0};
    int prof_steps_{0};
};

} // namespace trtmc
