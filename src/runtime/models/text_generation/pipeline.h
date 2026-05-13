#pragma once

// TextGenerationPipeline: serves ALL decoder-only LLMs.
// HF equivalent: TextGenerationPipeline (one class, many models).
//
// Composes: TrtModule (decoder) + KvCache + ITokenizer.
// The model-specific architecture (GQA, RoPE, SwiGLU, etc.) is baked into
// the TRT engine. This pipeline just runs prefill → decode loop.

#include "runtime/core/chat_template.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/inference_state.h"
#include "trtmc/runtime/sampler.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct TextGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    bool has_position_input{true};
    ChatTemplateFormat chat_template_format{ChatTemplateFormat::kNone};
    std::string token_id_name{"token_id"};
    std::string logits_output_name{"logits"};
    // runtime.* namespace (replaces TRTMC_DISABLE_CUDA_GRAPH, TRTMC_GPU_ARGMAX).
    // decoder_plugin::create() populates these from ctx.runtime_config.
    bool disable_cuda_graph{false};
    bool prefer_gpu_greedy{false};

    // Batched-prefill plumbing — populated when the bundle ships with a
    // dedicated prefill optimization profile. The runtime forwards the
    // whole prompt through `prefill_module` once (writing per-layer K/V
    // into the shared cache via write_prefill_kv) before falling back to
    // the per-token decode loop.
    std::string present_k_pattern{"present_k_{i}"};
    std::string present_v_pattern{"present_v_{i}"};
    int32_t prefill_max_length{0};
    int32_t num_layers{0};
    int32_t kv_dim{0};
};

// Populate the process-wide step-trace state from the resolved ConfigBundle.
// Called by decoder_plugin::create() before constructing the pipeline.
// Replaces the TRTMC_TEXT_STEP_TRACE_* env vars (deleted). Empty `path`
// keeps tracing disabled; a non-empty path truncates the target file.
void apply_text_trace_config_from_registry(const std::string& path, std::int32_t start_position,
                                           std::int32_t end_position, std::int32_t top_k);

class TextGenerationPipeline final : public IPipeline {
  public:
    struct DecoderContext {
        int32_t kv_rows{0};
        std::unique_ptr<TrtModule> module;
    };

    TextGenerationPipeline(std::unique_ptr<TrtModule> decoder,
                           std::unique_ptr<IInferenceState> state, TextGenConfig config,
                           cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer = nullptr,
                           std::string model_id_str = "",
                           std::unique_ptr<ISampler> sampler = nullptr,
                           std::shared_ptr<void> distributed_owner = nullptr);
    TextGenerationPipeline(std::vector<DecoderContext> decoders,
                           std::unique_ptr<IInferenceState> state, TextGenConfig config,
                           cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer = nullptr,
                           std::string model_id_str = "",
                           std::unique_ptr<ISampler> sampler = nullptr,
                           std::unique_ptr<TrtModule> prefill = nullptr,
                           std::shared_ptr<void> distributed_owner = nullptr);

    // Public API: takes raw text, returns typed result.
    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "TextGenerationPipeline"; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids, const GenerateConfig& cfg);

    // Argmax over logits (public for testing).
    static int32_t argmax(const std::vector<float>& logits);

  private:
    // Kept before TRT modules so it is destroyed after prefill_/decoders_.
    // TensorRT sampleDistCollective destroys its context/engine before
    // ncclCommDestroy; this member preserves that ordering for TP pipelines.
    std::shared_ptr<void> distributed_owner_;
    std::vector<DecoderContext> decoders_;
    std::unique_ptr<TrtModule> prefill_;
    std::unique_ptr<IInferenceState> state_;
    TextGenConfig config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<ISampler> sampler_;
    bool prefer_gpu_greedy_{false};
    const float* d_logits_ptr_{nullptr}; // device logits pointer (for GPU sampling)
    std::string logits_output_name_;
    int32_t active_decoder_index_{-1};
    bool state_bound_{false};

    // Internal: generate from token IDs with sampling parameters and timing.
    struct TimedGenResult {
        std::vector<int32_t> token_ids;
        double prefill_ms{0.0};
        double decode_ms{0.0};
    };
    TimedGenResult generate_from_ids(const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
                                     const SamplingParams& params, const GenerateConfig& cfg);

    // Run one decoder step: token_id → logits (D2H to host). Updates cache.
    void run_step(int32_t token_id, std::vector<float>& logits);

    // Run one decoder step: logits stay on device (d_logits_ptr_ updated).
    void run_step_device(int32_t token_id);

    // Decode loop (extracted for CCN).
    int32_t run_decode_loop(ISampler* sampler, const SamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens, bool gpu_sampling, const GenerateConfig& cfg,
                            int32_t prompt_token_count);
    int32_t select_decoder_index(int32_t desired_rows) const;
    TrtModule& bind_decoder_for_step();

    std::unique_ptr<ISampler> make_step_sampler(const SamplingParams& params);
    void run_prefill(const std::vector<int32_t>& input_ids, std::vector<float>& logits,
                     bool gpu_sampling);
    // Returns true if the batched prefill engine handled the prompt; false
    // means caller must fall back to the per-token decode loop.
    bool run_prefill_batched(const std::vector<int32_t>& input_ids, std::vector<float>& logits);
    bool should_stop_on_answer(const std::vector<int32_t>& output, int32_t prompt_token_count,
                               const GenerateConfig& cfg, int32_t steps, int32_t stop_interval,
                               bool is_eos) const;
    void log_decode_summary(int32_t steps, double ms) const;
};

} // namespace trtmc
