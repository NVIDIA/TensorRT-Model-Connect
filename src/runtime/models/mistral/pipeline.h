#pragma once

// Model-owned decoder text pipeline.
//
// Composes: TrtModule (decoder) + MistralKvCache + ITokenizer for this runtime
// plugin. Architecture-specific behavior remains in this model directory and
// in the TRT engine emitted by the matching family builder.

#include "runtime/models/mistral/inference_state.h"
#include "runtime/models/mistral/sampler.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class MistralKvCache;

struct MistralTextGenConfig {
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

    // Batched-prefill plumbing — populated when the bundle ships with a
    // dedicated prefill optimization profile. The runtime forwards the
    // whole prompt through `prefill_module` once (writing per-layer K/V
    // into the shared cache via write_prefill_kv) before falling back to
    // the per-token decode loop.
    std::string present_k_pattern{"present_k_{i}"};
    std::string present_v_pattern{"present_v_{i}"};
    int32_t prefill_max_length{0};
    int32_t prefill_profile_index{-1};
    std::string prefill_log_label;
    int32_t num_layers{0};
    int32_t kv_dim{0};
    int32_t mask_token_id{-1};
    int32_t diffusion_block_length{32};
    bool supports_text_diffusion{false};
};

// Populate the process-wide step-trace state from the resolved ConfigBundle.
// Called by decoder_plugin::create() before constructing the pipeline.
// Replaces the TRTMC_TEXT_STEP_TRACE_* env vars (deleted). Empty `path`
// keeps tracing disabled; a non-empty path truncates the target file.
void apply_text_trace_config_from_registry(const std::string& path, std::int32_t start_position,
                                           std::int32_t end_position, std::int32_t top_k);

class MistralTextGenerationPipeline final : public IPipeline {
  public:
    struct DecoderContext {
        int32_t kv_rows{0};
        std::unique_ptr<TrtModule> module;
    };

    MistralTextGenerationPipeline(std::unique_ptr<TrtModule> decoder,
                                  std::unique_ptr<MistralInferenceState> state,
                                  MistralTextGenConfig config, cudaStream_t stream,
                                  std::shared_ptr<ITokenizer> tokenizer = nullptr,
                                  std::string model_id_str = "",
                                  std::unique_ptr<MistralISampler> sampler = nullptr,
                                  std::shared_ptr<void> distributed_owner = nullptr);
    MistralTextGenerationPipeline(std::vector<DecoderContext> decoders,
                                  std::unique_ptr<MistralInferenceState> state,
                                  MistralTextGenConfig config, cudaStream_t stream,
                                  std::shared_ptr<ITokenizer> tokenizer = nullptr,
                                  std::string model_id_str = "",
                                  std::unique_ptr<MistralISampler> sampler = nullptr,
                                  std::unique_ptr<TrtModule> prefill = nullptr,
                                  std::unique_ptr<TrtModule> linear_spec_lora_prefill = nullptr,
                                  std::shared_ptr<void> distributed_owner = nullptr);

    // Public API: takes raw text, returns typed result.
    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "MistralTextGenerationPipeline"; }

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
    std::unique_ptr<TrtModule> linear_spec_lora_prefill_;
    std::unique_ptr<MistralInferenceState> state_;
    MistralTextGenConfig config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<MistralISampler> sampler_;
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
                                     const MistralSamplingParams& params,
                                     const GenerateConfig& cfg);
    TimedGenResult generate_diffusion_from_ids(const std::vector<int32_t>& input_ids,
                                               int32_t max_new_tokens,
                                               const MistralSamplingParams& params,
                                               const GenerateConfig& cfg);
    TimedGenResult generate_linear_spec_from_ids(const std::vector<int32_t>& input_ids,
                                                 int32_t max_new_tokens,
                                                 const MistralSamplingParams& params,
                                                 const GenerateConfig& cfg, bool use_lora_draft);
    std::string resolve_generation_mode(const GenerateConfig& cfg) const;
    void reset_generation_context();
    TrtModule& require_block_prefill(int32_t sq, TrtModule* prefill_override);
    MistralKvCache& require_block_kv_cache();
    void copy_block_logits(const TensorMap& outputs, std::vector<float>& logits) const;
    void append_prefill_kv(MistralKvCache& kv, TrtModule& prefill, int32_t sq);
    int32_t resolve_text_diffusion_block_length(const GenerateConfig& cfg, int32_t max_new_tokens,
                                                bool require_divisible) const;
    int32_t seed_next_token_from_prefill(const std::vector<int32_t>& input_ids,
                                         std::vector<float>& logits, int32_t vocab);
    void fill_diffusion_block(std::vector<int32_t>& block, std::vector<float>& logits,
                              int32_t block_len, int32_t vocab, bool use_threshold,
                              float threshold);
    int32_t verify_diffusion_block(const std::vector<int32_t>& block, std::vector<float>& logits,
                                   int32_t block_len, int32_t vocab);
    bool append_tokens_until_eos(const std::vector<int32_t>& tokens, std::vector<int32_t>& output,
                                 const MistralSamplingParams& params) const;
    void fill_linear_spec_block(std::vector<int32_t>& block, std::vector<float>& logits,
                                int32_t block_len, int32_t vocab, bool threshold_enabled,
                                float threshold, bool use_lora_draft);
    std::vector<int32_t> verify_linear_spec_block(const std::vector<int32_t>& block,
                                                  std::vector<float>& logits, int32_t block_len,
                                                  int32_t vocab);
    static int32_t count_linear_spec_accepts(const std::vector<int32_t>& ar_tokens,
                                             const std::vector<int32_t>& block);
    bool append_linear_spec_tokens(const std::vector<int32_t>& ar_tokens, int32_t emit_count,
                                   std::vector<int32_t>& output, int32_t& generated,
                                   const MistralSamplingParams& params) const;

    // Run one decoder step: token_id → logits (D2H to host). Updates cache.
    void run_step(int32_t token_id, std::vector<float>& logits);

    // Run one decoder step: logits stay on device (d_logits_ptr_ updated).
    void run_step_device(int32_t token_id);

    // Decode loop (extracted for CCN).
    int32_t run_decode_loop(MistralISampler* sampler, const MistralSamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens, bool gpu_sampling, const GenerateConfig& cfg,
                            int32_t prompt_token_count);
    int32_t select_decoder_index(int32_t desired_rows) const;
    TrtModule& bind_decoder_for_step();

    std::unique_ptr<MistralISampler> make_step_sampler(const MistralSamplingParams& params);
    void run_prefill(const std::vector<int32_t>& input_ids, std::vector<float>& logits,
                     bool gpu_sampling);
    void run_prefill_block(const std::vector<int32_t>& input_ids, bool bidirectional,
                           bool append_kv, std::vector<float>& logits,
                           TrtModule* prefill_override = nullptr);
    // Returns true if the batched prefill engine handled the prompt; false
    // means caller must fall back to the per-token decode loop.
    bool run_prefill_batched(const std::vector<int32_t>& input_ids, std::vector<float>& logits);
    void prime_decoder_after_batched_prefill(const std::vector<int32_t>& input_ids);
    bool should_stop_on_answer(const std::vector<int32_t>& output, int32_t prompt_token_count,
                               const GenerateConfig& cfg, int32_t steps, int32_t stop_interval,
                               bool is_eos) const;
    void log_decode_summary(int32_t steps, double ms) const;
};

} // namespace trtmc
