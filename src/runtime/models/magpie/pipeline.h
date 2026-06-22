#pragma once

// MagpiePipeline: Magpie TTS encoder-decoder pipeline with optional CFG.
// Uses TrtModule(encoder) + TrtModule(decoder) + KvCache + TrtModule(codec).

#include "runtime/core/trt_common.h"
#include "runtime/domains/audio/audio_configs.h"
#include "plugin_helpers.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/inference_state.h"
#include "trtmc/runtime/kv_cache.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trtmc {

class MagpiePipeline final : public IPipeline {
  public:
    MagpiePipeline(std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> decoder,
                   std::unique_ptr<IInferenceState> decoder_state, std::unique_ptr<TrtModule> codec,
                   std::unique_ptr<TrtModule> lt_module, std::unique_ptr<TrtModule> prefill_module,
                   std::unique_ptr<IInferenceState> decoder_state_uncond,
                   std::vector<CudaBuffer> cross_k, std::vector<CudaBuffer> cross_v,
                   std::vector<CudaBuffer> cross_k_uncond, std::vector<CudaBuffer> cross_v_uncond,
                   CudaBuffer encoder_output, CudaBuffer encoder_output_uncond,
                   std::vector<float> audio_embed, std::vector<float> text_embed,
                   std::vector<float> context_embed, std::vector<int32_t> context_lengths,
                   MagpieTTSConfig config, cudaStream_t stream,
                   std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    ~MagpiePipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "MagpiePipeline"; }
    using IPipeline::generate_audio_streaming;

  private:
    struct DecoderLoopState {
        int32_t hidden{0};
        int32_t num_cb{0};
        int32_t cb_size{0};
        int32_t total_logits{0};
        bool use_cfg{false};
        bool use_gpu_kernels{false};
        bool use_gpu_greedy{false};
        bool use_gpu_sampling{false};
        bool use_cross_attn_tracking{false};
        int32_t estimated_frames{0};
        int32_t finished_limit{0};
        int32_t max_source_positions{0};
        int32_t text_consumed_threshold{1};
        bool text_consumed{false};
        int32_t frames_past_text_consumed{0};
        int32_t max_peak_pos{0};
        std::vector<float> logits;
        std::vector<float> embed_buf;
        std::vector<float> cb_embed;
        std::string error;
        double prof_prefill_ms{0.0};
        double prof_embed_ms{0.0};
        double prof_trt_step_ms{0.0};
        double prof_sample_ms{0.0};
    };

    // Streaming generation: calls audio_callback with PCM chunks as they're
    // produced. Each chunk is (samples_ptr, num_samples, sample_rate).
    // The callback is invoked from the generation thread. Returns total samples.
    using AudioChunkCallback = std::function<void(const float*, int32_t, int32_t)>;
    int32_t generate_audio_streaming(const std::vector<int32_t>& text_ids, int32_t max_frames,
                                     AudioChunkCallback audio_callback, int32_t chunk_frames = 32);

    // --- Streaming helpers ---
    struct StreamingCodecState {
        int32_t num_cb{0};
        int32_t total_samples_output{0};
        int32_t total_frames{0};
        int32_t frames_at_last_flush{0};
    };

    bool streaming_decode_one_frame(DecoderLoopState& state, int32_t frame,
                                    std::vector<int32_t>& prev_decode_codes,
                                    std::vector<int32_t>& all_codes,
                                    StreamingCodecState& codec_state);
    void streaming_flush_codec(StreamingCodecState& codec_state,
                               const std::vector<int32_t>& all_codes,
                               const AudioChunkCallback& audio_callback, bool is_final);

    void run_encoder(const std::vector<int32_t>& text_ids);
    void compute_cross_kv();
    void bind_cross_kv();
    void compute_cross_kv_uncond();
    void bind_cross_kv_uncond();

    DecoderLoopState init_decoder_state() const;
    int32_t prefill_context(DecoderLoopState& state);
    std::vector<int32_t> run_decoder(int32_t max_frames);
    std::vector<float> run_codec(const std::vector<int32_t>& codes, int32_t num_frames);

    void run_decoder_step(const float* embed, int32_t embed_size, std::vector<float>& logits_out);
    void run_decoder_step_uncond(const float* embed, int32_t embed_size,
                                 std::vector<float>& logits_out);

    std::vector<int32_t> run_cpu_sampling_loop(DecoderLoopState& state, int32_t max_frames);
    void cpu_compute_frame_embed(DecoderLoopState& state, const std::vector<int32_t>& prev_codes);
    bool cpu_run_conditioned_step(DecoderLoopState& state, int32_t frame);
    bool cpu_sample_frame_codes(DecoderLoopState& state, std::vector<int32_t>& frame_codes,
                                bool& eos);

    // GPU greedy loop
    std::vector<int32_t> run_gpu_greedy_loop(DecoderLoopState& state, int32_t max_frames);
    bool gpu_greedy_frame_step(DecoderLoopState& state, int32_t frame, CudaBuffer& d_eos_flag);
    void gpu_greedy_update_text_consumed(DecoderLoopState& state, int32_t frame);

    // GPU sampling loop (top-k temperature sampling on device)
    std::vector<int32_t> run_gpu_sampling_loop(DecoderLoopState& state, int32_t max_frames);
    bool gpu_sampling_frame_step(DecoderLoopState& state, int32_t frame, CudaBuffer& d_eos_flag,
                                 std::vector<int32_t>& h_codes);

    // Unified GPU stop checking
    bool gpu_check_stop_conditions(DecoderLoopState& state, int32_t frame, CudaBuffer& d_eos_flag,
                                   int32_t& h_eos_flag, int32_t& gen_frames_actual);
    // GPU-side text completion update
    void gpu_update_text_completion(DecoderLoopState& state, int32_t frame);

    // CFG passes
    bool run_cfg_uncond_pass_gpu(DecoderLoopState& state, int32_t frame);
    bool run_cfg_uncond_pass_cpu(DecoderLoopState& state, int32_t frame);

    void update_text_completion(DecoderLoopState& state, int32_t frame);
    bool check_finished_limit(DecoderLoopState& state, int32_t frame);

    // Attention prior management (monotonic alignment)
    void init_attention_prior();
    void reset_attention_prior();
    void update_attention_prior(int32_t frame);
    int32_t detect_attended_peak(const std::vector<float>& align, int32_t text_len);
    void construct_attention_prior(std::vector<float>& prior, int32_t best_pos, int32_t text_len);
    void upload_attention_prior();

    // Batched prefill using optimization profile 1 of the decoder engine
    void init_prefill_context();
    bool prefill_context_batched(int32_t ctx_frames);

    // Prefill path helpers
    bool prefill_context_gpu(DecoderLoopState& state, int32_t ctx_frames, const char* label);
    bool prefill_context_cpu(DecoderLoopState& state, int32_t ctx_frames, const char* label);
    bool prefill_context_cfg(DecoderLoopState& state, int32_t ctx_frames);
    bool prefill_context_sequential(DecoderLoopState& state, int32_t ctx_frames);

    // Local transformer (codebook AR sampling)
    void init_local_transformer();
    bool sample_frame_codes_lt(DecoderLoopState& state, std::vector<int32_t>& frame_codes,
                               bool& eos);

    // Extracted helpers (CCN reduction)
    void lt_run_codebook_step(int32_t cb, const std::vector<float>& decoder_hidden,
                              std::vector<float>& logits);
    void init_prefill_buffers(int32_t N, int32_t W);
    void bind_prefill_cross_kv();

    // Constructor helpers
    void upload_embeddings_to_gpu();
    void init_cross_attn_resources();
    void init_cfg_logit_buffers();

    void apply_env_overrides();
    void ensure_cfg_resources();
    void run_cfg_encoder(const std::vector<int32_t>& text_ids);
    void log_decoder_profiling(const DecoderLoopState& state, int32_t ctx_frames,
                               int32_t gen_frames) const;
    void log_pipeline_profiling(int32_t num_frames, int32_t num_samples, double ms_encoder,
                                double ms_decoder, double ms_codec, double ms_total) const;
    void lookup_embed(const float* table, int32_t token_id, float* out) const;
    void sum_embeds(const float* a, const float* b, float* out) const;
    int32_t sample_top_k(const float* logits, int32_t vocab_size, float temperature, int32_t top_k);

    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<IInferenceState> decoder_state_;
    std::unique_ptr<TrtModule> codec_;

    std::unique_ptr<IInferenceState> decoder_state_uncond_;

    std::vector<CudaBuffer> cross_k_, cross_v_;
    std::vector<CudaBuffer> cross_k_uncond_, cross_v_uncond_;
    CudaBuffer encoder_output_, encoder_output_uncond_;

    CudaBuffer cross_attn_weights_, cross_attn_weights_scratch_;
    bool has_cross_attn_output_{false};

    std::vector<float> audio_embed_, text_embed_, context_embed_;
    std::vector<int32_t> context_lengths_;

    CudaBuffer audio_embed_device_, context_embed_device_;
    CudaBuffer device_codes_, device_full_argmax_, device_prev_codes_;
    CudaBuffer device_all_codes_;
    CudaBuffer device_logits_cond_, device_logits_uncond_;

    // GPU sampling: host-generated random values uploaded per frame
    CudaBuffer device_rand_vals_{0};

    // Attention prior (monotonic alignment, NeMo inference)
    CudaBuffer attn_prior_device_{0};        // [1, 1, max_source_positions] prior input
    CudaBuffer alignment_weights_device_{0}; // alignment output (avg of layers 3-6)
    CudaBuffer alignment_scratch_device_{0}; // scratch for uncond pass
    bool has_attn_prior_{false};
    bool has_alignment_output_{false};
    int32_t last_attended_pos_{0};
    std::vector<int32_t> attended_count_; // per-position visit count

    // Batched prefill (optimization profile 1)
    std::unique_ptr<TrtModule> prefill_module_; // profile-1 context for batched prefill
    int32_t prefill_ctx_len_{0};
    CudaBuffer prefill_mask_{0};      // [1, ctx_len, max_cache + ctx_len] causal mask
    CudaBuffer prefill_logits_{0};    // [ctx_len, output_size] scratch for prefill logits
    CudaBuffer prefill_positions_{0}; // [ctx_len] int32 positions 0..ctx_len-1
    bool prefill_ready_{false};

    // Local transformer TrtModule (codebook AR sampling)
    std::unique_ptr<TrtModule> lt_module_; // secondary engine for 1-layer LT
    CudaBuffer lt_cache_k_{0}, lt_cache_v_{0};
    CudaBuffer lt_present_k_{0}, lt_present_v_{0};
    CudaBuffer lt_output_{0}, lt_mask_{0}, lt_position_id_{0}, lt_input_embed_{0};
    // CFG: duplicate LT KV caches for unconditional path
    CudaBuffer lt_cache_k_uncond_{0}, lt_cache_v_uncond_{0};
    CudaBuffer lt_present_k_uncond_{0}, lt_present_v_uncond_{0};
    CudaBuffer lt_output_uncond_{0};
    std::vector<float> lt_in_proj_w_; // [decoder_hidden, lt_hidden] in_projection weight
    std::vector<float> lt_in_proj_b_; // [lt_hidden] bias
    std::vector<float> lt_out_proj_;  // packed: 8 x (weight [lt_hidden, cb_size] + bias [cb_size])
    std::vector<float> lt_pos_embed_; // [lt_max_pos, lt_hidden] position embeddings
    int32_t lt_hidden_{0};            // 256 typically
    int32_t lt_max_cache_{8};
    bool has_lt_{false}; // true if LT engine was loaded
    // Device buffer for decoder_hidden output from main decoder engine
    CudaBuffer decoder_hidden_buf_{0};        // [1, decoder_hidden] conditioned
    CudaBuffer decoder_hidden_buf_uncond_{0}; // [1, decoder_hidden] unconditional (CFG)
    bool has_decoder_hidden_output_{false};

    cudaStream_t stream_;
    MagpieTTSConfig config_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::mt19937 rng_;
    int32_t text_length_{0};
};

} // namespace trtmc
