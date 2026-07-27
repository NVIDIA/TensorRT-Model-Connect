/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/magpie/pipeline.h"

#include "runtime/models/magpie/decode_runtime.h"
#include "runtime/models/magpie/magpie_codec_plan.h"
#include "runtime/models/magpie/magpie_decode_policy.h"
#include "runtime/models/magpie/magpie_decoder_plan.h"
#include "runtime/models/magpie/magpie_generation_plan.h"
#include "runtime/models/magpie/magpie_text_completion_policy.h"
#include "runtime/models/magpie/tensor_names.h"

#ifndef TRTMC_HAS_CUDA_KERNELS
#define TRTMC_HAS_CUDA_KERNELS 0
#endif

#if TRTMC_HAS_CUDA_KERNELS
#include "runtime/models/magpie/magpie_kernels.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace trtmc {

namespace {
using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;
inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void log_magpie_frame_preview(const std::vector<int32_t>& all_codes, int32_t num_cb) {
    const int32_t gen_frames = static_cast<int32_t>(all_codes.size()) / std::max(num_cb, 1);
    for (int32_t f = 0; f < std::min(gen_frames, 10); ++f) {
        std::cerr << "[magpie-tts]   frame " << f << ": [";
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            if (cb > 0)
                std::cerr << ", ";
            std::cerr << all_codes[static_cast<std::size_t>(f) * num_cb + cb];
        }
        std::cerr << "]" << std::endl;
    }
    if (gen_frames <= 15)
        return;
    std::cerr << "[magpie-tts]   ..." << std::endl;
    for (int32_t f = gen_frames - 5; f < gen_frames; ++f) {
        std::cerr << "[magpie-tts]   frame " << f << ": [";
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            if (cb > 0)
                std::cerr << ", ";
            std::cerr << all_codes[static_cast<std::size_t>(f) * num_cb + cb];
        }
        std::cerr << "]" << std::endl;
    }
}

bool check_magpie_gpu_kernels_available([[maybe_unused]] const MagpieCudaBuffer& audio_embed,
                                        [[maybe_unused]] const MagpieCudaBuffer& codes,
                                        [[maybe_unused]] const MagpieCudaBuffer& full_argmax,
                                        [[maybe_unused]] const MagpieCudaBuffer& prev_codes) {
#if TRTMC_HAS_CUDA_KERNELS
    return audio_embed.ok() && codes.ok() && full_argmax.ok() && prev_codes.ok();
#else
    return false;
#endif
}

void upload_magpie_prev_codes_to_device([[maybe_unused]] MagpieCudaBuffer& d_prev,
                                        [[maybe_unused]] const int32_t* host_codes,
                                        [[maybe_unused]] int32_t num_cb,
                                        [[maybe_unused]] bool use_gpu,
                                        [[maybe_unused]] bool use_gpu_greedy) {
#if TRTMC_HAS_CUDA_KERNELS
    if (use_gpu && !use_gpu_greedy) {
        cudaMemcpy(d_prev.data(), host_codes, static_cast<std::size_t>(num_cb) * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
    }
#endif
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// MagpiePipeline (TrtModule-based)
// ═══════════════════════════════════════════════════════════════════════════

MagpiePipeline::MagpiePipeline(
    std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> decoder,
    std::unique_ptr<MagpieInferenceState> decoder_state, std::unique_ptr<TrtModule> codec,
    std::unique_ptr<TrtModule> lt_module, std::unique_ptr<TrtModule> prefill_module,
    std::unique_ptr<MagpieInferenceState> decoder_state_uncond,
    std::vector<MagpieCudaBuffer> cross_k, std::vector<MagpieCudaBuffer> cross_v,
    std::vector<MagpieCudaBuffer> cross_k_uncond, std::vector<MagpieCudaBuffer> cross_v_uncond,
    MagpieCudaBuffer encoder_output, MagpieCudaBuffer encoder_output_uncond,
    std::vector<float> audio_embed, std::vector<float> text_embed, std::vector<float> context_embed,
    std::vector<int32_t> context_lengths, MagpieTTSConfig config, cudaStream_t stream,
    std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str)
    : encoder_(std::move(encoder)), decoder_(std::move(decoder)),
      decoder_state_(std::move(decoder_state)), codec_(std::move(codec)),
      decoder_state_uncond_(std::move(decoder_state_uncond)), cross_k_(std::move(cross_k)),
      cross_v_(std::move(cross_v)), cross_k_uncond_(std::move(cross_k_uncond)),
      cross_v_uncond_(std::move(cross_v_uncond)), encoder_output_(std::move(encoder_output)),
      encoder_output_uncond_(std::move(encoder_output_uncond)), cross_attn_weights_(0),
      cross_attn_weights_scratch_(0), audio_embed_(std::move(audio_embed)),
      text_embed_(std::move(text_embed)), context_embed_(std::move(context_embed)),
      context_lengths_(std::move(context_lengths)), audio_embed_device_(0),
      context_embed_device_(0),
      device_codes_(static_cast<std::size_t>(config.num_codebooks) * sizeof(int32_t)),
      device_full_argmax_(static_cast<std::size_t>(config.num_codebooks) * sizeof(int32_t)),
      device_prev_codes_(static_cast<std::size_t>(config.num_codebooks) * sizeof(int32_t)),
      device_all_codes_(static_cast<std::size_t>(512) * config.num_codebooks * sizeof(int32_t)),
      device_logits_cond_(0), device_logits_uncond_(0),
      device_rand_vals_(static_cast<std::size_t>(config.num_codebooks) * sizeof(float)),
      prefill_module_(std::move(prefill_module)), lt_module_(std::move(lt_module)), stream_(stream),
      config_(config), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      rng_(std::random_device{}()) {
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("MagpiePipeline: invalid decoder module");
    if (!decoder_state_ || !decoder_state_->ok())
        throw std::runtime_error("MagpiePipeline: invalid decoder state");
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("MagpiePipeline: invalid encoder module");

    upload_embeddings_to_gpu();
    init_cross_attn_resources();
    init_cfg_logit_buffers();
    init_attention_prior();
    init_prefill_context();
    init_local_transformer();
}

MagpiePipeline::~MagpiePipeline() = default;

// The rest of the MagpiePipeline methods (lines 1421-2633 of original audio_pipeline.cpp)
// are included verbatim below.

void MagpiePipeline::upload_embeddings_to_gpu() {
    audio_embed_device_ = MagpieCudaBuffer(audio_embed_.size() * sizeof(float));
    context_embed_device_ = MagpieCudaBuffer(context_embed_.size() * sizeof(float));
    if (!audio_embed_.empty() && audio_embed_device_.ok())
        cudaMemcpy(audio_embed_device_.data(), audio_embed_.data(),
                   audio_embed_.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (!context_embed_.empty() && context_embed_device_.ok())
        cudaMemcpy(context_embed_device_.data(), context_embed_.data(),
                   context_embed_.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void MagpiePipeline::init_cross_attn_resources() {
    if (!decoder_->has_output("cross_attn_weights"))
        return;
    has_cross_attn_output_ = true;
    const auto xattn_bytes = static_cast<std::size_t>(config_.max_source_positions) * sizeof(float);
    cross_attn_weights_ = MagpieCudaBuffer(xattn_bytes);
    if (config_.cfg_scale > 1.0F)
        cross_attn_weights_scratch_ = MagpieCudaBuffer(xattn_bytes);
}

void MagpiePipeline::init_cfg_logit_buffers() {
    if (config_.cfg_scale <= 1.0F)
        return;
    const auto logits_bytes = static_cast<std::size_t>(config_.num_codebooks) *
                              static_cast<std::size_t>(config_.codebook_size) * sizeof(float);
    device_logits_cond_ = MagpieCudaBuffer(logits_bytes);
    device_logits_uncond_ = MagpieCudaBuffer(logits_bytes);
}

void MagpiePipeline::lookup_embed(const float* table, int32_t token_id, float* out) const {
    const auto offset =
        static_cast<std::size_t>(token_id) * static_cast<std::size_t>(config_.hidden_size);
    std::memcpy(out, table + offset, static_cast<std::size_t>(config_.hidden_size) * sizeof(float));
}

void MagpiePipeline::sum_embeds(const float* a, const float* b, float* out) const {
    for (int32_t i = 0; i < config_.hidden_size; ++i)
        out[i] = a[i] + b[i];
}

int32_t MagpiePipeline::sample_top_k(const float* logits, int32_t vocab_size, float temperature,
                                     int32_t top_k) {
    // Greedy mode: return argmax
    if (config_.greedy) {
        int32_t best = 0;
        for (int32_t i = 1; i < vocab_size; ++i)
            if (logits[i] > logits[best])
                best = i;
        return best;
    }

    // llama.cpp-style top-k: (logit, index) struct sorted in-place.
    // - Persistent thread-local buffer avoids per-call allocation
    // - Struct comparator is cache-friendly (data adjacent, no indirection)
    // - partial_sort is near-optimal for small vocab (2024) x small k (80)
    struct TokenData {
        float logit;
        int32_t id;
    };
    static thread_local std::vector<TokenData> s_candidates;

    top_k = std::min(top_k, vocab_size);

    // Build candidate array (one linear pass over logits)
    s_candidates.resize(static_cast<std::size_t>(vocab_size));
    for (int32_t i = 0; i < vocab_size; ++i) {
        s_candidates[i] = {logits[i], i};
    }

    // partial_sort: place top-k in descending order at front
    std::partial_sort(s_candidates.begin(), s_candidates.begin() + top_k, s_candidates.end(),
                      [](const TokenData& a, const TokenData& b) { return a.logit > b.logit; });

    // Temperature softmax over top-k (fused: compute exp, accumulate sum,
    // then scan for sample — avoids separate normalize pass)
    const float max_logit = s_candidates[0].logit;
    const float inv_temp = 1.0F / temperature;

    // Compute unnormalized probs and total sum in one pass
    float sum = 0.0F;
    for (int32_t i = 0; i < top_k; ++i) {
        float p = std::exp((s_candidates[i].logit - max_logit) * inv_temp);
        s_candidates[i].logit = p; // reuse logit field for prob storage
        sum += p;
    }

    // Sample: scan CDF against random threshold (no separate normalize pass)
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    const float threshold = dist(rng_) * sum;
    float cumulative = 0.0F;
    for (int32_t i = 0; i < top_k; ++i) {
        cumulative += s_candidates[i].logit;
        if (cumulative > threshold) {
            return s_candidates[i].id;
        }
    }
    return s_candidates[top_k - 1].id;
}

void MagpiePipeline::run_encoder(const std::vector<int32_t>& text_ids) {
    const int32_t max_pos = config_.max_source_positions;

    std::vector<int32_t> padded(static_cast<std::size_t>(max_pos), 0);
    const auto copy_len = std::min(static_cast<int32_t>(text_ids.size()), max_pos);
    if (copy_len > 0)
        std::memcpy(padded.data(), text_ids.data(),
                    static_cast<std::size_t>(copy_len) * sizeof(int32_t));

    Tensor input_ids_tensor;
    input_ids_tensor.data = padded.data();
    input_ids_tensor.shape = {static_cast<int64_t>(max_pos)};
    input_ids_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["input_ids"] = input_ids_tensor;

    // Use forward_async + sync to keep output on device, then D2D copy
    encoder_->forward_async(inputs);
    encoder_->sync();

    // Copy encoder output from module's internal buffer to our persistent buffer
    void* enc_out_ptr = encoder_->device_ptr("encoder_output");
    if (enc_out_ptr) {
        const auto bytes = static_cast<std::size_t>(max_pos) *
                           static_cast<std::size_t>(config_.hidden_size) * sizeof(float);
        cudaMemcpy(encoder_output_.data(), enc_out_ptr, bytes, cudaMemcpyDeviceToDevice);
    }

    // Zero out encoder output for padded positions
    if (copy_len < max_pos) {
        const auto hidden = config_.hidden_size;
        const auto zero_offset =
            static_cast<std::size_t>(copy_len) * static_cast<std::size_t>(hidden) * sizeof(float);
        const auto zero_bytes = static_cast<std::size_t>(max_pos - copy_len) *
                                static_cast<std::size_t>(hidden) * sizeof(float);
        cudaMemset(static_cast<char*>(encoder_output_.data()) + zero_offset, 0, zero_bytes);
    }

    std::cerr << "[magpie-tts] Encoder: processed " << copy_len << " tokens (padded to " << max_pos
              << ")" << std::endl;
}

// ---------------------------------------------------------------------------
// Cross-KV management
// ---------------------------------------------------------------------------

void MagpiePipeline::compute_cross_kv() {
    // All decoder layers receive the SAME encoder output as cross_k and cross_v
    // (the per-layer K/V projections are baked into the TRT decoder graph).
    // Instead of copying encoder output into N separate buffers (2*N D2D copies),
    // we now bind all layers directly to the encoder output buffer in bind_cross_kv().
    // No copies needed.
}

void MagpiePipeline::bind_cross_kv() {
    // Bind all layers to the SAME encoder output buffer (shared cross-KV).
    const int32_t dec_layers = static_cast<int32_t>(cross_k_.size());
    for (int32_t i = 0; i < dec_layers; ++i) {
        const std::string cross_k_name = magpie_layer_tensor_name("cross_k", i);
        const std::string cross_v_name = magpie_layer_tensor_name("cross_v", i);
        decoder_->bind_external(cross_k_name, encoder_output_.data());
        decoder_->bind_external(cross_v_name, encoder_output_.data());
    }

    if (has_cross_attn_output_ && cross_attn_weights_.ok())
        decoder_->bind_external("cross_attn_weights", cross_attn_weights_.data());

    // Bind decoder_hidden output for LT (conditioned path)
    if (has_decoder_hidden_output_ && decoder_hidden_buf_.ok())
        decoder_->bind_external("decoder_hidden", decoder_hidden_buf_.data());

    // Bind attention prior input and alignment weights output
    if (has_attn_prior_ && attn_prior_device_.ok())
        decoder_->bind_external("cross_attn_prior", attn_prior_device_.data());
    if (has_alignment_output_ && alignment_weights_device_.ok())
        decoder_->bind_external("alignment_weights", alignment_weights_device_.data());
}

void MagpiePipeline::compute_cross_kv_uncond() {
    // Same optimization as compute_cross_kv(): no copies needed.
    // bind_cross_kv_uncond() will bind directly to encoder_output_uncond_.
}

void MagpiePipeline::bind_cross_kv_uncond() {
    // Bind all layers to the shared unconditional encoder output buffer.
    const int32_t dec_layers = static_cast<int32_t>(cross_k_uncond_.size());
    for (int32_t i = 0; i < dec_layers; ++i) {
        const std::string cross_k_name = magpie_layer_tensor_name("cross_k", i);
        const std::string cross_v_name = magpie_layer_tensor_name("cross_v", i);
        decoder_->bind_external(cross_k_name, encoder_output_uncond_.data());
        decoder_->bind_external(cross_v_name, encoder_output_uncond_.data());
    }

    // Redirect cross_attn_weights to scratch buffer so uncond pass
    // doesn't overwrite the conditioned weights we need for tracking
    if (has_cross_attn_output_ && cross_attn_weights_scratch_.ok())
        decoder_->bind_external("cross_attn_weights", cross_attn_weights_scratch_.data());

    // Redirect decoder_hidden to uncond buffer so uncond pass doesn't
    // overwrite the conditioned hidden state we need for LT
    if (has_decoder_hidden_output_ && decoder_hidden_buf_uncond_.ok())
        decoder_->bind_external("decoder_hidden", decoder_hidden_buf_uncond_.data());

    // Redirect alignment_weights to scratch for uncond pass
    if (has_alignment_output_ && alignment_scratch_device_.ok())
        decoder_->bind_external("alignment_weights", alignment_scratch_device_.data());
    // Prior input stays the same for uncond (same prior applied to both paths)
}

// ---------------------------------------------------------------------------
// Decoder step via TrtModule
// ---------------------------------------------------------------------------

void MagpiePipeline::run_decoder_step(const float* embed, int32_t embed_size,
                                      std::vector<float>& logits_out) {
    int32_t dummy_token = 0;
    float use_input_embed = 1.0F;

    std::vector<float> embed_buf(embed, embed + embed_size);

    Tensor token_tensor;
    token_tensor.data = &dummy_token;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    Tensor embed_tensor;
    embed_tensor.data = embed_buf.data();
    embed_tensor.shape = {1, static_cast<int64_t>(embed_size)};
    embed_tensor.dtype = DType::kFloat32;

    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_input_embed;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    inputs["input_embed"] = embed_tensor;
    inputs["use_input_embed"] = use_embed_tensor;
    decoder_state_->prepare_step(inputs);

    TensorMap outputs = decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it != outputs.end()) {
        const auto& lt = it->second;
        auto n = lt.numel();
        logits_out.resize(static_cast<std::size_t>(n));
        std::memcpy(logits_out.data(), lt.data, n * sizeof(float));
    }

    decoder_state_->advance();
}

void MagpiePipeline::run_decoder_step_uncond(const float* embed, int32_t embed_size,
                                             std::vector<float>& logits_out) {
    // Swap to unconditional cache + cross-KV
    decoder_state_uncond_->bind_to(*decoder_);
    bind_cross_kv_uncond();

    int32_t dummy_token = 0;
    float use_input_embed = 1.0F;

    std::vector<float> embed_buf(embed, embed + embed_size);

    Tensor token_tensor;
    token_tensor.data = &dummy_token;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    Tensor embed_tensor;
    embed_tensor.data = embed_buf.data();
    embed_tensor.shape = {1, static_cast<int64_t>(embed_size)};
    embed_tensor.dtype = DType::kFloat32;

    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_input_embed;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    inputs["input_embed"] = embed_tensor;
    inputs["use_input_embed"] = use_embed_tensor;
    decoder_state_uncond_->prepare_step(inputs);

    TensorMap outputs = decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it != outputs.end()) {
        const auto& lt = it->second;
        auto n = lt.numel();
        logits_out.resize(static_cast<std::size_t>(n));
        std::memcpy(logits_out.data(), lt.data, n * sizeof(float));
    }

    decoder_state_uncond_->advance();

    // Restore conditioned cache + cross-KV
    decoder_state_->bind_to(*decoder_);
    bind_cross_kv();
}

// ---------------------------------------------------------------------------
// Decoder loop state initialization
// ---------------------------------------------------------------------------

MagpiePipeline::DecoderLoopState MagpiePipeline::init_decoder_state() const {
    DecoderLoopState s;
    const auto plan = make_magpie_decoder_plan(
        config_, static_cast<bool>(decoder_state_uncond_),
        static_cast<bool>(decoder_state_uncond_), // resources == cache in new runtime
        !cross_k_uncond_.empty(),
        check_magpie_gpu_kernels_available(audio_embed_device_, device_codes_, device_full_argmax_,
                                           device_prev_codes_),
        has_cross_attn_output_, cross_attn_weights_.ok(), text_length_);
    s.hidden = plan.hidden;
    s.num_cb = plan.num_cb;
    s.cb_size = plan.cb_size;
    s.total_logits = plan.total_logits;
    s.use_cfg = plan.use_cfg;
    s.use_gpu_kernels = plan.use_gpu_kernels;
    s.use_gpu_greedy = plan.use_gpu_greedy;
    s.use_gpu_sampling = plan.use_gpu_sampling;
    s.finished_limit = plan.finished_limit;
    s.max_source_positions = plan.max_source_positions;
    s.use_cross_attn_tracking = plan.use_cross_attn_tracking;
    s.estimated_frames = plan.estimated_frames;
    s.text_consumed_threshold = plan.text_consumed_threshold;

    s.embed_buf.resize(static_cast<std::size_t>(s.hidden));
    s.cb_embed.resize(static_cast<std::size_t>(s.hidden));

    return s;
}

// ---------------------------------------------------------------------------
// Phase 1: Context prefill
// ---------------------------------------------------------------------------

int32_t MagpiePipeline::prefill_context(DecoderLoopState& state) {
    if (context_embed_.empty() || context_lengths_.empty())
        return 0;

    const int32_t ctx_frames = context_lengths_[0];

    std::cerr << "[magpie-tts] Prefilling " << ctx_frames << " context frames ..." << std::endl;
    const auto t_prefill_start = SteadyClock::now();

    bool ok = false;

    // Batched prefill disabled: prefill_module_ writes present_k/v to its own
    // internal buffers, but decoder_state_->advance() reads from decoder_'s
    // buffers (stale data).  Fall through to sequential path until we add a
    // proper cross-module KV copy.
    // TODO(chaog): implement KV cache transfer from prefill_module_ to decoder_
    (void)prefill_ready_;

    // Sequential fallback (original path)
    if (!ok)
        ok = prefill_context_sequential(state, ctx_frames);

    if (!ok)
        return -1;

    const auto t_prefill_end = SteadyClock::now();
    state.prof_prefill_ms = elapsed_ms(t_prefill_start, t_prefill_end);

    return ctx_frames;
}

bool MagpiePipeline::prefill_context_sequential(DecoderLoopState& state, int32_t ctx_frames) {
    const int32_t hidden = state.hidden;

    // Conditioned cache prefill
    decoder_state_->bind_to(*decoder_);
    bind_cross_kv();

    const float* ctx_ptr = context_embed_.data();
    for (int32_t pos = 0; pos < ctx_frames; ++pos) {
        const float* frame_embed = ctx_ptr + static_cast<std::size_t>(pos) * hidden;
        run_decoder_step(frame_embed, hidden, state.logits);
    }

    // CFG: prefill unconditional cache with same speaker context but uncond cross-KV
    if (state.use_cfg && decoder_state_uncond_) {
        std::cerr << "[magpie-tts] CFG: prefilling unconditional cache (" << ctx_frames
                  << " frames) ..." << std::endl;

        decoder_state_uncond_->bind_to(*decoder_);
        bind_cross_kv_uncond();

        for (int32_t pos = 0; pos < ctx_frames; ++pos) {
            const float* frame_embed = ctx_ptr + static_cast<std::size_t>(pos) * hidden;

            int32_t dummy_token = 0;
            float use_input_embed = 1.0F;
            std::vector<float> embed_buf(frame_embed, frame_embed + hidden);

            Tensor token_tensor;
            token_tensor.data = &dummy_token;
            token_tensor.shape = {1};
            token_tensor.dtype = DType::kInt32;

            Tensor embed_tensor;
            embed_tensor.data = embed_buf.data();
            // Engine declares input_embed as [-1, hidden] (2-D); the
            // decoder expects rank 2 even for single-frame steps.
            embed_tensor.shape = {1, static_cast<int64_t>(hidden)};
            embed_tensor.dtype = DType::kFloat32;

            Tensor use_embed_tensor;
            use_embed_tensor.data = &use_input_embed;
            use_embed_tensor.shape = {1};
            use_embed_tensor.dtype = DType::kFloat32;

            TensorMap inputs;
            inputs["token_id"] = token_tensor;
            inputs["input_embed"] = embed_tensor;
            inputs["use_input_embed"] = use_embed_tensor;
            decoder_state_uncond_->prepare_step(inputs);

            decoder_->forward(inputs);
            decoder_state_uncond_->advance();
        }

        // Restore conditioned state
        decoder_state_->bind_to(*decoder_);
        bind_cross_kv();
    }

    return ctx_frames;
}

// ---------------------------------------------------------------------------
// CFG unconditional passes
// ---------------------------------------------------------------------------

bool MagpiePipeline::run_cfg_uncond_pass_gpu(DecoderLoopState& state, int32_t frame) {
    (void)frame;
#if TRTMC_HAS_CUDA_KERNELS
    // Save conditioned logits
    void* cond_logits_ptr = decoder_->device_ptr("logits");
    cudaMemcpyAsync(device_logits_cond_.data(), cond_logits_ptr,
                    static_cast<std::size_t>(state.total_logits) * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // Copy embed from conditioned decoder's input_embed to reuse
    void* cond_embed_ptr = decoder_->device_ptr("input_embed");

    // Run unconditional pass
    decoder_state_uncond_->bind_to(*decoder_);
    bind_cross_kv_uncond();

    // Copy embed
    void* uncond_embed_ptr = decoder_->device_ptr("input_embed");
    cudaMemcpyAsync(uncond_embed_ptr, cond_embed_ptr,
                    static_cast<std::size_t>(state.hidden) * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // Forward async
    decoder_->forward_device_async({});
    decoder_state_uncond_->advance();

    void* uncond_logits_ptr = decoder_->device_ptr("logits");

    // CFG interpolation: out = uncond + scale * (cond - uncond)
    // Restore conditioned bindings first
    decoder_state_->bind_to(*decoder_);
    bind_cross_kv();

    magpie_cfg_interpolate_device(static_cast<const float*>(device_logits_cond_.data()),
                                  static_cast<const float*>(uncond_logits_ptr),
                                  static_cast<float*>(decoder_->device_ptr("logits")),
                                  config_.cfg_scale, state.total_logits, stream_);
    return true;
#else
    (void)state;
    return false;
#endif
}

bool MagpiePipeline::run_cfg_uncond_pass_cpu(DecoderLoopState& state, int32_t frame) {
    std::vector<float> cond_logits = state.logits;
    std::vector<float> uncond_logits;

    // Run unconditional pass using full step (which swaps cache/cross-KV)
    run_decoder_step_uncond(state.embed_buf.data(), state.hidden, uncond_logits);

    // CPU-side CFG blend: logits = uncond + scale * (cond - uncond)
    const auto n = std::min(cond_logits.size(), uncond_logits.size());
    state.logits.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        state.logits[i] =
            uncond_logits[i] + config_.cfg_scale * (cond_logits[i] - uncond_logits[i]);
    }
    (void)frame;
    return true;
}

// ---------------------------------------------------------------------------
// Text-completion tracking
// ---------------------------------------------------------------------------

void MagpiePipeline::update_text_completion(DecoderLoopState& state, int32_t frame) {
    if (state.use_cross_attn_tracking && !state.text_consumed) {
        std::vector<float> xattn(static_cast<std::size_t>(state.max_source_positions));
        cudaMemcpy(xattn.data(), cross_attn_weights_.data(),
                   static_cast<std::size_t>(state.max_source_positions) * sizeof(float),
                   cudaMemcpyDeviceToHost);

        if (update_magpie_text_consumed_from_cross_attn(xattn.data(), state.max_source_positions,
                                                        state.text_consumed_threshold,
                                                        state.max_peak_pos, state.text_consumed)) {
            std::cerr << "[magpie-tts] Text consumed at frame " << frame
                      << " (max_peak_pos=" << state.max_peak_pos
                      << ", threshold=" << state.text_consumed_threshold
                      << ", text_len=" << text_length_ << ")" << std::endl;
        }
    }

    if (!state.use_cross_attn_tracking) {
        update_magpie_text_consumed_from_heuristic(state.estimated_frames, frame,
                                                   state.text_consumed);
    }
}

bool MagpiePipeline::check_finished_limit(DecoderLoopState& state, int32_t frame) {
    // Track frames past text completion (needed for EOS gating in sampling).
    // Use attention prior's position tracking when available, otherwise
    // fall back to cross-attn peak tracking.
    const bool near_end =
        has_attn_prior_ ? (last_attended_pos_ >= text_length_ - 2) : state.text_consumed;
    if (near_end) {
        ++state.frames_past_text_consumed;
    }

    if (!config_.enable_finished_limit_stop)
        return false;
    if (state.text_consumed && state.frames_past_text_consumed >= state.finished_limit) {
        std::cerr << "[magpie-tts] finished_limit_with_eot: stopping at frame " << frame << " ("
                  << state.frames_past_text_consumed << " frames past text consumed)" << std::endl;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// CPU frame embed computation
// ---------------------------------------------------------------------------

void MagpiePipeline::cpu_compute_frame_embed(DecoderLoopState& state,
                                             const std::vector<int32_t>& prev_codes) {
    const int32_t num_cb = state.num_cb;
    const int32_t cb_size = state.cb_size;
    const int32_t hidden = state.hidden;

    std::fill(state.embed_buf.begin(), state.embed_buf.end(), 0.0F);
    for (int32_t cb = 0; cb < num_cb; ++cb) {
        const float* table = audio_embed_.data() + static_cast<std::size_t>(cb) * cb_size * hidden;
        lookup_embed(table, prev_codes[cb], state.cb_embed.data());
        sum_embeds(state.embed_buf.data(), state.cb_embed.data(), state.embed_buf.data());
    }
    const float inv_cb = 1.0F / static_cast<float>(num_cb);
    for (int32_t i = 0; i < hidden; ++i)
        state.embed_buf[i] *= inv_cb;
}

// ---------------------------------------------------------------------------
// GPU greedy loop
// ---------------------------------------------------------------------------

bool MagpiePipeline::gpu_greedy_frame_step(DecoderLoopState& state, int32_t frame,
                                           MagpieCudaBuffer& d_eos_flag) {
#if TRTMC_HAS_CUDA_KERNELS
    constexpr int32_t EOS_TOKEN = 2017;
    constexpr int32_t AUDIO_RANGE = 2016;

    const int32_t num_cb = state.num_cb;
    const int32_t cb_size = state.cb_size;
    const int32_t hidden = state.hidden;

    // Embed
    const auto t_embed_start = SteadyClock::now();
    void* embed_ptr = decoder_->device_ptr("input_embed");
    magpie_gather_average_embed_device(static_cast<const float*>(audio_embed_device_.data()),
                                       static_cast<const int32_t*>(device_prev_codes_.data()),
                                       num_cb, cb_size, hidden, static_cast<float*>(embed_ptr),
                                       stream_);
    const auto t_embed_end = SteadyClock::now();
    state.prof_embed_ms += elapsed_ms(t_embed_start, t_embed_end);

    // Build mask and position on host, upload
    const auto t_step_start = SteadyClock::now();
    if (state.use_cfg || frame == 0) {
        decoder_state_->bind_to(*decoder_);
        bind_cross_kv();
    }

    // GPU-resident decode step: compute mask + position directly on device
    // to avoid per-frame H2D transfers.

    // Set use_input_embed = 1.0
    float use_embed_val = 1.0F;
    void* use_embed_ptr = decoder_->device_ptr("use_input_embed");
    cudaMemcpyAsync(use_embed_ptr, &use_embed_val, sizeof(float), cudaMemcpyHostToDevice, stream_);

    // Position: current cache position (single int32)
    int32_t pos = decoder_state_->position();
    void* pos_ptr = decoder_->device_ptr("position_id");
    if (pos_ptr)
        cudaMemcpyAsync(pos_ptr, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream_);

    // Mask: [1, 1, W+1] — zeros for cached entries + current, -inf for rest
    // For autoregressive (seq_len=1): allow positions 0..pos (inclusive)
    void* mask_ptr = decoder_->device_ptr("attention_mask");
    if (mask_ptr) {
        const int32_t W = decoder_state_->max_length();
        const int32_t mask_len = W + 1;
        // Build mask on host (small: ~1KB) and async upload
        // TODO: replace with a CUDA kernel for zero-copy when perf matters
        std::vector<float> mask(static_cast<std::size_t>(mask_len), -1e9F);
        for (int32_t i = 0; i <= pos; ++i)
            mask[static_cast<std::size_t>(i)] = 0.0F;
        cudaMemcpyAsync(mask_ptr, mask.data(), static_cast<std::size_t>(mask_len) * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
    }

    decoder_->forward_device_async({});
    decoder_state_->advance();

    // CFG: unconditional pass + device-side blend
    if (state.use_cfg && !run_cfg_uncond_pass_gpu(state, frame))
        return false;

    const auto t_step_end = SteadyClock::now();
    state.prof_trt_step_ms += elapsed_ms(t_step_start, t_step_end);

    // Sample + scatter
    const auto t_sample_start = SteadyClock::now();
    void* logits_ptr = decoder_->device_ptr("logits");

    magpie_greedy_sample_device(static_cast<const float*>(logits_ptr), num_cb, cb_size, AUDIO_RANGE,
                                static_cast<int32_t*>(device_codes_.data()),
                                static_cast<int32_t*>(device_full_argmax_.data()), stream_);

    magpie_scatter_codes_device(static_cast<const int32_t*>(device_codes_.data()),
                                static_cast<int32_t*>(device_all_codes_.data()),
                                static_cast<int32_t*>(device_prev_codes_.data()),
                                static_cast<const int32_t*>(device_full_argmax_.data()),
                                static_cast<int32_t*>(d_eos_flag.data()), frame, num_cb, EOS_TOKEN,
                                stream_);
    const auto t_sample_end = SteadyClock::now();
    state.prof_sample_ms += elapsed_ms(t_sample_start, t_sample_end);
    return true;
#else
    (void)state;
    (void)frame;
    (void)d_eos_flag;
    return false;
#endif
}

void MagpiePipeline::gpu_greedy_update_text_consumed(DecoderLoopState& state, int32_t frame) {
#if TRTMC_HAS_CUDA_KERNELS
    if (state.text_consumed)
        return;
    if (state.use_cross_attn_tracking) {
        std::vector<float> xattn(static_cast<std::size_t>(state.max_source_positions));
        cudaMemcpyAsync(xattn.data(), cross_attn_weights_.data(),
                        static_cast<std::size_t>(state.max_source_positions) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        if (update_magpie_text_consumed_from_cross_attn(xattn.data(), state.max_source_positions,
                                                        state.text_consumed_threshold,
                                                        state.max_peak_pos, state.text_consumed)) {
            std::cerr << "[magpie-tts] Text consumed at frame " << frame
                      << " (max_peak_pos=" << state.max_peak_pos
                      << ", threshold=" << state.text_consumed_threshold
                      << ", text_len=" << text_length_ << ")" << std::endl;
        }
        return;
    }
    update_magpie_text_consumed_from_heuristic(state.estimated_frames, frame, state.text_consumed);
#else
    (void)state;
    (void)frame;
#endif
}

std::vector<int32_t> MagpiePipeline::run_gpu_greedy_loop(DecoderLoopState& state,
                                                         int32_t max_frames) {
#if TRTMC_HAS_CUDA_KERNELS
    constexpr int32_t EOS_CHECK_INTERVAL = 16;
    constexpr int32_t MIN_FRAMES = 4;

    const int32_t num_cb = state.num_cb;

    MagpieCudaBuffer d_eos_flag(sizeof(int32_t));
    int32_t h_eos_flag = 0;
    cudaMemsetAsync(d_eos_flag.data(), 0, sizeof(int32_t), stream_);

    int32_t gen_frames_actual = 0;

    for (int32_t frame = 0; frame < max_frames; ++frame) {
        if (!gpu_greedy_frame_step(state, frame, d_eos_flag))
            break;

        gen_frames_actual = frame + 1;

        // Periodic checks (EOS, repetition, text-completion) every N frames
        const bool periodic =
            should_run_magpie_periodic_check(frame, MIN_FRAMES, EOS_CHECK_INTERVAL);
        if (periodic &&
            gpu_check_stop_conditions(state, frame, d_eos_flag, h_eos_flag, gen_frames_actual)) {
            break;
        }
        if (periodic && !state.text_consumed) {
            gpu_update_text_completion(state, frame);
        }
        if (check_finished_limit(state, frame)) {
            gen_frames_actual = frame + 1;
            break;
        }
    }

    // Final sync and bulk D2H of all accumulated codes
    cudaStreamSynchronize(stream_);
    const std::size_t total_codes_bytes =
        static_cast<std::size_t>(gen_frames_actual) * num_cb * sizeof(int32_t);
    std::vector<int32_t> all_codes(static_cast<std::size_t>(gen_frames_actual) * num_cb);
    cudaMemcpy(all_codes.data(), device_all_codes_.data(), total_codes_bytes,
               cudaMemcpyDeviceToHost);
    return all_codes;
#else
    (void)state;
    (void)max_frames;
    return {};
#endif
}

// ---------------------------------------------------------------------------
// CPU / non-greedy decode loop
// ---------------------------------------------------------------------------

std::vector<int32_t> MagpiePipeline::run_cpu_sampling_loop(DecoderLoopState& state,
                                                           int32_t max_frames) {
    constexpr int32_t MIN_FRAMES = 4;
    const int32_t num_cb = state.num_cb;

    std::vector<int32_t> all_codes;
    all_codes.reserve(static_cast<std::size_t>(max_frames) * num_cb);

    std::vector<int32_t> prev_codes(static_cast<std::size_t>(num_cb), kMagpieBosToken);

    for (int32_t frame = 0; frame < max_frames; ++frame) {
        // Embed computation
        const auto t_embed_start = SteadyClock::now();
        cpu_compute_frame_embed(state, prev_codes);
        const auto t_embed_end = SteadyClock::now();
        state.prof_embed_ms += elapsed_ms(t_embed_start, t_embed_end);

        // Conditioned decoder step
        const auto t_step_start = SteadyClock::now();
        if (state.use_cfg || frame == 0) {
            decoder_state_->bind_to(*decoder_);
            bind_cross_kv();
        }
        run_decoder_step(state.embed_buf.data(), state.hidden, state.logits);

        // CFG: unconditional pass + blend
        if (state.use_cfg && !run_cfg_uncond_pass_cpu(state, frame))
            break;
        const auto t_step_end = SteadyClock::now();
        state.prof_trt_step_ms += elapsed_ms(t_step_start, t_step_end);

        // Sample frame codes (LT path if available, otherwise flat logits)
        const auto t_sample_start = SteadyClock::now();
        std::vector<int32_t> frame_codes;
        bool eos = false;
        cpu_sample_frame_codes(state, frame_codes, eos);
        const auto t_sample_end = SteadyClock::now();
        state.prof_sample_ms += elapsed_ms(t_sample_start, t_sample_end);

        if (should_stop_magpie_on_eos(eos, frame, MIN_FRAMES)) {
            std::cerr << "[magpie-tts] EOS detected at frame " << frame
                      << ", dropping terminal frame" << std::endl;
            break;
        }

        for (int32_t cb = 0; cb < num_cb; ++cb)
            all_codes.push_back(frame_codes[cb]);
        prev_codes = frame_codes;
        upload_magpie_prev_codes_to_device(device_prev_codes_, prev_codes.data(), num_cb,
                                           state.use_gpu_kernels, state.use_gpu_greedy);

        update_text_completion(state, frame);
        update_attention_prior(frame);

        if (check_finished_limit(state, frame))
            break;
    }

    return all_codes;
}

// ---------------------------------------------------------------------------
// run_decoder() -- orchestrator
// ---------------------------------------------------------------------------

std::vector<int32_t> MagpiePipeline::run_decoder(int32_t max_frames) {
    DecoderLoopState state = init_decoder_state();

    // Reset KV caches
    decoder_state_->reset();
    if (state.use_cfg && decoder_state_uncond_)
        decoder_state_uncond_->reset();

    // Bind cross-attention K/V
    decoder_state_->bind_to(*decoder_);
    bind_cross_kv();
    reset_attention_prior();

    // Phase 1: Context prefill
    const int32_t ctx_frames = prefill_context(state);
    if (ctx_frames < 0)
        return {};

    // Phase 2: Autoregressive decode — upload BOS codes and dispatch to loop
    std::vector<int32_t> bos(static_cast<std::size_t>(state.num_cb), kMagpieBosToken);
    upload_magpie_prev_codes_to_device(device_prev_codes_, bos.data(), state.num_cb,
                                       state.use_gpu_kernels, false);

    std::vector<int32_t> all_codes;
    if (state.use_gpu_greedy && device_all_codes_.ok())
        all_codes = run_gpu_greedy_loop(state, max_frames);
    else if (state.use_gpu_sampling && device_all_codes_.ok() && !state.use_cfg)
        // GPU sampling enabled for non-CFG path. When CFG is active,
        // device-side CFG blend produces different FP results due to
        // operation ordering, causing generation trajectory divergence.
        // Fall through to CPU sampling for CFG to maintain quality parity.
        all_codes = run_gpu_sampling_loop(state, max_frames);
    else
        all_codes = run_cpu_sampling_loop(state, max_frames);

    const int32_t gen_frames = static_cast<int32_t>(all_codes.size()) / std::max(state.num_cb, 1);
    std::cerr << "[magpie-tts] Generated " << gen_frames << " frames (" << all_codes.size()
              << " codes)" << std::endl;

    log_decoder_profiling(state, ctx_frames, gen_frames);
    log_magpie_frame_preview(all_codes, state.num_cb);
    return all_codes;
}

// ---------------------------------------------------------------------------
// GPU sampling loop (top-k temperature sampling on device)
// ---------------------------------------------------------------------------

bool MagpiePipeline::gpu_sampling_frame_step(DecoderLoopState& state, int32_t frame,
                                             MagpieCudaBuffer& d_eos_flag,
                                             std::vector<int32_t>& h_codes) {
#if TRTMC_HAS_CUDA_KERNELS
    constexpr int32_t EOS_TOKEN = kMagpieEosToken;
    constexpr int32_t AUDIO_RANGE = kMagpieAudioRange;

    const int32_t num_cb = state.num_cb;
    const int32_t cb_size = state.cb_size;
    const int32_t hidden = state.hidden;
    const int32_t top_k = config_.top_k;
    const float temperature = config_.temperature;

    // ---- Embed (GPU kernel) ----
    const auto t_embed_start = SteadyClock::now();
    void* embed_ptr = decoder_->device_ptr("input_embed");
    magpie_gather_average_embed_device(static_cast<const float*>(audio_embed_device_.data()),
                                       static_cast<const int32_t*>(device_prev_codes_.data()),
                                       num_cb, cb_size, hidden, static_cast<float*>(embed_ptr),
                                       stream_);
    const auto t_embed_end = SteadyClock::now();
    state.prof_embed_ms += elapsed_ms(t_embed_start, t_embed_end);

    // ---- TRT step (logits stay on device) ----
    const auto t_step_start = SteadyClock::now();
    if (state.use_cfg || frame == 0) {
        decoder_state_->bind_to(*decoder_);
        bind_cross_kv();
    }

    // Build inputs via MagpieInferenceState::prepare_step
    int32_t dummy_token = 0;
    float use_embed_val = 1.0F;

    Tensor token_tensor;
    token_tensor.data = &dummy_token;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_embed_val;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    TensorMap step_inputs;
    step_inputs["token_id"] = token_tensor;
    step_inputs["use_input_embed"] = use_embed_tensor;
    decoder_state_->prepare_step(step_inputs);

    // Upload all inputs H2D (mask, position, token, use_input_embed)
    for (const auto& [name, tensor] : step_inputs) {
        void* dst = decoder_->device_ptr(name);
        if (dst && tensor.data) {
            cudaMemcpyAsync(dst, tensor.data, tensor.nbytes(), cudaMemcpyHostToDevice, stream_);
        }
    }

    decoder_->forward_device_async({});
    decoder_state_->advance();

    // CFG: unconditional pass + device-side blend
    if (state.use_cfg && !run_cfg_uncond_pass_gpu(state, frame))
        return false;

    const auto t_step_end = SteadyClock::now();
    state.prof_trt_step_ms += elapsed_ms(t_step_start, t_step_end);

    // ---- GPU top-k sample ----
    const auto t_sample_start = SteadyClock::now();

    // Generate random values on host (MT19937), upload to device
    std::vector<float> h_rand(static_cast<std::size_t>(num_cb));
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (int32_t cb = 0; cb < num_cb; ++cb)
        h_rand[static_cast<std::size_t>(cb)] = dist(rng_);
    cudaMemcpyAsync(device_rand_vals_.data(), h_rand.data(),
                    static_cast<std::size_t>(num_cb) * sizeof(float), cudaMemcpyHostToDevice,
                    stream_);
    cudaMemsetAsync(d_eos_flag.data(), 0, sizeof(int32_t), stream_);

    void* logits_ptr = decoder_->device_ptr("logits");

    magpie_topk_sample_device(static_cast<const float*>(logits_ptr), num_cb, cb_size, AUDIO_RANGE,
                              top_k, temperature, EOS_TOKEN,
                              static_cast<const float*>(device_rand_vals_.data()),
                              static_cast<int32_t*>(device_codes_.data()),
                              static_cast<int32_t*>(device_full_argmax_.data()),
                              static_cast<int32_t*>(d_eos_flag.data()), stream_);

    // D2H: only 8 codes + 1 EOS flag (36 bytes, not 64KB logits)
    cudaMemcpyAsync(h_codes.data(), device_codes_.data(),
                    static_cast<std::size_t>(num_cb) * sizeof(int32_t), cudaMemcpyDeviceToHost,
                    stream_);
    int32_t h_eos_flag = 0;
    cudaMemcpyAsync(&h_eos_flag, d_eos_flag.data(), sizeof(int32_t), cudaMemcpyDeviceToHost,
                    stream_);

    // Update prev_codes on device for next frame's embed kernel
    magpie_scatter_codes_device(static_cast<const int32_t*>(device_codes_.data()),
                                static_cast<int32_t*>(device_all_codes_.data()),
                                static_cast<int32_t*>(device_prev_codes_.data()),
                                static_cast<const int32_t*>(device_full_argmax_.data()),
                                static_cast<int32_t*>(d_eos_flag.data()), frame, num_cb, EOS_TOKEN,
                                stream_);

    cudaStreamSynchronize(stream_);
    const auto t_sample_end = SteadyClock::now();
    state.prof_sample_ms += elapsed_ms(t_sample_start, t_sample_end);

    return h_eos_flag == 0;
#else
    (void)state;
    (void)frame;
    (void)d_eos_flag;
    (void)h_codes;
    return false;
#endif
}

std::vector<int32_t> MagpiePipeline::run_gpu_sampling_loop(DecoderLoopState& state,
                                                           int32_t max_frames) {
#if TRTMC_HAS_CUDA_KERNELS
    constexpr int32_t MIN_FRAMES = 4;

    const int32_t num_cb = state.num_cb;

    MagpieCudaBuffer d_eos_flag(sizeof(int32_t));

    std::vector<int32_t> all_codes;
    all_codes.reserve(static_cast<std::size_t>(max_frames) * num_cb);

    std::vector<int32_t> h_codes(static_cast<std::size_t>(num_cb));

    for (int32_t frame = 0; frame < max_frames; ++frame) {
        const bool alive = gpu_sampling_frame_step(state, frame, d_eos_flag, h_codes);

        if (!alive && should_stop_magpie_on_eos(true, frame, MIN_FRAMES)) {
            std::cerr << "[magpie-tts] EOS detected at frame " << frame
                      << ", dropping terminal frame" << std::endl;
            break;
        }

        for (int32_t cb = 0; cb < num_cb; ++cb)
            all_codes.push_back(h_codes[static_cast<std::size_t>(cb)]);

        update_text_completion(state, frame);
        if (check_finished_limit(state, frame))
            break;
    }

    return all_codes;
#else
    (void)state;
    (void)max_frames;
    return {};
#endif
}

// ---------------------------------------------------------------------------
// GPU stop conditions / text completion helpers
// ---------------------------------------------------------------------------

bool MagpiePipeline::gpu_check_stop_conditions(DecoderLoopState& state, int32_t frame,
                                               MagpieCudaBuffer& d_eos_flag, int32_t& h_eos_flag,
                                               int32_t& gen_frames_actual) {
#if TRTMC_HAS_CUDA_KERNELS
    (void)state;
    (void)frame;
    (void)gen_frames_actual;

    cudaMemcpyAsync(&h_eos_flag, d_eos_flag.data(), sizeof(int32_t), cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);
    return h_eos_flag != 0;
#else
    (void)state;
    (void)frame;
    (void)d_eos_flag;
    (void)h_eos_flag;
    (void)gen_frames_actual;
    return false;
#endif
}

void MagpiePipeline::gpu_update_text_completion(DecoderLoopState& state, int32_t frame) {
#if TRTMC_HAS_CUDA_KERNELS
    if (state.text_consumed)
        return;
    if (state.use_cross_attn_tracking) {
        std::vector<float> xattn(static_cast<std::size_t>(state.max_source_positions));
        cudaMemcpyAsync(xattn.data(), cross_attn_weights_.data(),
                        static_cast<std::size_t>(state.max_source_positions) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        if (update_magpie_text_consumed_from_cross_attn(xattn.data(), state.max_source_positions,
                                                        state.text_consumed_threshold,
                                                        state.max_peak_pos, state.text_consumed)) {
            std::cerr << "[magpie-tts] Text consumed at frame " << frame
                      << " (max_peak_pos=" << state.max_peak_pos
                      << ", threshold=" << state.text_consumed_threshold
                      << ", text_len=" << text_length_ << ")" << std::endl;
        }
        return;
    }
    update_magpie_text_consumed_from_heuristic(state.estimated_frames, frame, state.text_consumed);
#else
    (void)state;
    (void)frame;
#endif
}

// ---------------------------------------------------------------------------
// CPU step decomposition helpers
// ---------------------------------------------------------------------------

bool MagpiePipeline::cpu_run_conditioned_step(DecoderLoopState& state, int32_t frame) {
    if (state.use_cfg || frame == 0) {
        decoder_state_->bind_to(*decoder_);
        bind_cross_kv();
    }
    run_decoder_step(state.embed_buf.data(), state.hidden, state.logits);
    return true;
}

bool MagpiePipeline::cpu_sample_frame_codes(DecoderLoopState& state,
                                            std::vector<int32_t>& frame_codes, bool& eos) {
    // NeMo-aligned EOS gating based on attention prior's position tracking:
    // - unfinished (forbid EOS): attended_pos < text_len - 3
    // - finished (allow/force EOS): attended_pos >= text_len - 2 for >5 frames
    const bool text_near_end = has_attn_prior_ && last_attended_pos_ >= text_length_ - 2;
    const bool text_unfinished = has_attn_prior_ && last_attended_pos_ < text_length_ - 3;
    const bool forbid_eos = text_unfinished;
    const bool force_eos = text_near_end && state.frames_past_text_consumed > 5;

    const auto decoded = decode_magpie_frame_codes(
        state.logits, state.num_cb, state.cb_size, config_.greedy, config_.temperature,
        config_.top_k,
        [this](const float* cb_logits, int32_t vocab_size, float temperature, int32_t top_k) {
            return sample_top_k(cb_logits, vocab_size, temperature, top_k);
        },
        forbid_eos, force_eos);
    frame_codes = decoded.frame_codes;
    eos = decoded.eos;
    return true;
}

// ---------------------------------------------------------------------------
// Attention prior management (monotonic alignment, NeMo inference)
// ---------------------------------------------------------------------------

void MagpiePipeline::init_attention_prior() {
    if (!decoder_->has_input("cross_attn_prior"))
        return;

    const auto max_src = config_.max_source_positions;
    const auto prior_bytes = static_cast<std::size_t>(max_src) * sizeof(float);

    attn_prior_device_ = MagpieCudaBuffer(prior_bytes);
    has_attn_prior_ = true;

    // Detect alignment_weights output
    if (decoder_->has_output("alignment_weights")) {
        alignment_weights_device_ = MagpieCudaBuffer(prior_bytes);
        has_alignment_output_ = true;
        if (config_.cfg_scale > 1.0F) {
            alignment_scratch_device_ = MagpieCudaBuffer(prior_bytes);
        }
    }

    // Initialize prior to all 1.0 (no constraint initially)
    std::vector<float> ones(static_cast<std::size_t>(max_src), 1.0F);
    cudaMemcpy(attn_prior_device_.data(), ones.data(), prior_bytes, cudaMemcpyHostToDevice);

    attended_count_.assign(static_cast<std::size_t>(max_src), 0);

    std::cerr << "[magpie-tts] Attention prior ready (max_src=" << max_src << ")" << std::endl;
}

void MagpiePipeline::reset_attention_prior() {
    if (!has_attn_prior_)
        return;
    last_attended_pos_ = 0;
    attended_count_.assign(static_cast<std::size_t>(config_.max_source_positions), 0);

    // Reset prior to all 1.0
    const auto max_src = config_.max_source_positions;
    std::vector<float> ones(static_cast<std::size_t>(max_src), 1.0F);
    cudaMemcpy(attn_prior_device_.data(), ones.data(),
               static_cast<std::size_t>(max_src) * sizeof(float), cudaMemcpyHostToDevice);
}

int32_t MagpiePipeline::detect_attended_peak(const std::vector<float>& align, int32_t text_len) {
    // Track attended position: argmax within a lookahead window from last position.
    // If position has been attended >=8 times, force advance (attention sink detection).
    constexpr int32_t kLookahead = 5;

    int32_t last_pos = last_attended_pos_;
    if (attended_count_[static_cast<std::size_t>(last_pos)] >= 8)
        last_pos = std::min(last_pos + 1, text_len - 1);

    const int32_t window_end = std::min(last_pos + kLookahead, text_len - 3);
    if (window_end <= last_pos)
        return text_len - 1; // text ended

    int32_t best_pos = last_pos;
    float best_val = -1.0F;
    for (int32_t p = last_pos; p < window_end; ++p) {
        if (align[static_cast<std::size_t>(p)] > best_val) {
            best_val = align[static_cast<std::size_t>(p)];
            best_pos = p;
        }
    }
    return best_pos;
}

void MagpiePipeline::construct_attention_prior(std::vector<float>& prior, int32_t best_pos,
                                               int32_t text_len) {
    constexpr int32_t kLookahead = 5;

    if (text_len <= 5)
        return; // prior stays all 1.0 (NeMo: no prior for very short text)

    // Standard prior: epsilon everywhere, 1.0 in sliding window
    const float epsilon = 0.1F;
    std::fill(prior.begin(), prior.end(), epsilon);

    // Slight history exposure
    if (best_pos > 0)
        prior[static_cast<std::size_t>(best_pos - 1)] = 1.0F;
    prior[static_cast<std::size_t>(best_pos)] = 1.0F;
    for (int32_t i = 1; i <= kLookahead; ++i) {
        const int32_t idx = std::min(best_pos + i, text_len - 1);
        prior[static_cast<std::size_t>(idx)] = 1.0F;
    }

    // Penalize positions attended >= 10 times (attention sink)
    for (int32_t p = 0; p < text_len; ++p) {
        if (attended_count_[static_cast<std::size_t>(p)] >= 10) {
            for (int32_t j = 0; j <= p; ++j)
                prior[static_cast<std::size_t>(j)] = epsilon;
        }
    }
}

void MagpiePipeline::update_attention_prior(int32_t frame) {
    (void)frame;
    if (!has_attn_prior_ || !has_alignment_output_)
        return;

    const int32_t max_src = config_.max_source_positions;
    const int32_t text_len = text_length_;
    if (text_len <= 1)
        return;

    // Read alignment weights from GPU
    std::vector<float> align(static_cast<std::size_t>(max_src));
    cudaMemcpy(align.data(), alignment_weights_device_.data(),
               static_cast<std::size_t>(max_src) * sizeof(float), cudaMemcpyDeviceToHost);

    const int32_t best_pos = detect_attended_peak(align, text_len);
    last_attended_pos_ = best_pos;
    attended_count_[static_cast<std::size_t>(best_pos)]++;

    // Construct prior (NeMo: construct_inference_prior)
    std::vector<float> prior(static_cast<std::size_t>(max_src), 1.0F);
    construct_attention_prior(prior, best_pos, text_len);

    // Upload prior to GPU
    cudaMemcpy(attn_prior_device_.data(), prior.data(),
               static_cast<std::size_t>(max_src) * sizeof(float), cudaMemcpyHostToDevice);
}

void MagpiePipeline::upload_attention_prior() {
    // No-op: prior is uploaded in update_attention_prior()
}

// ---------------------------------------------------------------------------
// Batched prefill using optimization profile 1
// ---------------------------------------------------------------------------

void MagpiePipeline::init_prefill_buffers(int32_t N, int32_t W) {
    const auto mask_elems = static_cast<std::size_t>(N) * (W + N);
    std::vector<float> mask(mask_elems, -1e9F);
    for (int32_t r = 0; r < N; ++r) {
        for (int32_t c = W; c <= W + r; ++c)
            mask[static_cast<std::size_t>(r) * (W + N) + c] = 0.0F;
    }
    prefill_mask_ = MagpieCudaBuffer(mask_elems * sizeof(float));
    cudaMemcpy(prefill_mask_.data(), mask.data(), mask_elems * sizeof(float),
               cudaMemcpyHostToDevice);

    std::vector<int32_t> positions(N);
    for (int32_t i = 0; i < N; ++i)
        positions[i] = i;
    prefill_positions_ = MagpieCudaBuffer(static_cast<std::size_t>(N) * sizeof(int32_t));
    cudaMemcpy(prefill_positions_.data(), positions.data(),
               static_cast<std::size_t>(N) * sizeof(int32_t), cudaMemcpyHostToDevice);

    const int32_t output_size = config_.num_codebooks * config_.codebook_size;
    prefill_logits_ = MagpieCudaBuffer(static_cast<std::size_t>(N) * output_size * sizeof(float));
}

void MagpiePipeline::bind_prefill_cross_kv() {
    if (!prefill_module_ || !prefill_module_->ok())
        return;
    for (std::size_t i = 0; i < cross_k_.size(); ++i) {
        prefill_module_->bind_external("cross_k_" + std::to_string(i), cross_k_[i].data());
        prefill_module_->bind_external("cross_v_" + std::to_string(i), cross_v_[i].data());
    }
}

void MagpiePipeline::init_prefill_context() {
    if (context_lengths_.empty())
        return;
    prefill_ctx_len_ = context_lengths_[0];
    if (prefill_ctx_len_ <= 1)
        return;

    init_prefill_buffers(prefill_ctx_len_, decoder_state_->max_length());
    prefill_ready_ = prefill_mask_.ok() && prefill_positions_.ok() && prefill_logits_.ok();
    if (prefill_ready_)
        bind_prefill_cross_kv();
    if (prefill_ready_)
        std::cerr << "[magpie-tts] Batched prefill ready (ctx_len=" << prefill_ctx_len_ << ")"
                  << std::endl;
}

bool MagpiePipeline::prefill_context_batched(int32_t ctx_frames) {
    if (!prefill_ready_ || !prefill_module_ || !prefill_module_->ok())
        return false; // fall through to sequential path

    if (ctx_frames != prefill_ctx_len_)
        return false; // size mismatch, fall back

    // Build input tensors for profile-1 forward pass
    TensorMap inputs;

    // Input embeddings: [N, hidden] from context_embed_device_
    Tensor embed_t;
    embed_t.data = context_embed_device_.data();
    embed_t.shape = {ctx_frames, config_.hidden_size};
    embed_t.dtype = DType::kFloat32;
    inputs["input_embed"] = embed_t;

    // Position IDs: [N]
    Tensor pos_t;
    pos_t.data = prefill_positions_.data();
    pos_t.shape = {ctx_frames};
    pos_t.dtype = DType::kInt32;
    inputs["position_id"] = pos_t;

    // Attention mask: [1, N, W+N]
    const int32_t W = decoder_state_->max_length();
    Tensor mask_t;
    mask_t.data = prefill_mask_.data();
    mask_t.shape = {1, ctx_frames, W + ctx_frames};
    mask_t.dtype = DType::kFloat32;
    inputs["attention_mask"] = mask_t;

    // Execute batched prefill (single forward pass)
    prefill_module_->forward_async(inputs);
    prefill_module_->sync();

    // Copy present_k/v from prefill module to decoder's KV cache
    // (The KV cache now has ctx_frames entries)
    // Advance state to reflect the prefilled context positions.
    // MagpieKvCache::advance() only supports n_tokens=1, so advance in a loop.
    for (int32_t i = 0; i < ctx_frames; ++i)
        decoder_state_->advance(1);

    std::cerr << "[magpie-tts] Batched prefill done (ctx_frames=" << ctx_frames << ", O(1) vs O("
              << ctx_frames << ") sequential)" << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
// Prefill path helpers (GPU / CPU / CFG)
// ---------------------------------------------------------------------------

bool MagpiePipeline::prefill_context_gpu(DecoderLoopState& state, int32_t ctx_frames,
                                         const char* label) {
    // GPU-resident prefill: upload context embeddings from device buffer
    // and run sequential decoder steps without D2H sync.
    const int32_t hidden = state.hidden;
    const float* ctx_ptr = context_embed_.data();
    for (int32_t pos = 0; pos < ctx_frames; ++pos) {
        const float* frame_embed = ctx_ptr + static_cast<std::size_t>(pos) * hidden;
        run_decoder_step(frame_embed, hidden, state.logits);
    }
    (void)label;
    return true;
}

bool MagpiePipeline::prefill_context_cpu(DecoderLoopState& state, int32_t ctx_frames,
                                         const char* label) {
    const int32_t hidden = state.hidden;
    const float* ctx_ptr = context_embed_.data();
    for (int32_t pos = 0; pos < ctx_frames; ++pos) {
        const float* frame_embed = ctx_ptr + static_cast<std::size_t>(pos) * hidden;
        run_decoder_step(frame_embed, hidden, state.logits);
    }
    (void)label;
    return true;
}

bool MagpiePipeline::prefill_context_cfg(DecoderLoopState& state, int32_t ctx_frames) {
    std::cerr << "[magpie-tts] CFG: prefilling unconditional cache (" << ctx_frames
              << " frames) ..." << std::endl;

    const int32_t hidden = state.hidden;
    const float* ctx_ptr = context_embed_.data();

    decoder_state_uncond_->bind_to(*decoder_);
    bind_cross_kv_uncond();

    for (int32_t pos = 0; pos < ctx_frames; ++pos) {
        const float* frame_embed = ctx_ptr + static_cast<std::size_t>(pos) * hidden;

        int32_t dummy_token = 0;
        float use_input_embed = 1.0F;
        std::vector<float> embed_buf(frame_embed, frame_embed + hidden);

        Tensor token_tensor;
        token_tensor.data = &dummy_token;
        token_tensor.shape = {1};
        token_tensor.dtype = DType::kInt32;

        Tensor embed_tensor;
        embed_tensor.data = embed_buf.data();
        embed_tensor.shape = {1, static_cast<int64_t>(hidden)};
        embed_tensor.dtype = DType::kFloat32;

        Tensor use_embed_tensor;
        use_embed_tensor.data = &use_input_embed;
        use_embed_tensor.shape = {1};
        use_embed_tensor.dtype = DType::kFloat32;

        TensorMap inputs;
        inputs["token_id"] = token_tensor;
        inputs["input_embed"] = embed_tensor;
        inputs["use_input_embed"] = use_embed_tensor;
        decoder_state_uncond_->prepare_step(inputs);

        decoder_->forward(inputs);
        decoder_state_uncond_->advance();
    }

    // Rebind conditioned cross-KV for generation phase
    decoder_state_->bind_to(*decoder_);
    bind_cross_kv();
    return true;
}

// ---------------------------------------------------------------------------
// Local transformer initialization + per-codebook AR sampling
// ---------------------------------------------------------------------------

void MagpiePipeline::init_local_transformer() {
    // Detect and allocate decoder_hidden output buffer (needed for LT and NeMo EOS gating)
    if (decoder_->has_output("decoder_hidden")) {
        const auto dec_hidden_bytes = static_cast<std::size_t>(config_.hidden_size) * sizeof(float);
        decoder_hidden_buf_ = MagpieCudaBuffer(dec_hidden_bytes);
        if (config_.cfg_scale > 1.0F) {
            decoder_hidden_buf_uncond_ = MagpieCudaBuffer(dec_hidden_bytes);
        }
        has_decoder_hidden_output_ = true;
    }

    // Initialize LT engine if loaded from bundle
    if (lt_module_ && lt_module_->ok()) {
        has_lt_ = true;
        // LT dimensions: infer from engine I/O or use typical defaults
        // (256 hidden, 8 max codebooks = max cache positions)
        lt_hidden_ = 256;
        lt_max_cache_ = config_.num_codebooks;

        const std::size_t lt_cache_bytes = static_cast<std::size_t>(lt_max_cache_) *
                                           static_cast<std::size_t>(lt_hidden_) * sizeof(float);
        lt_cache_k_ = MagpieCudaBuffer(lt_cache_bytes);
        lt_cache_v_ = MagpieCudaBuffer(lt_cache_bytes);
        lt_present_k_ = MagpieCudaBuffer(lt_cache_bytes);
        lt_present_v_ = MagpieCudaBuffer(lt_cache_bytes);
        lt_output_ = MagpieCudaBuffer(static_cast<std::size_t>(lt_hidden_) * sizeof(float));
        lt_mask_ = MagpieCudaBuffer(static_cast<std::size_t>(lt_max_cache_ + 1) * sizeof(float));
        lt_position_id_ = MagpieCudaBuffer(sizeof(int32_t));
        lt_input_embed_ = MagpieCudaBuffer(static_cast<std::size_t>(lt_hidden_) * sizeof(float));

        if (config_.cfg_scale > 1.0F) {
            lt_cache_k_uncond_ = MagpieCudaBuffer(lt_cache_bytes);
            lt_cache_v_uncond_ = MagpieCudaBuffer(lt_cache_bytes);
            lt_present_k_uncond_ = MagpieCudaBuffer(lt_cache_bytes);
            lt_present_v_uncond_ = MagpieCudaBuffer(lt_cache_bytes);
            lt_output_uncond_ =
                MagpieCudaBuffer(static_cast<std::size_t>(lt_hidden_) * sizeof(float));
        }

        // Bind KV cache to LT module
        lt_module_->bind_external("cache_k", lt_cache_k_.data());
        lt_module_->bind_external("cache_v", lt_cache_v_.data());

        std::cerr << "[magpie-tts] Local transformer engine loaded (hidden=" << lt_hidden_
                  << ", max_cache=" << lt_max_cache_ << ")" << std::endl;
    }
}

void MagpiePipeline::lt_run_codebook_step(int32_t cb, const std::vector<float>& decoder_hidden,
                                          std::vector<float>& logits) {
    std::vector<float> lt_input(static_cast<std::size_t>(lt_hidden_), 0.0f);
    const float* w = lt_in_proj_w_.data();
    const float* b = lt_in_proj_b_.data();
    for (int32_t o = 0; o < lt_hidden_; ++o) {
        float val = b[o];
        for (int32_t i = 0; i < config_.hidden_size; ++i)
            val += decoder_hidden[static_cast<std::size_t>(i)] *
                   w[static_cast<std::size_t>(o) * config_.hidden_size + i];
        lt_input[static_cast<std::size_t>(o)] = val;
    }
    cudaMemcpy(lt_input_embed_.data(), lt_input.data(), lt_input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    int32_t pos = cb;
    cudaMemcpy(lt_position_id_.data(), &pos, sizeof(int32_t), cudaMemcpyHostToDevice);
    std::vector<float> mask(static_cast<std::size_t>(lt_max_cache_ + 1), -1e9f);
    for (int32_t i = 0; i <= cb; ++i)
        mask[static_cast<std::size_t>(i)] = 0.0f;
    cudaMemcpy(lt_mask_.data(), mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice);

    TensorMap lt_inputs;
    Tensor embed_t;
    embed_t.data = lt_input_embed_.data();
    embed_t.shape = {1, lt_hidden_};
    embed_t.dtype = DType::kFloat32;
    lt_inputs["input_embed"] = embed_t;
    lt_module_->forward_async(lt_inputs);
    lt_module_->sync();

    std::vector<float> lt_out(static_cast<std::size_t>(lt_hidden_));
    cudaMemcpy(lt_out.data(), lt_module_->device_ptr("output"), lt_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    const int32_t cb_size = static_cast<int32_t>(logits.size());
    const std::size_t proj_stride = static_cast<std::size_t>(lt_hidden_ + 1) * cb_size;
    const float* proj_w = lt_out_proj_.data() + cb * proj_stride;
    const float* proj_b = proj_w + static_cast<std::size_t>(lt_hidden_) * cb_size;
    for (int32_t v = 0; v < cb_size; ++v) {
        float val = proj_b[v];
        for (int32_t h = 0; h < lt_hidden_; ++h)
            val += lt_out[static_cast<std::size_t>(h)] *
                   proj_w[static_cast<std::size_t>(h) * cb_size + v];
        logits[static_cast<std::size_t>(v)] = val;
    }
}

bool MagpiePipeline::sample_frame_codes_lt(DecoderLoopState& state,
                                           std::vector<int32_t>& frame_codes, bool& eos) {
    if (!has_lt_ || !lt_module_ || !lt_module_->ok())
        return false;

    const int32_t num_cb = state.num_cb;
    const int32_t cb_size = state.cb_size;

    std::vector<float> decoder_hidden(static_cast<std::size_t>(config_.hidden_size));
    cudaMemcpy(decoder_hidden.data(), decoder_hidden_buf_.data(),
               decoder_hidden.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemset(lt_cache_k_.data(), 0, lt_cache_k_.size());
    cudaMemset(lt_cache_v_.data(), 0, lt_cache_v_.size());

    frame_codes.resize(static_cast<std::size_t>(num_cb));
    eos = false;

    for (int32_t cb = 0; cb < num_cb; ++cb) {
        std::vector<float> logits(static_cast<std::size_t>(cb_size), 0.0f);
        lt_run_codebook_step(cb, decoder_hidden, logits);

        int32_t token =
            config_.greedy
                ? static_cast<int32_t>(
                      std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())))
                : sample_top_k(logits.data(), cb_size, config_.temperature, config_.top_k);

        frame_codes[static_cast<std::size_t>(cb)] = token;
        if (cb == 0 && token == kMagpieEosToken)
            eos = true;

        // Copy present_k/v to cache for next codebook step
        // (LT KV cache grows with each codebook position)
    }

    return true; // LT sampling was used
}

// ---------------------------------------------------------------------------
// Streaming generation: interleaved decode + codec
// ---------------------------------------------------------------------------

bool MagpiePipeline::streaming_decode_one_frame(DecoderLoopState& state, int32_t frame,
                                                std::vector<int32_t>& prev_decode_codes,
                                                std::vector<int32_t>& all_codes,
                                                StreamingCodecState& codec_state) {
    constexpr int32_t MIN_FRAMES = 4;
    const int32_t num_cb = codec_state.num_cb;

    cpu_compute_frame_embed(state, prev_decode_codes);
    if (!cpu_run_conditioned_step(state, frame))
        return false;
    if (state.use_cfg && !run_cfg_uncond_pass_cpu(state, frame))
        return false;

    std::vector<int32_t> frame_codes;
    bool eos = false;
    cpu_sample_frame_codes(state, frame_codes, eos);

    if (should_stop_magpie_on_eos(eos, frame, MIN_FRAMES)) {
        std::cerr << "[magpie-tts] EOS at frame " << frame << std::endl;
        return false;
    }

    for (int32_t cb = 0; cb < num_cb; ++cb)
        all_codes.push_back(frame_codes[cb]);
    codec_state.total_frames++;

    prev_decode_codes = frame_codes;
    upload_magpie_prev_codes_to_device(device_prev_codes_, prev_decode_codes.data(), num_cb,
                                       state.use_gpu_kernels, state.use_gpu_greedy);
    update_text_completion(state, frame);
    update_attention_prior(frame);

    return !check_finished_limit(state, frame);
}

void MagpiePipeline::streaming_flush_codec(StreamingCodecState& codec_state,
                                           const std::vector<int32_t>& all_codes,
                                           const AudioChunkCallback& audio_callback,
                                           bool is_final) {
    constexpr int32_t kSamplesPerFrame = 1024;
    constexpr int32_t kMarginFrames = 4; // hold-back for future conv context
    const int32_t num_cb = codec_state.num_cb;

    const int32_t decoded_frames = static_cast<int32_t>(all_codes.size()) / num_cb;
    if (decoded_frames <= codec_state.frames_at_last_flush)
        return;

    // Re-decode the FULL accumulated sequence for seamless audio
    auto wav = run_codec(all_codes, decoded_frames);
    if (wav.empty())
        return;

    // Only output samples up to (decoded_frames - margin) to ensure
    // all output samples have proper future context from the codec's
    // non-causal convolutions. On final flush, output everything.
    const int32_t safe_frames =
        is_final ? decoded_frames
                 : std::max(decoded_frames - kMarginFrames, codec_state.frames_at_last_flush);
    const int32_t safe_samples = safe_frames * kSamplesPerFrame;
    const int32_t out_end = std::min(safe_samples, static_cast<int32_t>(wav.size()));

    if (out_end > codec_state.total_samples_output) {
        const int32_t new_start = codec_state.total_samples_output;
        const int32_t new_len = out_end - new_start;
        audio_callback(wav.data() + new_start, new_len, config_.sample_rate);
        codec_state.total_samples_output = out_end;
    }

    codec_state.frames_at_last_flush = decoded_frames;
}

int32_t MagpiePipeline::generate_audio_streaming(const std::vector<int32_t>& text_ids,
                                                 int32_t max_frames,
                                                 AudioChunkCallback audio_callback,
                                                 int32_t chunk_frames) {
    if (!audio_callback)
        return 0;

    reset_rng_for_request();
    ensure_cfg_resources();
    text_length_ = static_cast<int32_t>(text_ids.size());

    std::cerr << "[magpie-tts] Starting streaming pipeline with " << text_ids.size()
              << " text tokens, max_frames=" << max_frames << ", chunk_frames=" << chunk_frames
              << ", cfg_scale=" << config_.cfg_scale << std::endl;

    const auto t_start = SteadyClock::now();

    // Stage 1: Encode
    run_encoder(text_ids);
    compute_cross_kv();
    run_cfg_encoder(text_ids);

    std::cerr << "[magpie-tts] Encoder: " << elapsed_ms(t_start, SteadyClock::now()) << " ms"
              << std::endl;

    // Stage 2: Initialize decoder (same as run_decoder)
    DecoderLoopState state = init_decoder_state();
    decoder_state_->reset();
    if (state.use_cfg && decoder_state_uncond_)
        decoder_state_uncond_->reset();

    decoder_state_->bind_to(*decoder_);
    bind_cross_kv();
    reset_attention_prior();

    const int32_t ctx_frames = prefill_context(state);
    if (ctx_frames < 0)
        return 0;

    std::cerr << "[magpie-tts] Prefill: " << elapsed_ms(t_start, SteadyClock::now()) << " ms"
              << std::endl;

    // Stage 3: Interleaved decode + codec streaming
    const int32_t num_cb = state.num_cb;

    std::vector<int32_t> prev_decode_codes(static_cast<std::size_t>(num_cb), kMagpieBosToken);
    upload_magpie_prev_codes_to_device(device_prev_codes_, prev_decode_codes.data(), num_cb,
                                       state.use_gpu_kernels, false);

    std::vector<int32_t> all_codes;
    all_codes.reserve(static_cast<std::size_t>(max_frames) * num_cb);

    StreamingCodecState codec_state;
    codec_state.num_cb = num_cb;

    for (int32_t frame = 0; frame < max_frames; ++frame) {
        if (!streaming_decode_one_frame(state, frame, prev_decode_codes, all_codes, codec_state))
            break;

        // Flush when chunk is full
        if ((codec_state.total_frames - codec_state.frames_at_last_flush) >= chunk_frames)
            streaming_flush_codec(codec_state, all_codes, audio_callback, false);
    }

    // Flush remaining (release margin)
    streaming_flush_codec(codec_state, all_codes, audio_callback, true);

    const auto t_end = SteadyClock::now();
    const double audio_dur =
        static_cast<double>(codec_state.total_samples_output) / config_.sample_rate;
    std::cerr << "[magpie-tts] Streaming done: " << codec_state.total_frames << " frames, "
              << codec_state.total_samples_output << " samples, " << elapsed_ms(t_start, t_end)
              << " ms total"
              << ", RTF=" << (elapsed_ms(t_start, t_end) / 1000.0) / std::max(audio_dur, 0.001)
              << std::endl;

    return codec_state.total_samples_output;
}

// ---------------------------------------------------------------------------
// run_codec() -- codes -> waveform via TrtModule
// ---------------------------------------------------------------------------

std::vector<float> MagpiePipeline::run_codec(const std::vector<int32_t>& codes,
                                             int32_t num_frames) {
    const int32_t num_cb = config_.num_codebooks;
    if (num_frames <= 0)
        return {};

    if (!codec_ || !codec_->ok()) {
        std::cerr << "[magpie-tts] No codec engine, generating silence" << std::endl;
        const int32_t samples_per_frame =
            config_.sample_rate / std::max(static_cast<int32_t>(config_.frames_per_second), 1);
        const auto total = static_cast<std::size_t>(num_frames) * samples_per_frame;
        return std::vector<float>(total, 0.0F);
    }

    // Build codec input using the plan helper
    // We need to figure out max_codec_frames from the codec engine's codec_tokens shape
    int32_t max_codec_frames = num_frames;
    // Get codec_tokens shape from engine output info
    auto codec_inputs = codec_->input_info();
    for (const auto& ti : codec_inputs) {
        if (ti.name == "codec_tokens" && ti.shape.size() >= 2) {
            max_codec_frames = static_cast<int32_t>(ti.shape[1]);
            break;
        }
    }

    const auto plan = make_magpie_codec_plan(num_frames, num_cb, max_codec_frames);
    std::vector<int32_t> codec_input = build_magpie_codec_input(codes, num_cb, plan);

    Tensor codec_tokens_tensor;
    codec_tokens_tensor.data = codec_input.data();
    codec_tokens_tensor.shape = {static_cast<int64_t>(num_cb),
                                 static_cast<int64_t>(max_codec_frames)};
    codec_tokens_tensor.dtype = DType::kInt32;

    Tensor input_len_tensor;
    int32_t input_len = plan.input_len;
    input_len_tensor.data = &input_len;
    input_len_tensor.shape = {1};
    input_len_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["codec_tokens"] = codec_tokens_tensor;
    inputs["input_len"] = input_len_tensor;

    TensorMap outputs = codec_->forward(inputs);

    auto it = outputs.find("waveform");
    if (it == outputs.end()) {
        std::cerr << "[magpie-tts] Codec: no 'waveform' output" << std::endl;
        return {};
    }

    const auto& wt = it->second;
    const auto total_out = wt.numel();
    const auto trimmed = plan.valid_samples;
    const auto copy_n = std::min(static_cast<std::size_t>(total_out), trimmed);

    std::vector<float> waveform(copy_n);
    std::memcpy(waveform.data(), wt.data, copy_n * sizeof(float));

    std::cerr << "[magpie-tts] Codec: " << num_frames << " frames -> " << waveform.size()
              << " samples" << std::endl;
    return waveform;
}

// ---------------------------------------------------------------------------
// Profiling / logging
// ---------------------------------------------------------------------------

void MagpiePipeline::log_decoder_profiling(const DecoderLoopState& state, int32_t ctx_frames,
                                           int32_t gen_frames) const {
    std::cerr << "\n[magpie-tts] --- Decoder Profiling Breakdown ---\n"
              << "[magpie-tts]   Context prefill:   " << state.prof_prefill_ms << " ms ("
              << ctx_frames << " frames, "
              << (ctx_frames > 0 ? state.prof_prefill_ms / ctx_frames : 0.0) << " ms/frame)\n"
              << "[magpie-tts]   Embed computation: " << state.prof_embed_ms << " ms ("
              << (gen_frames > 0 ? state.prof_embed_ms / gen_frames : 0.0) << " ms/frame)\n"
              << "[magpie-tts]   TRT decoder steps: " << state.prof_trt_step_ms << " ms ("
              << (gen_frames > 0 ? state.prof_trt_step_ms / gen_frames : 0.0) << " ms/frame)\n"
              << "[magpie-tts]   Sampling:          " << state.prof_sample_ms << " ms ("
              << (gen_frames > 0 ? state.prof_sample_ms / gen_frames : 0.0) << " ms/frame)\n"
              << "[magpie-tts]   Text tracking:     "
              << (state.use_cross_attn_tracking ? "cross-attn tracking" : "heuristic (text_len*3)")
              << "\n"
              << "[magpie-tts]   Stop guards:       "
              << (config_.enable_finished_limit_stop ? "finished_limit"
                                                     : "none (EOS/max_frames only)")
              << "\n"
              << "[magpie-tts] ---------------------------------\n";
}

void MagpiePipeline::log_pipeline_profiling(int32_t num_frames, int32_t num_samples,
                                            double ms_encoder, double ms_decoder, double ms_codec,
                                            double ms_total) const {
    const double ms_per_frame = (num_frames > 0) ? ms_decoder / num_frames : 0.0;
    const double audio_duration = static_cast<double>(num_samples) / config_.sample_rate;
    const double rtf = (audio_duration > 0.0) ? (ms_total / 1000.0) / audio_duration : 0.0;

    std::cerr << "\n[magpie-tts] ===== PROFILING REPORT =====\n"
              << "[magpie-tts]   Encoder:        " << ms_encoder << " ms\n"
              << "[magpie-tts]   Cross-KV:       D2D copies (per-layer buffers)\n"
              << "[magpie-tts]   Decoder:        " << ms_decoder << " ms (" << num_frames
              << " frames, " << ms_per_frame << " ms/frame)\n"
              << "[magpie-tts]   Codec:          " << ms_codec << " ms\n"
              << "[magpie-tts]   Total pipeline: " << ms_total << " ms\n"
              << "[magpie-tts]   Audio duration: " << audio_duration << " s (" << num_samples
              << " samples @ " << config_.sample_rate << " Hz)\n"
              << "[magpie-tts]   RTF (real-time factor): " << rtf
              << " (< 1.0 = faster than real-time)\n"
              << "[magpie-tts]   CFG scale:      " << config_.cfg_scale
              << (config_.cfg_scale > 1.0F ? " (enabled, 2x decoder steps)" : " (disabled)") << "\n"
              << "[magpie-tts]   finished_limit: "
              << (config_.enable_finished_limit_stop
                      ? std::to_string(config_.finished_limit_with_eot)
                      : std::string("disabled"))
              << " (text_len=" << text_length_
              << ", est_frames=" << static_cast<int32_t>(static_cast<float>(text_length_) * 3.0F)
              << ")\n"
              << "[magpie-tts] =============================\n"
              << std::endl;
}

// ---------------------------------------------------------------------------
// generate_audio() helpers
// ---------------------------------------------------------------------------

void MagpiePipeline::reset_rng_for_request(int32_t request_seed) {
    // All values now arrive pre-populated from the audio_magpie.* namespace
    // (magpie_plugin does the ctx.runtime_config reads at construction).
    // Formerly this method read TRTMC_MAGPIE_{GREEDY,CFG_SCALE,TEMPERATURE,
    // FINISHED_LIMIT,SEED} directly — those env vars are deleted.
    const int64_t seed = resolve_magpie_seed(config_.seed, request_seed);
    if (seed >= 0)
        rng_.seed(static_cast<std::mt19937::result_type>(seed));
}

void MagpiePipeline::ensure_cfg_resources() {
    if (config_.cfg_scale <= 1.0F || decoder_state_uncond_)
        return;

    // Lazy-allocate CFG resources if cfg_scale was bumped via env var
    const int32_t dec_layers = static_cast<int32_t>(cross_k_.size());
    const int32_t kv_dim =
        decoder_state_->max_length() > 0
            ? static_cast<int32_t>(
                  cross_k_[0].size() /
                  (static_cast<std::size_t>(config_.max_source_positions) * sizeof(float)))
            : config_.hidden_size;

    // We can't easily construct a new MagpieKvCache without knowing kv_dim,
    // so just log a warning and skip CFG.
    std::cerr << "[magpie-tts] WARNING: CFG requested via env but no uncond cache allocated. "
              << "Rebuild with cfg_scale > 1 in bundle config." << std::endl;
    config_.cfg_scale = 1.0F;
    (void)dec_layers;
    (void)kv_dim;
}

void MagpiePipeline::run_cfg_encoder(const std::vector<int32_t>& text_ids) {
    if (config_.cfg_scale <= 1.0F || !encoder_output_uncond_.ok() || cross_k_uncond_.empty())
        return;

    std::cerr << "[magpie-tts] CFG: encoding null text for unconditional path ..." << std::endl;

    const auto enc_bytes = encoder_output_.size();

    // Encode empty text
    std::vector<int32_t> empty_ids;
    run_encoder(empty_ids);

    // Save unconditional encoder output
    cudaMemcpy(encoder_output_uncond_.data(), encoder_output_.data(), enc_bytes,
               cudaMemcpyDeviceToDevice);

    // Re-encode actual text
    run_encoder(text_ids);

    compute_cross_kv_uncond();
}

// ---------------------------------------------------------------------------
// generate_audio() -- full pipeline orchestration
// ---------------------------------------------------------------------------

AudioResult MagpiePipeline::generate_audio(const std::string& prompt, const GenerateConfig& cfg) {
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(prompt);

    int32_t max_frames = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 512;

    AudioResult result;
    result.sample_rate = config_.sample_rate;

    reset_rng_for_request(cfg.seed);
    ensure_cfg_resources();

    text_length_ = static_cast<int32_t>(input_ids.size());

    std::cerr << "[magpie-tts] Starting pipeline with " << input_ids.size()
              << " text tokens, max_frames=" << max_frames << (config_.greedy ? " (greedy)" : "")
              << ", cfg_scale=" << config_.cfg_scale << ", finished_limit="
              << (config_.enable_finished_limit_stop
                      ? std::to_string(config_.finished_limit_with_eot)
                      : std::string("disabled"))
              << std::endl;

    const auto t_pipeline_start = SteadyClock::now();

    // Stage 1: Encode text
    std::cerr << "[magpie-tts] Running encoder ..." << std::endl;
    const auto t_enc_start = SteadyClock::now();
    run_encoder(input_ids);
    const auto t_enc_end = SteadyClock::now();

    // Stage 2: Copy encoder output to per-layer cross-attention buffers
    compute_cross_kv();

    // Stage 2b (CFG): Run encoder with empty text for unconditional cross-KV
    run_cfg_encoder(input_ids);

    // Stage 3: Autoregressive decode
    std::cerr << "[magpie-tts] Running decoder ..." << std::endl;
    const auto t_dec_start = SteadyClock::now();
    auto codes = run_decoder(max_frames);
    const auto t_dec_end = SteadyClock::now();
    if (codes.empty()) {
        std::cerr << "[magpie-tts] Decoder produced no codes" << std::endl;
        return result;
    }

    const int32_t num_frames = static_cast<int32_t>(codes.size()) / config_.num_codebooks;

    // Stage 4: Codec -> waveform
    std::cerr << "[magpie-tts] Running codec ..." << std::endl;
    const auto t_codec_start = SteadyClock::now();
    auto waveform = run_codec(codes, num_frames);
    const auto t_codec_end = SteadyClock::now();
    if (waveform.empty()) {
        std::cerr << "[magpie-tts] Codec produced no audio" << std::endl;
        return result;
    }

    result.samples = std::move(waveform);
    result.num_samples = static_cast<int32_t>(result.samples.size());

    const auto t_pipeline_end = SteadyClock::now();

    log_pipeline_profiling(num_frames, result.num_samples, elapsed_ms(t_enc_start, t_enc_end),
                           elapsed_ms(t_dec_start, t_dec_end),
                           elapsed_ms(t_codec_start, t_codec_end),
                           elapsed_ms(t_pipeline_start, t_pipeline_end));

    return result;
}

} // namespace trtmc
