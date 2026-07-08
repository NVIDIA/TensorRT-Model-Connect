/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/canary/pipeline.h"

#include "plugin_helpers.h"
#include "runtime/models/canary/canary_cross_kv_apply.h"
#include "runtime/models/canary/canary_cross_kv_plan.h"
#include "runtime/models/canary/canary_decode_policy.h"
#include "runtime/models/canary/canary_host_plan.h"
#include "runtime/models/canary/canary_mel_spectrogram.h"
#include "runtime/models/canary/decode_runtime.h"
#include "trtmc/tokenizer.h"
#include "utils/wav_reader.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

using CanaryClock = std::chrono::steady_clock;

bool canary_stage_timing_enabled() {
    const char* value = std::getenv("TRTMC_CANARY_STAGE_TIMING");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

double elapsed_ms(CanaryClock::time_point start, CanaryClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// CanaryPipeline
// ═══════════════════════════════════════════════════════════════════════════

CanaryPipeline::CanaryPipeline(std::unique_ptr<TrtModule> encoder,
                               std::unique_ptr<TrtModule> decoder,
                               std::unique_ptr<CanaryInferenceState> state,
                               CanaryConfig canary_config, int32_t hidden_size,
                               int32_t num_decoder_layers, MelFilterbank mel_fb, int32_t mel_n_fft,
                               int32_t mel_hop_length, int32_t mel_chunk_length,
                               int32_t mel_sampling_rate, cudaStream_t stream,
                               std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str)
    : encoder_(std::move(encoder)), decoder_(std::move(decoder)), state_(std::move(state)),
      canary_config_(std::move(canary_config)), hidden_size_(hidden_size),
      num_decoder_layers_(num_decoder_layers),
      mel_fb_(std::make_unique<MelFilterbank>(std::move(mel_fb))), mel_n_fft_(mel_n_fft),
      mel_hop_length_(mel_hop_length), mel_chunk_length_(mel_chunk_length),
      mel_sampling_rate_(mel_sampling_rate), stream_(stream), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("CanaryPipeline: invalid encoder module");
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("CanaryPipeline: invalid decoder module");
    if (!state_ || !state_->ok())
        throw std::runtime_error("CanaryPipeline: invalid inference state");

    // Allocate cross-attention K/V device buffers
    cross_kv_bytes_ = static_cast<std::size_t>(canary_config_.max_source_positions) *
                      static_cast<std::size_t>(hidden_size_) * sizeof(float);

    cross_k_ptrs_.resize(static_cast<std::size_t>(num_decoder_layers_), nullptr);
    cross_v_ptrs_.resize(static_cast<std::size_t>(num_decoder_layers_), nullptr);
    for (int32_t i = 0; i < num_decoder_layers_; ++i) {
        cudaMalloc(&cross_k_ptrs_[static_cast<std::size_t>(i)], cross_kv_bytes_);
        cudaMalloc(&cross_v_ptrs_[static_cast<std::size_t>(i)], cross_kv_bytes_);
    }
}

CanaryPipeline::~CanaryPipeline() {
    for (auto* ptr : cross_k_ptrs_) {
        if (ptr)
            cudaFree(ptr);
    }
    for (auto* ptr : cross_v_ptrs_) {
        if (ptr)
            cudaFree(ptr);
    }
}

TextResult CanaryPipeline::transcribe(const float* audio_data, int32_t num_samples,
                                      int32_t max_new_tokens, int32_t input_sample_rate) {
    const bool report_stage_timing = canary_stage_timing_enabled();
    const auto transcribe_start = CanaryClock::now();

    // Step 0: Resample if needed
    const float* samples_ptr = audio_data;
    int32_t samples_count = num_samples;
    std::vector<float> resampled_buf;

    if (input_sample_rate > 0 && input_sample_rate != mel_sampling_rate_) {
        std::cerr << "[canary] Resampling audio from " << input_sample_rate << " Hz to "
                  << mel_sampling_rate_ << " Hz" << std::endl;
        resampled_buf =
            resample_linear(audio_data, num_samples, input_sample_rate, mel_sampling_rate_);
        samples_ptr = resampled_buf.data();
        samples_count = static_cast<int32_t>(resampled_buf.size());
    }
    const auto resample_end = CanaryClock::now();

    // Step 1: Extract mel spectrogram
    canary::MelResult mel;
    if (mel_fb_ && !mel_fb_->data.empty()) {
        mel =
            canary::extract_mel_spectrogram(samples_ptr, samples_count, mel_fb_->data.data(),
                                            mel_fb_->n_freq_bins, mel_fb_->n_mel_bins, mel_n_fft_,
                                            mel_hop_length_, mel_chunk_length_, mel_sampling_rate_);
    }
    const auto mel_end = CanaryClock::now();

    if (mel.data.empty()) {
        return TextResult{"[mel extraction failed]", {}};
    }

    // Step 2: Run encoder. The mel is chunk-padded, so mel.n_frames is the full
    // (padded) length; mel.valid_frames is the real audio length, used to mask
    // the padded tail in self-attention and to zero-pad the cross-attention K/V.
    const int32_t valid_mel_frames = mel.valid_frames > 0 ? mel.valid_frames : mel.n_frames;
    std::cerr << "[canary] Running encoder ..." << std::endl;
    run_encoder(mel.data.data(), mel.n_mels, mel.n_frames, valid_mel_frames);
    const auto encoder_end = CanaryClock::now();

    // Compute actual encoder sequence length for masking
    const int32_t mel_full = resolve_canary_expected_mel_length(canary_config_);
    int32_t actual_enc_seq_len = compute_canary_actual_encoder_length(
        valid_mel_frames, mel_full, canary_config_.max_source_positions);
    if (actual_enc_seq_len > 0) {
        std::cerr << "[canary] Actual encoder seq len: " << actual_enc_seq_len << " / "
                  << canary_config_.max_source_positions << std::endl;
    }

    // Step 3: Set up cross-attention K/V
    std::cerr << "[canary] Computing cross-attention K/V ..." << std::endl;
    setup_cross_attention(actual_enc_seq_len);
    const auto cross_kv_end = CanaryClock::now();

    // Step 4: Run decoder
    std::vector<int32_t> initial_tokens = make_canary_initial_decoder_tokens(canary_config_);
    std::cerr << "[canary] Running decoder ..." << std::endl;
    auto output_ids = run_decoder(initial_tokens, max_new_tokens);
    const auto decoder_end = CanaryClock::now();

    // Step 5: Decode token IDs
    TextResult out;
    out.token_ids = std::move(output_ids);
    if (tokenizer_ && !out.token_ids.empty()) {
        out.text = tokenizer_->decode(out.token_ids);
    }
    const auto tokenize_end = CanaryClock::now();

    if (report_stage_timing) {
        std::ostringstream timing;
        timing << "[trtmc.canary_timing.json] {\"resample_ms\":"
               << elapsed_ms(transcribe_start, resample_end) << ",\"mel_ms\":"
               << elapsed_ms(resample_end, mel_end) << ",\"encoder_ms\":"
               << elapsed_ms(mel_end, encoder_end) << ",\"cross_kv_ms\":"
               << elapsed_ms(encoder_end, cross_kv_end) << ",\"decoder_ms\":"
               << elapsed_ms(cross_kv_end, decoder_end) << ",\"tokenizer_ms\":"
               << elapsed_ms(decoder_end, tokenize_end) << ",\"total_ms\":"
               << elapsed_ms(transcribe_start, tokenize_end) << '}';
        std::cerr << timing.str() << std::endl;
    }
    return out;
}

void CanaryPipeline::run_encoder(const float* mel_data, int32_t mel_bins, int32_t mel_length,
                                 int32_t valid_mel_frames) {
    const int32_t expected_length = resolve_canary_expected_mel_length(canary_config_);
    const std::size_t mel_size =
        static_cast<std::size_t>(mel_bins) * static_cast<std::size_t>(expected_length);

    // Prepare mel input (pad if needed)
    std::vector<float> mel_host;
    if (mel_length == expected_length) {
        mel_host.assign(mel_data, mel_data + mel_size);
    } else {
        mel_host = build_canary_padded_mel_input(mel_data, mel_bins, mel_length, expected_length);
    }

    // Build input TensorMap
    TensorMap inputs;
    Tensor mel_tensor;
    mel_tensor.data = mel_host.data();
    mel_tensor.shape = {mel_bins, expected_length};
    mel_tensor.dtype = DType::kFloat32;
    inputs["mel_features"] = mel_tensor;

    // Optional encoder_mask input
    const int32_t enc_seq = canary_config_.max_source_positions;
    std::vector<float> enc_mask;
    if (encoder_->has_input("encoder_mask")) {
        int32_t actual_enc =
            compute_canary_actual_encoder_length(valid_mel_frames, expected_length, enc_seq);
        if (actual_enc <= 0)
            actual_enc = enc_seq;
        enc_mask = build_canary_encoder_mask_values(enc_seq, actual_enc);

        Tensor mask_tensor;
        mask_tensor.data = enc_mask.data();
        mask_tensor.shape = {static_cast<int64_t>(enc_mask.size())};
        mask_tensor.dtype = DType::kFloat32;
        inputs["encoder_mask"] = mask_tensor;
    }

    // Run encoder (we need the output to stay on device, so use forward_async + sync)
    encoder_->forward_async(inputs);
    encoder_->sync();
}

void CanaryPipeline::setup_cross_attention(int32_t actual_enc_seq_len) {
    // Get encoder output device pointer
    void* enc_output_device = encoder_->device_ptr("encoder_output");

    // Apply cross-KV plan: optionally zero-pad encoder output, then copy to each layer
    const auto plan = make_canary_cross_kv_plan(canary_config_.max_source_positions, hidden_size_,
                                                actual_enc_seq_len);

    std::string error;
    const bool ok = apply_canary_cross_kv_plan(
        plan, static_cast<std::size_t>(num_decoder_layers_),
        [enc_output_device](std::size_t valid_bytes, std::size_t pad_bytes) {
            return cudaMemset(static_cast<char*>(enc_output_device) + valid_bytes, 0, pad_bytes) ==
                   cudaSuccess;
        },
        [this, enc_output_device](std::size_t layer, CanaryCrossKvBufferKind kind,
                                  std::size_t bytes) {
            void* dst =
                kind == CanaryCrossKvBufferKind::K ? cross_k_ptrs_[layer] : cross_v_ptrs_[layer];
            return cudaMemcpy(dst, enc_output_device, bytes, cudaMemcpyDeviceToDevice) ==
                   cudaSuccess;
        },
        error);
    if (!ok) {
        throw std::runtime_error(error);
    }

    // Bind cross-K/V to decoder module
    for (int32_t i = 0; i < num_decoder_layers_; ++i) {
        const std::string suffix = "_" + std::to_string(i);
        decoder_->bind_external("cross_k" + suffix, cross_k_ptrs_[static_cast<std::size_t>(i)]);
        decoder_->bind_external("cross_v" + suffix, cross_v_ptrs_[static_cast<std::size_t>(i)]);
    }
}

std::vector<int32_t> CanaryPipeline::run_decoder(const std::vector<int32_t>& initial_tokens,
                                                 int32_t max_new_tokens) {
    state_->reset();
    state_->bind_to(*decoder_);

    const int32_t eot_id = canary_config_.eot_token_id;

    auto result = run_canary_decode_loop(
        initial_tokens, max_new_tokens, eot_id,
        [this](int32_t token, std::vector<float>& logits, std::string&) {
            run_decoder_step(token, logits);
            return true;
        },
        [](const std::vector<float>& logits) { return canary_select_argmax_token(logits); });

    if (result.prefill_failed) {
        std::cerr << "[canary] Prefill step failed: " << result.error << std::endl;
    } else if (result.decode_failed) {
        std::cerr << "[canary] Decode step failed: " << result.error << std::endl;
    }

    return result.output_ids;
}

void CanaryPipeline::run_decoder_step(int32_t token_id, std::vector<float>& logits) {
    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    state_->prepare_step(inputs);

    TensorMap outputs = decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end()) {
        throw std::runtime_error("CanaryPipeline: no 'logits' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data, num_logits * sizeof(float));

    state_->advance();
}

} // namespace trtmc
