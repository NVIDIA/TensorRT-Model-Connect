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
#include "runtime/models/canary/canary_request.h"
#include "runtime/models/canary/decode_runtime.h"
#include "trtmc/tokenizer.h"
#include "utils/wav_reader.h"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
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

struct CanaryStageTimestamps {
    CanaryClock::time_point transcribe_start;
    CanaryClock::time_point resample_end;
    CanaryClock::time_point mel_end;
    CanaryClock::time_point encoder_end;
    CanaryClock::time_point cross_kv_end;
    CanaryClock::time_point decoder_end;
    CanaryClock::time_point tokenize_end;
};

bool is_canary_control_token_start(const std::string& text, std::size_t position) {
    return text[position] == '<' && position + 1 < text.size() && text[position + 1] == '|';
}

bool is_canary_control_token_end(const std::string& text, std::size_t position) {
    return text[position] == '>' && position > 0 && text[position - 1] == '|';
}

std::string remove_punctuation_outside_control_tokens(const std::string& text) {
    std::string cleaned;
    cleaned.reserve(text.size());
    bool in_control_token = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (!in_control_token && is_canary_control_token_start(text, i))
            in_control_token = true;
        if (in_control_token || std::ispunct(ch) == 0)
            cleaned.push_back(static_cast<char>(ch));
        if (in_control_token && is_canary_control_token_end(text, i))
            in_control_token = false;
    }
    return cleaned;
}

void report_canary_stage_timing(bool enabled, const CanaryStageTimestamps& timestamps) {
    if (!enabled)
        return;
    std::ostringstream timing;
    timing << "[trtmc.canary_timing.json] {\"resample_ms\":"
           << elapsed_ms(timestamps.transcribe_start, timestamps.resample_end)
           << ",\"mel_ms\":" << elapsed_ms(timestamps.resample_end, timestamps.mel_end)
           << ",\"encoder_ms\":" << elapsed_ms(timestamps.mel_end, timestamps.encoder_end)
           << ",\"cross_kv_ms\":" << elapsed_ms(timestamps.encoder_end, timestamps.cross_kv_end)
           << ",\"decoder_ms\":" << elapsed_ms(timestamps.cross_kv_end, timestamps.decoder_end)
           << ",\"tokenizer_ms\":" << elapsed_ms(timestamps.decoder_end, timestamps.tokenize_end)
           << ",\"total_ms\":" << elapsed_ms(timestamps.transcribe_start, timestamps.tokenize_end)
           << '}';
    std::cerr << timing.str() << std::endl;
}

void validate_canary_audio_input(const float* audio_data, int32_t num_samples) {
    if (audio_data == nullptr || num_samples <= 0) {
        throw std::invalid_argument("Canary transcription requires non-empty audio samples");
    }
}

void validate_canary_output_budget(const CanaryConfig& model, const TranscriptionConfig& cfg,
                                   std::size_t prompt_tokens) {
    const int32_t available_output_tokens =
        model.max_target_positions - static_cast<int32_t>(prompt_tokens);
    if (available_output_tokens <= 0 || cfg.max_output_tokens > available_output_tokens) {
        throw std::invalid_argument("Canary max_output_tokens must be in [1, " +
                                    std::to_string(std::max(available_output_tokens, 0)) +
                                    "] after accounting for the decoder prompt");
    }
}

double validate_canary_input_duration(int32_t num_samples, int32_t sample_rate,
                                      const TranscriptionConfig& cfg) {
    const double duration_seconds =
        static_cast<double>(num_samples) / static_cast<double>(sample_rate);
    if (cfg.max_input_duration_seconds > 0.0F &&
        duration_seconds > static_cast<double>(cfg.max_input_duration_seconds) + 1.0e-6) {
        throw std::invalid_argument("Canary input duration " + std::to_string(duration_seconds) +
                                    " seconds exceeds max_input_duration_seconds=" +
                                    std::to_string(cfg.max_input_duration_seconds));
    }
    return duration_seconds;
}

double resolve_canary_segment_duration(double input_duration_seconds, double model_segment_seconds,
                                       const TranscriptionConfig& cfg) {
    const double segment_seconds = cfg.segment_duration_seconds > 0.0F
                                       ? static_cast<double>(cfg.segment_duration_seconds)
                                       : model_segment_seconds;
    if (segment_seconds <= 0.0 || segment_seconds > model_segment_seconds) {
        throw std::invalid_argument(
            "Canary segment_duration_seconds must be > 0 and <= the bundle limit of " +
            std::to_string(model_segment_seconds) + " seconds");
    }
    if (cfg.segment_duration_seconds <= 0.0F && input_duration_seconds > model_segment_seconds) {
        throw std::invalid_argument(
            "Canary input exceeds the bundle's single-segment limit of " +
            std::to_string(model_segment_seconds) +
            " seconds; set segment_duration_seconds to enable segmented decoding");
    }
    return segment_seconds;
}

int32_t canary_segment_sample_count(double segment_seconds, int32_t sample_rate) {
    const int64_t segment_samples =
        static_cast<int64_t>(std::llround(segment_seconds * static_cast<double>(sample_rate)));
    if (segment_samples <= 0 || segment_samples > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("Canary segment duration produces an invalid sample count");
    }
    return static_cast<int32_t>(segment_samples);
}

void append_canary_transcription_segment(TextResult& combined, TextResult segment, int64_t offset,
                                         int32_t count, int32_t sample_rate,
                                         const TranscriptionConfig& cfg) {
    if (!cfg.punctuation) {
        segment.text = remove_punctuation_outside_control_tokens(segment.text);
    }
    if (cfg.timestamps) {
        TranscriptionSegment timed;
        timed.start_seconds = static_cast<double>(offset) / static_cast<double>(sample_rate);
        timed.end_seconds = static_cast<double>(offset + count) / static_cast<double>(sample_rate);
        timed.text = segment.text;
        timed.token_ids = segment.token_ids;
        combined.segments.push_back(std::move(timed));
    }
    if (!combined.text.empty() && !segment.text.empty()) {
        combined.text += '\n';
    }
    combined.text += segment.text;
    combined.token_ids.insert(combined.token_ids.end(), segment.token_ids.begin(),
                              segment.token_ids.end());
    combined.prefill_ms += segment.prefill_ms;
    combined.decode_ms += segment.decode_ms;
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

    // Decoder shapes are stable within each dynamic cache-row bucket. Capture
    // the TensorRT enqueue on the first step and replay it until a shape change
    // invalidates the graph and triggers a recapture.
    if (!canary_config_.disable_cuda_graph)
        decoder_->enable_cuda_graph();

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
    TranscriptionConfig cfg;
    cfg.max_output_tokens = max_new_tokens;
    cfg.input_sample_rate = input_sample_rate;
    return transcribe(audio_data, num_samples, cfg);
}

TextResult CanaryPipeline::transcribe(const float* audio_data, int32_t num_samples,
                                      const TranscriptionConfig& cfg) {
    validate_canary_audio_input(audio_data, num_samples);

    auto initial_tokens = make_canary_request_tokens(canary_config_, cfg, tokenizer_.get());
    validate_canary_output_budget(canary_config_, cfg, initial_tokens.size());

    const int32_t sample_rate =
        cfg.input_sample_rate > 0 ? cfg.input_sample_rate : mel_sampling_rate_;
    const double duration_seconds = validate_canary_input_duration(num_samples, sample_rate, cfg);
    const double model_segment_seconds = static_cast<double>(mel_chunk_length_);
    const double segment_seconds =
        resolve_canary_segment_duration(duration_seconds, model_segment_seconds, cfg);
    const int32_t segment_samples = canary_segment_sample_count(segment_seconds, sample_rate);

    TextResult combined;
    for (int64_t offset = 0; offset < num_samples; offset += segment_samples) {
        const int32_t count = static_cast<int32_t>(
            std::min<int64_t>(segment_samples, static_cast<int64_t>(num_samples) - offset));
        auto segment =
            transcribe_segment(audio_data + offset, count, sample_rate, initial_tokens,
                               cfg.max_output_tokens, cfg.beam_size, cfg.beam_length_penalty);
        append_canary_transcription_segment(combined, std::move(segment), offset, count,
                                            sample_rate, cfg);
    }
    return combined;
}

TextResult CanaryPipeline::transcribe_segment(const float* audio_data, int32_t num_samples,
                                              int32_t input_sample_rate,
                                              const std::vector<int32_t>& initial_tokens,
                                              int32_t max_output_tokens, int32_t beam_size,
                                              float beam_length_penalty) {
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
    std::cerr << "[canary] Running decoder ..." << std::endl;
    auto output_ids =
        run_decoder(initial_tokens, max_output_tokens, beam_size, beam_length_penalty);
    const auto decoder_end = CanaryClock::now();

    // Step 5: Decode token IDs
    TextResult out;
    out.token_ids = std::move(output_ids);
    if (tokenizer_ && !out.token_ids.empty()) {
        out.text = tokenizer_->decode(out.token_ids);
    }
    const auto tokenize_end = CanaryClock::now();

    report_canary_stage_timing(report_stage_timing,
                               {transcribe_start, resample_end, mel_end, encoder_end, cross_kv_end,
                                decoder_end, tokenize_end});
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
                                                 int32_t max_new_tokens, int32_t beam_size,
                                                 float beam_length_penalty) {
    if (beam_size > 1) {
        ensure_beam_state_capacity(beam_size);
        auto result = run_canary_beam_search(
            initial_tokens, max_new_tokens, canary_config_.eot_token_id, beam_size,
            beam_length_penalty,
            [this](const std::vector<int32_t>& prefix, std::vector<float>& logits,
                   std::string& error) {
                try {
                    state_->reset();
                    state_->bind_to(*decoder_);
                    for (const int32_t token : prefix)
                        run_decoder_step(token, logits);
                    beam_states_a_.front()->copy_from(*state_);
                    return true;
                } catch (const std::exception& e) {
                    error = e.what();
                    return false;
                }
            },
            [this](int32_t generation, int32_t parent_slot, int32_t child_slot, int32_t token,
                   std::vector<float>& logits, std::string& error) {
                try {
                    auto& parents = generation % 2 == 0 ? beam_states_a_ : beam_states_b_;
                    auto& children = generation % 2 == 0 ? beam_states_b_ : beam_states_a_;
                    state_->copy_from(*parents.at(static_cast<std::size_t>(parent_slot)));
                    run_decoder_step(token, logits);
                    children.at(static_cast<std::size_t>(child_slot))->copy_from(*state_);
                    return true;
                } catch (const std::exception& e) {
                    error = e.what();
                    return false;
                }
            });
        if (result.prefill_failed || result.decode_failed)
            throw std::runtime_error("Canary beam search failed: " + result.error);
        return result.output_ids;
    }

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

void CanaryPipeline::ensure_beam_state_capacity(int32_t beam_size) {
    const auto target = static_cast<std::size_t>(beam_size);
    while (beam_states_a_.size() < target) {
        auto state_a = state_->create_empty();
        auto state_b = state_->create_empty();
        if (!state_a || !state_b || !state_a->ok() || !state_b->ok())
            throw std::runtime_error("CanaryPipeline: failed to allocate beam inference state");
        beam_states_a_.push_back(std::move(state_a));
        beam_states_b_.push_back(std::move(state_b));
    }
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
