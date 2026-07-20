/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_speech_streaming/pipeline.h"

#include "audio_helpers.h"
#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "plugin_helpers.h"
#include "utils/wav_reader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

using RnntClock = std::chrono::steady_clock;

struct RnntStageTotals {
    double frontend_ms{0.0};
    double encoder_and_transfer_ms{0.0};
    double predictor_joint_and_transfer_ms{0.0};
    double text_ms{0.0};
};

bool rnnt_stage_timing_enabled() {
    const char* value = std::getenv("TRTMC_RNNT_STAGE_TIMING");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

double elapsed_ms(RnntClock::time_point start, RnntClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void report_rnnt_stage_timing(bool enabled, const char* mode, const RnntStageTotals& stages,
                              double total_ms, int64_t source_samples, int64_t resampled_samples,
                              int64_t mel_frames) {
    if (!enabled) {
        return;
    }
    std::ostringstream timing;
    timing << "[trtmc.rnnt_timing.json] {\"mode\":\"" << mode
           << "\",\"frontend_ms\":" << stages.frontend_ms
           << ",\"encoder_and_transfer_ms\":" << stages.encoder_and_transfer_ms
           << ",\"predictor_joint_and_transfer_ms\":" << stages.predictor_joint_and_transfer_ms
           << ",\"text_ms\":" << stages.text_ms << ",\"total_ms\":" << total_ms
           << ",\"source_samples\":" << source_samples
           << ",\"resampled_samples\":" << resampled_samples << ",\"mel_frames\":" << mel_frames
           << '}';
    std::cerr << timing.str() << std::endl;
}

Tensor make_tensor(void* data, std::vector<int64_t> shape, DType dtype) {
    Tensor t;
    t.data = data;
    t.shape = std::move(shape);
    t.dtype = dtype;
    return t;
}

int32_t infer_encoder_frames(const Tensor& encoder_output, int32_t hidden_size) {
    if (hidden_size <= 0)
        return 0;
    const auto elems = static_cast<int32_t>(encoder_output.numel());
    return elems > 0 ? elems / hidden_size : 0;
}

int32_t argmax_token(const std::vector<float>& logits) {
    if (logits.empty())
        return -1;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

int32_t subsampled_frame_count(int32_t frames, bool causal) {
    if (frames <= 0)
        return 0;
    for (int i = 0; i < 3; ++i)
        frames = causal ? (frames / 2 + 1) : ((frames + 2 - 3) / 2 + 1);
    return frames;
}

bool rnnt_stream_debug_enabled() {
    const char* v = std::getenv("TRTMC_RNNT_STREAM_DEBUG");
    return v && std::string(v) != "0";
}

rnnt::MelSpectrogramOptions make_rnnt_mel_options(const RnntConfig& config) {
    rnnt::MelSpectrogramOptions options;
    options.n_fft = config.mel_n_fft;
    options.win_length = config.mel_win_length;
    options.hop_length = config.mel_hop_length;
    options.chunk_length_s = config.mel_chunk_length;
    options.sample_rate = config.sample_rate;
    options.symmetric_window = true;
    options.center_window_in_fft = true;
    options.preemphasis = config.mel_preemph;
    options.log_scale = rnnt::MelLogScale::kNaturalLog;
    return options;
}

void validate_rnnt_module(const std::unique_ptr<TrtModule>& module, const char* name) {
    if (!module || !module->ok())
        throw std::runtime_error(std::string("RnntPipeline: invalid ") + name + " module");
}

void validate_rnnt_core_config(const RnntConfig& config, const MelFilterbank& mel_fb) {
    if (mel_fb.data.empty())
        throw std::runtime_error("RnntPipeline: missing mel filterbank bundle section");
    if (config.encoder_hidden_size <= 0 || config.pred_hidden_size <= 0 ||
        config.pred_num_layers <= 0)
        throw std::runtime_error("RnntPipeline: invalid RNNT dimensions in config");
    if (config.blank_id < 0)
        throw std::runtime_error("RnntPipeline: invalid blank token id");
}

bool has_streaming_encoder_sections(const std::map<int32_t, std::string>& steady_sections,
                                    const std::map<int32_t, std::string>& first_sections) {
    return !steady_sections.empty() || !first_sections.empty();
}

void validate_rnnt_streaming_config(const RnntConfig& config, bool has_streaming_sections) {
    if (!has_streaming_sections)
        return;
    if (config.encoder_layers <= 0 || config.streaming_cache_left <= 0 ||
        config.streaming_time_cache <= 0)
        throw std::runtime_error("RnntPipeline: invalid RNNT streaming dimensions in config");
}

void print_vector_stats(const char* label, const std::vector<float>& values) {
    if (values.empty()) {
        std::cerr << "[rnnt.stream] " << label << " empty\n";
        return;
    }
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    double sum_abs = 0.0;
    for (float v : values)
        sum_abs += std::abs(static_cast<double>(v));
    std::cerr << "[rnnt.stream] " << label << " count=" << values.size() << " min=" << *min_it
              << " max=" << *max_it
              << " mean_abs=" << (sum_abs / static_cast<double>(values.size())) << "\n";
}

} // namespace

class RnntTranscriptionStream final : public ITranscriptionStream {
  public:
    RnntTranscriptionStream(RnntPipeline& pipeline, TranscriptionStreamConfig cfg)
        : pipeline_(pipeline), cfg_(cfg),
          schedule_(make_nemotron_streaming_schedule(
              cfg_.att_context_left, cfg_.att_context_right,
              cfg_.input_sample_rate > 0 ? cfg_.input_sample_rate : pipeline.config_.sample_rate,
              pipeline.config_.mel_hop_length, pipeline.config_.subsampling_factor)),
          transcribe_start_(RnntClock::now()),
          feature_state_(pipeline_.mel_fb_->data.data(), pipeline_.mel_fb_->n_freq_bins,
                         pipeline_.mel_fb_->n_mel_bins, make_rnnt_mel_options(pipeline_.config_),
                         cfg_.input_sample_rate > 0 ? cfg_.input_sample_rate
                                                    : pipeline_.config_.sample_rate) {
        const auto state_elems = static_cast<std::size_t>(pipeline_.config_.pred_num_layers) *
                                 pipeline_.config_.pred_hidden_size;
        state_h_.assign(state_elems, 0.0F);
        state_c_.assign(state_elems, 0.0F);
        const auto decode_start = RnntClock::now();
        pred_output_ = pipeline_.run_predictor(pipeline_.config_.blank_id, state_h_, state_c_);
        stages_.predictor_joint_and_transfer_ms += elapsed_ms(decode_start, RnntClock::now());
        reset_encoder_cache();
    }

    TranscriptionStreamResult accept_audio(const float* audio_samples, int32_t num_samples,
                                           bool is_final = false) override {
        if (num_samples < 0)
            throw std::runtime_error("RNNT streaming transcription received negative sample count");
        if (num_samples > 0 && audio_samples == nullptr)
            throw std::runtime_error("RNNT streaming transcription received null audio");
        if (finished_)
            throw std::runtime_error("RNNT streaming transcription stream is already finished");

        if (num_samples > 0) {
            const auto frontend_start = RnntClock::now();
            feature_state_.accept_audio(audio_samples, num_samples);
            stages_.frontend_ms += elapsed_ms(frontend_start, RnntClock::now());
            accepted_samples_ += num_samples;
        }
        ++chunk_index_;

        if (is_final)
            return finish();

        const int32_t shift = use_first_step_plan() ? schedule_.first_shift_mel_frames
                                                    : schedule_.next_shift_mel_frames;
        const int32_t retained_lookahead =
            schedule_.next_shift_mel_frames - schedule_.first_shift_mel_frames;
        const int32_t required_mel = shift + retained_lookahead;
        if (available_mel_frames() - next_mel_start_ < required_mel)
            return make_result(false);

        return process_ready(false);
    }

    TranscriptionStreamResult finish() override {
        if (finished_)
            return final_;
        finished_ = true;

        final_ = process_ready(true);
        final_.is_final = true;
        const auto feature_stats = feature_state_.stats();
        report_rnnt_stage_timing(
            rnnt_stage_timing_enabled(), "streaming", stages_,
            elapsed_ms(transcribe_start_, RnntClock::now()), feature_stats.accepted_source_samples,
            feature_stats.generated_resampled_samples, feature_stats.computed_mel_frames);
        return final_;
    }

    void reset() override {
        transcribe_start_ = RnntClock::now();
        stages_ = {};
        feature_state_.reset();
        accepted_samples_ = 0;
        chunk_index_ = 0;
        finished_ = false;
        final_ = {};
        next_mel_start_ = 0;
        emitted_.clear();
        std::fill(state_h_.begin(), state_h_.end(), 0.0F);
        std::fill(state_c_.begin(), state_c_.end(), 0.0F);
        const auto decode_start = RnntClock::now();
        pred_output_ = pipeline_.run_predictor(pipeline_.config_.blank_id, state_h_, state_c_);
        stages_.predictor_joint_and_transfer_ms += elapsed_ms(decode_start, RnntClock::now());
        reset_encoder_cache();
    }

    TranscriptionStreamConfig config() const override { return cfg_; }

  private:
    bool use_first_step_plan() const {
        return next_mel_start_ == 0 && !cfg_.pad_and_drop_preencoded;
    }

    int32_t token_limit() const { return cfg_.max_new_tokens > 0 ? cfg_.max_new_tokens : 256; }

    bool token_limit_reached() const {
        return static_cast<int32_t>(emitted_.size()) >= token_limit();
    }

    int32_t current_shift_mel_frames(bool first_step) const {
        return first_step ? schedule_.first_shift_mel_frames : schedule_.next_shift_mel_frames;
    }

    int32_t current_pre_encode_cache_mel_frames(bool first_step) const {
        return first_step ? schedule_.first_pre_encode_cache_mel_frames
                          : schedule_.next_pre_encode_cache_mel_frames;
    }

    int32_t current_drop_extra_pre_encoded(bool first_step) const {
        return first_step ? 0 : schedule_.drop_extra_pre_encoded;
    }

    bool should_process_chunk(int32_t remaining, int32_t shift, bool final) const {
        if (remaining <= 0)
            return false;
        if (!final && remaining < shift)
            return false;
        return true;
    }

    int32_t available_mel_frames() const { return feature_state_.available_frames(); }

    std::vector<float> make_chunk_mel(int32_t start_frame, int32_t valid_new_frames,
                                      bool first_step) const {
        const int32_t chunk_frames =
            first_step ? schedule_.first_shift_mel_frames : schedule_.next_shift_mel_frames;
        const int32_t pre = first_step ? schedule_.first_pre_encode_cache_mel_frames
                                       : schedule_.next_pre_encode_cache_mel_frames;
        const int32_t total = pre + chunk_frames;
        std::vector<float> chunk(static_cast<std::size_t>(pipeline_.config_.num_mel_bins) * total,
                                 0.0F);
        for (int32_t m = 0; m < pipeline_.config_.num_mel_bins; ++m) {
            for (int32_t p = 0; p < pre; ++p) {
                const int32_t src_frame = start_frame - pre + p;
                if (src_frame >= 0 && src_frame < feature_state_.frame_count()) {
                    chunk[static_cast<std::size_t>(m) * total + p] =
                        feature_state_.value(m, src_frame);
                }
            }
            for (int32_t p = 0; p < valid_new_frames; ++p) {
                const int32_t src_frame = start_frame + p;
                if (src_frame >= 0 && src_frame < feature_state_.frame_count()) {
                    chunk[static_cast<std::size_t>(m) * total + pre + p] =
                        feature_state_.value(m, src_frame);
                }
            }
        }
        return chunk;
    }

    TranscriptionStreamResult make_result(bool is_final) {
        const auto text_start = RnntClock::now();
        TranscriptionStreamResult out;
        out.token_ids = emitted_;
        if (pipeline_.tokenizer_ && !out.token_ids.empty())
            out.text = pipeline_.tokenizer_->decode(out.token_ids);
        out.is_final = is_final;
        out.chunk_index = chunk_index_;
        out.accepted_samples = accepted_samples_;
        out.sample_rate = cfg_.input_sample_rate;
        stages_.text_ms += elapsed_ms(text_start, RnntClock::now());
        return out;
    }

    int32_t valid_query_frames_for_chunk(int32_t valid_new_frames, bool first_step) const {
        const int32_t valid_total_mel =
            current_pre_encode_cache_mel_frames(first_step) + valid_new_frames;
        const int32_t drop_extra = current_drop_extra_pre_encoded(first_step);
        int32_t valid_query_frames =
            subsampled_frame_count(valid_total_mel, pipeline_.config_.causal_downsampling) -
            drop_extra;
        return std::max(0, std::min(valid_query_frames, schedule_.valid_encoder_frames));
    }

    void trace_streaming_chunk(const std::vector<float>& enc, std::size_t before_tokens,
                               int32_t valid_new_frames, int32_t valid_query_frames) const {
        std::cerr << "[rnnt.stream] chunk=" << chunk_index_ << " mel_start=" << next_mel_start_
                  << " valid_new_mel=" << valid_new_frames << " valid_query=" << valid_query_frames
                  << " cache_len=" << cache_last_channel_len_
                  << " emitted_delta=" << (emitted_.size() - before_tokens)
                  << " emitted_total=" << emitted_.size() << "\n";
        print_vector_stats("encoder", enc);
    }

    bool process_next_chunk(int32_t actual_mel_frames, bool final) {
        const bool first_step = use_first_step_plan();
        const int32_t shift = current_shift_mel_frames(first_step);
        const int32_t remaining = actual_mel_frames - next_mel_start_;
        if (!should_process_chunk(remaining, shift, final))
            return false;

        const int32_t valid_new_frames = std::min(shift, remaining);
        const int32_t valid_query_frames =
            valid_query_frames_for_chunk(valid_new_frames, first_step);
        if (valid_query_frames <= 0)
            return false;

        const auto frontend_start = RnntClock::now();
        feature_state_.ensure_frames(next_mel_start_ + valid_new_frames, final);
        auto chunk = make_chunk_mel(next_mel_start_, valid_new_frames, first_step);
        stages_.frontend_ms += elapsed_ms(frontend_start, RnntClock::now());
        std::vector<float> next_channel;
        std::vector<float> next_time;
        const auto before_tokens = emitted_.size();
        const auto encoder_start = RnntClock::now();
        auto enc = pipeline_.run_streaming_encoder(
            cfg_.att_context_right, chunk, cache_last_channel_, cache_last_time_,
            cache_last_channel_len_, valid_query_frames, first_step, next_channel, next_time);
        stages_.encoder_and_transfer_ms += elapsed_ms(encoder_start, RnntClock::now());
        const auto decode_start = RnntClock::now();
        pipeline_.decode_encoder_frames(enc, valid_query_frames, token_limit(), pred_output_,
                                        state_h_, state_c_, emitted_);
        stages_.predictor_joint_and_transfer_ms += elapsed_ms(decode_start, RnntClock::now());
        if (rnnt_stream_debug_enabled())
            trace_streaming_chunk(enc, before_tokens, valid_new_frames, valid_query_frames);

        cache_last_channel_ = std::move(next_channel);
        cache_last_time_ = std::move(next_time);
        cache_last_channel_len_ =
            std::min(pipeline_.config_.streaming_cache_left,
                     cache_last_channel_len_ + schedule_.valid_encoder_frames);
        next_mel_start_ += valid_new_frames;
        return valid_new_frames == shift;
    }

    TranscriptionStreamResult process_ready(bool final) {
        if (accepted_samples_ == 0)
            return make_result(final);
        const int32_t actual_mel_frames = available_mel_frames();

        while (!token_limit_reached()) {
            if (!process_next_chunk(actual_mel_frames, final))
                break;
        }
        return make_result(final);
    }

    void reset_encoder_cache() {
        const auto channel_count = static_cast<std::size_t>(pipeline_.config_.encoder_layers) *
                                   pipeline_.config_.streaming_cache_left *
                                   pipeline_.config_.encoder_hidden_size;
        const auto time_count = static_cast<std::size_t>(pipeline_.config_.encoder_layers) *
                                pipeline_.config_.encoder_hidden_size *
                                pipeline_.config_.streaming_time_cache;
        cache_last_channel_.assign(channel_count, 0.0F);
        cache_last_time_.assign(time_count, 0.0F);
        cache_last_channel_len_ = 0;
    }

    RnntPipeline& pipeline_;
    TranscriptionStreamConfig cfg_;
    RnntStreamingSchedule schedule_;
    RnntClock::time_point transcribe_start_;
    rnnt::IncrementalMelSpectrogram feature_state_;
    RnntStageTotals stages_;
    std::vector<float> cache_last_channel_;
    std::vector<float> cache_last_time_;
    std::vector<float> pred_output_;
    std::vector<float> state_h_;
    std::vector<float> state_c_;
    std::vector<int32_t> emitted_;
    int64_t accepted_samples_{0};
    int32_t next_mel_start_{0};
    int32_t cache_last_channel_len_{0};
    int32_t chunk_index_{0};
    bool finished_{false};
    TranscriptionStreamResult final_;
};

RnntPipeline::RnntPipeline(std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> predictor,
                           std::unique_ptr<TrtModule> joint,
                           std::unique_ptr<TrtModule> prompt_kernel,
                           std::map<int32_t, std::string> streaming_encoder_sections,
                           IBackend* backend, ModuleCreateOptions module_options,
                           std::map<int32_t, std::string> streaming_first_encoder_sections,
                           std::string bundle_path, RnntConfig config, MelFilterbank mel_fb,
                           cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
                           std::string model_id_str)
    : encoder_(std::move(encoder)), predictor_(std::move(predictor)), joint_(std::move(joint)),
      prompt_kernel_(std::move(prompt_kernel)),
      streaming_encoder_sections_(std::move(streaming_encoder_sections)),
      streaming_first_encoder_sections_(std::move(streaming_first_encoder_sections)),
      backend_(backend), module_options_(module_options), bundle_path_(std::move(bundle_path)),
      config_(std::move(config)), mel_fb_(std::make_unique<MelFilterbank>(std::move(mel_fb))),
      stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)) {
    validate_rnnt_module(encoder_, "encoder");
    validate_rnnt_module(predictor_, "predictor");
    validate_rnnt_module(joint_, "joint");
    validate_rnnt_core_config(config_, *mel_fb_);
    validate_rnnt_streaming_config(
        config_, has_streaming_encoder_sections(streaming_encoder_sections_,
                                                streaming_first_encoder_sections_));
    if (config_.has_prompt_kernel) {
        if (!prompt_kernel_ || !prompt_kernel_->ok())
            throw std::runtime_error(
                "RnntPipeline: prompt_kernel module required when has_prompt_kernel=true");
        if (config_.num_prompts <= 0)
            throw std::runtime_error("RnntPipeline: invalid num_prompts for prompt_kernel variant");
    }
}

int32_t RnntPipeline::resolve_prompt_index(const std::string& tag) const {
    if (!config_.has_prompt_kernel)
        return 0;
    if (tag.empty())
        return 0;
    auto it = config_.prompt_dictionary.find(tag);
    if (it == config_.prompt_dictionary.end())
        throw std::invalid_argument("Unsupported language tag '" + tag + "'");
    return it->second;
}

std::vector<float> RnntPipeline::run_prompt_kernel(const float* encoder_frame) {
    if (!prompt_kernel_)
        throw std::runtime_error("RnntPipeline: run_prompt_kernel called without engine");
    const int32_t enc_dim = config_.encoder_hidden_size;
    const int32_t num_p = config_.num_prompts;

    TensorMap inputs;
    inputs["encoder_frame"] =
        make_tensor(const_cast<float*>(encoder_frame), {1, enc_dim}, DType::kFloat32);
    inputs["prompt_onehot"] = make_tensor(prompt_onehot_.data(), {1, num_p}, DType::kFloat32);

    auto outputs = prompt_kernel_->forward(inputs);
    auto it = outputs.find("prompt_kernel_output");
    if (it == outputs.end())
        throw std::runtime_error("RnntPipeline: prompt_kernel missing 'prompt_kernel_output'");
    const auto* src = static_cast<const float*>(it->second.data);
    const auto count = static_cast<std::size_t>(it->second.numel());
    return std::vector<float>(src, src + count);
}

RnntPipeline::~RnntPipeline() = default;

void RnntPipeline::setup_prompt_state(const std::string& language) {
    if (!config_.has_prompt_kernel) {
        prompt_onehot_.clear();
        prompt_index_ = -1;
        return;
    }
    prompt_index_ = resolve_prompt_index(language);
    prompt_onehot_.assign(static_cast<std::size_t>(config_.num_prompts), 0.0F);
    if (prompt_index_ >= 0 && prompt_index_ < config_.num_prompts)
        prompt_onehot_[static_cast<std::size_t>(prompt_index_)] = 1.0F;
}

std::unique_ptr<ITranscriptionStream>
RnntPipeline::create_transcription_stream(const TranscriptionStreamConfig& cfg) {
    (void)make_nemotron_streaming_schedule(cfg.att_context_left, cfg.att_context_right,
                                           cfg.input_sample_rate, config_.mel_hop_length,
                                           config_.subsampling_factor);

    setup_prompt_state(cfg.language);

    if (cfg.online_normalization)
        throw std::runtime_error("RNNT streaming transcription does not support "
                                 "online_normalization for this checkpoint");
    if (!cfg.use_cache || !cfg.use_feature_cache)
        throw std::runtime_error("RNNT streaming transcription parity requires both encoder cache "
                                 "and feature cache to match NeMo cache-aware streaming");
    if (streaming_encoders_.find(cfg.att_context_right) == streaming_encoders_.end() &&
        streaming_encoder_sections_.find(cfg.att_context_right) ==
            streaming_encoder_sections_.end()) {
        throw std::runtime_error(
            "RNNT cache-aware streaming requires a bundle with streaming encoder cache "
            "inputs/outputs for the requested att_context_size");
    }
    if (!cfg.pad_and_drop_preencoded &&
        streaming_first_encoders_.find(cfg.att_context_right) == streaming_first_encoders_.end() &&
        streaming_first_encoder_sections_.find(cfg.att_context_right) ==
            streaming_first_encoder_sections_.end()) {
        throw std::runtime_error(
            "RNNT cache-aware streaming requires a first-step streaming encoder plan for "
            "NeMo pad_and_drop_preencoded=false parity");
    }
    (void)streaming_encoder_for(cfg.att_context_right, !cfg.pad_and_drop_preencoded);
    return std::make_unique<RnntTranscriptionStream>(*this, cfg);
}

TextResult RnntPipeline::transcribe(const float* audio_data, int32_t num_samples,
                                    int32_t max_new_tokens, int32_t input_sample_rate) {
    const bool report_timing = rnnt_stage_timing_enabled();
    const auto transcribe_start = RnntClock::now();
    RnntStageTotals stages;

    int32_t actual_mel_frames = 0;
    auto mel = extract_padded_mel(audio_data, num_samples, input_sample_rate, actual_mel_frames);
    const auto frontend_end = RnntClock::now();
    stages.frontend_ms = elapsed_ms(transcribe_start, frontend_end);
    if (mel.empty())
        return TextResult{"[mel extraction failed]", {}};

    auto encoder_output = run_encoder(mel, actual_mel_frames);
    const auto encoder_end = RnntClock::now();
    stages.encoder_and_transfer_ms = elapsed_ms(frontend_end, encoder_end);
    const int32_t encoder_frames =
        static_cast<int32_t>(encoder_output.size()) / config_.encoder_hidden_size;
    if (encoder_frames <= 0)
        return TextResult{"[encoder produced no frames]", {}};

    const auto state_elems =
        static_cast<std::size_t>(config_.pred_num_layers) * config_.pred_hidden_size;
    std::vector<float> state_h(state_elems, 0.0F);
    std::vector<float> state_c(state_elems, 0.0F);
    std::vector<float> pred_output = run_predictor(config_.blank_id, state_h, state_c);

    std::vector<int32_t> emitted;
    const int32_t token_limit = max_new_tokens > 0 ? max_new_tokens : 256;
    emitted.reserve(static_cast<std::size_t>(token_limit));

    decode_encoder_frames(encoder_output, encoder_frames, token_limit, pred_output, state_h,
                          state_c, emitted);
    const auto decode_end = RnntClock::now();
    stages.predictor_joint_and_transfer_ms = elapsed_ms(encoder_end, decode_end);

    TextResult out;
    out.token_ids = std::move(emitted);
    if (tokenizer_ && !out.token_ids.empty())
        out.text = tokenizer_->decode(out.token_ids);
    const auto text_end = RnntClock::now();
    stages.text_ms = elapsed_ms(decode_end, text_end);
    const int64_t resampled_samples =
        input_sample_rate > 0 && input_sample_rate != config_.sample_rate
            ? static_cast<int64_t>(num_samples) * config_.sample_rate / input_sample_rate
            : num_samples;
    report_rnnt_stage_timing(report_timing, "offline", stages,
                             elapsed_ms(transcribe_start, text_end), num_samples, resampled_samples,
                             actual_mel_frames);
    return out;
}

std::vector<float> RnntPipeline::extract_padded_mel(const float* audio_data, int32_t num_samples,
                                                    int32_t input_sample_rate,
                                                    int32_t& actual_frames) const {
    const float* samples_ptr = audio_data;
    int32_t samples_count = num_samples;
    std::vector<float> resampled;
    if (input_sample_rate > 0 && input_sample_rate != config_.sample_rate) {
        std::cerr << "[rnnt] Resampling audio from " << input_sample_rate << " Hz to "
                  << config_.sample_rate << " Hz" << std::endl;
        resampled =
            resample_linear(audio_data, num_samples, input_sample_rate, config_.sample_rate);
        samples_ptr = resampled.data();
        samples_count = static_cast<int32_t>(resampled.size());
    }

    rnnt::MelResult mel = rnnt::extract_rnnt_mel_spectrogram(
        samples_ptr, samples_count, mel_fb_->data.data(), mel_fb_->n_freq_bins, mel_fb_->n_mel_bins,
        config_.mel_n_fft, config_.mel_win_length, config_.mel_hop_length, config_.mel_chunk_length,
        config_.sample_rate, config_.mel_preemph);
    actual_frames = std::min(mel.n_frames, std::max(0, samples_count / config_.mel_hop_length));
    if (mel.data.empty())
        return {};

    const int32_t target_frames = config_.mel_length > 0 ? config_.mel_length : mel.n_frames;
    std::vector<float> padded(static_cast<std::size_t>(mel.n_mels) * target_frames, 0.0F);
    const int32_t copy_frames = std::min(mel.n_frames, target_frames);
    for (int32_t m = 0; m < mel.n_mels; ++m) {
        std::memcpy(padded.data() + static_cast<std::size_t>(m) * target_frames,
                    mel.data.data() + static_cast<std::size_t>(m) * mel.n_frames,
                    static_cast<std::size_t>(copy_frames) * sizeof(float));
    }
    return padded;
}

std::vector<float> RnntPipeline::run_encoder(const std::vector<float>& mel, int32_t actual_frames) {
    TensorMap inputs;
    inputs["mel_features"] =
        make_tensor(const_cast<float*>(mel.data()), {config_.num_mel_bins, config_.mel_length},
                    DType::kFloat32);

    std::vector<float> encoder_mask;
    if (encoder_->has_input("encoder_mask")) {
        const int32_t max_frames = std::max(
            1, config_.encoder_seq_len > 0
                   ? config_.encoder_seq_len
                   : subsampled_frame_count(config_.mel_length, config_.causal_downsampling));
        int32_t actual_encoder_frames =
            std::max(1, subsampled_frame_count(actual_frames, config_.causal_downsampling));
        actual_encoder_frames = std::min(actual_encoder_frames, max_frames);
        encoder_mask.assign(static_cast<std::size_t>(max_frames) * max_frames, -10000.0F);
        for (int32_t q = 0; q < actual_encoder_frames; ++q) {
            const int32_t k_begin = std::max(0, q - config_.att_context_left);
            const int32_t k_end =
                std::min(actual_encoder_frames - 1, q + config_.att_context_right);
            for (int32_t k = k_begin; k <= k_end; ++k)
                encoder_mask[static_cast<std::size_t>(q) * max_frames + k] = 0.0F;
        }
        inputs["encoder_mask"] =
            make_tensor(encoder_mask.data(), {1, max_frames, max_frames}, DType::kFloat32);
    }

    auto outputs = encoder_->forward(inputs);
    auto it = outputs.find("encoder_output");
    if (it == outputs.end())
        throw std::runtime_error("RnntPipeline: encoder missing 'encoder_output'");
    const auto frames = infer_encoder_frames(it->second, config_.encoder_hidden_size);
    const auto count = static_cast<std::size_t>(frames) * config_.encoder_hidden_size;
    const auto* src = static_cast<const float*>(it->second.data);
    return std::vector<float>(src, src + count);
}

TrtModule& RnntPipeline::streaming_encoder_for(int32_t right_context, bool first_step) {
    auto& encoders = first_step ? streaming_first_encoders_ : streaming_encoders_;
    auto& sections = first_step ? streaming_first_encoder_sections_ : streaming_encoder_sections_;
    auto cached = encoders.find(right_context);
    if (cached != encoders.end() && cached->second)
        return *cached->second;

    auto section_it = sections.find(right_context);
    if (section_it == sections.end() || section_it->second.empty())
        throw std::runtime_error("RnntPipeline: missing streaming encoder plan for requested "
                                 "att_context_size");
    if (bundle_path_.empty())
        throw std::runtime_error(
            "RnntPipeline: missing bundle path for lazy streaming encoder load");

    const std::string label = std::string("rnnt streaming encoder ") +
                              (first_step ? "first ctx" : "ctx") + std::to_string(right_context);
    BundleFile bundle = ReadBundleFile(bundle_path_);
    const auto* plan = find_section(bundle, section_it->second);
    auto loaded = load_trt_module_from_plan(backend_, plan, label.c_str(), module_options_);
    auto* module = loaded.module.get();
    encoders[right_context] = std::move(loaded.module);
    return *module;
}

std::vector<float> RnntPipeline::run_streaming_encoder(
    int32_t right_context, const std::vector<float>& mel,
    const std::vector<float>& cache_last_channel, const std::vector<float>& cache_last_time,
    int32_t cache_last_channel_len, int32_t valid_query_frames, bool first_step,
    std::vector<float>& next_channel, std::vector<float>& next_time) {
    TrtModule& encoder = streaming_encoder_for(right_context, first_step);

    const auto schedule = make_nemotron_streaming_schedule(
        config_.att_context_left, right_context, config_.sample_rate, config_.mel_hop_length,
        config_.subsampling_factor);
    const int32_t query_frames = schedule.valid_encoder_frames;
    const int32_t cache_frames = config_.streaming_cache_left;
    const int32_t key_frames = cache_frames + query_frames;
    const int32_t mel_frames =
        first_step ? schedule.first_pre_encode_cache_mel_frames + schedule.first_shift_mel_frames
                   : schedule.next_pre_encode_cache_mel_frames + schedule.next_shift_mel_frames;

    std::vector<float> encoder_mask(static_cast<std::size_t>(query_frames) * key_frames, -10000.0F);
    const int32_t valid_cache_begin = cache_frames - std::max(0, cache_last_channel_len);
    for (int32_t q = 0; q < valid_query_frames; ++q) {
        const int32_t key_end = cache_frames + std::min(valid_query_frames - 1, q + right_context);
        for (int32_t k = valid_cache_begin; k <= key_end; ++k) {
            if (k >= 0 && k < key_frames)
                encoder_mask[static_cast<std::size_t>(q) * key_frames + k] = 0.0F;
        }
    }

    TensorMap inputs;
    inputs["mel_features"] = make_tensor(const_cast<float*>(mel.data()),
                                         {config_.num_mel_bins, mel_frames}, DType::kFloat32);
    inputs["cache_last_channel"] = make_tensor(
        const_cast<float*>(cache_last_channel.data()),
        {config_.encoder_layers, cache_frames, config_.encoder_hidden_size}, DType::kFloat32);
    inputs["cache_last_time"] = make_tensor(
        const_cast<float*>(cache_last_time.data()),
        {config_.encoder_layers, config_.encoder_hidden_size, config_.streaming_time_cache},
        DType::kFloat32);
    inputs["encoder_mask"] =
        make_tensor(encoder_mask.data(), {1, query_frames, key_frames}, DType::kFloat32);

    auto outputs = encoder.forward(inputs);
    auto enc_it = outputs.find("encoder_output");
    auto ch_it = outputs.find("cache_last_channel_next");
    auto tm_it = outputs.find("cache_last_time_next");
    if (enc_it == outputs.end() || ch_it == outputs.end() || tm_it == outputs.end())
        throw std::runtime_error("RnntPipeline: streaming encoder missing required outputs");

    const auto* enc_src = static_cast<const float*>(enc_it->second.data);
    const auto enc_count = static_cast<std::size_t>(query_frames) * config_.encoder_hidden_size;
    std::vector<float> enc(enc_src, enc_src + enc_count);

    const auto* ch_src = static_cast<const float*>(ch_it->second.data);
    next_channel.assign(ch_src, ch_src + ch_it->second.numel());
    const auto* tm_src = static_cast<const float*>(tm_it->second.data);
    next_time.assign(tm_src, tm_src + tm_it->second.numel());
    return enc;
}

void RnntPipeline::decode_encoder_frames(const std::vector<float>& encoder_output,
                                         int32_t frame_count, int32_t token_limit,
                                         std::vector<float>& pred_output,
                                         std::vector<float>& state_h, std::vector<float>& state_c,
                                         std::vector<int32_t>& emitted) {
    for (int32_t frame = 0;
         frame < frame_count && static_cast<int32_t>(emitted.size()) < token_limit; ++frame) {
        const float* enc_ptr =
            encoder_output.data() + static_cast<std::size_t>(frame) * config_.encoder_hidden_size;
        // Multilingual variant: replace the encoder frame with the prompt_kernel
        // MLP output (Linear 1152 -> 2048 -> ReLU -> 1024) before the joint.
        std::vector<float> projected;
        const float* enc_frame = enc_ptr;
        if (config_.has_prompt_kernel) {
            projected = run_prompt_kernel(enc_ptr);
            enc_frame = projected.data();
        }
        int32_t symbols_this_frame = 0;
        while (static_cast<int32_t>(emitted.size()) < token_limit) {
            auto logits = run_joint(enc_frame, pred_output.data());
            const int32_t token = argmax_token(logits);
            if (token < 0)
                break;

            const auto decision = make_rnnt_greedy_decision(
                token, config_.blank_id, symbols_this_frame, config_.max_symbols_per_step);
            if (decision.emit_token) {
                emitted.push_back(token);
                pred_output = run_predictor(token, state_h, state_c);
                ++symbols_this_frame;
            }
            if (decision.advance_frame)
                break;
        }
    }
}

std::vector<float> RnntPipeline::run_predictor(int32_t token_id, std::vector<float>& state_h,
                                               std::vector<float>& state_c) {
    TensorMap inputs;
    inputs["token_id"] = make_tensor(&token_id, {1}, DType::kInt32);
    const auto layer_stride = static_cast<std::size_t>(config_.pred_hidden_size);
    for (int32_t layer = 0; layer < config_.pred_num_layers; ++layer) {
        const std::string suffix = "_" + std::to_string(layer);
        inputs["state_h" + suffix] =
            make_tensor(state_h.data() + static_cast<std::size_t>(layer) * layer_stride,
                        {1, config_.pred_hidden_size}, DType::kFloat32);
        inputs["state_c" + suffix] =
            make_tensor(state_c.data() + static_cast<std::size_t>(layer) * layer_stride,
                        {1, config_.pred_hidden_size}, DType::kFloat32);
    }

    auto outputs = predictor_->forward(inputs);
    auto pred_it = outputs.find("pred_output");
    if (pred_it == outputs.end())
        throw std::runtime_error("RnntPipeline: predictor missing 'pred_output'");

    for (int32_t layer = 0; layer < config_.pred_num_layers; ++layer) {
        const std::string suffix = "_" + std::to_string(layer);
        auto h_it = outputs.find("next_h" + suffix);
        auto c_it = outputs.find("next_c" + suffix);
        if (h_it == outputs.end() || c_it == outputs.end())
            throw std::runtime_error("RnntPipeline: predictor missing next state outputs");
        std::memcpy(state_h.data() + static_cast<std::size_t>(layer) * layer_stride,
                    h_it->second.data, layer_stride * sizeof(float));
        std::memcpy(state_c.data() + static_cast<std::size_t>(layer) * layer_stride,
                    c_it->second.data, layer_stride * sizeof(float));
    }

    const auto* pred = static_cast<const float*>(pred_it->second.data);
    return std::vector<float>(pred, pred + config_.pred_hidden_size);
}

std::vector<float> RnntPipeline::run_joint(const float* encoder_frame, const float* pred_output) {
    TensorMap inputs;
    inputs["encoder_frame"] = make_tensor(const_cast<float*>(encoder_frame),
                                          {1, config_.encoder_hidden_size}, DType::kFloat32);
    inputs["pred_output"] = make_tensor(const_cast<float*>(pred_output),
                                        {1, config_.pred_hidden_size}, DType::kFloat32);
    auto outputs = joint_->forward(inputs);
    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("RnntPipeline: joint missing 'logits'");
    const auto* logits = static_cast<const float*>(it->second.data);
    const auto count = static_cast<std::size_t>(it->second.numel());
    return std::vector<float>(logits, logits + count);
}

} // namespace trtmc
