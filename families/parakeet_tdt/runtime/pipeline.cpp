/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/parakeet_tdt/runtime/pipeline.h"

#include "families/parakeet_tdt/runtime/audio_helpers.h"
#include "families/parakeet_tdt/runtime/audio_input.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace trtmc::parakeet_tdt {
namespace {
const internal::ConfigField fields[] = {
    {"max_new_tokens", internal::ConfigKind::I64, internal::ConfigValue{int64_t{256}},
     "Maximum emitted transcript tokens"},
};
void validate_float_outputs(const TensorMap& outputs) {
    for (const auto& item : outputs) {
        const auto& tensor = item.second;
        if (tensor.dtype != DType::kFloat32 || tensor.data == nullptr || tensor.shape.empty() ||
            std::any_of(tensor.shape.begin(), tensor.shape.end(), [](auto d) { return d <= 0; }))
            throw std::runtime_error("Parakeet TDT requires nonempty FP32 host outputs: " +
                                     item.first);
    }
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
} // namespace
TdtPipeline::TdtPipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> predictor,
                         std::unique_ptr<ITrtModule> joint, TdtConfig config, MelFilterbank mel,
                         std::shared_ptr<ITokenizer> tokenizer)
    : encoder_(std::move(encoder)), predictor_(std::move(predictor)), joint_(std::move(joint)),
      config_(std::move(config)), mel_fb_(std::make_unique<MelFilterbank>(std::move(mel))),
      tokenizer_(std::move(tokenizer)) {
    if (!encoder_ || !predictor_ || !joint_ || !encoder_->ok() || !predictor_->ok() ||
        !joint_->ok() || !tokenizer_)
        throw std::invalid_argument("Parakeet TDT requires three valid engines and a tokenizer");
    validate_tdt_mel_geometry(config_, mel_fb_->n_freq_bins, mel_fb_->n_mel_bins);
    if (mel_fb_->data.size() != static_cast<size_t>(mel_fb_->n_freq_bins) * mel_fb_->n_mel_bins ||
        config_.sample_rate != 16000 || config_.encoder_hidden_size <= 0 ||
        config_.pred_hidden_size <= 0 || config_.pred_num_layers <= 0 || config_.blank_id < 0 ||
        config_.max_symbols_per_step <= 0 || config_.duration_values.empty() ||
        std::any_of(
            config_.duration_values.begin(), config_.duration_values.end(),
            [](auto d) { return d < 0; }))
        throw std::invalid_argument("invalid Parakeet TDT runtime dimensions");
}
std::vector<internal::TaskInstance> TdtPipeline::task_bindings() {
    return {internal::bind<internal::ISpeechTranscription>(*this, fields)};
}
TextResult TdtPipeline::run(const internal::SpeechTranscriptionRequest& request,
                            internal::ConfigView options) {
    internal::validate_config(fields, options);
    const auto limit = internal::config_get<int64_t>(options, fields, "max_new_tokens").value();
    if (limit <= 0 || limit > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("Parakeet TDT max_new_tokens must be a positive int32");
    const auto audio = tdt::prepare_audio(request);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto start = std::chrono::steady_clock::now();
    int32_t actual_frames = 0;
    auto mel =
        extract_padded_mel(audio.data(), static_cast<int32_t>(audio.size()), 16000, actual_frames);
    if (mel.empty())
        throw std::runtime_error("Parakeet TDT mel extraction returned no features");
    auto encoded = run_encoder(mel, actual_frames);
    const auto after_encoder = std::chrono::steady_clock::now();
    std::vector<float> state_h(
        static_cast<size_t>(config_.pred_num_layers) * config_.pred_hidden_size, 0);
    auto state_c = state_h;
    auto predicted = run_predictor(config_.blank_id, state_h, state_c);
    TextResult result;
    decode_encoder_frames(
        encoded, static_cast<int32_t>(encoded.size() / config_.encoder_hidden_size),
        static_cast<int32_t>(limit), predicted, state_h, state_c, result.token_ids);
    result.text = tokenizer_->decode(result.token_ids);
    result.prefill_ms = std::chrono::duration<double, std::milli>(after_encoder - start).count();
    result.decode_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - after_encoder)
            .count();
    return result;
}
std::vector<float> TdtPipeline::extract_padded_mel(const float* audio_data, int32_t num_samples,
                                                   int32_t input_sample_rate,
                                                   int32_t& actual_frames) const {
    const float* samples_ptr = audio_data;
    int32_t samples_count = num_samples;
    std::vector<float> resampled;
    if (input_sample_rate > 0 && input_sample_rate != config_.sample_rate) {
        std::cerr << "[tdt] Resampling audio from " << input_sample_rate << " Hz to "
                  << config_.sample_rate << " Hz" << std::endl;
        resampled =
            tdt::resample_linear(audio_data, num_samples, input_sample_rate, config_.sample_rate);
        samples_ptr = resampled.data();
        samples_count = static_cast<int32_t>(resampled.size());
    }

    tdt::MelResult mel = tdt::extract_tdt_mel_spectrogram(
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

std::vector<float> TdtPipeline::run_encoder(const std::vector<float>& mel, int32_t actual_frames) {
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
            const int32_t k_begin =
                config_.att_context_left < 0 ? 0 : std::max(0, q - config_.att_context_left);
            const int32_t k_end =
                config_.att_context_right < 0
                    ? actual_encoder_frames - 1
                    : std::min(actual_encoder_frames - 1, q + config_.att_context_right);
            for (int32_t k = k_begin; k <= k_end; ++k)
                encoder_mask[static_cast<std::size_t>(q) * max_frames + k] = 0.0F;
        }
        inputs["encoder_mask"] =
            make_tensor(encoder_mask.data(), {1, max_frames, max_frames}, DType::kFloat32);
    }

    auto outputs = encoder_->forward(inputs);
    validate_float_outputs(outputs);
    auto it = outputs.find("encoder_output");
    if (it == outputs.end())
        throw std::runtime_error("TdtPipeline: encoder missing 'encoder_output'");
    if (it->second.numel() <= 0 || it->second.numel() % config_.encoder_hidden_size != 0)
        throw std::runtime_error("TdtPipeline: encoder output size does not match config");
    const auto frames = infer_encoder_frames(it->second, config_.encoder_hidden_size);
    const auto valid_frames = std::min(
        frames, std::max(1, subsampled_frame_count(actual_frames, config_.causal_downsampling)));
    const auto count = static_cast<std::size_t>(valid_frames) * config_.encoder_hidden_size;
    const auto* src = static_cast<const float*>(it->second.data);
    return std::vector<float>(src, src + count);
}

void TdtPipeline::decode_encoder_frames(const std::vector<float>& encoder_output,
                                        int32_t frame_count, int32_t token_limit,
                                        std::vector<float>& pred_output,
                                        std::vector<float>& state_h, std::vector<float>& state_c,
                                        std::vector<int32_t>& emitted) {
    int32_t frame = 0;
    while (frame < frame_count && static_cast<int32_t>(emitted.size()) < token_limit) {
        const float* enc_ptr =
            encoder_output.data() + static_cast<std::size_t>(frame) * config_.encoder_hidden_size;
        const float* enc_frame = enc_ptr;
        int32_t symbols_on_frame = 0;
        while (static_cast<int32_t>(emitted.size()) < token_limit) {
            const auto logits = run_joint(enc_frame, pred_output.data());
            const auto token_count = static_cast<std::size_t>(config_.blank_id + 1);
            if (logits.size() != token_count + config_.duration_values.size())
                throw std::runtime_error("TdtPipeline: joint output sizes do not match config");
            const std::vector<float> token_logits(logits.begin(), logits.begin() + token_count);
            const std::vector<float> duration_logits(logits.begin() + token_count, logits.end());
            const int32_t token = argmax_token(token_logits);
            const int32_t duration_index = argmax_token(duration_logits);
            if (token < 0)
                break;

            const auto decision = make_tdt_greedy_decision(
                token, duration_index, config_.duration_values, config_.blank_id);
            if (decision.emit_token) {
                emitted.push_back(token);
                ++symbols_on_frame;
                pred_output = run_predictor(token, state_h, state_c);
            }
            frame += decision.frame_advance;
            if (decision.frame_advance > 0)
                break;
            if (symbols_on_frame >= config_.max_symbols_per_step) {
                ++frame;
                break;
            }
        }
    }
}

std::vector<float> TdtPipeline::run_predictor(int32_t token_id, std::vector<float>& state_h,
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
    validate_float_outputs(outputs);
    auto pred_it = outputs.find("pred_output");
    if (pred_it == outputs.end())
        throw std::runtime_error("TdtPipeline: predictor missing 'pred_output'");

    if (pred_it->second.numel() != static_cast<size_t>(config_.pred_hidden_size))
        throw std::runtime_error("TdtPipeline: predictor output size does not match config");
    for (int32_t layer = 0; layer < config_.pred_num_layers; ++layer) {
        const std::string suffix = "_" + std::to_string(layer);
        auto h_it = outputs.find("next_h" + suffix);
        auto c_it = outputs.find("next_c" + suffix);
        if (h_it == outputs.end() || c_it == outputs.end())
            throw std::runtime_error("TdtPipeline: predictor missing next state outputs");
        if (h_it->second.numel() != static_cast<size_t>(config_.pred_hidden_size) ||
            c_it->second.numel() != static_cast<size_t>(config_.pred_hidden_size))
            throw std::runtime_error(
                "TdtPipeline: predictor state output size does not match config");
        std::memcpy(state_h.data() + static_cast<std::size_t>(layer) * layer_stride,
                    h_it->second.data, layer_stride * sizeof(float));
        std::memcpy(state_c.data() + static_cast<std::size_t>(layer) * layer_stride,
                    c_it->second.data, layer_stride * sizeof(float));
    }

    const auto* pred = static_cast<const float*>(pred_it->second.data);
    return std::vector<float>(pred, pred + config_.pred_hidden_size);
}

std::vector<float> TdtPipeline::run_joint(const float* encoder_frame, const float* pred_output) {
    TensorMap inputs;
    inputs["encoder_frame"] = make_tensor(const_cast<float*>(encoder_frame),
                                          {1, config_.encoder_hidden_size}, DType::kFloat32);
    inputs["pred_output"] = make_tensor(const_cast<float*>(pred_output),
                                        {1, config_.pred_hidden_size}, DType::kFloat32);
    auto outputs = joint_->forward(inputs);
    validate_float_outputs(outputs);
    auto token_it = outputs.find("token_logits");
    auto duration_it = outputs.find("duration_logits");
    if (token_it == outputs.end() || duration_it == outputs.end())
        throw std::runtime_error("TdtPipeline: joint missing token_logits or duration_logits");
    if (token_it->second.numel() != static_cast<size_t>(config_.blank_id) + 1 ||
        duration_it->second.numel() != config_.duration_values.size())
        throw std::runtime_error("TdtPipeline: joint output sizes do not match config");
    const auto* tokens = static_cast<const float*>(token_it->second.data);
    const auto* durations = static_cast<const float*>(duration_it->second.data);
    std::vector<float> logits(tokens, tokens + token_it->second.numel());
    logits.insert(logits.end(), durations, durations + duration_it->second.numel());
    return logits;
}
} // namespace trtmc::parakeet_tdt
