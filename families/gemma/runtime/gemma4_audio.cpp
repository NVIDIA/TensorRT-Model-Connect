/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/gemma/runtime/gemma4_audio.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace trtmc {
namespace {

// CPU-side radix-two FFT. No device kernel or platform-specific model policy.
void fft(std::vector<std::complex<double>>& values,
         const std::vector<std::complex<double>>& roots) {
    const auto size = values.size();
    for (std::size_t i = 1, reversed = 0; i < size; ++i) {
        std::size_t bit = size >> 1;
        for (; reversed & bit; bit >>= 1)
            reversed ^= bit;
        reversed ^= bit;
        if (i < reversed)
            std::swap(values[i], values[reversed]);
    }
    for (std::size_t width = 2;; width *= 2) {
        for (std::size_t start = 0; start < size; start += width)
            for (std::size_t i = 0; i < width / 2; ++i) {
                const auto even = values[start + i];
                const auto odd = roots[i * (size / width)] * values[start + i + width / 2];
                values[start + i] = even + odd;
                values[start + i + width / 2] = even - odd;
            }
        if (width == size)
            break;
    }
}

void validate_frontend(const GemmaAudioConfig& c) {
    if (c.waveform) {
        if (c.sample_rate != 16000 || c.mel_bins <= 0 || c.max_frames <= 0)
            throw std::invalid_argument("Gemma4 invalid unified waveform geometry");
        return;
    }
    if (c.sample_rate != 16000 || c.frame_length < 2 || c.hop_length < 1 ||
        c.fft_length < c.frame_length || (c.fft_length & (c.fft_length - 1)) || c.mel_bins <= 0 ||
        c.max_frames <= 0 || c.max_frames % 4 ||
        c.window.size() != static_cast<std::size_t>(c.frame_length) ||
        c.mel_filters.size() != static_cast<std::size_t>(c.fft_length / 2 + 1) * c.mel_bins ||
        c.mean.size() != static_cast<std::size_t>(c.mel_bins) || c.stddev.size() != c.mean.size() ||
        !std::isfinite(c.input_scale) || !std::isfinite(c.preemphasis) ||
        !std::isfinite(c.mel_floor) || c.mel_floor <= 0 ||
        !std::all_of(
            c.mean.begin(), c.mean.end(), [](double v) { return std::isfinite(v); }) ||
        !std::all_of(
            c.stddev.begin(), c.stddev.end(), [](double v) { return std::isfinite(v) && v > 0; }))
        throw std::invalid_argument("Gemma4 invalid native audio frontend configuration");
}

} // namespace

GemmaAudioFeatures gemma4_audio_features(const internal::AudioView& audio,
                                         const GemmaAudioConfig& c) {
    validate_frontend(c);
    if (audio.channels != 1 ||
        audio.sample_rate.value_or(c.sample_rate) != static_cast<uint32_t>(c.sample_rate) ||
        !audio.samples.data() || audio.samples.empty() || audio.samples.size() > INT32_MAX)
        throw std::invalid_argument("Gemma4 requires nonempty 16 kHz mono float PCM");
    for (float sample : audio.samples)
        if (!std::isfinite(sample))
            throw std::invalid_argument("Gemma4 PCM samples must be finite");
    const int64_t samples = static_cast<int64_t>(audio.samples.size());
    if (c.waveform) {
        const int64_t frames = (samples + c.mel_bins - 1) / c.mel_bins;
        if (frames > c.max_frames)
            throw std::invalid_argument("Gemma4 audio exceeds the engine frame capacity");
        GemmaAudioFeatures result;
        result.soft_tokens = static_cast<int>(frames);
        result.mask.assign(c.max_frames, 0);
        std::fill_n(result.mask.begin(), frames, 1);
        result.mel.assign(static_cast<std::size_t>(c.max_frames) * c.mel_bins, 0);
        std::copy(audio.samples.begin(), audio.samples.end(), result.mel.begin());
        return result;
    }
    const int64_t available = samples + c.frame_length / 2 - (int64_t{c.frame_length} + 1);
    if (available < 0)
        throw std::invalid_argument("Gemma4 audio is shorter than one valid analysis frame");
    const int64_t frames = available / c.hop_length + 1;
    if (frames > c.max_frames)
        throw std::invalid_argument("Gemma4 audio exceeds the engine frame capacity");
    GemmaAudioFeatures result;
    result.soft_tokens = static_cast<int>((frames + 3) / 4);
    result.mask.assign(c.max_frames, 0);
    std::fill_n(result.mask.begin(), frames, 1);
    result.mel.assign(static_cast<std::size_t>(c.max_frames) * c.mel_bins, 0);
    const double pi = std::acos(-1.0);
    std::vector<std::complex<double>> roots(c.fft_length / 2);
    for (std::size_t i = 0; i < roots.size(); ++i)
        roots[i] = std::polar(1.0, -2 * pi * i / c.fft_length);
    std::vector<std::complex<double>> frame(c.fft_length);
    std::vector<double> magnitude(c.fft_length / 2 + 1);
    const auto sample_at = [&](int64_t index) {
        if (index < 0 || index >= samples)
            return 0.0F;
        return audio.samples[static_cast<std::size_t>(index)] * static_cast<float>(c.input_scale);
    };
    for (int64_t t = 0; t < frames; ++t) {
        std::fill(frame.begin(), frame.end(), std::complex<double>{});
        const int64_t start = t * c.hop_length - c.frame_length / 2;
        for (int j = 0; j < c.frame_length; ++j) {
            float value = sample_at(start + j);
            if (c.preemphasis > 0) {
                const float preemphasis = static_cast<float>(c.preemphasis);
                if (c.preemphasis_htk)
                    value = j == 0 ? value * (1.0F - preemphasis)
                                   : value - preemphasis * sample_at(start + j - 1);
                else
                    value = sample_at(start + j + 1) - preemphasis * value;
            }
            // The provider rounds waveform/window multiplication to float32
            // before the double-precision FFT and mel accumulation.
            frame[j] = static_cast<double>(value * c.window[j]);
        }
        fft(frame, roots);
        for (std::size_t k = 0; k < magnitude.size(); ++k)
            magnitude[k] = std::abs(frame[k]); // Magnitude, not power spectrum.
        for (int bin = 0; bin < c.mel_bins; ++bin) {
            double mel = 0;
            for (std::size_t k = 0; k < magnitude.size(); ++k)
                mel += magnitude[k] * c.mel_filters[k * c.mel_bins + bin];
            const double normalized = (std::log(mel + c.mel_floor) - c.mean[bin]) / c.stddev[bin];
            result.mel[static_cast<std::size_t>(t) * c.mel_bins + bin] =
                static_cast<float>(normalized);
        }
    }
    return result;
}

GemmaAudioEncoder::GemmaAudioEncoder(std::unique_ptr<ITrtModule> module, GemmaAudioConfig config)
    : module_(std::move(module)), config_(std::move(config)) {
    validate_frontend(config_);
    if (!module_ || !module_->ok() || config_.hidden_size <= 0)
        throw std::runtime_error("Gemma4 invalid native audio engine");
}

std::vector<float> GemmaAudioEncoder::encode(const GemmaAudioFeatures& input) {
    const int output_tokens = config_.waveform ? config_.max_frames : config_.max_frames / 4;
    if (input.mel.size() != static_cast<std::size_t>(config_.max_frames) * config_.mel_bins ||
        input.mask.size() != static_cast<std::size_t>(config_.max_frames))
        throw std::invalid_argument("Gemma4 audio feature geometry does not match its engine");
    TensorMap inputs;
    inputs["input_features"] = {const_cast<float*>(input.mel.data()),
                                {config_.max_frames, config_.mel_bins},
                                DType::kFloat32};
    inputs["input_features_mask"] = {
        const_cast<int32_t*>(input.mask.data()), {config_.max_frames}, DType::kInt32};
    const auto outputs = module_->forward(inputs);
    const auto& features = outputs.at("audio_features");
    const auto& mask = outputs.at("audio_feature_mask");
    if (features.dtype != DType::kFloat32 || mask.dtype != DType::kInt32 ||
        features.numel() != static_cast<std::size_t>(output_tokens) * config_.hidden_size ||
        mask.numel() != static_cast<std::size_t>(output_tokens))
        throw std::runtime_error("Gemma4 audio engine output contract mismatch");
    const auto* values = static_cast<const float*>(features.data);
    const auto* valid = static_cast<const int32_t*>(mask.data);
    std::vector<float> result;
    for (int i = 0; i < output_tokens; ++i)
        if (valid[i])
            result.insert(result.end(), values + i * config_.hidden_size,
                          values + (i + 1) * config_.hidden_size);
    if (result.size() != static_cast<std::size_t>(input.soft_tokens) * config_.hidden_size ||
        !std::all_of(result.begin(), result.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Gemma4 audio engine returned invalid conditioned features");
    return result;
}

} // namespace trtmc
