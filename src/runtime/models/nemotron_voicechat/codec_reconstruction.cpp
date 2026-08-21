/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/codec_reconstruction.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {
namespace nemotron_voicechat {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int32_t kIstftPadding = (kCodecFftSize - kCodecHopLength) / 2; // six samples
constexpr int32_t kStreamingTrim = 2 * kCodecHopLength;                  // eight samples

float periodic_hann(int32_t sample) {
    return static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(sample) /
                                                    static_cast<double>(kCodecFftSize))));
}

float bounded_magnitude(float logit) {
    // Exact stable form of Speech commit
    // 097dfe9e2f55baf653b83035868bdc89849f1b47,
    // nemo/collections/speechlm2/modules/ear_tts_vae_codec.py::Latent2Wav.forward:
    //   100 * exp(-softplus(-logit + log(100)))
    constexpr float kMaximum = 100.0F;
    constexpr float kLogMaximum = 4.605170185988091368F;
    const float shifted = logit - kLogMaximum;
    if (shifted >= 0.0F) {
        const float exp_negative = std::exp(-shifted);
        return kMaximum / (1.0F + exp_negative);
    }
    const float exp_positive = std::exp(shifted);
    return kMaximum * exp_positive / (1.0F + exp_positive);
}

std::array<float, kCodecFftSize> irfft16(const CodecReconstruction::Spectrum& spectrum) {
    std::array<float, kCodecFftSize> output{};
    for (int32_t sample = 0; sample < kCodecFftSize; ++sample) {
        double value = static_cast<double>(spectrum[0].real());
        value += static_cast<double>(spectrum[kCodecSpectralBins - 1].real()) *
                 ((sample & 1) == 0 ? 1.0 : -1.0);
        for (int32_t bin = 1; bin < kCodecSpectralBins - 1; ++bin) {
            const double angle = 2.0 * kPi * static_cast<double>(bin) *
                                 static_cast<double>(sample) / static_cast<double>(kCodecFftSize);
            const double real = static_cast<double>(spectrum[static_cast<std::size_t>(bin)].real());
            const double imag = static_cast<double>(spectrum[static_cast<std::size_t>(bin)].imag());
            value += 2.0 * (real * std::cos(angle) - imag * std::sin(angle));
        }
        output[static_cast<std::size_t>(sample)] =
            static_cast<float>(value / static_cast<double>(kCodecFftSize));
    }
    return output;
}

} // namespace

const std::array<CodecCacheBinding, kCodecConvBlocks>& codec_cache_bindings() {
    static constexpr std::array<CodecCacheBinding, kCodecConvBlocks> bindings = {{
        {"codec_cache_in_0", "codec_cache_out_0", 1536},
        {"codec_cache_in_1", "codec_cache_out_1", 1536},
        {"codec_cache_in_2", "codec_cache_out_2", 1536},
        {"codec_cache_in_3", "codec_cache_out_3", 768},
        {"codec_cache_in_4", "codec_cache_out_4", 768},
        {"codec_cache_in_5", "codec_cache_out_5", 768},
        {"codec_cache_in_6", "codec_cache_out_6", 384},
        {"codec_cache_in_7", "codec_cache_out_7", 384},
        {"codec_cache_in_8", "codec_cache_out_8", 384},
    }};
    return bindings;
}

CodecCausalCache::CodecCausalCache() {
    for (std::size_t block = 0; block < current_.size(); ++block) {
        const std::size_t elements =
            static_cast<std::size_t>(kCodecConvCacheChannels[block]) * kCodecConvCacheWidth;
        current_[block].resize(elements);
        next_[block].resize(elements);
    }
    reset();
}

std::size_t CodecCausalCache::checked_block(int32_t block) {
    if (block < 0 || block >= kCodecConvBlocks) {
        throw std::out_of_range("VoiceChat codec cache block out of range: " +
                                std::to_string(block));
    }
    return static_cast<std::size_t>(block);
}

void CodecCausalCache::reset() {
    for (auto& cache : current_)
        std::fill(cache.begin(), cache.end(), 0.0F);
    for (auto& cache : next_)
        std::fill(cache.begin(), cache.end(), 0.0F);
}

void CodecCausalCache::commit() {
    current_.swap(next_);
    for (auto& cache : next_)
        std::fill(cache.begin(), cache.end(), 0.0F);
}

const float* CodecCausalCache::current_data(int32_t block) const {
    return current_[checked_block(block)].data();
}

float* CodecCausalCache::next_data(int32_t block) {
    return next_[checked_block(block)].data();
}

std::size_t CodecCausalCache::element_count(int32_t block) const {
    return current_[checked_block(block)].size();
}

CodecReconstruction::CodecReconstruction() {
    reset();
}

CodecReconstruction::Spectrum CodecReconstruction::decode_spectrum(const float* spectral_params,
                                                                   int32_t spectral_frames,
                                                                   int32_t frame) {
    Spectrum spectrum{};
    for (int32_t bin = 0; bin < kCodecSpectralBins; ++bin) {
        const float magnitude_logit =
            spectral_params[static_cast<std::size_t>(bin) * spectral_frames + frame];
        const float phase =
            spectral_params[static_cast<std::size_t>(kCodecSpectralBins + bin) * spectral_frames +
                            frame];
        if (!std::isfinite(magnitude_logit) || !std::isfinite(phase)) {
            throw std::invalid_argument("VoiceChat spectral parameters must be finite");
        }
        const float magnitude = bounded_magnitude(magnitude_logit);
        const float real = magnitude * std::cos(phase);
        const float imag =
            (bin == 0 || bin == kCodecSpectralBins - 1) ? 0.0F : magnitude * std::sin(phase);
        spectrum[static_cast<std::size_t>(bin)] = {real, imag};
    }
    return spectrum;
}

std::vector<float> CodecReconstruction::inverse_stft(const std::vector<Spectrum>& spectra) {
    if (spectra.empty())
        return {};

    const std::size_t output_size =
        static_cast<std::size_t>(spectra.size() - 1) * kCodecHopLength + kCodecFftSize;
    std::vector<float> overlap_add(output_size, 0.0F);
    std::vector<float> envelope(output_size, 0.0F);

    std::array<float, kCodecFftSize> window{};
    for (int32_t sample = 0; sample < kCodecFftSize; ++sample)
        window[static_cast<std::size_t>(sample)] = periodic_hann(sample);

    for (std::size_t frame = 0; frame < spectra.size(); ++frame) {
        const auto time = irfft16(spectra[frame]);
        const std::size_t offset = frame * kCodecHopLength;
        for (int32_t sample = 0; sample < kCodecFftSize; ++sample) {
            const float win = window[static_cast<std::size_t>(sample)];
            // Exact scalar form of ear_tts_vae_codec.py::spec_to_wav at the
            // Speech commit above when constrain_value_range=True:
            // torch.where(ifft >= 0, minimum(ifft, window),
            //             maximum(ifft, -window)).
            const float constrained = std::clamp(time[static_cast<std::size_t>(sample)], -win, win);
            overlap_add[offset + static_cast<std::size_t>(sample)] += constrained * win;
            envelope[offset + static_cast<std::size_t>(sample)] += win * win;
        }
    }

    if (output_size <= 2 * static_cast<std::size_t>(kIstftPadding))
        return {};
    std::vector<float> waveform(output_size - 2 * kIstftPadding);
    for (std::size_t index = kIstftPadding; index < output_size - kIstftPadding; ++index) {
        const float normalization = envelope[index];
        if (!(normalization > 1e-11F))
            throw std::runtime_error("VoiceChat ISTFT window envelope contains a zero");
        waveform[index - kIstftPadding] = overlap_add[index] / normalization;
    }
    return waveform;
}

std::vector<float> CodecReconstruction::trim_streaming_padding(std::vector<float> waveform) {
    if (waveform.size() < 2 * static_cast<std::size_t>(kStreamingTrim))
        throw std::runtime_error("VoiceChat ISTFT output is shorter than streaming padding");
    return std::vector<float>(waveform.begin() + kStreamingTrim, waveform.end() - kStreamingTrim);
}

std::vector<float> CodecReconstruction::push(const float* spectral_params, int32_t codec_frames) {
    if (codec_frames <= 0)
        throw std::invalid_argument("VoiceChat codec_frames must be positive");
    if (spectral_params == nullptr)
        throw std::invalid_argument("VoiceChat spectral_params must not be null");

    const int32_t spectral_frames = codec_frames * kCodecSpectralFramesPerFrame;
    std::vector<Spectrum> spectra;
    spectra.reserve(static_cast<std::size_t>(kCodecSpectralCacheFrames + spectral_frames));
    spectra.insert(spectra.end(), spectral_cache_.begin(), spectral_cache_.end());
    for (int32_t frame = 0; frame < spectral_frames; ++frame)
        spectra.push_back(decode_spectrum(spectral_params, spectral_frames, frame));

    std::copy_n(spectra.end() - kCodecSpectralCacheFrames, kCodecSpectralCacheFrames,
                spectral_cache_.begin());
    active_ = true;

    auto waveform = trim_streaming_padding(inverse_stft(spectra));
    const std::size_t expected = static_cast<std::size_t>(codec_frames) * kCodecFrameSamples;
    if (waveform.size() != expected) {
        throw std::runtime_error("VoiceChat codec produced an unexpected waveform length");
    }
    return waveform;
}

std::vector<float> CodecReconstruction::push(const std::vector<float>& spectral_params,
                                             int32_t codec_frames) {
    if (codec_frames <= 0)
        throw std::invalid_argument("VoiceChat codec_frames must be positive");
    const std::size_t expected = static_cast<std::size_t>(kCodecSpectralChannels) * codec_frames *
                                 kCodecSpectralFramesPerFrame;
    if (spectral_params.size() != expected) {
        throw std::invalid_argument("VoiceChat spectral_params size does not match codec_frames");
    }
    return push(spectral_params.data(), codec_frames);
}

std::vector<float> CodecReconstruction::flush() {
    if (!active_)
        return {};

    std::vector<Spectrum> spectra;
    spectra.reserve(kCodecSpectralCacheFrames + kCodecFlushFrames);
    spectra.insert(spectra.end(), spectral_cache_.begin(), spectral_cache_.end());
    spectra.resize(kCodecSpectralCacheFrames + kCodecFlushFrames); // zero spectra on the right
    auto tail = trim_streaming_padding(inverse_stft(spectra));
    reset();
    if (tail.size() != static_cast<std::size_t>(kCodecFlushFrames * kCodecHopLength)) {
        throw std::runtime_error("VoiceChat codec flush produced an unexpected tail length");
    }
    return tail;
}

void CodecReconstruction::reset() {
    for (auto& spectrum : spectral_cache_)
        spectrum.fill({0.0F, 0.0F});
    active_ = false;
}

} // namespace nemotron_voicechat
} // namespace trtmc
