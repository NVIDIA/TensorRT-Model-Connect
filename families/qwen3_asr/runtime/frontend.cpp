/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_asr/runtime/frontend.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace trtmc::qwen3_asr {
namespace {

constexpr double kPi = 3.14159265358979323846;

using Complex = std::complex<double>;

void fft_inplace(std::vector<Complex>& values, bool inverse) {
    const std::size_t size = values.size();
    for (std::size_t i = 1, j = 0; i < size; ++i) {
        std::size_t bit = size >> 1;
        for (; (j & bit) != 0; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(values[i], values[j]);
    }
    for (std::size_t length = 2; length <= size; length <<= 1) {
        const double angle = (inverse ? 2.0 : -2.0) * kPi / static_cast<double>(length);
        const Complex step(std::cos(angle), std::sin(angle));
        const std::size_t half = length / 2;
        for (std::size_t offset = 0; offset < size; offset += length) {
            Complex twiddle(1.0, 0.0);
            for (std::size_t i = 0; i < half; ++i) {
                const Complex even = values[offset + i];
                const Complex odd = values[offset + i + half] * twiddle;
                values[offset + i] = even + odd;
                values[offset + i + half] = even - odd;
                twiddle *= step;
            }
        }
    }
    if (inverse) {
        const double scale = 1.0 / static_cast<double>(size);
        for (auto& value : values)
            value *= scale;
    }
}

// Power spectrum of an n-point real frame for any n, by Bluestein's chirp-z transform over a
// power-of-two FFT, in double precision.
class PowerSpectrum {
  public:
    explicit PowerSpectrum(std::int32_t n) : n_(n) {
        if (n_ < 2)
            throw std::invalid_argument("qwen3_asr: FFT size must be at least 2");
        size_ = 1;
        while (size_ < static_cast<std::size_t>(2 * n_ - 1))
            size_ <<= 1;
        chirp_.resize(static_cast<std::size_t>(n_));
        kernel_.assign(size_, Complex(0.0, 0.0));
        const auto period = 2 * static_cast<std::int64_t>(n_);
        for (std::int32_t i = 0; i < n_; ++i) {
            // e^{i pi j^2 / n} is periodic in j^2 with period 2n; reducing keeps the angle exact.
            const auto square = (static_cast<std::int64_t>(i) * i) % period;
            const double angle = kPi * static_cast<double>(square) / static_cast<double>(n_);
            chirp_[static_cast<std::size_t>(i)] = Complex(std::cos(angle), -std::sin(angle));
            const Complex kernel(std::cos(angle), std::sin(angle));
            kernel_[static_cast<std::size_t>(i)] = kernel;
            if (i != 0)
                kernel_[size_ - static_cast<std::size_t>(i)] = kernel;
        }
        fft_inplace(kernel_, false);
        work_.resize(size_);
    }

    void compute(const float* frame, double* power) {
        std::fill(work_.begin(), work_.end(), Complex(0.0, 0.0));
        for (std::int32_t i = 0; i < n_; ++i) {
            const auto index = static_cast<std::size_t>(i);
            work_[index] = static_cast<double>(frame[i]) * chirp_[index];
        }
        fft_inplace(work_, false);
        for (std::size_t i = 0; i < size_; ++i)
            work_[i] *= kernel_[i];
        fft_inplace(work_, true);
        for (std::int32_t k = 0; k <= n_ / 2; ++k) {
            const auto index = static_cast<std::size_t>(k);
            power[index] = std::norm(work_[index] * chirp_[index]);
        }
    }

  private:
    std::int32_t n_;
    std::size_t size_{0};
    std::vector<Complex> chirp_;
    std::vector<Complex> kernel_;
    std::vector<Complex> work_;
};

void validate(const FrontendConfig& config, const MelFilterbank& filters) {
    if (config.n_fft < 4 || config.n_fft % 2 != 0 || config.hop_length <= 0 ||
        config.min_samples < 0 || config.chunk_frames <= 0 || config.window_chunks <= 0)
        throw std::invalid_argument("qwen3_asr: invalid front-end geometry");
    if (filters.n_freq != config.n_fft / 2 + 1 || filters.n_mel <= 0 ||
        filters.weights.size() !=
            static_cast<std::size_t>(filters.n_freq) * static_cast<std::size_t>(filters.n_mel))
        throw std::invalid_argument("qwen3_asr: mel filterbank does not match n_fft");
}

} // namespace

std::int32_t post_cnn_length(std::int32_t frames) {
    if (frames < 0)
        throw std::invalid_argument("qwen3_asr: negative frame count");
    for (int stage = 0; stage < 3; ++stage)
        frames = frames > 0 ? (frames - 1) / 2 + 1 : 0;
    return frames;
}

std::int64_t audio_token_count(std::int64_t samples, const FrontendConfig& config) {
    if (samples <= 0 || config.hop_length <= 0 || config.chunk_frames <= 0)
        throw std::invalid_argument("qwen3_asr: invalid clip length or front-end geometry");
    const std::int64_t frames = std::max(samples, config.min_samples) / config.hop_length;
    const std::int64_t full = frames / config.chunk_frames;
    const auto rest = static_cast<std::int32_t>(frames % config.chunk_frames);
    return full * post_cnn_length(config.chunk_frames) + post_cnn_length(rest);
}

namespace detail {
std::vector<double> power_spectrum(const std::vector<float>& frame) {
    PowerSpectrum plan(static_cast<std::int32_t>(frame.size()));
    std::vector<double> power(frame.size() / 2 + 1);
    plan.compute(frame.data(), power.data());
    return power;
}
} // namespace detail

LogMel log_mel(const std::vector<float>& mono, const FrontendConfig& config,
               const MelFilterbank& filters) {
    validate(config, filters);
    if (mono.empty())
        throw std::invalid_argument("qwen3_asr: audio is empty");
    std::vector<float> audio = mono;
    if (static_cast<std::int64_t>(audio.size()) < config.min_samples)
        audio.resize(static_cast<std::size_t>(config.min_samples), 0.0F);
    const auto samples = static_cast<std::int64_t>(audio.size());
    const std::int32_t half = config.n_fft / 2;
    if (samples <= half)
        throw std::invalid_argument("qwen3_asr: audio is shorter than the reflection padding");
    if (samples / config.hop_length > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("qwen3_asr: audio is too long");

    // Reflection padding of torch.stft(center=True): index -j reads j, index n-1+j reads n-1-j.
    std::vector<float> padded(static_cast<std::size_t>(samples + 2 * half));
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(padded.size()); ++i) {
        std::int64_t source = i - half;
        if (source < 0)
            source = -source;
        else if (source >= samples)
            source = 2 * (samples - 1) - source;
        padded[static_cast<std::size_t>(i)] = audio[static_cast<std::size_t>(source)];
    }

    std::vector<float> window(static_cast<std::size_t>(config.n_fft));
    for (std::int32_t i = 0; i < config.n_fft; ++i) {
        const double phase = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(config.n_fft);
        window[static_cast<std::size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(phase));
    }

    LogMel result;
    result.n_mel = filters.n_mel;
    result.frames = static_cast<std::int32_t>(samples / config.hop_length);
    result.values.assign(
        static_cast<std::size_t>(result.n_mel) * static_cast<std::size_t>(result.frames), 0.0F);
    PowerSpectrum spectrum(config.n_fft);
    std::vector<float> frame(static_cast<std::size_t>(config.n_fft));
    std::vector<double> power(static_cast<std::size_t>(filters.n_freq));
    std::vector<double> mel(static_cast<std::size_t>(filters.n_mel));
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::int32_t t = 0; t < result.frames; ++t) {
        const auto start =
            static_cast<std::size_t>(t) * static_cast<std::size_t>(config.hop_length);
        for (std::int32_t i = 0; i < config.n_fft; ++i) {
            const auto index = static_cast<std::size_t>(i);
            frame[index] = padded[start + index] * window[index];
        }
        spectrum.compute(frame.data(), power.data());
        std::fill(mel.begin(), mel.end(), 0.0);
        for (std::int32_t f = 0; f < filters.n_freq; ++f) {
            const double value =
                static_cast<double>(static_cast<float>(power[static_cast<std::size_t>(f)]));
            if (value == 0.0)
                continue;
            const float* row = filters.weights.data() + static_cast<std::size_t>(f) * filters.n_mel;
            for (std::int32_t m = 0; m < filters.n_mel; ++m)
                mel[static_cast<std::size_t>(m)] += value * static_cast<double>(row[m]);
        }
        for (std::int32_t m = 0; m < filters.n_mel; ++m) {
            const float energy =
                std::max(static_cast<float>(mel[static_cast<std::size_t>(m)]), 1e-10F);
            const float logged = std::log10(energy);
            result.values[static_cast<std::size_t>(m) * static_cast<std::size_t>(result.frames) +
                          static_cast<std::size_t>(t)] = logged;
            maximum = std::max(maximum, logged);
        }
    }
    const float floor = maximum - 8.0F;
    for (auto& value : result.values)
        value = (std::max(value, floor) + 4.0F) / 4.0F;
    return result;
}

EncoderBatch encoder_batch(const LogMel& mel, const FrontendConfig& config) {
    if (mel.frames <= 0 || mel.n_mel <= 0 ||
        mel.values.size() !=
            static_cast<std::size_t>(mel.n_mel) * static_cast<std::size_t>(mel.frames))
        throw std::invalid_argument("qwen3_asr: invalid log-mel features");
    const std::int32_t chunk = config.chunk_frames;
    const std::int32_t chunks = (mel.frames + chunk - 1) / chunk;
    EncoderBatch batch;
    batch.windows = (chunks + config.window_chunks - 1) / config.window_chunks;
    const auto slots =
        static_cast<std::size_t>(batch.windows) * static_cast<std::size_t>(config.window_chunks);
    batch.features.assign(
        slots * static_cast<std::size_t>(mel.n_mel) * static_cast<std::size_t>(chunk), 0.0F);
    batch.chunk_tokens.assign(slots, 0);
    for (std::int32_t c = 0; c < chunks; ++c) {
        const std::int32_t first = c * chunk;
        const std::int32_t valid = std::min(chunk, mel.frames - first);
        const std::int32_t tokens = post_cnn_length(valid);
        batch.chunk_tokens[static_cast<std::size_t>(c)] = tokens;
        batch.audio_tokens += tokens;
        for (std::int32_t m = 0; m < mel.n_mel; ++m) {
            const float* source =
                mel.values.data() + static_cast<std::size_t>(m) * mel.frames + first;
            float* target =
                batch.features.data() +
                (static_cast<std::size_t>(c) * mel.n_mel + static_cast<std::size_t>(m)) * chunk;
            std::copy(source, source + valid, target);
        }
    }
    return batch;
}

} // namespace trtmc::qwen3_asr
