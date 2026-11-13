/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_frontend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace trtmc::nemotron_omni {
namespace {
constexpr int kFft = 512;
constexpr int kHop = 160;
constexpr int kWindow = 400;
constexpr int kBins = 257;
constexpr int kMel = 128;
constexpr double kPi = 3.14159265358979323846;

// CPU reference-style radix-two FFT, not a GPU specialization or custom kernel.
void fft(std::array<std::complex<double>, kFft>& x) {
    for (int i = 1, j = 0; i < kFft; ++i) {
        int bit = kFft >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(x[i], x[j]);
    }
    for (int length = 2; length <= kFft; length <<= 1) {
        const auto step = std::polar(1.0, -2.0 * kPi / length);
        for (int start = 0; start < kFft; start += length) {
            std::complex<double> phase{1.0, 0.0};
            for (int j = 0; j < length / 2; ++j) {
                const auto a = x[start + j];
                const auto b = phase * x[start + j + length / 2];
                x[start + j] = a + b;
                x[start + j + length / 2] = a - b;
                phase *= step;
            }
        }
    }
}
} // namespace

AudioFeatures audio_features(const std::vector<float>& samples, const std::vector<float>& filters,
                             std::size_t max_frames) {
    const auto frames = samples.size() / kHop;
    if (frames < 2 || frames > max_frames ||
        samples.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Omni audio requires at least 20 ms within the configured capacity");
    }
    if (filters.size() != kMel * kBins ||
        !std::all_of(filters.begin(), filters.end(),
                     [](float x) { return std::isfinite(x) && x >= 0; }) ||
        !std::all_of(samples.begin(), samples.end(), [](float x) { return std::isfinite(x); })) {
        throw std::invalid_argument(
            "Omni audio requires finite PCM and a complete nonnegative mel bank");
    }
    std::array<float, kFft> window{};
    for (int i = 0; i < kWindow; ++i) {
        window[(kFft - kWindow) / 2 + i] =
            0.5F * (1.0F - std::cos(static_cast<float>(i) *
                                    static_cast<float>(2.0 * kPi / (kWindow - 1))));
    }
    AudioFeatures result{std::vector<float>(frames * kMel), frames};
    for (std::size_t frame = 0; frame < frames; ++frame) {
        std::array<std::complex<double>, kFft> spectrum{};
        for (int i = 0; i < kFft; ++i) {
            const auto source = static_cast<long long>(frame * kHop) + i - kFft / 2;
            float sample = 0;
            if (source >= 0 && source < static_cast<long long>(samples.size())) {
                sample = samples[static_cast<std::size_t>(source)];
                if (source > 0)
                    sample -= 0.97F * samples[static_cast<std::size_t>(source - 1)];
            }
            spectrum[i] = static_cast<double>(sample * window[i]);
        }
        fft(spectrum);
        std::array<float, kBins> power{};
        for (int i = 0; i < kBins; ++i)
            power[i] = static_cast<float>(std::norm(spectrum[i]));
        for (int mel = 0; mel < kMel; ++mel) {
            float sum = 0;
            for (int i = 0; i < kBins; ++i)
                sum += filters[mel * kBins + i] * power[i];
            result.values[frame * kMel + mel] = std::log(sum + 5.960464477539063e-08F);
        }
    }
    for (int mel = 0; mel < kMel; ++mel) {
        double mean = 0;
        for (std::size_t t = 0; t < frames; ++t)
            mean += result.values[t * kMel + mel];
        mean /= static_cast<double>(frames);
        double variance = 0;
        for (std::size_t t = 0; t < frames; ++t) {
            const double difference = result.values[t * kMel + mel] - mean;
            variance += difference * difference;
        }
        const double scale = std::sqrt(variance / static_cast<double>(frames - 1)) + 1e-5;
        for (std::size_t t = 0; t < frames; ++t) {
            auto& value = result.values[t * kMel + mel];
            value = static_cast<float>((value - mean) / scale);
        }
    }
    return result;
}

AudioPositions audio_positions(int tokens, int hidden, int heads) {
    if (tokens < 1 || tokens > 8192 || hidden < 2 || hidden % 2 || heads < 1 || hidden > 16384 ||
        heads > 256 || hidden % heads) {
        throw std::invalid_argument("Invalid Omni acoustic position dimensions");
    }
    AudioPositions result;
    result.values.resize(static_cast<std::size_t>(2 * tokens - 1) * hidden);
    result.indices.resize(static_cast<std::size_t>(heads) * tokens * tokens);
    for (int row = 0; row < 2 * tokens - 1; ++row) {
        for (int channel = 0; channel < hidden; channel += 2) {
            const float frequency = 1.0F / std::pow(10000.0F, static_cast<float>(channel) / hidden);
            const float angle = static_cast<float>(tokens - 1 - row) * frequency;
            result.values[static_cast<std::size_t>(row) * hidden + channel] = std::sin(angle);
            result.values[static_cast<std::size_t>(row) * hidden + channel + 1] = std::cos(angle);
        }
    }
    for (int head = 0; head < heads; ++head)
        for (int row = 0; row < tokens; ++row)
            for (int col = 0; col < tokens; ++col)
                result.indices[(static_cast<std::size_t>(head) * tokens + row) * tokens + col] =
                    tokens - 1 - row + col;
    return result;
}
} // namespace trtmc::nemotron_omni
