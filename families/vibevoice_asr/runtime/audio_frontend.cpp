/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vibevoice_asr/runtime/audio_frontend.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace trtmc::vibevoice_asr {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::int64_t kZeroCrossings = 32;
constexpr double kKaiserBeta = 8.6;

double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    const double quarter = 0.25 * x * x;
    for (int k = 1; k < 64; ++k) {
        term *= quarter / (static_cast<double>(k) * k);
        sum += term;
        if (term < 1e-17 * sum)
            break;
    }
    return sum;
}

} // namespace

std::vector<float> downmix(const float* interleaved, std::size_t frames, std::uint32_t channels) {
    if (interleaved == nullptr || frames == 0 || channels == 0)
        throw std::invalid_argument("VibeVoice-ASR requires nonempty interleaved PCM");
    std::vector<float> mono(frames);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double sum = 0.0;
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            const float value = interleaved[frame * channels + channel];
            if (!std::isfinite(value))
                throw std::invalid_argument("VibeVoice-ASR PCM must contain only finite samples");
            sum += value;
        }
        mono[frame] = static_cast<float>(sum / channels);
    }
    return mono;
}

std::vector<float> resample(const std::vector<float>& input, std::int32_t from_rate,
                            std::int32_t to_rate) {
    if (from_rate <= 0 || to_rate <= 0)
        throw std::invalid_argument("VibeVoice-ASR sample rates must be positive");
    if (from_rate == to_rate || input.empty())
        return input;
    const std::int64_t divisor = std::gcd<std::int64_t>(from_rate, to_rate);
    const std::int64_t up = to_rate / divisor;
    const std::int64_t down = from_rate / divisor;
    const std::int64_t count = static_cast<std::int64_t>(input.size());
    const std::int64_t outputs = (count * up + down - 1) / down;
    // Low-pass at the lower Nyquist rate, designed on the up-sampled grid.
    const std::int64_t spacing = std::max(up, down);
    const std::int64_t half_width = kZeroCrossings * spacing;
    const double norm = bessel_i0(kKaiserBeta);
    auto tap = [&](std::int64_t offset) {
        const double x = static_cast<double>(offset) / static_cast<double>(spacing);
        const double sinc = offset == 0 ? 1.0 : std::sin(kPi * x) / (kPi * x);
        const double ratio = static_cast<double>(offset) / static_cast<double>(half_width);
        const double window =
            bessel_i0(kKaiserBeta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio)));
        return sinc * window / norm * static_cast<double>(up) / static_cast<double>(spacing);
    };
    // One tap table per output phase: offsets phase + up * m within the window.
    const std::int64_t taps = 2 * (half_width / up) + 2;
    std::vector<double> table(static_cast<std::size_t>(up * taps), 0.0);
    std::vector<std::int64_t> first(static_cast<std::size_t>(up), 0);
    for (std::int64_t phase = 0; phase < up; ++phase) {
        // Input j contributes to output centre c = n * down + phase with offset c - j * up.
        const std::int64_t lowest = -((half_width - phase) / up); // m with phase - up * m <= hw
        first[static_cast<std::size_t>(phase)] = lowest;
        for (std::int64_t t = 0; t < taps; ++t) {
            const std::int64_t offset = phase - up * (lowest + t);
            if (offset >= -half_width && offset <= half_width)
                table[static_cast<std::size_t>(phase * taps + t)] = tap(offset);
        }
    }
    std::vector<float> output(static_cast<std::size_t>(outputs));
    for (std::int64_t k = 0; k < outputs; ++k) {
        const std::int64_t centre = k * down;
        const std::int64_t base = centre / up;
        const std::int64_t phase = centre % up;
        const double* weights = &table[static_cast<std::size_t>(phase * taps)];
        const std::int64_t start = base + first[static_cast<std::size_t>(phase)];
        double sum = 0.0;
        for (std::int64_t t = 0; t < taps; ++t) {
            const std::int64_t j = start + t;
            if (j >= 0 && j < count)
                sum += weights[t] * input[static_cast<std::size_t>(j)];
        }
        output[static_cast<std::size_t>(k)] = static_cast<float>(sum);
    }
    return output;
}

void normalize_loudness(std::vector<float>& samples, double target_dbfs, double eps) {
    if (samples.empty())
        return;
    double energy = 0.0;
    for (const float value : samples)
        energy += static_cast<double>(value) * value;
    const float rms = static_cast<float>(std::sqrt(energy / static_cast<double>(samples.size())));
    const float gain =
        static_cast<float>(std::pow(10.0, target_dbfs / 20.0)) / (rms + static_cast<float>(eps));
    float peak = 0.0F;
    for (float& value : samples) {
        value *= gain;
        peak = std::max(peak, std::fabs(value));
    }
    if (peak > 1.0F) {
        const float scale = peak + static_cast<float>(eps);
        for (float& value : samples)
            value /= scale;
    }
}

} // namespace trtmc::vibevoice_asr
