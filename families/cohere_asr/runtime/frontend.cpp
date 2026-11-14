/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cohere_asr/runtime/frontend.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>

namespace trtmc::cohere_asr {
namespace {
constexpr float kPi = 3.14159265358979323846F;
// Match the CPU float32 normal convention used by the pinned reference:
// MT19937 24-bit uniforms and paired Box-Muller groups of sixteen values.
void normal_block(float* data) {
    for (int j = 0; j < 8; ++j) {
        const float radius = std::sqrt(-2.0F * std::log(1.0F - data[j]));
        const float angle = 2.0F * kPi * data[j + 8];
        data[j] = radius * std::cos(angle);
        data[j + 8] = radius * std::sin(angle);
    }
}
void fft(std::vector<std::complex<double>>& x) {
    constexpr int n = 512;
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(x[i], x[j]);
    }
    for (int length = 2; length <= n; length <<= 1) {
        const double angle = -2.0 * 3.14159265358979323846 / length;
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        for (int begin = 0; begin < n; begin += length) {
            std::complex<double> twiddle(1.0, 0.0);
            for (int k = 0; k < length / 2; ++k) {
                const auto a = x[begin + k], b = x[begin + k + length / 2] * twiddle;
                x[begin + k] = a + b;
                x[begin + k + length / 2] = a - b;
                twiddle *= step;
            }
        }
    }
}
} // namespace
std::vector<float> normal_noise(std::size_t count) {
    if (count < 16)
        throw std::invalid_argument("cohere_asr dither requires at least 16 samples");
    std::mt19937 generator(static_cast<std::uint32_t>(count));
    auto uniform = [&]() {
        return static_cast<float>(generator() & 0xffffffU) * (1.0F / 16777216.0F);
    };
    std::vector<float> result(count);
    for (auto& value : result)
        value = uniform();
    for (std::size_t i = 0; i + 15 < count; i += 16)
        normal_block(result.data() + i);
    if (count % 16) {
        auto* tail = result.data() + count - 16;
        for (int i = 0; i < 16; ++i)
            tail[i] = uniform();
        normal_block(tail);
    }
    return result;
}
std::vector<float> features(const float* samples, std::size_t count,
                            const std::vector<float>& window, const std::vector<float>& filters) {
    if (!samples || count < 512 || count > 560000 || window.size() != 400 ||
        filters.size() != 128 * 257)
        throw std::invalid_argument(
            "cohere_asr requires 512..560000 samples and its checkpoint frontend buffers");
    auto waveform = normal_noise(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(samples[i]))
            throw std::invalid_argument("cohere_asr audio contains non-finite samples");
        waveform[i] = samples[i] + 1e-5F * waveform[i];
    }
    for (std::size_t i = count - 1; i > 0; --i)
        waveform[i] -= 0.97F * waveform[i - 1];
    const std::size_t frames = count / 160;
    std::vector<float> result(128 * frames), powers(257);
    std::vector<std::complex<double>> buffer(512);
    for (std::size_t t = 0; t < frames; ++t) {
        for (int k = 0; k < 512; ++k) {
            auto index = static_cast<std::int64_t>(t * 160) + k - 256;
            const float w = k >= 56 && k < 456 ? window[k - 56] : 0.0F;
            const float sample =
                index >= 0 && index < static_cast<std::int64_t>(count) ? waveform[index] : 0.0F;
            buffer[k] = std::complex<double>(sample * w, 0.0);
        }
        fft(buffer);
        for (int k = 0; k < 257; ++k)
            powers[k] = static_cast<float>(std::norm(buffer[k]));
        for (int mel = 0; mel < 128; ++mel) {
            float power = 0.0F;
            for (int k = 0; k < 257; ++k)
                power += filters[mel * 257 + k] * powers[k];
            result[mel * frames + t] = std::log(power + 0x1p-24F);
        }
    }
    for (int mel = 0; mel < 128; ++mel) {
        double mean = 0.0, variance = 0.0;
        for (std::size_t t = 0; t < frames; ++t)
            mean += result[mel * frames + t];
        mean /= frames;
        for (std::size_t t = 0; t < frames; ++t) {
            const double d = result[mel * frames + t] - mean;
            variance += d * d;
        }
        const double stddev = std::sqrt(variance / (frames - 1)) + 1e-5;
        for (std::size_t t = 0; t < frames; ++t)
            result[mel * frames + t] =
                static_cast<float>((result[mel * frames + t] - mean) / stddev);
    }
    return result;
}
} // namespace trtmc::cohere_asr
