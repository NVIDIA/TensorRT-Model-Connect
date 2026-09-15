/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace trtmc::cosyvoice3 {
namespace {
constexpr double pi = 3.14159265358979323846;
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
std::vector<float> coefficient(const nlohmann::json& data, const char* name, size_t count) {
    auto values = data.at(name).get<std::vector<float>>();
    require(values.size() == count, "CosyVoice3 invalid preprocessing coefficient shape");
    for (auto x : values)
        require(std::isfinite(x), "CosyVoice3 nonfinite coefficient");
    return values;
}
int reflect(int i, int length) {
    while (i < 0 || i >= length)
        i = i < 0 ? -i : 2 * length - i - 2;
    return i;
}
// Direct real DFT: ordinary CPU signal processing, no learned execution.
// Coefficients are computed once per transform, not once per audio frame.
struct Spectrum {
    int n, bins;
    std::vector<double> real, imag;
    explicit Spectrum(int size)
        : n(size), bins(size / 2 + 1), real(size * bins), imag(size * bins) {
        for (int k = 0; k < bins; ++k)
            for (int t = 0; t < n; ++t) {
                real[k * n + t] = std::cos(2 * pi * k * t / n);
                imag[k * n + t] = -std::sin(2 * pi * k * t / n);
            }
    }
    std::vector<float> power(const std::vector<float>& input) const {
        std::vector<float> result(bins);
        for (int k = 0; k < bins; ++k) {
            double re = 0, im = 0;
            for (int t = 0; t < n; ++t) {
                re += input[t] * real[k * n + t];
                im += input[t] * imag[k * n + t];
            }
            result[k] = static_cast<float>(re * re + im * im);
        }
        return result;
    }
};
std::vector<float> project(const std::vector<float>& spectrum, const std::vector<float>& bank,
                           int channels) {
    const size_t bins = spectrum.size();
    std::vector<float> result(channels);
    for (int c = 0; c < channels; ++c) {
        double sum = 0;
        for (size_t k = 0; k < bins; ++k)
            sum += double(bank[c * bins + k]) * spectrum[k];
        result[c] = static_cast<float>(sum);
    }
    return result;
}
} // namespace

std::vector<float> resample_reference(const std::vector<float>& samples, int source, int target) {
    require(source >= 16000 && source <= 192000 && target > 0,
            "CosyVoice3 unsupported reference sample rate");
    if (source == target)
        return samples;
    int divisor = std::gcd(source, target), orig = source / divisor, next = target / divisor;
    double base = std::min(orig, next) * .99;
    int width = static_cast<int>(std::ceil(6 * orig / base));
    const size_t length = (samples.size() * target + source - 1) / source;
    std::vector<float> result(length);
    // Equivalent polyphase sinc/Hann kernel to transforms.Resample's FP32 cache.
    for (int phase = 0; phase < next; ++phase) {
        double offset = static_cast<float>(-float(phase) / next);
        const int begin = std::max(0, int(std::ceil(width + orig * (-6 / base - offset))));
        const int end =
            std::min(2 * width + orig, int(std::floor(width + orig * (6 / base - offset))) + 1);
        std::vector<float> kernel(end - begin);
        for (int k = begin; k < end; ++k) {
            // torch creates the phase offsets in default float32 before promotion.
            double t = std::clamp((offset + double(k - width) / orig) * base, -6., 6.);
            double window = std::pow(std::cos(t * pi / 12), 2);
            double angle = t * pi;
            kernel[k - begin] = static_cast<float>((angle == 0 ? 1 : std::sin(angle) / angle) *
                                                   window * base / orig);
        }
        for (size_t j = phase; j < length; j += next) {
            int64_t start = int64_t(j / next) * orig - width + begin;
            double sum = 0;
            for (int k = 0; k < int(kernel.size()); ++k) {
                int64_t i = start + k;
                if (i >= 0 && i < int64_t(samples.size()))
                    sum += double(samples[i]) * kernel[k];
            }
            result[j] = static_cast<float>(sum);
        }
    }
    return result;
}

ReferenceFeatures reference_features(const AudioReference& request, const nlohmann::json& data) {
    require(request.sample_rate >= 16000 && request.sample_rate <= 192000,
            "CosyVoice3 reference sample rate must be 16000 through 192000 Hz");
    double seconds = double(request.samples.size()) / request.sample_rate;
    require(seconds >= .1 && seconds <= 30, "CosyVoice3 reference must be 0.1 through 30 seconds");
    for (float value : request.samples)
        require(std::isfinite(value), "CosyVoice3 nonfinite reference audio");
    require(data.at("schema") == 1, "CosyVoice3 unsupported frontend coefficients");
    auto s16 = resample_reference(request.samples, request.sample_rate, 16000);
    auto s24 = resample_reference(request.samples, request.sample_rate, 24000);
    auto kw = coefficient(data, "kaldi_window", 400), ww = coefficient(data, "whisper_window", 400);
    auto aw = coefficient(data, "acoustic_window", 1920);
    auto kb = coefficient(data, "kaldi_bank", 80 * 257),
         wb = coefficient(data, "whisper_bank", 128 * 201);
    auto ab = coefficient(data, "acoustic_bank", 80 * 961);
    ReferenceFeatures out;
    out.speaker_frames = 1 + (int(s16.size()) - 400) / 160;
    out.token_frames = int(s16.size()) / 160;
    out.mel_frames = (int(s24.size()) + 1440 - 1920) / 480 + 1;
    Spectrum kaldi(512), whisper(400), acoustic(1920);
    out.speaker.resize(out.speaker_frames * 80);
    out.tokens.resize(out.token_frames * 128);
    out.mel.resize(out.mel_frames * 80);
    for (int f = 0; f < out.speaker_frames; ++f) {
        std::vector<float> frame(512);
        double mean = 0;
        for (int t = 0; t < 400; ++t)
            mean += s16[f * 160 + t];
        float dc = static_cast<float>(mean / 400);
        for (int t = 0; t < 400; ++t) {
            float current = s16[f * 160 + t] - dc,
                  previous = s16[f * 160 + std::max(0, t - 1)] - dc;
            frame[t] = (current - .97f * previous) * kw[t];
        }
        auto values = project(kaldi.power(frame), kb, 80);
        for (int c = 0; c < 80; ++c)
            out.speaker[f * 80 + c] =
                std::log(std::max(values[c], std::numeric_limits<float>::epsilon()));
    }
    for (int c = 0; c < 80; ++c) {
        double sum = 0;
        for (int f = 0; f < out.speaker_frames; ++f)
            sum += out.speaker[f * 80 + c];
        float mean = static_cast<float>(sum / out.speaker_frames);
        for (int f = 0; f < out.speaker_frames; ++f)
            out.speaker[f * 80 + c] -= mean;
    }
    for (int f = 0; f < out.token_frames; ++f) {
        std::vector<float> frame(400);
        for (int t = 0; t < 400; ++t)
            frame[t] = s16[reflect(f * 160 + t - 200, int(s16.size()))] * ww[t];
        auto values = project(whisper.power(frame), wb, 128);
        for (int c = 0; c < 128; ++c)
            out.tokens[c * out.token_frames + f] = std::log10(std::max(values[c], 1e-10f));
    }
    float peak = *std::max_element(out.tokens.begin(), out.tokens.end());
    for (auto& x : out.tokens)
        x = (std::max(x, peak - 8) + 4) / 4;
    for (int f = 0; f < out.mel_frames; ++f) {
        std::vector<float> frame(1920);
        for (int t = 0; t < 1920; ++t)
            frame[t] = s24[reflect(f * 480 + t - 720, int(s24.size()))] * aw[t];
        auto power = acoustic.power(frame);
        for (auto& x : power)
            x = std::sqrt(x + 1e-9f);
        auto values = project(power, ab, 80);
        for (int c = 0; c < 80; ++c)
            out.mel[f * 80 + c] = std::log(std::max(values[c], 1e-5f));
    }
    return out;
}
} // namespace trtmc::cosyvoice3
