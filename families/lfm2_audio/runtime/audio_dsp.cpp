/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_audio/runtime/audio_dsp.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace trtmc::lfm2_audio {
namespace {
constexpr double kPi = 3.1415926535897932384626433832795;
using Complex = std::complex<double>;

// Mixed-radix CPU FFT for signal framing only. Learned model computation is
// represented in the family-owned TensorRT networks.
void fft(std::vector<Complex>& data, bool inverse) {
    const std::size_t n = data.size();
    if (n <= 1)
        return;
    std::size_t radix = 2;
    while (radix < n && n % radix != 0)
        ++radix;
    const std::size_t width = n / radix;
    std::vector<std::vector<Complex>> parts(radix, std::vector<Complex>(width));
    for (std::size_t r = 0; r < radix; ++r) {
        for (std::size_t j = 0; j < width; ++j)
            parts[r][j] = data[j * radix + r];
        fft(parts[r], inverse);
    }
    for (std::size_t k = 0; k < n; ++k) {
        Complex result{};
        for (std::size_t r = 0; r < radix; ++r) {
            const double angle = (inverse ? 2 : -2) * kPi * double(k * r) / double(n);
            result += parts[r][k % width] * std::polar(1.0, angle);
        }
        data[k] = result;
    }
}

void validate_audio(const float* audio, std::size_t count, std::int32_t rate) {
    if (audio == nullptr || count == 0 || rate <= 0 || rate > 384000)
        throw std::invalid_argument(
            "LFM audio requires nonempty mono audio and a valid sample rate");
    if (!std::all_of(audio, audio + count, [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("LFM audio samples must be finite");
}

// Fixed-order blocked FP32 reduction for the reference preprocessing contract.
// Keep four independent groups of four lanes, with bounded partial sums. A
// serial sum drifts on the constant log floor and normalization magnifies that
// drift by 1 / 1e-5. The final masked frame participates as zero in this sum.
float feature_sum(const std::vector<float>& values) {
    const auto blocks = values.size() / 16;
    std::size_t ceil_log2 = 0;
    for (auto n = blocks > 0 ? blocks - 1 : 0; n; n >>= 1)
        ++ceil_log2;
    const auto shift = std::max<std::size_t>(4, ceil_log2 / 4);
    const std::size_t block_count = std::size_t{1} << shift;
    float accumulators[4][4][4]{};
    for (std::size_t block = 0; block < blocks; ++block) {
        for (int group = 0; group < 4; ++group)
            for (int lane = 0; lane < 4; ++lane)
                accumulators[0][group][lane] += values[block * 16 + group * 4 + lane];
        if ((block + 1) % block_count == 0) {
            for (int level = 1; level < 4; ++level) {
                for (int group = 0; group < 4; ++group)
                    for (int lane = 0; lane < 4; ++lane) {
                        accumulators[level][group][lane] += accumulators[level - 1][group][lane];
                        accumulators[level - 1][group][lane] = 0;
                    }
                if (((block + 1) >> (level * shift)) % block_count != 0)
                    break;
            }
        }
    }
    for (int level = 1; level < 4; ++level)
        for (int group = 0; group < 4; ++group)
            for (int lane = 0; lane < 4; ++lane)
                accumulators[0][group][lane] += accumulators[level][group][lane];
    const auto vectors = values.size() / 4;
    for (std::size_t vector = blocks * 4; vector < vectors; ++vector)
        for (int lane = 0; lane < 4; ++lane)
            accumulators[0][0][lane] += values[vector * 4 + lane];
    for (int group = 1; group < 4; ++group)
        for (int lane = 0; lane < 4; ++lane)
            accumulators[0][0][lane] += accumulators[0][group][lane];
    float total = 0;
    for (std::size_t i = vectors * 4; i < values.size(); ++i)
        total += values[i];
    for (int lane = 0; lane < 4; ++lane)
        total += accumulators[0][0][lane];
    return total;
}
} // namespace

std::vector<float> resample(const float* audio, std::size_t count, std::int32_t source_rate) {
    validate_audio(audio, count, source_rate);
    if (source_rate == 16000)
        return {audio, audio + count};
    // torchaudio's default sinc_interp_hann kernel: width 6, rolloff .99.
    const int gcd = std::gcd(source_rate, 16000);
    const int source = source_rate / gcd, target = 16000 / gcd;
    const double base = std::min(source, target) * 0.99;
    const int width = static_cast<int>(std::ceil(6.0 * source / base));
    const std::size_t output_count = (count * 16000 + source_rate - 1) / source_rate;
    std::vector<float> result(output_count);
    std::vector<std::vector<float>> kernels(target, std::vector<float>(2 * width + source));
    for (int phase = 0; phase < target; ++phase) {
        for (int offset = -width; offset < width + source; ++offset) {
            double t = (double(offset) / source - double(phase) / target) * base;
            t = std::clamp(t, -6.0, 6.0);
            const double window = std::pow(std::cos(t * kPi / 12), 2);
            t *= kPi;
            kernels[phase][offset + width] =
                static_cast<float>((t == 0 ? 1 : std::sin(t) / t) * window * base / source);
        }
    }
    for (std::size_t out = 0; out < output_count; ++out) {
        const auto center = static_cast<std::int64_t>(out / target) * source;
        const auto& kernel = kernels[out % target];
        float sum = 0;
        for (int offset = -width; offset < width + source; ++offset) {
            const auto at = center + offset;
            if (at >= 0 && static_cast<std::size_t>(at) < count)
                sum += audio[at] * kernel[offset + width];
        }
        result[out] = sum;
    }
    return result;
}

MelFeatures extract_mel(const float* audio, std::size_t count, std::int32_t source_rate,
                        const std::vector<float>& filters) {
    if (filters.size() != 128 * 257)
        throw std::invalid_argument("invalid LFM audio mel filters");
    auto signal = resample(audio, count, source_rate);
    if (signal.size() < 1280)
        throw std::invalid_argument("LFM audio requires at least 80 ms of input");
    const int valid_frames = static_cast<int>(signal.size() / 160);
    const int frames = valid_frames + 1;
    std::vector<float> pre(signal.size());
    pre[0] = signal[0];
    for (std::size_t i = 1; i < signal.size(); ++i)
        pre[i] = signal[i] - 0.97F * signal[i - 1];
    std::vector<float> features(static_cast<std::size_t>(frames) * 128);
    std::vector<Complex> frame(512);
    float window[512]{};
    for (int i = 0; i < 400; ++i)
        window[i + 56] = static_cast<float>(0.5 - 0.5 * std::cos(2 * kPi * i / 399));
    for (int t = 0; t < frames; ++t) {
        for (int i = 0; i < 512; ++i) {
            std::int64_t at = static_cast<std::int64_t>(t) * 160 + i - 256;
            frame[i] = (at < 0 || static_cast<std::size_t>(at) >= pre.size())
                           ? 0
                           : static_cast<double>(pre[at] * window[i]);
        }
        fft(frame, false);
        float power[257];
        for (int f = 0; f < 257; ++f) {
            const float re = static_cast<float>(frame[f].real()),
                        im = static_cast<float>(frame[f].imag());
            // The reference computes magnitude followed by squaring.
            const float magnitude = std::sqrt(re * re + im * im);
            power[f] = magnitude * magnitude;
        }
        for (int m = 0; m < 128; ++m) {
            float sum = 0;
            for (int f = 0; f < 257; ++f)
                sum += filters[m * 257 + f] * power[f];
            features[t * 128 + m] = std::log(sum + 0x1p-24F);
        }
    }
    std::vector<float> reduction(frames);
    for (int m = 0; m < 128; ++m) {
        for (int t = 0; t < valid_frames; ++t)
            reduction[t] = features[t * 128 + m];
        const float mean = feature_sum(reduction) / valid_frames;
        for (int t = 0; t < valid_frames; ++t) {
            const float d = features[t * 128 + m] - mean;
            reduction[t] = d * d;
        }
        const float sigma = std::sqrt(feature_sum(reduction) / (valid_frames - 1)) + 1e-5F;
        for (int t = 0; t < valid_frames; ++t)
            features[t * 128 + m] = (features[t * 128 + m] - mean) / sigma;
        features[valid_frames * 128 + m] = 0;
    }
    return {std::move(features), frames};
}

StreamingIstft::StreamingIstft() {
    for (int i = 0; i < 1280; ++i)
        window_[i] = static_cast<float>(0.5 - 0.5 * std::cos(2 * kPi * i / 1280));
}

void StreamingIstft::reset() {
    signal_.fill(0);
    envelope_.fill(0);
    frames_ = finalized_ = 0;
    finished_ = false;
}

void StreamingIstft::flush(std::size_t end, std::vector<float>& output) {
    for (; finalized_ < end; ++finalized_) {
        const auto slot = finalized_ % 1280;
        if (finalized_ >= 480) {
            if (envelope_[slot] <= 1e-11F)
                throw std::runtime_error("LFM audio overlap-add envelope is zero");
            output.push_back(signal_[slot] / envelope_[slot]);
        }
        signal_[slot] = envelope_[slot] = 0;
    }
}

std::vector<float> StreamingIstft::append(const float* spectra, std::size_t frames) {
    if (finished_)
        throw std::invalid_argument("LFM audio ISTFT is finished; reset before appending");
    if (frames && spectra == nullptr)
        throw std::invalid_argument("LFM audio spectra must not be null");
    std::vector<float> output;
    output.reserve(frames * 320);
    std::vector<Complex> bins(1280);
    for (std::size_t t = 0; t < frames; ++t) {
        const float* spectrum = spectra + t * 1282;
        for (int f = 0; f <= 640; ++f) {
            const float magnitude = std::exp(spectrum[f]);
            const float re = magnitude * std::cos(spectrum[f + 641]);
            const float im = magnitude * std::sin(spectrum[f + 641]);
            bins[f] = {re, (f == 0 || f == 640) ? 0 : im};
            if (f > 0 && f < 640)
                bins[1280 - f] = std::conj(bins[f]);
        }
        fft(bins, true);
        for (int i = 0; i < 1280; ++i) {
            const auto slot = (frames_ * 320 + i) % 1280;
            signal_[slot] += static_cast<float>(bins[i].real() / 1280) * window_[i];
            envelope_[slot] += window_[i] * window_[i];
        }
        ++frames_;
        // The next frame starts at frames_*hop and therefore cannot modify
        // this prefix. The first 480 samples are the reference's left crop.
        flush(frames_ * 320, output);
    }
    return output;
}

std::vector<float> StreamingIstft::finish() {
    if (finished_)
        return {};
    finished_ = true;
    std::vector<float> output;
    if (frames_)
        flush(frames_ * 320 + 480, output);
    return output;
}

std::vector<float> inverse_spectrogram(const std::vector<float>& spectra) {
    if (spectra.size() % 1282 != 0)
        throw std::invalid_argument("invalid LFM audio spectrum shape");
    StreamingIstft synthesis;
    auto result = synthesis.append(spectra.data(), spectra.size() / 1282);
    auto tail = synthesis.finish();
    result.insert(result.end(), tail.begin(), tail.end());
    return result;
}
} // namespace trtmc::lfm2_audio
