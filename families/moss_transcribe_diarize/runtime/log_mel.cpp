/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/moss_transcribe_diarize/runtime/log_mel.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace trtmc::moss_transcribe_diarize {
namespace {

constexpr double kPi = 3.14159265358979323846;
// log10 floor and dynamic range of the Whisper feature extractor.
constexpr double kMelFloor = 1e-10;
constexpr double kDynamicRange = 8.0;
// Host threads computing the frames of one chunk.
constexpr std::size_t kMaxWorkers = 8;

std::size_t next_power_of_two(std::size_t value) {
    std::size_t result = 1;
    while (result < value)
        result <<= 1;
    return result;
}

} // namespace

MelFilterBank parse_mel_filter_bank(const std::vector<char>& bytes) {
    std::uint32_t shape[2] = {0, 0};
    if (bytes.size() < sizeof(shape))
        throw std::runtime_error("mel_filters.bin is truncated");
    std::memcpy(shape, bytes.data(), sizeof(shape));
    MelFilterBank bank;
    bank.freq_bins = static_cast<std::int32_t>(shape[0]);
    bank.mel_bins = static_cast<std::int32_t>(shape[1]);
    const auto count = static_cast<std::size_t>(shape[0]) * shape[1];
    if (count == 0 || bytes.size() != sizeof(shape) + count * sizeof(float))
        throw std::runtime_error("mel_filters.bin size does not match its shape");
    bank.weights.resize(count);
    std::memcpy(bank.weights.data(), bytes.data() + sizeof(shape), count * sizeof(float));
    return bank;
}

Dft::Dft(std::size_t length) : length_(length), padded_(next_power_of_two(2 * length - 1)) {
    if (length == 0)
        throw std::invalid_argument("DFT length must be positive");
    chirp_.resize(length_);
    kernel_spectrum_.assign(padded_, {0.0, 0.0});
    const auto period = 2 * length_;
    for (std::size_t n = 0; n < length_; ++n) {
        // n^2 is reduced modulo 2N first, so the phase stays exact for large n.
        const double phase =
            kPi * static_cast<double>((n * n) % period) / static_cast<double>(length_);
        chirp_[n] = {std::cos(phase), -std::sin(phase)};
        kernel_spectrum_[n] = std::conj(chirp_[n]);
        if (n != 0)
            kernel_spectrum_[padded_ - n] = std::conj(chirp_[n]);
    }
    twiddles_.resize(padded_ / 2);
    for (std::size_t k = 0; k < twiddles_.size(); ++k) {
        const double angle = -2.0 * kPi * static_cast<double>(k) / static_cast<double>(padded_);
        twiddles_[k] = {std::cos(angle), std::sin(angle)};
    }
    fft(kernel_spectrum_, false);
}

void Dft::fft(std::vector<std::complex<double>>& data, bool inverse) const {
    const std::size_t size = data.size();
    for (std::size_t i = 1, j = 0; i < size; ++i) {
        std::size_t bit = size >> 1;
        for (; (j & bit) != 0; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(data[i], data[j]);
    }
    for (std::size_t span = 2; span <= size; span <<= 1) {
        const std::size_t half = span / 2;
        const std::size_t stride = size / span;
        for (std::size_t offset = 0; offset < size; offset += span) {
            for (std::size_t i = 0; i < half; ++i) {
                const auto twiddle =
                    inverse ? std::conj(twiddles_[i * stride]) : twiddles_[i * stride];
                const auto even = data[offset + i];
                const auto odd = data[offset + i + half] * twiddle;
                data[offset + i] = even + odd;
                data[offset + i + half] = even - odd;
            }
        }
    }
    if (inverse) {
        const double scale = 1.0 / static_cast<double>(size);
        for (auto& value : data)
            value *= scale;
    }
}

void Dft::forward(std::vector<std::complex<double>>& values) const {
    if (values.size() != length_)
        throw std::invalid_argument("DFT input length mismatch");
    std::vector<std::complex<double>> work(padded_, {0.0, 0.0});
    for (std::size_t n = 0; n < length_; ++n)
        work[n] = values[n] * chirp_[n];
    fft(work, false);
    for (std::size_t i = 0; i < padded_; ++i)
        work[i] *= kernel_spectrum_[i];
    fft(work, true);
    for (std::size_t k = 0; k < length_; ++k)
        values[k] = work[k] * chirp_[k];
}

LogMelExtractor::LogMelExtractor(MelFilterBank bank, std::int32_t n_fft, std::int32_t hop_length,
                                 std::int32_t chunk_samples, std::int32_t mel_frames)
    : bank_(std::move(bank)), n_fft_(n_fft), hop_length_(hop_length), chunk_samples_(chunk_samples),
      mel_frames_(mel_frames), dft_(n_fft > 0 ? static_cast<std::size_t>(n_fft) : 1) {
    if (n_fft <= 0 || n_fft % 2 != 0 || hop_length <= 0 || chunk_samples <= n_fft)
        throw std::invalid_argument("invalid log-mel geometry");
    if (bank_.freq_bins != n_fft / 2 + 1 || bank_.mel_bins <= 0 ||
        bank_.weights.size() != static_cast<std::size_t>(bank_.freq_bins) * bank_.mel_bins)
        throw std::invalid_argument("mel filter bank does not match n_fft");
    // The centred STFT yields chunk_samples / hop + 1 frames; the last is dropped.
    if (mel_frames != chunk_samples / hop_length)
        throw std::invalid_argument("mel frame count does not match the chunk length");

    window_.resize(static_cast<std::size_t>(n_fft));
    for (std::int32_t n = 0; n < n_fft; ++n)
        window_[n] = 0.5 - 0.5 * std::cos(2.0 * kPi * n / static_cast<double>(n_fft));

    first_bin_.assign(bank_.mel_bins, 0);
    end_bin_.assign(bank_.mel_bins, 0);
    for (std::int32_t m = 0; m < bank_.mel_bins; ++m) {
        std::int32_t first = bank_.freq_bins;
        std::int32_t end = 0;
        for (std::int32_t k = 0; k < bank_.freq_bins; ++k) {
            if (bank_.weights[static_cast<std::size_t>(k) * bank_.mel_bins + m] != 0.0F) {
                first = std::min(first, k);
                end = k + 1;
            }
        }
        first_bin_[m] = std::min(first, end);
        end_bin_[m] = end;
    }
}

void LogMelExtractor::frame_pair_power(const double* first, const double* second,
                                       std::vector<std::complex<double>>& work, double* power_first,
                                       double* power_second) const {
    // Two real frames share one complex transform: z = a + i b.
    for (std::int32_t n = 0; n < n_fft_; ++n)
        work[n] = {first[n] * window_[n], second[n] * window_[n]};
    dft_.forward(work);
    const std::int32_t bins = n_fft_ / 2 + 1;
    for (std::int32_t k = 0; k < bins; ++k) {
        const auto z = work[k];
        const auto mirror = std::conj(work[(n_fft_ - k) % n_fft_]);
        const auto a = 0.5 * (z + mirror);
        const auto b = std::complex<double>(0.0, -0.5) * (z - mirror);
        power_first[k] = std::norm(a);
        power_second[k] = std::norm(b);
    }
}

std::vector<float> LogMelExtractor::chunk(const float* samples, std::int64_t count) const {
    if (count < 0 || count > chunk_samples_ || (count > 0 && samples == nullptr))
        throw std::invalid_argument("log-mel chunk must hold at most one chunk of samples");
    const std::int32_t pad = n_fft_ / 2;
    const std::size_t length = static_cast<std::size_t>(chunk_samples_);
    std::vector<double> padded(length + static_cast<std::size_t>(n_fft_), 0.0);
    for (std::int64_t i = 0; i < count; ++i)
        padded[static_cast<std::size_t>(pad + i)] = samples[i];
    // Reflect padding (edge excluded) of the zero-padded chunk on both sides.
    for (std::int32_t i = 0; i < pad; ++i) {
        padded[static_cast<std::size_t>(i)] = padded[static_cast<std::size_t>(2 * pad - i)];
        padded[length + pad + i] = padded[length + pad - 2 - i];
    }

    const std::int32_t bins = n_fft_ / 2 + 1;
    const std::size_t mels = static_cast<std::size_t>(bank_.mel_bins);
    const std::size_t frames = static_cast<std::size_t>(mel_frames_);
    const std::size_t pairs = (frames + 1) / 2;
    std::vector<double> logs(mels * frames, 0.0);
    const std::vector<double> silence(static_cast<std::size_t>(n_fft_), 0.0);
    // Frame pairs are independent. Workers take pairs from a shared counter and
    // each keeps its own maximum, so the values do not depend on how many
    // workers start; a helper that cannot be started leaves its pairs to the
    // others. Errors are rethrown after every started worker has joined.
    const std::size_t workers =
        std::clamp<std::size_t>(std::thread::hardware_concurrency(), 1, kMaxWorkers);
    std::vector<double> peaks(workers, -std::numeric_limits<double>::infinity());
    std::vector<std::exception_ptr> failures(workers);
    std::atomic<std::size_t> next_pair{0};
    auto run = [&](std::size_t worker) {
        try {
            std::vector<std::complex<double>> work(static_cast<std::size_t>(n_fft_));
            std::vector<double> power_a(bins), power_b(bins);
            double peak = -std::numeric_limits<double>::infinity();
            for (std::size_t pair = next_pair++; pair < pairs; pair = next_pair++) {
                const std::size_t t = 2 * pair;
                const double* first = padded.data() + t * hop_length_;
                const bool paired = t + 1 < frames;
                const double* second =
                    paired ? padded.data() + (t + 1) * hop_length_ : silence.data();
                frame_pair_power(first, second, work, power_a.data(), power_b.data());
                for (std::size_t m = 0; m < mels; ++m) {
                    double sum_a = 0.0;
                    double sum_b = 0.0;
                    for (std::int32_t k = first_bin_[m]; k < end_bin_[m]; ++k) {
                        const double weight = bank_.weights[static_cast<std::size_t>(k) * mels + m];
                        sum_a += weight * power_a[k];
                        sum_b += weight * power_b[k];
                    }
                    const double log_a = std::log10(std::max(sum_a, kMelFloor));
                    logs[m * frames + t] = log_a;
                    peak = std::max(peak, log_a);
                    if (paired) {
                        const double log_b = std::log10(std::max(sum_b, kMelFloor));
                        logs[m * frames + t + 1] = log_b;
                        peak = std::max(peak, log_b);
                    }
                }
            }
            peaks[worker] = peak;
        } catch (...) {
            failures[worker] = std::current_exception();
            next_pair = pairs;
        }
    };
    std::vector<std::thread> helpers;
    helpers.reserve(workers - 1);
    for (std::size_t worker = 1; worker < workers; ++worker) {
        try {
            helpers.emplace_back(run, worker);
        } catch (...) {
            break; // The started workers and this thread still cover every pair.
        }
    }
    run(0);
    for (auto& helper : helpers)
        helper.join();
    for (const auto& failure : failures) {
        if (failure)
            std::rethrow_exception(failure);
    }
    const double peak = *std::max_element(peaks.begin(), peaks.end());
    std::vector<float> features(logs.size());
    const double floor = peak - kDynamicRange;
    for (std::size_t i = 0; i < logs.size(); ++i)
        features[i] = static_cast<float>((std::max(logs[i], floor) + 4.0) / 4.0);
    return features;
}

} // namespace trtmc::moss_transcribe_diarize
