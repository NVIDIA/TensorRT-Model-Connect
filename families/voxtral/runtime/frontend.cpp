/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/voxtral/runtime/frontend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace trtmc::voxtral {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::size_t smallest_factor(std::size_t size) {
    for (std::size_t factor : {2U, 3U, 5U}) {
        if (size % factor == 0)
            return factor;
    }
    return size;
}

// Nonzero bin range of every mel filter; a Slaney bank overlaps only neighbours.
struct SparseFilters {
    std::vector<std::int32_t> first, last;
};

SparseFilters sparse_filters(const std::vector<float>& filters, std::int32_t bins,
                             std::int32_t mels) {
    SparseFilters sparse{std::vector<std::int32_t>(static_cast<std::size_t>(mels), 0),
                         std::vector<std::int32_t>(static_cast<std::size_t>(mels), -1)};
    for (std::int32_t mel = 0; mel < mels; ++mel) {
        for (std::int32_t bin = 0; bin < bins; ++bin) {
            if (filters[static_cast<std::size_t>(bin) * mels + mel] == 0.0F)
                continue;
            if (sparse.last[mel] < 0)
                sparse.first[mel] = bin;
            sparse.last[mel] = bin;
        }
    }
    return sparse;
}

void validate(const FrontendConfig& config, std::size_t filter_values) {
    if (config.n_fft <= 0 || config.n_fft % 2 != 0 || config.hop_length <= 0 ||
        config.num_mel_bins <= 0 || config.chunk_samples <= config.n_fft ||
        config.chunk_samples % config.hop_length != 0 ||
        config.chunk_samples / config.hop_length != config.mel_frames_per_chunk)
        throw std::invalid_argument("voxtral front-end geometry is inconsistent");
    const auto bins = static_cast<std::size_t>(config.n_fft / 2 + 1);
    if (filter_values != bins * static_cast<std::size_t>(config.num_mel_bins))
        throw std::invalid_argument("voxtral mel filter bank does not match n_fft and mel bins");
}

} // namespace

Fft::Fft(std::int32_t length) : length_(length) {
    if (length <= 0)
        throw std::invalid_argument("FFT length must be positive");
    auto remainder = static_cast<std::size_t>(length);
    for (std::size_t factor : {2U, 3U, 5U}) {
        while (remainder % factor == 0)
            remainder /= factor;
    }
    if (remainder != 1)
        throw std::invalid_argument("FFT length must factor into 2, 3 and 5");
    twiddles_.resize(static_cast<std::size_t>(length));
    for (std::size_t index = 0; index < twiddles_.size(); ++index)
        twiddles_[index] = std::polar(1.0, -2.0 * kPi * static_cast<double>(index) / length);
    scratch_.resize(static_cast<std::size_t>(length));
}

void Fft::forward(const std::complex<double>* input, std::complex<double>* output) {
    transform(input, 1, static_cast<std::size_t>(length_), output);
}

// Decimation in time: the `factor` interleaved subsequences are transformed into
// consecutive blocks of `output` and then combined with the twiddle factors.
void Fft::transform(const std::complex<double>* input, std::size_t stride, std::size_t size,
                    std::complex<double>* output) {
    if (size == 1) {
        output[0] = input[0];
        return;
    }
    const std::size_t factor = smallest_factor(size);
    const std::size_t block = size / factor;
    for (std::size_t r = 0; r < factor; ++r)
        transform(input + r * stride, stride * factor, block, output + r * block);
    const auto length = static_cast<std::size_t>(length_);
    const std::size_t step = length / size;
    std::complex<double> rotated[5];
    for (std::size_t k = 0; k < block; ++k) {
        for (std::size_t r = 0; r < factor; ++r)
            rotated[r] = output[r * block + k] * twiddles_[(r * k * step) % length];
        for (std::size_t q = 0; q < factor; ++q) {
            std::complex<double> sum = 0.0;
            for (std::size_t r = 0; r < factor; ++r)
                sum += rotated[r] * twiddles_[(r * q * block * step) % length];
            scratch_[q * block + k] = sum;
        }
    }
    std::copy(scratch_.begin(), scratch_.begin() + static_cast<std::ptrdiff_t>(size), output);
}

std::vector<float> downmix(const float* samples, std::size_t count, std::uint32_t channels) {
    if (samples == nullptr || count == 0)
        throw std::invalid_argument("voxtral requires nonempty PCM audio");
    if (channels == 0 || count % channels != 0)
        throw std::invalid_argument(
            "voxtral PCM must hold whole frames of a positive channel count");
    if (channels == 1)
        return {samples, samples + count};
    std::vector<float> mono(count / channels);
    for (std::size_t frame = 0; frame < mono.size(); ++frame) {
        double sum = 0.0;
        for (std::uint32_t channel = 0; channel < channels; ++channel)
            sum += samples[frame * channels + channel];
        mono[frame] = static_cast<float>(sum / channels);
    }
    return mono;
}

std::int64_t chunk_count(std::size_t samples, std::int64_t chunk_samples) {
    if (chunk_samples <= 0)
        throw std::invalid_argument("chunk length must be positive");
    const auto chunk = static_cast<std::size_t>(chunk_samples);
    return std::max<std::int64_t>(1, static_cast<std::int64_t>((samples + chunk - 1) / chunk));
}

std::vector<float> log_mel_features(const std::vector<float>& mono,
                                    const std::vector<float>& filters,
                                    const FrontendConfig& config) {
    validate(config, filters.size());
    const std::int32_t bins = config.n_fft / 2 + 1;
    const std::int32_t mels = config.num_mel_bins;
    const std::int64_t chunks = chunk_count(mono.size(), config.chunk_samples);
    const std::int64_t length = chunks * config.chunk_samples;
    const std::int64_t pad = config.n_fft / 2;
    const std::int64_t frames = chunks * config.mel_frames_per_chunk;
    const auto available = static_cast<std::int64_t>(mono.size());
    // Reflect padding (edge excluded) of the zero-padded clip.
    auto sample = [&](std::int64_t index) -> double {
        if (index < 0)
            index = -index;
        else if (index >= length)
            index = 2 * (length - 1) - index;
        return index < available ? static_cast<double>(mono[static_cast<std::size_t>(index)]) : 0.0;
    };
    std::vector<double> window(static_cast<std::size_t>(config.n_fft));
    for (std::size_t n = 0; n < window.size(); ++n)
        window[n] = 0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(n) / config.n_fft);
    const auto sparse = sparse_filters(filters, bins, mels);

    Fft fft(config.n_fft);
    std::vector<std::complex<double>> input(static_cast<std::size_t>(config.n_fft));
    std::vector<std::complex<double>> spectrum(input.size());
    std::vector<double> power(static_cast<std::size_t>(bins));
    std::vector<double> log_spec(static_cast<std::size_t>(frames) * mels);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int64_t frame = 0; frame < frames; ++frame) {
        const std::int64_t start = frame * config.hop_length - pad;
        for (std::size_t n = 0; n < input.size(); ++n)
            input[n] = {sample(start + static_cast<std::int64_t>(n)) * window[n], 0.0};
        fft.forward(input.data(), spectrum.data());
        for (std::int32_t bin = 0; bin < bins; ++bin)
            power[static_cast<std::size_t>(bin)] =
                std::norm(spectrum[static_cast<std::size_t>(bin)]);
        for (std::int32_t mel = 0; mel < mels; ++mel) {
            double energy = 0.0;
            for (std::int32_t bin = sparse.first[mel]; bin <= sparse.last[mel]; ++bin)
                energy += static_cast<double>(filters[static_cast<std::size_t>(bin) * mels + mel]) *
                          power[static_cast<std::size_t>(bin)];
            const double value = std::log10(std::max(energy, 1e-10));
            log_spec[static_cast<std::size_t>(mel) * frames + frame] = value;
            maximum = std::max(maximum, value);
        }
    }
    const double floor = maximum - 8.0;
    const std::int64_t per_chunk = config.mel_frames_per_chunk;
    std::vector<float> features(log_spec.size());
    for (std::int64_t chunk = 0; chunk < chunks; ++chunk) {
        for (std::int32_t mel = 0; mel < mels; ++mel) {
            const double* source = log_spec.data() + static_cast<std::size_t>(mel) * frames +
                                   static_cast<std::size_t>(chunk * per_chunk);
            float* target =
                features.data() + static_cast<std::size_t>((chunk * mels + mel) * per_chunk);
            for (std::int64_t frame = 0; frame < per_chunk; ++frame)
                target[frame] = static_cast<float>((std::max(source[frame], floor) + 4.0) / 4.0);
        }
    }
    return features;
}

} // namespace trtmc::voxtral
