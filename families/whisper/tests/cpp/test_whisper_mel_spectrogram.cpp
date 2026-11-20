/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/whisper/runtime/whisper_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

struct Golden {
    int mel;
    int frame;
    float value;
};

void check_values(const trtmc::whisper::MelResult& result, std::initializer_list<Golden> expected) {
    const bool valid_shape = (result.n_mels == 80 || result.n_mels == 128) &&
                             result.n_frames == 3000 &&
                             result.data.size() == static_cast<std::size_t>(result.n_mels) * 3000;
    check(valid_shape, "30-second output has the expected mel shape");
    if (!valid_shape) {
        return;
    }
    for (const auto& point : expected) {
        const float actual = result.data[point.mel * result.n_frames + point.frame];
        if (!std::isfinite(actual) || std::abs(actual - point.value) > 3e-5F) {
            std::fprintf(stderr, "FAIL: mel %d frame %d: expected %.8f, got %.8f\n", point.mel,
                         point.frame, point.value, actual);
            ++failures;
        }
    }
}

trtmc::whisper::MelResult extract(const std::vector<float>& audio, int mels) {
    // A sparse test filterbank selects FFT bin m + 1 for mel m.
    std::vector<float> filters(201 * mels, 0.0F);
    for (int m = 0; m < mels; ++m) {
        filters[(m + 1) * mels + m] = 1.0F;
    }
    return trtmc::whisper::extract_mel_spectrogram(audio.data(), static_cast<int32_t>(audio.size()),
                                                   filters.data(), 201, mels, 400, 160, 30, 16000);
}

std::vector<float> tone(int samples) {
    std::vector<float> audio(samples);
    constexpr double pi2 = 6.28318530717958647692;
    for (int i = 0; i < samples; ++i) {
        audio[i] = static_cast<float>(0.25 * std::cos(pi2 * 3.0 * i / 400.0) +
                                      0.15 * std::sin(pi2 * 7.0 * i / 400.0));
    }
    return audio;
}

// Goldens from Transformers 5.2.0 WhisperFeatureExtractor with the sparse
// filterbank above and n_fft=400, hop_length=160, sampling_rate=16000,
// chunk_length=30. The signal has nonzero endpoints so center padding is visible.
void test_center_padding(int mels) {
    const auto short_result = extract(tone(16000), mels);
    check_values(short_result, {{0, 0, 1.22718477F},
                                {2, 0, 1.72748566F},
                                {6, 0, 0.91715926F},
                                {0, 1, 0.67869687F},
                                {0, 2999, -0.27251434F}});

    auto audio = tone(480000);
    const auto full_result = extract(audio, mels);
    check_values(full_result, {{0, 0, 1.22718477F},
                               {0, 2999, 0.70422518F},
                               {2, 2999, 1.69706297F},
                               {6, 2999, 1.58521879F}});

    audio.resize(480160, 1.0F);
    check(extract(audio, mels).data == full_result.data,
          "samples beyond the 30-second chunk do not affect its reflection");
}

void test_discarded_frame_does_not_set_log_floor(int mels) {
    std::vector<float> audio(480000, 0.0F);
    for (int i = 0; i < 160; ++i) {
        audio[480000 - 160 + i] = static_cast<float>(i + 1) / 320.0F;
    }
    const auto result = extract(audio, mels);
    check_values(result, {{0, 0, -0.40593076F},
                          {79, 1000, -0.40593076F},
                          {0, 2999, 1.59406924F},
                          {2, 2999, 1.10322535F},
                          {6, 2999, 0.74954677F}});
}

void test_filterbank_is_read_only_for_nonzero_power() {
    const std::vector<float> silence(16000, 0.0F);
    for (const std::vector<float>* audio :
         {static_cast<const std::vector<float>*>(nullptr), &silence}) {
        const auto result = trtmc::whisper::extract_mel_spectrogram(
            audio == nullptr ? nullptr : audio->data(),
            audio == nullptr ? 0 : static_cast<int32_t>(audio->size()), nullptr, 201, 80, 400, 160,
            30, 16000);
        check(result.n_mels == 80 && result.n_frames == 3000 &&
                  std::all_of(result.data.begin(), result.data.end(),
                              [](float value) { return value == -1.5F; }),
              "empty or silent input gives the silence floor without reading the filterbank");
    }
}

void test_empty_and_single_sample(int mels) {
    const auto empty = extract({}, mels);
    check(empty.n_mels == mels && empty.n_frames == 3000, "empty input keeps output shape");
    check(std::all_of(empty.data.begin(), empty.data.end(),
                      [](float value) { return value == -1.5F; }),
          "empty input produces the silence floor");
    check_values(extract({0.5F}, mels),
                 {{0, 0, 0.84948498F}, {0, 1, 0.33946735F}, {0, 2999, -1.15051508F}});
}

// The frontend's straightforward definition, which the production code must
// equal bit for bit: per frame, the Hann window, the power spectrum by a
// radix-2 transform that updates its twiddle factor inline in every butterfly
// group (through Bluestein's two transforms when the length is not a power of
// two), the dense filterbank product, then the last frame dropped and the
// log-mel normalization.
namespace definition {

void fft_radix2_inplace(std::vector<std::complex<double>>& values, bool inverse) {
    const std::size_t size = values.size();
    for (std::size_t i = 1, j = 0; i < size; ++i) {
        std::size_t bit = size >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }

    constexpr double kPi2 = 2.0 * 3.14159265358979323846;
    for (std::size_t length = 2; length <= size; length <<= 1) {
        const double angle = (inverse ? kPi2 : -kPi2) / static_cast<double>(length);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        const std::size_t half = length / 2;
        for (std::size_t offset = 0; offset < size; offset += length) {
            std::complex<double> twiddle(1.0, 0.0);
            for (std::size_t i = 0; i < half; ++i) {
                const auto even = values[offset + i];
                const auto odd = values[offset + i + half] * twiddle;
                values[offset + i] = even + odd;
                values[offset + i + half] = even - odd;
                twiddle *= step;
            }
        }
    }

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(size);
        for (auto& value : values) {
            value *= scale;
        }
    }
}

std::vector<float> rfft_power(const std::vector<float>& input) {
    const int32_t n = static_cast<int32_t>(input.size());
    const int32_t bins = n / 2 + 1;
    std::vector<float> power(static_cast<std::size_t>(bins), 0.0F);
    if (n <= 0) {
        return power;
    }

    std::vector<std::complex<double>> workspace;
    if ((n & (n - 1)) == 0) {
        workspace.resize(static_cast<std::size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            workspace[static_cast<std::size_t>(i)] = {static_cast<double>(input[i]), 0.0};
        }
        fft_radix2_inplace(workspace, false);
        for (int32_t k = 0; k < bins; ++k) {
            power[k] = static_cast<float>(std::norm(workspace[static_cast<std::size_t>(k)]));
        }
        return power;
    }

    std::size_t fft_size = 1;
    while (fft_size < static_cast<std::size_t>(2 * n - 1)) {
        fft_size <<= 1;
    }
    std::vector<std::complex<double>> chirp(static_cast<std::size_t>(n));
    std::vector<std::complex<double>> kernel_fft(fft_size, {0.0, 0.0});
    constexpr double kPi = 3.14159265358979323846;
    for (int32_t i = 0; i < n; ++i) {
        const double index = static_cast<double>(i);
        const double angle = kPi * index * index / static_cast<double>(n);
        chirp[static_cast<std::size_t>(i)] = {std::cos(angle), -std::sin(angle)};
        const std::complex<double> kernel(std::cos(angle), std::sin(angle));
        kernel_fft[static_cast<std::size_t>(i)] = kernel;
        if (i != 0) {
            kernel_fft[fft_size - static_cast<std::size_t>(i)] = kernel;
        }
    }
    fft_radix2_inplace(kernel_fft, false);

    workspace.assign(fft_size, {0.0, 0.0});
    for (int32_t i = 0; i < n; ++i) {
        workspace[static_cast<std::size_t>(i)] =
            static_cast<double>(input[i]) * chirp[static_cast<std::size_t>(i)];
    }
    fft_radix2_inplace(workspace, false);
    for (std::size_t i = 0; i < fft_size; ++i) {
        workspace[i] *= kernel_fft[i];
    }
    fft_radix2_inplace(workspace, true);
    for (int32_t k = 0; k < bins; ++k) {
        const auto value =
            workspace[static_cast<std::size_t>(k)] * chirp[static_cast<std::size_t>(k)];
        power[k] = static_cast<float>(std::norm(value));
    }
    return power;
}

struct MelCase {
    int32_t n_fft;
    int32_t hop_length;
    int32_t chunk_length_s;
    int32_t sample_rate;
    int32_t n_mel_bins;
};

std::vector<float> mel(const std::vector<float>& audio, const std::vector<float>& filters,
                       const MelCase& c) {
    const int32_t n_freq_bins = c.n_fft / 2 + 1;
    const int32_t n_samples = static_cast<int32_t>(audio.size());
    const int32_t chunk_samples = c.chunk_length_s * c.sample_rate;
    const int32_t valid_audio = std::min(n_samples, chunk_samples);
    const int32_t pad = c.n_fft / 2;
    const int32_t frames_to_compute =
        valid_audio > 0 ? 1 + (pad + valid_audio - 1) / c.hop_length : 0;

    std::vector<float> chunk(static_cast<std::size_t>(chunk_samples), 0.0F);
    std::copy_n(audio.begin(), valid_audio, chunk.begin());
    std::vector<float> padded(static_cast<std::size_t>(chunk_samples + 2 * pad), 0.0F);
    std::copy(chunk.begin(), chunk.end(), padded.begin() + pad);
    for (int32_t i = 0; i < pad; ++i) {
        padded[static_cast<std::size_t>(i)] = chunk[static_cast<std::size_t>(pad - i)];
        padded[static_cast<std::size_t>(pad + chunk_samples + i)] =
            chunk[static_cast<std::size_t>(chunk_samples - 2 - i)];
    }

    std::vector<float> window(static_cast<std::size_t>(c.n_fft));
    constexpr double kPi2 = 2.0 * 3.14159265358979323846;
    for (int32_t i = 0; i < c.n_fft; ++i) {
        window[static_cast<std::size_t>(i)] = static_cast<float>(
            0.5 * (1.0 - std::cos(kPi2 * static_cast<double>(i) / static_cast<double>(c.n_fft))));
    }

    const int32_t n_frames = 1 + (static_cast<int32_t>(padded.size()) - c.n_fft) / c.hop_length;
    std::vector<float> raw(static_cast<std::size_t>(c.n_mel_bins) * n_frames, 0.0F);
    std::vector<float> frame(static_cast<std::size_t>(c.n_fft));
    std::vector<float> frame_mel(static_cast<std::size_t>(c.n_mel_bins));
    for (int32_t t = 0; t < std::min(frames_to_compute, n_frames); ++t) {
        for (int32_t i = 0; i < c.n_fft; ++i) {
            frame[static_cast<std::size_t>(i)] =
                padded[static_cast<std::size_t>(t * c.hop_length + i)] *
                window[static_cast<std::size_t>(i)];
        }
        const auto power = rfft_power(frame);
        std::fill(frame_mel.begin(), frame_mel.end(), 0.0F);
        for (int32_t f = 0; f < n_freq_bins; ++f) {
            const float p = power[static_cast<std::size_t>(f)];
            if (p == 0.0F) {
                continue;
            }
            const float* row = filters.data() + static_cast<std::size_t>(f) * c.n_mel_bins;
            for (int32_t m = 0; m < c.n_mel_bins; ++m) {
                frame_mel[static_cast<std::size_t>(m)] += p * row[m];
            }
        }
        for (int32_t m = 0; m < c.n_mel_bins; ++m) {
            raw[static_cast<std::size_t>(m) * n_frames + t] =
                frame_mel[static_cast<std::size_t>(m)];
        }
    }

    const int32_t n_frames_out = n_frames > 1 ? n_frames - 1 : n_frames;
    std::vector<float> out(static_cast<std::size_t>(c.n_mel_bins) * n_frames_out);
    for (int32_t m = 0; m < c.n_mel_bins; ++m) {
        std::copy_n(raw.begin() + static_cast<std::ptrdiff_t>(m) * n_frames, n_frames_out,
                    out.begin() + static_cast<std::ptrdiff_t>(m) * n_frames_out);
    }
    float global_max = -1e10F;
    for (float& value : out) {
        value = std::log10(std::max(value, 1e-10F));
        if (value > global_max) {
            global_max = value;
        }
    }
    const float floor = global_max - 8.0F;
    for (float& value : out) {
        value = std::max(value, floor);
        value = (value + 4.0F) / 4.0F;
    }
    return out;
}

} // namespace definition

// Equal bit for bit, except that a NaN matches any NaN: which operand's NaN an
// addition returns follows the operand order the compiler picks, so a NaN's
// sign and payload bits are not part of the definition. Its place is.
bool same_values(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0 &&
            !(std::isnan(a[i]) && std::isnan(b[i]))) {
            return false;
        }
    }
    return true;
}

// Frames whose rounding shows in the output: a pure tone on a bin leaves the
// other bins as rounding noise, so one differing double anywhere in the
// transform changes a float there; sums that cancel do the same.
std::vector<float> power_input(int size, int kind, std::mt19937& rng) {
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    constexpr double kPi2 = 2.0 * 3.14159265358979323846;
    const float infinity = std::numeric_limits<float>::infinity();
    std::vector<float> x(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) {
        const double t = static_cast<double>(i);
        float value = 0.0F;
        switch (kind) {
        case 0: // noise
            value = uniform(rng);
            break;
        case 1: // a cosine on a bin
            value = static_cast<float>(std::cos(kPi2 * 3.0 * t / size));
            break;
        case 2: // a sine on a bin
            value = static_cast<float>(std::sin(kPi2 * 7.0 * t / size));
            break;
        case 3: // constant
            value = 1.0F;
            break;
        case 4: // alternating sign
            value = (i % 2) != 0 ? -1.0F : 1.0F;
            break;
        case 5: // one impulse
            value = i == 0 ? 1.0F : 0.0F;
            break;
        case 6: // silence
            value = 0.0F;
            break;
        case 7: // negative zeros
            value = -0.0F;
            break;
        case 8: // a power past the float range
            value = 1e30F * uniform(rng);
            break;
        case 9: // a power below it
            value = 1e-30F * uniform(rng);
            break;
        case 10: // a loud tone
            value = 30000.0F * static_cast<float>(std::cos(kPi2 * 50.0 * t / 400.0 + 0.3));
            break;
        case 11: // half silent
            value = i < size / 2 ? uniform(rng) : 0.0F;
            break;
        case 12: // sums that cancel
            value = (i % 3) == 0 ? 1.0F : -0.5F;
            break;
        case 13: // near the largest float
            value = 3.0e38F * uniform(rng);
            break;
        case 14: // one infinity
            value = i == size / 3 ? infinity : uniform(rng);
            break;
        case 15: // two infinities of opposite sign
            value = i == size / 3 ? -infinity : (i == size / 2 ? infinity : uniform(rng));
            break;
        case 16: // one NaN
            value = i == size / 5 ? std::numeric_limits<float>::quiet_NaN() : uniform(rng);
            break;
        default: // every sample infinite
            value = infinity;
            break;
        }
        x[static_cast<std::size_t>(i)] = value;
    }
    return x;
}

// Where no multiply-add is contracted (x86-64 by default) every operation
// rounds the same in any compile context, so the power spectrum must equal the
// definition bit for bit. Where FMAs are contracted (__FP_FAST_FMA), which
// product fuses depends on how the compiler shapes each loop, so this file's
// own copy of the definition can round differently from the library: it does
// on x86-64 with -mfma. A double rounding moves a power by at most one float
// ulp, or changes a bin that holds only rounding noise, so there a bin may
// differ by one float ulp or by 1e-12 of its frame's peak power: far below any
// wrong twiddle factor, order or scale. On aarch64 the two round alike (GCC
// 13, -O2 and -O3), and the comparison stays exact: it is what holds the
// twiddle table to its two arrays there.
#if defined(__FP_FAST_FMA) && !defined(__aarch64__)
constexpr double kPowerPeakTolerance = 1e-12;
#else
constexpr double kPowerPeakTolerance = 0.0;
#endif

bool same_power(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (same_values(actual, expected)) {
        return true;
    }
    if (kPowerPeakTolerance == 0.0 || actual.size() != expected.size()) {
        return false;
    }
    float peak = 0.0F;
    for (const float value : expected) {
        if (std::isfinite(value)) {
            peak = std::max(peak, value);
        }
    }
    const double tolerance = kPowerPeakTolerance * static_cast<double>(peak);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0 &&
                !(std::isnan(actual[i]) && std::isnan(expected[i]))) {
                return false;
            }
        } else if (std::nextafter(actual[i], expected[i]) != expected[i] &&
                   std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i])) >
                       tolerance) {
            return false;
        }
    }
    return true;
}

void test_power_spectrum_equals_definition() {
    std::mt19937 rng(20261116);
    // 400 is Whisper's frame. The others cover the direct radix-2 path at
    // every size from 1 to 1024 and Bluestein's path at seven transform sizes
    // from 8 to 2048.
    for (const int size : {1,   2,   3,   4,   5,   7,   8,   16,  25,  32,  64,  100,
                           128, 160, 200, 256, 399, 400, 401, 480, 512, 513, 800, 1024}) {
        for (int kind = 0; kind < 18; ++kind) {
            const auto input = power_input(size, kind, rng);
            if (!same_power(trtmc::whisper::detail::rfft_power(input),
                            definition::rfft_power(input))) {
                std::fprintf(stderr, "FAIL: power spectrum of %d samples, input kind %d\n", size,
                             kind);
                ++failures;
            }
        }
    }
}

// Overlapping triangular filters over the frequency bins: each row holds at
// most two adjacent nonzero coefficients, as the Slaney filterbank's rows do.
std::vector<float> triangular_filterbank(int mels) {
    std::vector<float> filters(201 * static_cast<std::size_t>(mels), 0.0F);
    for (int f = 0; f < 201; ++f) {
        const double position = static_cast<double>(f) * (mels + 1) / 201.0;
        const int below = static_cast<int>(position);
        const float weight = static_cast<float>(position - below);
        if (below >= 1 && below <= mels) {
            filters[static_cast<std::size_t>(f) * mels + (below - 1)] = 1.0F - weight;
        }
        if (below < mels && weight != 0.0F) {
            filters[static_cast<std::size_t>(f) * mels + below] = weight;
        }
    }
    return filters;
}

void check_mel_equals_definition(const std::vector<float>& audio, const std::vector<float>& filters,
                                 const definition::MelCase& c, const std::string& what) {
    const auto actual = trtmc::whisper::extract_mel_spectrogram(
        audio.data(), static_cast<int32_t>(audio.size()), filters.data(), c.n_fft / 2 + 1,
        c.n_mel_bins, c.n_fft, c.hop_length, c.chunk_length_s, c.sample_rate);
    const auto expected = definition::mel(audio, filters, c);
    const int32_t frames = c.chunk_length_s * c.sample_rate / c.hop_length;
    const bool same = actual.n_mels == c.n_mel_bins && actual.n_frames == frames &&
                      same_values(actual.data, expected);
    if (!same) {
        std::fprintf(stderr, "FAIL: mel of %zu samples, frame %d, hop %d, %d bins, %s\n",
                     audio.size(), c.n_fft, c.hop_length, c.n_mel_bins, what.c_str());
        ++failures;
    }
}

void check_mel_equals_definition(const std::vector<float>& audio, const std::vector<float>& filters,
                                 int mels, const std::string& what) {
    check_mel_equals_definition(audio, filters, definition::MelCase{400, 160, 30, 16000, mels},
                                what);
}

void test_mel_equals_definition() {
    std::mt19937 rng(75918);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    const auto noise = [&](int samples) {
        std::vector<float> audio(static_cast<std::size_t>(samples));
        for (float& value : audio) {
            value = uniform(rng);
        }
        return audio;
    };
    const auto sparse80 = triangular_filterbank(80);
    const auto sparse128 = triangular_filterbank(128);
    std::vector<float> dense(201 * 80);
    for (float& value : dense) {
        value = 0.5F + 0.5F * uniform(rng);
    }
    auto zero_rows = sparse80;
    for (int f = 0; f < 201; f += 3) {
        std::fill_n(zero_rows.begin() + static_cast<std::ptrdiff_t>(f) * 80, 80, 0.0F);
    }
    // Every row is nonzero, and none reaches the first or the last column.
    auto inner_columns = dense;
    for (int f = 0; f < 201; ++f) {
        inner_columns[static_cast<std::size_t>(f) * 80] = 0.0F;
        inner_columns[static_cast<std::size_t>(f) * 80 + 79] = 0.0F;
    }
    auto interior_zero = dense;
    for (int f = 0; f < 201; ++f) {
        interior_zero[static_cast<std::size_t>(f) * 80 + 40] = 0.0F;
        interior_zero[static_cast<std::size_t>(f) * 80 + (f % 80)] = 0.0F;
    }
    // The ends of a row's span are whatever is not zero: a negative, a small
    // or a NaN coefficient there still counts. Each row is four columns wide.
    // Negative ends share their columns with other rows' positive weights (a
    // negative sum would be floored) and small ends sit alone in theirs. NaN
    // ends are in every tenth row only, so most columns keep finite values.
    std::vector<float> negative_edges(201 * 80, 0.0F);
    std::vector<float> nan_edges(201 * 80, 0.0F);
    std::vector<float> small_edges(201 * 80, 0.0F);
    for (int f = 0; f < 201; ++f) {
        const int column = (f * 5) % 77;
        float* row = negative_edges.data() + static_cast<std::size_t>(f) * 80;
        row[column] = -0.25F;
        row[column + 1] = 0.5F;
        row[column + 2] = 0.25F;
        row[column + 3] = -0.125F;
        row = nan_edges.data() + static_cast<std::size_t>(f) * 80;
        row[column] = 0.125F;
        row[column + 1] = 0.5F;
        row[column + 2] = 0.25F;
        row[column + 3] = 0.125F;
        if ((f % 10) == 3) {
            row[column + ((f % 20) == 3 ? 0 : 3)] = std::numeric_limits<float>::quiet_NaN();
        }
        if (f < 20) {
            row = small_edges.data() + static_cast<std::size_t>(f) * 80 + 4 * f;
            row[0] = 5e-7F;
            row[1] = 0.5F;
            row[2] = 0.25F;
            row[3] = 5e-7F;
        }
    }

    // Lengths around the frame, the hop and the 30-second chunk.
    for (const int samples :
         {0, 1, 2, 159, 160, 161, 199, 200, 201, 400, 1600, 479999, 480000, 480160}) {
        check_mel_equals_definition(noise(samples), sparse80, 80, "noise");
    }
    // Every other input and filterbank on a short and a 3-second clip.
    for (const int samples : {1600, 48000}) {
        const auto audio = noise(samples);
        check_mel_equals_definition(audio, sparse128, 128, "noise, 128 bins");
        check_mel_equals_definition(audio, dense, 80, "noise, dense filterbank");
        check_mel_equals_definition(audio, zero_rows, 80, "noise, filterbank with zero rows");
        check_mel_equals_definition(audio, inner_columns, 80, "noise, empty outer columns");
        check_mel_equals_definition(audio, interior_zero, 80,
                                    "noise, zeros inside the filterbank rows");
        check_mel_equals_definition(audio, negative_edges, 80,
                                    "noise, negative coefficients at span ends");
        check_mel_equals_definition(audio, small_edges, 80,
                                    "noise, small coefficients at span ends");
        check_mel_equals_definition(audio, nan_edges, 80, "noise, NaN coefficients at span ends");
        check_mel_equals_definition(tone(samples), sparse80, 80, "tone");
        check_mel_equals_definition(std::vector<float>(static_cast<std::size_t>(samples), 0.0F),
                                    sparse80, 80, "silence");

        auto loud = tone(samples);
        for (float& value : loud) {
            value *= 60000.0F;
        }
        check_mel_equals_definition(loud, sparse80, 80, "loud tone");

        // A power that is not finite keeps the whole filterbank row.
        auto broken = audio;
        broken[static_cast<std::size_t>(samples / 2)] = std::numeric_limits<float>::infinity();
        check_mel_equals_definition(broken, sparse80, 80, "one infinite sample");
        check_mel_equals_definition(broken, zero_rows, 80, "one infinite sample, zero rows");
        check_mel_equals_definition(broken, inner_columns, 80,
                                    "one infinite sample, empty outer columns");
        broken = audio;
        broken[static_cast<std::size_t>(samples / 3)] = std::numeric_limits<float>::quiet_NaN();
        check_mel_equals_definition(broken, sparse80, 80, "one NaN sample");
        broken = audio;
        for (float& value : broken) {
            value *= 3.0e38F;
        }
        check_mel_equals_definition(broken, sparse80, 80, "samples near the largest float");
        // A power that is not finite is the first to need a filterbank row,
        // and finite powers need it afterwards: an infinite first sample, then
        // a tone too loud for a float power in its own bins only.
        broken = audio;
        broken[0] = std::numeric_limits<float>::infinity();
        check_mel_equals_definition(broken, sparse80, 80, "an infinite first sample");
        check_mel_equals_definition(broken, inner_columns, 80,
                                    "an infinite first sample, empty outer columns");
        broken = audio;
        const auto loud_start = tone(800);
        for (std::size_t i = 0; i < loud_start.size(); ++i) {
            broken[i] = 3.0e18F * loud_start[i];
        }
        check_mel_equals_definition(broken, sparse80, 80, "a start too loud for a float power");
        check_mel_equals_definition(broken, inner_columns, 80,
                                    "a start too loud for a float power, empty outer columns");

        // Two different NaNs and an infinity in one frame.
        broken = audio;
        const std::uint32_t payload = 0x7fc12345U;
        std::memcpy(&broken[static_cast<std::size_t>(samples / 2)], &payload, sizeof(float));
        broken[static_cast<std::size_t>(samples / 2 + 40)] =
            -std::numeric_limits<float>::quiet_NaN();
        broken[static_cast<std::size_t>(samples / 2 + 100)] =
            std::numeric_limits<float>::infinity();
        check_mel_equals_definition(broken, sparse80, 80, "two NaNs and an infinity in one frame");
    }

    // Other frame geometries than Whisper's, each over many frames of one
    // extraction: a power-of-two frame (the direct radix-2 path), a frame
    // that is neither, and a hop that does not divide the chunk.
    for (const definition::MelCase& c :
         {definition::MelCase{512, 160, 1, 16000, 40}, definition::MelCase{256, 128, 2, 8000, 23},
          definition::MelCase{200, 80, 1, 8000, 16}, definition::MelCase{400, 150, 1, 16000, 80}}) {
        std::vector<float> filters(static_cast<std::size_t>(c.n_fft / 2 + 1) * c.n_mel_bins);
        for (float& value : filters) {
            value = uniform(rng) > 0.3F ? 0.0F : 0.5F + 0.5F * uniform(rng);
        }
        const int chunk = c.chunk_length_s * c.sample_rate;
        for (const int samples : {chunk / 3, chunk, chunk + 7}) {
            check_mel_equals_definition(noise(samples), filters, c, "noise");
            check_mel_equals_definition(tone(samples), filters, c, "tone");
        }
    }
}

} // namespace

int main() {
    for (const int mels : {80, 128}) {
        test_center_padding(mels);
        test_discarded_frame_does_not_set_log_floor(mels);
        test_empty_and_single_sample(mels);
    }
    test_filterbank_is_read_only_for_nonzero_power();
    test_power_spectrum_equals_definition();
    test_mel_equals_definition();
    if (failures == 0) {
        std::printf("Whisper mel frontend tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
