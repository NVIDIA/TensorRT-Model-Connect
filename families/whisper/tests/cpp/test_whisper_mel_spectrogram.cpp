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

// The frame lengths the production code transforms at half their size by a
// mixed-radix transform of the sample pairs: even, not a power of two, and
// with no prime factor of the half other than 2, 3 and 5. Whisper's frame of
// 400 samples is one. Every other length keeps a radix-2 transform.
bool takes_mixed_radix_path(int32_t n) {
    if (n <= 0 || n % 2 != 0 || (n & (n - 1)) == 0) {
        return false;
    }
    int32_t rest = n / 2;
    for (const int32_t factor : {2, 3, 5}) {
        while (rest % factor == 0) {
            rest /= factor;
        }
    }
    return rest == 1;
}

// The frontend's straightforward definition, which the production code must
// equal bit for bit: per frame, the Hann window, the power spectrum by a
// radix-2 transform that updates its twiddle factor inline in every butterfly
// group (through Bluestein's two transforms when the length is not a power of
// two), the dense filterbank product, then the last frame dropped and the
// log-mel normalization.
//
// A frame length on the mixed-radix path is the exception. Its production
// power spectrum is held to the exact transform within one float ulp
// (same_as_exact_power), not to this definition's transform bit for bit, and
// mel() takes such a frame's power values from the production seam: the
// window, the padding, the filterbank product, the dropped frame and the
// normalization stay held bit for bit there too.
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
        const auto power = takes_mixed_radix_path(c.n_fft)
                               ? trtmc::whisper::detail::rfft_power(frame)
                               : rfft_power(frame);
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

// The exact power spectrum of a frame: a naive O(n^2) transform accumulated
// in long double, each power rounded to float once. The roots of unity come
// from angles folded into [0, pi / 4], and the index k * j is reduced mod n
// in integers.
class ExactPower {
  public:
    explicit ExactPower(int size)
        : size_(size), cos_(static_cast<std::size_t>(size)), sin_(static_cast<std::size_t>(size)) {
        const long double quarter_pi = std::atan(1.0L);
        for (int r = 0; r < size_; ++r) {
            const int octant = 8 * r / size_;
            const int rest = 8 * r % size_;
            const bool mirrored = (octant & 1) != 0;
            const long double angle = quarter_pi *
                                      static_cast<long double>(mirrored ? size_ - rest : rest) /
                                      static_cast<long double>(size_);
            long double c = std::cos(angle);
            long double s = std::sin(angle);
            if (mirrored) {
                std::swap(c, s);
            }
            // A quarter turn takes (cos, sin) to (-sin, cos).
            for (int quarter = octant / 2; quarter > 0; --quarter) {
                const long double turned = -s;
                s = c;
                c = turned;
            }
            cos_[static_cast<std::size_t>(r)] = c;
            sin_[static_cast<std::size_t>(r)] = s;
        }
    }

    std::vector<float> operator()(const std::vector<float>& input) const {
        std::vector<float> power(static_cast<std::size_t>(size_ / 2 + 1));
        for (int k = 0; k <= size_ / 2; ++k) {
            long double real = 0.0L;
            long double imag = 0.0L;
            int r = 0;
            for (int j = 0; j < size_; ++j) {
                const long double value = input[static_cast<std::size_t>(j)];
                real += value * cos_[static_cast<std::size_t>(r)];
                imag -= value * sin_[static_cast<std::size_t>(r)];
                r += k;
                if (r >= size_) {
                    r -= size_;
                }
            }
            power[static_cast<std::size_t>(k)] = static_cast<float>(real * real + imag * imag);
        }
        return power;
    }

  private:
    int size_;
    std::vector<long double> cos_;
    std::vector<long double> sin_;
};

// The reference is exact only where long double is wider than double: it has
// 64 mantissa digits on x86-64 and 113 on aarch64. Where it has fewer, a test
// that needs the reference says so in one line and is skipped.
constexpr int kExactReferenceDigits = 64;

bool has_exact_reference(const char* test) {
    const int digits = std::numeric_limits<long double>::digits;
    if (digits >= kExactReferenceDigits) {
        return true;
    }
    std::printf("skipped, %s: long double has %d mantissa digits here and the exact reference "
                "needs %d\n",
                test, digits, kExactReferenceDigits);
    return false;
}

// How many floats apart two values that are not NaN sit on the ordered line
// of floats: +0 and -0 share a place, and an infinity is one step past the
// largest finite value.
std::int64_t ulp_distance(float a, float b) {
    const auto place = [](float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const std::int64_t magnitude = bits & 0x7fffffffU;
        return (bits >> 31) != 0 ? -magnitude : magnitude;
    };
    const std::int64_t distance = place(a) - place(b);
    return distance < 0 ? -distance : distance;
}

float finite_peak(const std::vector<float>& power) {
    float peak = 0.0F;
    for (const float value : power) {
        if (std::isfinite(value)) {
            peak = std::max(peak, value);
        }
    }
    return peak;
}

// On the mixed-radix path the reference is the exact transform, and no longer
// the Bluestein definition, which is the less accurate of the two there
// (test_mixed_radix_power_against_bluestein_definition measures by how much
// they differ). The production code forms each power in double, far inside a
// float ulp of the exact value, and rounds it to float; the exact power is
// rounded once. Such a double rounding lands on the single rounding or on a
// float next to it, so the bound is one float ulp. A bin that holds only the
// transform's own rounding noise is the exception (a tone on a bin leaves
// every other bin as one): its double is not within a float ulp of the exact
// value, and it may differ by kMixedRadixNoise of the frame's peak power
// instead, which only a bin more than 150 dB below the peak can do by more
// than one ulp.
constexpr double kMixedRadixNoise = 1e-22;

bool within_exact_power(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }
    const double tolerance = kMixedRadixNoise * static_cast<double>(finite_peak(expected));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::isnan(actual[i])) {
            return false;
        }
        const double difference =
            std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        if (ulp_distance(actual[i], expected[i]) > 1 && !(difference <= tolerance)) {
            return false;
        }
    }
    return true;
}

// A frame that holds a sample that is not finite must give NaN in every bin.
bool same_as_exact_power(const std::vector<float>& actual, const std::vector<float>& input,
                         const ExactPower& exact) {
    if (!std::all_of(input.begin(), input.end(),
                     [](float value) { return std::isfinite(value); })) {
        return actual.size() == input.size() / 2 + 1 &&
               std::all_of(actual.begin(), actual.end(),
                           [](float value) { return std::isnan(value); });
    }
    return within_exact_power(actual, exact(input));
}

void test_power_spectrum_equals_definition() {
    std::mt19937 rng(20261116);
    const bool exact_reference =
        has_exact_reference("the mixed-radix lengths of the power spectrum test");
    // 400 is Whisper's frame: it and 100, 160, 200, 480 and 800 take the
    // mixed-radix path and are held to the exact transform. The others are
    // held to the definition. They cover the direct radix-2 path at every
    // size from 1 to 1024 and Bluestein's path at eight transform sizes from
    // 8 to 2048; 14, 202 and 398 are even lengths that keep Bluestein's path,
    // their half having another prime factor than 2, 3 and 5.
    for (const int size : {1,   2,   3,   4,   5,   7,   8,   16,  25,  32,   64, 100, 128, 160,
                           200, 256, 399, 400, 401, 480, 512, 513, 800, 1024, 14, 101, 202, 398}) {
        const ExactPower exact(size);
        for (int kind = 0; kind < 18; ++kind) {
            const auto input = power_input(size, kind, rng);
            if (takes_mixed_radix_path(size) && !exact_reference) {
                continue;
            }
            const auto actual = trtmc::whisper::detail::rfft_power(input);
            const bool same = takes_mixed_radix_path(size)
                                  ? same_as_exact_power(actual, input, exact)
                                  : same_power(actual, definition::rfft_power(input));
            if (!same) {
                std::fprintf(stderr, "FAIL: power spectrum of %d samples, input kind %d\n", size,
                             kind);
                ++failures;
            }
        }
    }
}

// Frames as the front end forms them: `size` samples of a 16 kHz signal that
// start at sample `start`, times the Hann window, as a float product. The
// noise comes from the generator's raw bits, so a frame is the same wherever
// the C library's cos and sin agree.
std::vector<float> signal_frame(int size, int kind, int start, std::mt19937& rng) {
    constexpr double kPi2 = 2.0 * 3.14159265358979323846;
    std::vector<float> frame(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) {
        const float noise = static_cast<float>(rng() >> 8) / 8388608.0F - 1.0F;
        const double t = static_cast<double>(start + i) / 16000.0;
        float sample = 0.0F;
        switch (kind) {
        case 0: // noise at full scale
            sample = noise;
            break;
        case 1: // noise 40 dB down
            sample = 1e-2F * noise;
            break;
        case 2: // noise 80 dB down
            sample = 1e-4F * noise;
            break;
        case 3: // three tones between bins, and noise 80 dB down
            sample = static_cast<float>(0.3 * std::sin(kPi2 * 440.0 * t) +
                                        0.2 * std::sin(kPi2 * 1000.0 * t) +
                                        0.1 * std::sin(kPi2 * 3217.3 * t)) +
                     1e-4F * noise;
            break;
        default: // a clean chirp from 50 Hz to 7.9 kHz over 30 seconds
            sample = static_cast<float>(
                0.4 * std::sin(kPi2 * (50.0 * t + 0.5 * (7850.0 / 30.0) * t * t)));
            break;
        }
        const float window = static_cast<float>(
            0.5 * (1.0 - std::cos(kPi2 * static_cast<double>(i) / static_cast<double>(size))));
        frame[static_cast<std::size_t>(i)] = sample * window;
    }
    return frame;
}
constexpr int kSignalKinds = 5;

// Frames whose powers sit at the two ends of the float range, where it shows
// if a power is formed or scaled in float before it is rounded: it then
// overflows in the two binades below the largest float, and is rounded twice
// among the subnormals. Two impulses, x[1] = a and x[4] = b, give the powers
// a^2 + b^2 + 2 a b cos(6 pi k / n), from (a - b)^2 to (a + b)^2. Noise of
// amplitude c / sqrt(n) has a mean power of c^2 / 3.
std::vector<float> edge_frame(int size, int kind, std::mt19937& rng) {
    std::vector<float> frame(static_cast<std::size_t>(size), 0.0F);
    const auto noise = [&](float amplitude) {
        const float scale = amplitude / std::sqrt(static_cast<float>(size));
        for (float& sample : frame) {
            sample = scale * (static_cast<float>(rng() >> 8) / 8388608.0F - 1.0F);
        }
    };
    switch (kind) {
    case 0: // powers from 6.4e37 to 2.56e38, in the top two binades and under them
        frame[1] = 1.2e19F;
        frame[4] = 0.4e19F;
        break;
    case 1: // a mean power of 1.3e38: the strongest bins pass the largest float
        noise(2e19F);
        break;
    case 2: // powers from 1e-40 to 9e-40, subnormal
        frame[1] = 2e-20F;
        frame[4] = 1e-20F;
        break;
    case 3: // powers from 9e-46 to 8.1e-45, one to six times the smallest subnormal
        frame[1] = 6e-23F;
        frame[4] = 3e-23F;
        break;
    default: // a mean power of 1.3e-40: every bin subnormal
        noise(2e-20F);
        break;
    }
    return frame;
}
constexpr int kEdgeKinds = 5;

// The mixed-radix path against the exact transform on frames like the front
// end's, at Whisper's frame and at lengths whose halves cover the stage
// plans: radix 3 and radix 5 alone, each after radix 2 and after radix 4,
// repeated, and all four together. Then on the frames at the ends of the
// float range.
void test_mixed_radix_power_spectrum_is_exact() {
    if (!has_exact_reference("the mixed-radix path against the exact transform")) {
        return;
    }
    std::mt19937 rng(20261117);
    for (const int size : {6,   10,  12,  18,  20,  24,  30,  36,  48,  50,  60,  72,  96,
                           150, 162, 250, 300, 320, 360, 384, 400, 480, 600, 640, 720, 800}) {
        check(takes_mixed_radix_path(size), "the length is on the mixed-radix path");
        const ExactPower exact(size);
        const int frames = size == 400 ? 4 : 1;
        for (int kind = 0; kind < kSignalKinds; ++kind) {
            for (int index = 0; index < frames; ++index) {
                const auto input = signal_frame(size, kind, 120000 * index + 977 * kind, rng);
                if (!same_as_exact_power(trtmc::whisper::detail::rfft_power(input), input, exact)) {
                    std::fprintf(stderr,
                                 "FAIL: exact power spectrum of %d samples, signal kind %d, "
                                 "frame %d\n",
                                 size, kind, index);
                    ++failures;
                }
            }
        }
    }

    // How many exact powers of the edge frames fall in the binade under the
    // largest float, in the one under that, past the largest float, and among
    // the subnormals: each place must be reached.
    const float largest = std::numeric_limits<float>::max();
    const float smallest_normal = std::numeric_limits<float>::min();
    long top_binade = 0;
    long second_binade = 0;
    long overflowed = 0;
    long subnormal = 0;
    for (const int size : {12, 100, 400, 480}) {
        const ExactPower exact(size);
        for (int kind = 0; kind < kEdgeKinds; ++kind) {
            const auto input = edge_frame(size, kind, rng);
            const auto expected = exact(input);
            for (const float power : expected) {
                top_binade += power > largest / 2.0F && power <= largest ? 1 : 0;
                second_binade += power > largest / 4.0F && power <= largest / 2.0F ? 1 : 0;
                overflowed += std::isinf(power) ? 1 : 0;
                subnormal += power > 0.0F && power < smallest_normal ? 1 : 0;
            }
            if (!within_exact_power(trtmc::whisper::detail::rfft_power(input), expected)) {
                std::fprintf(stderr,
                             "FAIL: exact power spectrum of %d samples, float range edge kind "
                             "%d\n",
                             size, kind);
                ++failures;
            }
        }
    }
    check(top_binade > 0 && second_binade > 0 && overflowed > 0 && subnormal > 0,
          "the edge frames reach the top two binades of the float range, the overflow and the "
          "subnormals");
}

// What the mixed-radix path changed for Whisper's frame, measured against the
// Bluestein definition the production code equalled bit for bit before it:
// over a fixed set of frames, how many power values differ and by how much.
// A value may differ by one float ulp. It may differ by more only within
// kBluesteinError of its frame's peak power, which keeps such a difference
// to bins far below the peak. There the Bluestein path's own error is many
// ulps of the bin, and a clean chirp has such bins: against the exact
// transform the definition is more than one ulp off in about 700 of these
// values, the mixed-radix path in none. On these frames the largest
// difference of more than one ulp was measured at 4.4e-21 of the peak power,
// the same in every build tried: x86-64 from -O0 to -O3, with and without
// contracted multiply-adds, and aarch64 at -O2 and -O3. The bound leaves a
// factor of about 200 above that for another compiler's or C library's
// rounding. The counts are printed so that they can be quoted.
constexpr double kBluesteinError = 1e-18;

void test_mixed_radix_power_against_bluestein_definition() {
    std::mt19937 rng(400);
    long frames = 0;
    long values = 0;
    long differing = 0;
    long beyond_one_ulp = 0;
    std::int64_t largest_ulp = 0;
    double largest_of_peak = 0.0;
    bool within_bound = true;
    for (int kind = 0; kind < kSignalKinds; ++kind) {
        const int count = kind == kSignalKinds - 1 ? 64 : 16;
        for (int index = 0; index < count; ++index) {
            const auto input = signal_frame(400, kind, (480000 / count) * index, rng);
            const auto actual = trtmc::whisper::detail::rfft_power(input);
            const auto expected = definition::rfft_power(input);
            const double peak = static_cast<double>(finite_peak(expected));
            ++frames;
            for (std::size_t i = 0; i < expected.size(); ++i) {
                ++values;
                if (std::memcmp(&actual[i], &expected[i], sizeof(float)) == 0) {
                    continue;
                }
                ++differing;
                if (std::isnan(actual[i]) || std::isnan(expected[i]) || !(peak > 0.0)) {
                    within_bound = false;
                    continue;
                }
                const std::int64_t ulp = ulp_distance(actual[i], expected[i]);
                const double of_peak =
                    std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i])) /
                    peak;
                largest_ulp = std::max(largest_ulp, ulp);
                if (ulp > 1) {
                    ++beyond_one_ulp;
                    largest_of_peak = std::max(largest_of_peak, of_peak);
                    if (!(of_peak <= kBluesteinError)) {
                        within_bound = false;
                    }
                }
            }
        }
    }
    std::printf("mixed-radix path against the Bluestein definition, %ld frames of 400 samples: "
                "%ld of %ld power values differ, by at most %lld ulp; %ld by more than one ulp, "
                "at most %.3g of their frame's peak power\n",
                frames, differing, values, static_cast<long long>(largest_ulp), beyond_one_ulp,
                largest_of_peak);
    check(within_bound, "a power of Whisper's frame is within one ulp of the Bluestein "
                        "definition's, or within 1e-18 of the frame's peak power");
}

// A frame on the mixed-radix path that holds a sample that is not finite
// gives NaN in every bin, wherever the sample sits and whatever the other
// samples are.
void test_mixed_radix_non_finite_frame_is_all_nan() {
    std::mt19937 rng(20261118);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const auto all_nan = [](const std::vector<float>& frame) {
        const auto power = trtmc::whisper::detail::rfft_power(frame);
        return power.size() == frame.size() / 2 + 1 &&
               std::all_of(power.begin(), power.end(),
                           [](float value) { return std::isnan(value); });
    };
    for (const int size : {6, 10, 12, 100, 400, 480}) {
        // The other samples: noise, silence, a constant, values near the
        // largest float.
        for (int base = 0; base < 4; ++base) {
            std::vector<float> clean(static_cast<std::size_t>(size));
            for (float& value : clean) {
                const float noise = static_cast<float>(rng() >> 8) / 8388608.0F - 1.0F;
                value = base == 0 ? noise : base == 1 ? 0.0F : base == 2 ? 1.0F : 3.0e38F * noise;
            }
            bool every_bin_nan = true;
            for (const float value : {nan, infinity, -infinity}) {
                // One sample: the first, the last, and an even and an odd
                // place in the middle.
                for (const int place : {0, 1, size / 2 - 1, size / 2, size - 2, size - 1}) {
                    auto frame = clean;
                    frame[static_cast<std::size_t>(place)] = value;
                    every_bin_nan = every_bin_nan && all_nan(frame);
                }
                // Two samples, in the same and in the other half of the
                // sample pairs.
                for (const float other : {nan, infinity, -infinity}) {
                    for (const int place : {1, 2, size / 2, size - 1}) {
                        auto frame = clean;
                        frame[0] = value;
                        frame[static_cast<std::size_t>(place)] = other;
                        every_bin_nan = every_bin_nan && all_nan(frame);
                    }
                }
                every_bin_nan = every_bin_nan && all_nan(std::vector<float>(clean.size(), value));
            }
            if (!every_bin_nan) {
                std::fprintf(stderr,
                             "FAIL: a frame of %d samples with a sample that is not finite "
                             "(other samples kind %d) has a bin that is not NaN\n",
                             size, base);
                ++failures;
            }
        }
    }
}

// The rule must not take a finite frame for one that is not. A constant frame
// of the largest float, positive or negative, makes the largest sums a frame
// can: none of its bins may be NaN, and its zero-frequency power, the square
// of the frame's sum, is past the float range.
void test_mixed_radix_largest_finite_frame_has_no_nan() {
    const float largest = std::numeric_limits<float>::max();
    for (const int size : {12, 100, 400, 480, 800}) {
        for (const float value : {largest, -largest}) {
            const auto power = trtmc::whisper::detail::rfft_power(
                std::vector<float>(static_cast<std::size_t>(size), value));
            const bool finite_rule = power.size() == static_cast<std::size_t>(size / 2 + 1) &&
                                     std::none_of(power.begin(), power.end(),
                                                  [](float bin) { return std::isnan(bin); }) &&
                                     power[0] == std::numeric_limits<float>::infinity();
            if (!finite_rule) {
                std::fprintf(stderr,
                             "FAIL: a constant frame of %d samples of %g has a NaN bin, or its "
                             "zero-frequency power is not +inf\n",
                             size, static_cast<double>(value));
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
    // extraction: a power-of-two frame (the direct radix-2 path), another
    // frame on the mixed-radix path, a hop that does not divide the chunk,
    // and a frame on Bluestein's path.
    for (const definition::MelCase& c :
         {definition::MelCase{512, 160, 1, 16000, 40}, definition::MelCase{256, 128, 2, 8000, 23},
          definition::MelCase{200, 80, 1, 8000, 16}, definition::MelCase{400, 150, 1, 16000, 80},
          definition::MelCase{202, 80, 1, 8000, 16}}) {
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
    test_mixed_radix_power_spectrum_is_exact();
    test_mixed_radix_power_against_bluestein_definition();
    test_mixed_radix_non_finite_frame_is_all_nan();
    test_mixed_radix_largest_finite_frame_has_no_nan();
    test_mel_equals_definition();
    if (failures == 0) {
        std::printf("Whisper mel frontend tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
