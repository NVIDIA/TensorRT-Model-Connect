/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/whisper/runtime/whisper_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace trtmc {
namespace whisper {

namespace {

std::vector<float> make_hann_window(int32_t length) {
    std::vector<float> window(length);
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t i = 0; i < length; ++i) {
        window[i] = static_cast<float>(
            0.5 * (1.0 - std::cos(pi2 * static_cast<double>(i) / static_cast<double>(length))));
    }
    return window;
}

bool is_power_of_two(int32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

// In-place radix-2 transform of one power-of-two size in one direction, without
// the inverse transform's 1/size scale. The bit-reversed order and every
// stage's twiddle factors are tabulated once per instance: stage `length` uses
// w_i = step^i (i < length / 2), produced by the recurrence w_0 = 1,
// w_{i+1} = w_i * step, the same complex operations in the same order as an
// inline per-butterfly update. The values are then the inline update's bit for
// bit wherever the compiler fuses the same multiply-adds in both shapes: with
// the Release flags it fuses none on x86-64, and the same ones on aarch64
// (GCC 13).
class Radix2Transform {
  public:
    Radix2Transform(std::size_t size, bool inverse) : size_(size) {
        bit_reversed_.resize(size_, 0);
        for (std::size_t i = 1, j = 0; i < size_; ++i) {
            std::size_t bit = size_ >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            bit_reversed_[i] = j;
        }
        // The real and imaginary parts go to separate tables: stored side by
        // side as std::complex, the aarch64 build (GCC 13) fuses multiply-adds
        // differently and no longer matches the inline update bit for bit.
        twiddles_real_.reserve(size_ > 1 ? size_ - 1 : 0);
        twiddles_imag_.reserve(size_ > 1 ? size_ - 1 : 0);
        constexpr double kPi2 = 2.0 * 3.14159265358979323846;
        for (std::size_t length = 2; length <= size_; length <<= 1) {
            const double angle = (inverse ? kPi2 : -kPi2) / static_cast<double>(length);
            const std::complex<double> step(std::cos(angle), std::sin(angle));
            std::complex<double> twiddle(1.0, 0.0);
            for (std::size_t i = 0; i < length / 2; ++i) {
                twiddles_real_.push_back(twiddle.real());
                twiddles_imag_.push_back(twiddle.imag());
                twiddle *= step;
            }
        }
    }

    // Where natural-order element i sits once the values are bit-reversed
    // (the permutation is its own inverse).
    std::size_t bit_reversed(std::size_t i) const { return bit_reversed_[i]; }

    // The whole transform of values in natural order.
    void operator()(std::complex<double>* values) const {
        for (std::size_t i = 1; i < size_; ++i) {
            const std::size_t j = bit_reversed_[i];
            if (i < j) {
                std::swap(values[i], values[j]);
            }
        }
        butterflies(values);
    }

    // The butterfly stages alone, for values already in bit-reversed order.
    void butterflies(std::complex<double>* values) const {
        const double* stage_real = twiddles_real_.data();
        const double* stage_imag = twiddles_imag_.data();
        for (std::size_t length = 2; length <= size_; length <<= 1) {
            const std::size_t half = length / 2;
            for (std::size_t offset = 0; offset < size_; offset += length) {
                for (std::size_t i = 0; i < half; ++i) {
                    const std::complex<double> twiddle(stage_real[i], stage_imag[i]);
                    const auto even = values[offset + i];
                    const auto odd = values[offset + i + half] * twiddle;
                    values[offset + i] = even + odd;
                    values[offset + i + half] = even - odd;
                }
            }
            stage_real += half;
            stage_imag += half;
        }
    }

  private:
    std::size_t size_;
    std::vector<std::size_t> bit_reversed_;
    std::vector<double> twiddles_real_;
    std::vector<double> twiddles_imag_;
};

// exp(-2 pi i k / n) for k < n. The angle is folded into the first eighth of a
// turn before the cosine and sine are taken, so their argument stays within
// [0, pi / 4] and the roots at multiples of a quarter turn are exact.
void unit_root(std::size_t k, std::size_t n, double& real, double& imag) {
    constexpr double kQuarterPi = 0.78539816339744830962;
    const std::size_t eighths = 8 * k;
    const std::size_t octant = eighths / n;
    const std::size_t rest = eighths % n;
    const bool mirrored = (octant & 1U) != 0;
    const double angle =
        kQuarterPi * static_cast<double>(mirrored ? n - rest : rest) / static_cast<double>(n);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    double cosine = c;
    double sine = s;
    switch (octant) {
    case 0:
        break;
    case 1:
        cosine = s;
        sine = c;
        break;
    case 2:
        cosine = -s;
        sine = c;
        break;
    case 3:
        cosine = -c;
        sine = s;
        break;
    case 4:
        cosine = -c;
        sine = -s;
        break;
    case 5:
        cosine = -s;
        sine = -c;
        break;
    case 6:
        cosine = s;
        sine = -c;
        break;
    default:
        cosine = c;
        sine = -s;
        break;
    }
    real = cosine;
    imag = 0.0 - sine;
}

// The butterflies below work on (real, imaginary) pairs of doubles in place.
// p0 .. p[radix - 1] point at one value of each of the stage's sub-blocks;
// a1 .. are the values of sub-blocks 1 .. already multiplied by their twiddle
// factors. Every product is written out on the real and imaginary parts, so
// none goes through std::complex's operator*.

inline void combine2(double* p0, double* p1, double a1r, double a1i) {
    const double a0r = p0[0];
    const double a0i = p0[1];
    p0[0] = a0r + a1r;
    p0[1] = a0i + a1i;
    p1[0] = a0r - a1r;
    p1[1] = a0i - a1i;
}

inline void combine3(double* p0, double* p1, double* p2, double a1r, double a1i, double a2r,
                     double a2i) {
    constexpr double kSin = 0.86602540378443864676; // sin(2 pi / 3)
    const double a0r = p0[0];
    const double a0i = p0[1];
    const double t1r = a1r + a2r;
    const double t1i = a1i + a2i;
    const double t2r = a1r - a2r;
    const double t2i = a1i - a2i;
    p0[0] = a0r + t1r;
    p0[1] = a0i + t1i;
    const double mr = a0r - 0.5 * t1r;
    const double mi = a0i - 0.5 * t1i;
    const double ur = kSin * t2r;
    const double ui = kSin * t2i;
    p1[0] = mr + ui;
    p1[1] = mi - ur;
    p2[0] = mr - ui;
    p2[1] = mi + ur;
}

inline void combine4(double* p0, double* p1, double* p2, double* p3, double a1r, double a1i,
                     double a2r, double a2i, double a3r, double a3i) {
    const double a0r = p0[0];
    const double a0i = p0[1];
    const double t0r = a0r + a2r;
    const double t0i = a0i + a2i;
    const double t1r = a0r - a2r;
    const double t1i = a0i - a2i;
    const double t2r = a1r + a3r;
    const double t2i = a1i + a3i;
    const double t3r = a1r - a3r;
    const double t3i = a1i - a3i;
    p0[0] = t0r + t2r;
    p0[1] = t0i + t2i;
    p2[0] = t0r - t2r;
    p2[1] = t0i - t2i;
    p1[0] = t1r + t3i;
    p1[1] = t1i - t3r;
    p3[0] = t1r - t3i;
    p3[1] = t1i + t3r;
}

inline void combine5(double* p0, double* p1, double* p2, double* p3, double* p4, double a1r,
                     double a1i, double a2r, double a2i, double a3r, double a3i, double a4r,
                     double a4i) {
    constexpr double kCos1 = 0.30901699437494742410;  // cos(2 pi / 5)
    constexpr double kSin1 = 0.95105651629515357212;  // sin(2 pi / 5)
    constexpr double kCos2 = -0.80901699437494742410; // cos(4 pi / 5)
    constexpr double kSin2 = 0.58778525229247312917;  // sin(4 pi / 5)
    const double a0r = p0[0];
    const double a0i = p0[1];
    const double t1r = a1r + a4r;
    const double t1i = a1i + a4i;
    const double t2r = a2r + a3r;
    const double t2i = a2i + a3i;
    const double t3r = a1r - a4r;
    const double t3i = a1i - a4i;
    const double t4r = a2r - a3r;
    const double t4i = a2i - a3i;
    p0[0] = a0r + t1r + t2r;
    p0[1] = a0i + t1i + t2i;
    const double br = a0r + kCos1 * t1r + kCos2 * t2r;
    const double bi = a0i + kCos1 * t1i + kCos2 * t2i;
    const double ur = kSin1 * t3r + kSin2 * t4r;
    const double ui = kSin1 * t3i + kSin2 * t4i;
    p1[0] = br + ui;
    p1[1] = bi - ur;
    p4[0] = br - ui;
    p4[1] = bi + ur;
    const double cr = a0r + kCos2 * t1r + kCos1 * t2r;
    const double ci = a0i + kCos2 * t1i + kCos1 * t2i;
    const double vr = kSin2 * t3r - kSin1 * t4r;
    const double vi = kSin2 * t3i - kSin1 * t4i;
    p2[0] = cr + vi;
    p2[1] = ci - vr;
    p3[0] = cr - vi;
    p3[1] = ci + vr;
}

// In-place forward transform of one size whose prime factors are all in
// {2, 3, 5}, in double precision, decimation in time: the values are loaded in
// a digit-reversed order and each stage combines `radix` neighbouring
// sub-blocks of `span` values. The stages run radix 4 first, then one radix 2,
// then radix 3, then radix 5. Stage twiddles exp(-2 pi i q j / (radix * span))
// (sub-block q in 1 .. radix - 1, position j in 1 .. span - 1) are tabulated
// once per instance, each from its own angle; position 0 needs none.
class MixedRadixTransform {
  public:
    explicit MixedRadixTransform(std::size_t size) : size_(size) {
        if (size_ == 0) {
            return;
        }
        std::size_t rest = size_;
        std::size_t span = 1;
        auto add_stages = [&](std::size_t radix, bool repeat) {
            while (rest % radix == 0) {
                Stage stage;
                stage.radix = radix;
                stage.span = span;
                stage.twiddles = twiddles_.size();
                const std::size_t block = radix * span;
                for (std::size_t j = 1; j < span; ++j) {
                    for (std::size_t q = 1; q < radix; ++q) {
                        double real = 0.0;
                        double imag = 0.0;
                        unit_root(q * j, block, real, imag);
                        twiddles_.push_back(real);
                        twiddles_.push_back(imag);
                    }
                }
                stages_.push_back(stage);
                span = block;
                rest /= radix;
                if (!repeat) {
                    break;
                }
            }
        };
        add_stages(4, true);
        add_stages(2, false);
        add_stages(3, true);
        add_stages(5, true);
        // rest is 1 here: the plan builds this transform for such sizes only.
        sources_.resize(size_);
        fill_sources(0, 0, 1, stages_.size());
    }

    std::size_t size() const { return size_; }

    // The natural-order input index whose value slot i takes before the stages.
    std::size_t source(std::size_t i) const { return sources_[i]; }

    // All stages over `size` (real, imaginary) pairs loaded in source() order.
    void butterflies(double* values) const {
        for (const Stage& stage : stages_) {
            const double* twiddles = twiddles_.data() + stage.twiddles;
            const std::size_t span = stage.span;
            const std::size_t step = 2 * span;
            for (std::size_t block = 0; block < size_; block += stage.radix * span) {
                double* p0 = values + 2 * block;
                const double* w = twiddles;
                if (stage.radix == 4) {
                    double* p1 = p0 + step;
                    double* p2 = p1 + step;
                    double* p3 = p2 + step;
                    combine4(p0, p1, p2, p3, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]);
                    for (std::size_t j = 1; j < span; ++j, w += 6) {
                        p0 += 2;
                        p1 += 2;
                        p2 += 2;
                        p3 += 2;
                        combine4(p0, p1, p2, p3, p1[0] * w[0] - p1[1] * w[1],
                                 p1[0] * w[1] + p1[1] * w[0], p2[0] * w[2] - p2[1] * w[3],
                                 p2[0] * w[3] + p2[1] * w[2], p3[0] * w[4] - p3[1] * w[5],
                                 p3[0] * w[5] + p3[1] * w[4]);
                    }
                } else if (stage.radix == 5) {
                    double* p1 = p0 + step;
                    double* p2 = p1 + step;
                    double* p3 = p2 + step;
                    double* p4 = p3 + step;
                    combine5(p0, p1, p2, p3, p4, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0],
                             p4[1]);
                    for (std::size_t j = 1; j < span; ++j, w += 8) {
                        p0 += 2;
                        p1 += 2;
                        p2 += 2;
                        p3 += 2;
                        p4 += 2;
                        combine5(p0, p1, p2, p3, p4, p1[0] * w[0] - p1[1] * w[1],
                                 p1[0] * w[1] + p1[1] * w[0], p2[0] * w[2] - p2[1] * w[3],
                                 p2[0] * w[3] + p2[1] * w[2], p3[0] * w[4] - p3[1] * w[5],
                                 p3[0] * w[5] + p3[1] * w[4], p4[0] * w[6] - p4[1] * w[7],
                                 p4[0] * w[7] + p4[1] * w[6]);
                    }
                } else if (stage.radix == 3) {
                    double* p1 = p0 + step;
                    double* p2 = p1 + step;
                    combine3(p0, p1, p2, p1[0], p1[1], p2[0], p2[1]);
                    for (std::size_t j = 1; j < span; ++j, w += 4) {
                        p0 += 2;
                        p1 += 2;
                        p2 += 2;
                        combine3(p0, p1, p2, p1[0] * w[0] - p1[1] * w[1],
                                 p1[0] * w[1] + p1[1] * w[0], p2[0] * w[2] - p2[1] * w[3],
                                 p2[0] * w[3] + p2[1] * w[2]);
                    }
                } else {
                    double* p1 = p0 + step;
                    combine2(p0, p1, p1[0], p1[1]);
                    for (std::size_t j = 1; j < span; ++j, w += 2) {
                        p0 += 2;
                        p1 += 2;
                        combine2(p0, p1, p1[0] * w[0] - p1[1] * w[1], p1[0] * w[1] + p1[1] * w[0]);
                    }
                }
            }
        }
    }

  private:
    struct Stage {
        std::size_t radix{0};
        std::size_t span{0};     // values per sub-block
        std::size_t twiddles{0}; // offset of the stage's table in twiddles_
    };

    // The block of `stages` stages at `slot` transforms the inputs
    // source, source + stride, ...: its last stage splits them into `radix`
    // interleaved sub-sequences, one per sub-block.
    void fill_sources(std::size_t slot, std::size_t source, std::size_t stride,
                      std::size_t stages) {
        if (stages == 0) {
            sources_[slot] = source;
            return;
        }
        const Stage& stage = stages_[stages - 1];
        for (std::size_t q = 0; q < stage.radix; ++q) {
            fill_sources(slot + q * stage.span, source + q * stride, stride * stage.radix,
                         stages - 1);
        }
    }

    std::size_t size_;
    std::vector<Stage> stages_;
    std::vector<std::size_t> sources_;
    std::vector<double> twiddles_;
};

// Size of the mixed-radix transform behind a frame of length n, or 0 when the
// frame keeps a radix-2 path. The frame takes the mixed-radix path, with a
// transform of half its length, when n is even, is not a power of two, and
// n / 2 has no prime factor other than 2, 3 and 5.
std::size_t mixed_radix_transform_size(int32_t n) {
    if (n <= 0 || n % 2 != 0 || is_power_of_two(n)) {
        return 0;
    }
    int32_t rest = n / 2;
    for (const int32_t factor : {2, 3, 5}) {
        while (rest % factor == 0) {
            rest /= factor;
        }
    }
    return rest == 1 ? static_cast<std::size_t>(n / 2) : 0;
}

// Size of the radix-2 transforms behind a frame of length n: n itself when it
// is a power of two, else the next power of two that holds the 2n - 1 values
// of Bluestein's algorithm.
std::size_t transform_size(int32_t n) {
    if (n <= 0) {
        return 0;
    }
    if (is_power_of_two(n)) {
        return static_cast<std::size_t>(n);
    }
    std::size_t size = 1;
    while (size < static_cast<std::size_t>(2 * n - 1)) {
        size <<= 1;
    }
    return size;
}

class RfftPowerPlan {
  public:
    explicit RfftPowerPlan(int32_t n)
        : n_(n), mixed_radix_(mixed_radix_transform_size(n)),
          fft_size_(mixed_radix_.size() != 0 ? 0 : transform_size(n)), forward_(fft_size_, false),
          inverse_(is_power_of_two(n) ? 0 : fft_size_, true), workspace_(fft_size_),
          pair_values_(2 * mixed_radix_.size()) {
        if (n_ <= 0 || is_power_of_two(n_)) {
            return;
        }

        if (mixed_radix_.size() != 0) {
            const std::size_t half = mixed_radix_.size();
            half_twiddles_.resize(2 * (half + 1));
            for (std::size_t k = 0; k <= half; ++k) {
                double real = 0.0;
                double imag = 0.0;
                unit_root(k, static_cast<std::size_t>(n_), real, imag);
                half_twiddles_[2 * k] = imag;
                half_twiddles_[2 * k + 1] = 0.0 - real;
            }
            return;
        }

        chirp_.resize(static_cast<std::size_t>(n_));
        kernel_fft_.assign(fft_size_, {0.0, 0.0});
        constexpr double kPi = 3.14159265358979323846;
        for (int32_t i = 0; i < n_; ++i) {
            const double index = static_cast<double>(i);
            const double angle = kPi * index * index / static_cast<double>(n_);
            chirp_[static_cast<std::size_t>(i)] = {std::cos(angle), -std::sin(angle)};
            const std::complex<double> kernel(std::cos(angle), std::sin(angle));
            kernel_fft_[static_cast<std::size_t>(i)] = kernel;
            if (i != 0) {
                kernel_fft_[fft_size_ - static_cast<std::size_t>(i)] = kernel;
            }
        }
        forward_(kernel_fft_.data());
    }

    void execute(const float* input, int32_t n_out, float* power_out) {
        const int32_t bins = std::min(n_out, n_ / 2 + 1);
        if (n_ <= 0) {
            std::fill(power_out, power_out + n_out, 0.0F);
            return;
        }

        if (is_power_of_two(n_)) {
            // Loaded in bit-reversed order, the frame needs the butterfly
            // stages alone.
            for (int32_t i = 0; i < n_; ++i) {
                workspace_[forward_.bit_reversed(static_cast<std::size_t>(i))] = {
                    static_cast<double>(input[i]), 0.0};
            }
            forward_.butterflies(workspace_.data());
            write_power(bins, n_out, power_out);
            return;
        }

        if (mixed_radix_.size() != 0) {
            write_mixed_radix_power(input, bins, power_out);
            std::fill(power_out + bins, power_out + n_out, 0.0F);
            return;
        }

        std::fill(workspace_.begin(), workspace_.end(), std::complex<double>(0.0, 0.0));
        for (int32_t i = 0; i < n_; ++i) {
            workspace_[forward_.bit_reversed(static_cast<std::size_t>(i))] =
                static_cast<double>(input[i]) * chirp_[static_cast<std::size_t>(i)];
        }
        forward_.butterflies(workspace_.data());
        for (std::size_t i = 0; i < fft_size_; ++i) {
            workspace_[i] *= kernel_fft_[i];
        }
        inverse_(workspace_.data());
        // Only the first `bins` values of the inverse transform are read, so
        // only they take its 1/size scale.
        const double scale = 1.0 / static_cast<double>(fft_size_);
        for (int32_t k = 0; k < bins; ++k) {
            workspace_[static_cast<std::size_t>(k)] *= scale;
        }
        for (int32_t k = 0; k < bins; ++k) {
            const auto value =
                workspace_[static_cast<std::size_t>(k)] * chirp_[static_cast<std::size_t>(k)];
            power_out[k] = static_cast<float>(std::norm(value));
        }
        std::fill(power_out + bins, power_out + n_out, 0.0F);
    }

  private:
    void write_power(int32_t bins, int32_t n_out, float* power_out) const {
        for (int32_t k = 0; k < bins; ++k) {
            power_out[k] = static_cast<float>(std::norm(workspace_[static_cast<std::size_t>(k)]));
        }
        std::fill(power_out + bins, power_out + n_out, 0.0F);
    }

    // The mixed-radix path. The pairs z[j] = x[2j] + i x[2j + 1] are
    // transformed at size n / 2 to Z. With C[k] = conj(Z[(n / 2 - k) mod n / 2])
    // and E, O the spectra of the even and of the odd samples:
    //   2 E[k] = Z[k] + C[k],  2i O[k] = Z[k] - C[k],
    //   2 X[k] = 2 E[k] + (-i exp(-2 pi i k / n)) (2i O[k]),
    // for k in 0 .. n / 2 (Z[n / 2] is Z[0]). The power takes the exact factor
    // 1/4.
    void write_mixed_radix_power(const float* input, int32_t bins, float* power_out) {
        double* values = pair_values_.data();
        const std::size_t size = mixed_radix_.size();
        for (std::size_t i = 0; i < size; ++i) {
            const std::size_t source = 2 * mixed_radix_.source(i);
            values[2 * i] = static_cast<double>(input[source]);
            values[2 * i + 1] = static_cast<double>(input[source + 1]);
        }
        mixed_radix_.butterflies(values);
        // A frame that holds a sample that is not finite gives NaN in every
        // bin. Z[0] tells exactly whether it does: its real and imaginary
        // parts are the sums of the even and of the odd samples, reached by
        // additions alone (position 0 of a stage takes no twiddle factor).
        // Finite floats cannot overflow such a sum of doubles, and a term
        // that is not finite leaves it not finite.
        if (!std::isfinite(values[0]) || !std::isfinite(values[1])) {
            std::fill(power_out, power_out + bins, std::numeric_limits<float>::quiet_NaN());
            return;
        }
        const double* twiddles = half_twiddles_.data();
        for (int32_t k = 0; k < bins; ++k) {
            const std::size_t index = static_cast<std::size_t>(k);
            const std::size_t at = index == size ? 0 : index;
            const std::size_t mirror = index == 0 ? 0 : size - index;
            const double zr = values[2 * at];
            const double zi = values[2 * at + 1];
            const double cr = values[2 * mirror];
            const double ci = -values[2 * mirror + 1];
            const double er = zr + cr;
            const double ei = zi + ci;
            const double dr = zr - cr;
            const double di = zi - ci;
            const double wr = twiddles[2 * index];
            const double wi = twiddles[2 * index + 1];
            const double real = er + (wr * dr - wi * di);
            const double imag = ei + (wr * di + wi * dr);
            power_out[k] = static_cast<float>(0.25 * (real * real + imag * imag));
        }
    }

    int32_t n_;
    // The mixed-radix path's transform of the sample pairs; empty (size 0)
    // for a length on a radix-2 path.
    MixedRadixTransform mixed_radix_;
    std::size_t fft_size_;
    Radix2Transform forward_;
    // Bluestein's second transform; empty unless the length takes Bluestein's
    // path.
    Radix2Transform inverse_;
    std::vector<std::complex<double>> chirp_;
    std::vector<std::complex<double>> kernel_fft_;
    std::vector<std::complex<double>> workspace_;
    // The mixed-radix transform's (real, imaginary) pairs.
    std::vector<double> pair_values_;
    // -i exp(-2 pi i k / n) for k in 0 .. n / 2, as (real, imaginary) pairs;
    // empty unless the length takes the mixed-radix path.
    std::vector<double> half_twiddles_;
};

std::vector<float> build_center_padded_audio(const float* samples, int32_t n_samples,
                                             int32_t chunk_length_s, int32_t sample_rate,
                                             int32_t n_fft) {
    const int32_t audio_length = chunk_length_s * sample_rate;
    std::vector<float> audio_padded(audio_length, 0.0F);
    const int32_t copy_len = std::min(n_samples, audio_length);
    if (copy_len > 0) {
        std::memcpy(audio_padded.data(), samples, copy_len * sizeof(float));
    }

    const int32_t pad_size = n_fft / 2;
    const int32_t padded_length = pad_size + audio_length + pad_size;
    std::vector<float> padded(padded_length, 0.0F);
    std::memcpy(padded.data() + pad_size, audio_padded.data(), audio_length * sizeof(float));
    // STFT centers the fixed-length waveform with reflection, excluding each edge sample.
    for (int32_t i = 0; i < pad_size; ++i) {
        padded[i] = audio_padded[pad_size - i];
        padded[pad_size + audio_length + i] = audio_padded[audio_length - 2 - i];
    }
    return padded;
}

// The half-open column span [first, last) of one filterbank row outside which
// every coefficient is zero (an all-zero row is an empty span).
std::pair<int32_t, int32_t> nonzero_filter_columns(const float* row, int32_t n_mel_bins) {
    int32_t first = 0;
    while (first < n_mel_bins && row[first] == 0.0F) {
        ++first;
    }
    int32_t last = n_mel_bins;
    while (last > first && row[last - 1] == 0.0F) {
        --last;
    }
    return {first, last};
}

std::vector<float> compute_mel_spectrogram(const std::vector<float>& padded,
                                           const std::vector<float>& window,
                                           const float* mel_filters, int32_t n_fft,
                                           int32_t hop_length, int32_t n_freq_bins,
                                           int32_t n_mel_bins, int32_t frames_to_compute,
                                           int32_t& n_frames_raw) {
    n_frames_raw = 1 + (static_cast<int32_t>(padded.size()) - n_fft) / hop_length;
    std::vector<float> mel_spec(static_cast<std::size_t>(n_mel_bins) * n_frames_raw, 0.0F);
    std::vector<float> windowed(n_fft);
    std::vector<float> frame_power(n_freq_bins);
    std::vector<float> frame_mel(n_mel_bins);
    RfftPowerPlan fft_plan(n_fft);
    // Each row's nonzero column span, found when a frame first needs the row
    // (first == -1 until then).
    std::vector<std::pair<int32_t, int32_t>> filter_columns(static_cast<std::size_t>(n_freq_bins),
                                                            {-1, -1});

    const int32_t computed_frames = std::clamp(frames_to_compute, 0, n_frames_raw);
    for (int32_t t = 0; t < computed_frames; ++t) {
        const int32_t start = t * hop_length;
        for (int32_t i = 0; i < n_fft; ++i) {
            windowed[static_cast<std::size_t>(i)] =
                padded[static_cast<std::size_t>(start + i)] * window[static_cast<std::size_t>(i)];
        }

        fft_plan.execute(windowed.data(), n_freq_bins, frame_power.data());

        std::fill(frame_mel.begin(), frame_mel.end(), 0.0F);
        for (int32_t f = 0; f < n_freq_bins; ++f) {
            const float p = frame_power[static_cast<std::size_t>(f)];
            if (p == 0.0F) {
                continue;
            }
            const float* filter_row = mel_filters + static_cast<std::size_t>(f) * n_mel_bins;
            // A finite power adds exactly +0 through a zero coefficient, so
            // only the row's nonzero column span changes frame_mel. A
            // non-finite power keeps the full row (inf * 0 is NaN).
            auto& columns = filter_columns[static_cast<std::size_t>(f)];
            if (columns.first < 0) {
                columns = nonzero_filter_columns(filter_row, n_mel_bins);
            }
            const auto [first, last] = columns;
            const bool finite = std::isfinite(p);
            const int32_t column_end = finite ? last : n_mel_bins;
            for (int32_t m = finite ? first : 0; m < column_end; ++m) {
                frame_mel[static_cast<std::size_t>(m)] +=
                    p * filter_row[static_cast<std::size_t>(m)];
            }
        }
        for (int32_t m = 0; m < n_mel_bins; ++m) {
            mel_spec[static_cast<std::size_t>(m) * n_frames_raw + t] =
                frame_mel[static_cast<std::size_t>(m)];
        }
    }
    return mel_spec;
}

void normalize_log_mel_inplace(std::vector<float>& mel_spec) {
    float global_max = -1e10F;
    for (float& value : mel_spec) {
        value = std::log10(std::max(value, 1e-10F));
        if (value > global_max) {
            global_max = value;
        }
    }

    const float floor = global_max - 8.0F;
    for (float& value : mel_spec) {
        value = std::max(value, floor);
        value = (value + 4.0F) / 4.0F;
    }
}

// Drops every row's last frame in place: row m moves from m * n_frames_raw
// down to m * n_frames_out, never past a row still to be moved.
void trim_last_frame(std::vector<float>& mel_spec, int32_t n_mel_bins, int32_t n_frames_raw,
                     int32_t& n_frames_out) {
    n_frames_out = n_frames_raw;
    if (n_frames_raw <= 1) {
        return;
    }

    n_frames_out = n_frames_raw - 1;
    for (int32_t m = 1; m < n_mel_bins; ++m) {
        std::memmove(mel_spec.data() + static_cast<std::size_t>(m) * n_frames_out,
                     mel_spec.data() + static_cast<std::size_t>(m) * n_frames_raw,
                     n_frames_out * sizeof(float));
    }
    mel_spec.resize(static_cast<std::size_t>(n_mel_bins) * n_frames_out);
}

} // namespace

namespace detail {

std::vector<float> rfft_power(const std::vector<float>& input) {
    const int32_t n = static_cast<int32_t>(input.size());
    const int32_t n_out = n / 2 + 1;
    std::vector<float> power(static_cast<std::size_t>(n_out), 0.0F);
    if (n > 0) {
        RfftPowerPlan plan(n);
        plan.execute(input.data(), n_out, power.data());
    }
    return power;
}

} // namespace detail

MelResult extract_mel_spectrogram(const float* samples, int32_t n_samples, const float* mel_filters,
                                  int32_t n_freq_bins, int32_t n_mel_bins, int32_t n_fft,
                                  int32_t hop_length, int32_t chunk_length_s, int32_t sample_rate) {
    const int32_t expected_freq_bins = n_fft / 2 + 1;
    const int32_t freq_bins = n_freq_bins == expected_freq_bins ? n_freq_bins : expected_freq_bins;
    const int32_t chunk_samples = chunk_length_s * sample_rate;
    const int32_t valid_audio = std::min(std::max(n_samples, 0), chunk_samples);
    const int32_t pad_size = n_fft / 2;
    const int32_t frames_to_compute =
        valid_audio > 0 && hop_length > 0 ? 1 + (pad_size + valid_audio - 1) / hop_length : 0;
    const std::vector<float> padded =
        build_center_padded_audio(samples, n_samples, chunk_length_s, sample_rate, n_fft);

    int32_t n_frames_raw = 0;
    std::vector<float> mel_spec =
        compute_mel_spectrogram(padded, make_hann_window(n_fft), mel_filters, n_fft, hop_length,
                                freq_bins, n_mel_bins, frames_to_compute, n_frames_raw);
    int32_t n_frames_out = 0;
    trim_last_frame(mel_spec, n_mel_bins, n_frames_raw, n_frames_out);
    normalize_log_mel_inplace(mel_spec);

    MelResult result;
    result.data = std::move(mel_spec);
    result.n_mels = n_mel_bins;
    result.n_frames = n_frames_out;
    return result;
}

} // namespace whisper
} // namespace trtmc
