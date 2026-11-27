/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmfold2/runtime/sampler.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace trtmc::esmfold2 {
namespace {

using Mat4 = std::array<std::array<double, 4>, 4>;

// The reference rounds every float32 (torch) and double (Python) operation on its own. A compiler
// may fuse a product into the following addition (GCC does by default where the ISA has FMA, as on
// aarch64), which skips the product's rounding; the volatile round trip keeps it on every target.
template <typename T>
T rounded(T value) {
    volatile T stored = value;
    return stored;
}

// Jacobi eigen-decomposition of a symmetric 4x4 matrix; returns the eigenvector of the largest
// eigenvalue.
std::array<double, 4> dominantEigenvector(Mat4 matrix) {
    Mat4 vectors{};
    for (int i = 0; i < 4; ++i)
        vectors[i][i] = 1.0;
    for (int sweep = 0; sweep < 64; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 4; ++p)
            for (int q = p + 1; q < 4; ++q)
                off += matrix[p][q] * matrix[p][q];
        if (off < 1.0e-30)
            break;
        for (int p = 0; p < 4; ++p) {
            for (int q = p + 1; q < 4; ++q) {
                if (std::fabs(matrix[p][q]) < 1.0e-300)
                    continue;
                const double theta = (matrix[q][q] - matrix[p][p]) / (2.0 * matrix[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int k = 0; k < 4; ++k) {
                    const double kp = matrix[k][p];
                    const double kq = matrix[k][q];
                    matrix[k][p] = c * kp - s * kq;
                    matrix[k][q] = s * kp + c * kq;
                }
                for (int k = 0; k < 4; ++k) {
                    const double pk = matrix[p][k];
                    const double qk = matrix[q][k];
                    matrix[p][k] = c * pk - s * qk;
                    matrix[q][k] = s * pk + c * qk;
                }
                for (int k = 0; k < 4; ++k) {
                    const double kp = vectors[k][p];
                    const double kq = vectors[k][q];
                    vectors[k][p] = c * kp - s * kq;
                    vectors[k][q] = s * kp + c * kq;
                }
            }
        }
    }
    int best = 0;
    for (int i = 1; i < 4; ++i) {
        if (matrix[i][i] > matrix[best][best])
            best = i;
    }
    return {vectors[0][best], vectors[1][best], vectors[2][best], vectors[3][best]};
}

} // namespace

NoiseSchedule noise_schedule(const SamplerSettings& settings, std::int32_t configured_steps) {
    if (configured_steps < 1)
        throw std::invalid_argument("ESMFold2 needs at least one sampling step");
    std::vector<float> raw;
    if (configured_steps == 1) {
        raw = {static_cast<float>(settings.sigma_max_ratio * settings.sigma_data), 0.0F};
    } else {
        const double inverse = 1.0 / settings.exponent;
        const double high = std::pow(settings.sigma_max_ratio, inverse);
        const double span = std::pow(settings.sigma_min_ratio, inverse) - high;
        // The cap filter below drops NaN silently, so overflow must be caught before it.
        if (!std::isfinite(high) || !std::isfinite(span))
            throw std::invalid_argument("ESMFold2 noise schedule overflows for exponent " +
                                        std::to_string(settings.exponent));
        const float last = static_cast<float>(configured_steps - 1);
        for (std::int32_t step = 0; step < configured_steps; ++step) {
            const float ramp = static_cast<float>(step) / last;
            // torch: high + (ramp / (steps - 1)) * span, two float32 roundings; pow(base, 7) turns
            // one ulp of base into several of the sigma.
            const float base = rounded(ramp * static_cast<float>(span)) + static_cast<float>(high);
            raw.push_back(static_cast<float>(settings.sigma_data) *
                          std::pow(base, static_cast<float>(settings.exponent)));
        }
        raw.push_back(0.0F);
    }
    for (const float sigma : raw) {
        if (!std::isfinite(sigma))
            throw std::invalid_argument("ESMFold2 noise schedule of " +
                                        std::to_string(configured_steps) +
                                        " steps has a non-finite level");
    }
    NoiseSchedule schedule;
    const float cap = static_cast<float>(settings.sigma_cap);
    schedule.sigmas.push_back(cap);
    for (const float sigma : raw) {
        if (sigma <= cap)
            schedule.sigmas.push_back(sigma);
    }
    // Every step divides by its noise level: the levels must fall strictly, and only the last may
    // be zero.
    for (std::size_t index = 0; index < schedule.sigmas.size(); ++index) {
        const float sigma = schedule.sigmas[index];
        const bool last = index + 1 == schedule.sigmas.size();
        if (!std::isfinite(sigma) || (last ? sigma != 0.0F : sigma <= 0.0F) ||
            (index > 0 && sigma >= schedule.sigmas[index - 1]))
            throw std::invalid_argument("ESMFold2 noise schedule of " +
                                        std::to_string(configured_steps) +
                                        " steps does not fall strictly to zero");
    }
    if (schedule.sigmas.size() < 2)
        throw std::invalid_argument("ESMFold2 noise schedule has no sampling step");
    const float gamma = static_cast<float>(settings.gamma_0);
    for (const float sigma : schedule.sigmas)
        schedule.gammas.push_back(static_cast<double>(sigma) > settings.gamma_min ? gamma : 0.0F);
    return schedule;
}

StepLevels step_levels(const SamplerSettings& settings, const NoiseSchedule& schedule,
                       std::int32_t step) {
    if (step < 0 || step >= schedule.steps())
        throw std::out_of_range("ESMFold2 sampling step out of range");
    const auto index = static_cast<std::size_t>(step);
    const double sigma_prev = schedule.sigmas[index];
    StepLevels levels;
    levels.sigma_next = schedule.sigmas[index + 1U];
    levels.noise_level = sigma_prev * (1.0 + static_cast<double>(schedule.gammas[index + 1U]));
    // Python rounds both squares, so a step with gamma 0 gets exactly no churn; fused, it would get
    // the square's rounding error.
    const double excess =
        rounded(levels.noise_level * levels.noise_level) - rounded(sigma_prev * sigma_prev);
    levels.churn_std = settings.noise_scale * std::pow(excess > 0.0 ? excess : 0.0, 0.5);
    return levels;
}

Mat3 rotation_from_quaternion(const std::array<float, 4>& quaternion) {
    float squared = 0.0F;
    for (const float value : quaternion)
        squared += value * value;
    const float scale = std::sqrt(squared);
    const float sign = quaternion[0] < 0.0F ? -scale : scale;
    std::array<float, 4> q{};
    for (std::size_t i = 0; i < 4; ++i)
        q[i] = quaternion[i] / sign;
    const float r = q[0];
    const float i = q[1];
    const float j = q[2];
    const float k = q[3];
    const float norm = r * r + i * i + j * j + k * k;
    const float two_s = 2.0F / norm;
    Mat3 rotation{};
    rotation[0] = {1.0F - two_s * (j * j + k * k), two_s * (i * j - k * r),
                   two_s * (i * k + j * r)};
    rotation[1] = {two_s * (i * j + k * r), 1.0F - two_s * (i * i + k * k),
                   two_s * (j * k - i * r)};
    rotation[2] = {two_s * (i * k - j * r), two_s * (j * k + i * r),
                   1.0F - two_s * (i * i + j * j)};
    return rotation;
}

void center_random_augmentation(std::vector<float>& coords, std::int32_t samples,
                                std::int32_t atoms, const std::vector<float>& atom_mask,
                                const std::vector<Mat3>& rotations,
                                const std::vector<float>& translations) {
    const auto atom_count = static_cast<std::size_t>(atoms);
    double valid = 0.0;
    for (std::size_t atom = 0; atom < atom_count; ++atom)
        valid += atom_mask[atom];
    const float denominator = static_cast<float>(valid < 1.0 ? 1.0 : valid);
    for (std::int32_t sample = 0; sample < samples; ++sample) {
        float* rows = coords.data() + static_cast<std::size_t>(sample) * atom_count * 3U;
        std::array<double, 3> sum{};
        for (std::size_t atom = 0; atom < atom_count; ++atom)
            for (std::size_t axis = 0; axis < 3; ++axis)
                sum[axis] += static_cast<double>(rows[atom * 3U + axis] * atom_mask[atom]);
        std::array<float, 3> mean{};
        for (std::size_t axis = 0; axis < 3; ++axis)
            mean[axis] = static_cast<float>(sum[axis]) / denominator;
        const Mat3& rotation = rotations[static_cast<std::size_t>(sample)];
        const float* shift = translations.data() + static_cast<std::size_t>(sample) * 3U;
        for (std::size_t atom = 0; atom < atom_count; ++atom) {
            std::array<float, 3> centered{};
            for (std::size_t axis = 0; axis < 3; ++axis)
                centered[axis] = rows[atom * 3U + axis] - mean[axis];
            for (std::size_t column = 0; column < 3; ++column) {
                double value = 0.0;
                for (std::size_t inner = 0; inner < 3; ++inner)
                    value += static_cast<double>(centered[inner]) *
                             static_cast<double>(rotation[inner][column]);
                rows[atom * 3U + column] = static_cast<float>(value) + shift[column];
            }
        }
    }
}

void weighted_rigid_align(std::vector<float>& coords, const std::vector<float>& target,
                          std::int32_t samples, std::int32_t atoms,
                          const std::vector<float>& atom_mask) {
    const auto atom_count = static_cast<std::size_t>(atoms);
    double weight = 0.0;
    for (std::size_t atom = 0; atom < atom_count; ++atom)
        weight += atom_mask[atom];
    const double denominator = weight < 1.0e-8 ? 1.0e-8 : weight;
    for (std::int32_t sample = 0; sample < samples; ++sample) {
        const std::size_t offset = static_cast<std::size_t>(sample) * atom_count * 3U;
        float* source = coords.data() + offset;
        const float* goal = target.data() + offset;
        std::array<double, 3> source_mean{};
        std::array<double, 3> goal_mean{};
        for (std::size_t atom = 0; atom < atom_count; ++atom) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                source_mean[axis] +=
                    atom_mask[atom] * static_cast<double>(source[atom * 3U + axis]);
                goal_mean[axis] += atom_mask[atom] * static_cast<double>(goal[atom * 3U + axis]);
            }
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
            source_mean[axis] /= denominator;
            goal_mean[axis] /= denominator;
        }
        // S[a][b] = sum_i w_i c_ia t_ib over the centered source c and target t.
        std::array<std::array<double, 3>, 3> s{};
        for (std::size_t atom = 0; atom < atom_count; ++atom) {
            if (atom_mask[atom] == 0.0F)
                continue;
            for (std::size_t a = 0; a < 3; ++a) {
                const double c = static_cast<double>(source[atom * 3U + a]) - source_mean[a];
                for (std::size_t b = 0; b < 3; ++b) {
                    const double t = static_cast<double>(goal[atom * 3U + b]) - goal_mean[b];
                    s[a][b] += atom_mask[atom] * c * t;
                }
            }
        }
        // Horn's quaternion method: the top eigenvector rotates the source onto the target.
        const Mat4 horn{{
            {s[0][0] + s[1][1] + s[2][2], s[1][2] - s[2][1], s[2][0] - s[0][2], s[0][1] - s[1][0]},
            {s[1][2] - s[2][1], s[0][0] - s[1][1] - s[2][2], s[0][1] + s[1][0], s[2][0] + s[0][2]},
            {s[2][0] - s[0][2], s[0][1] + s[1][0], -s[0][0] + s[1][1] - s[2][2], s[1][2] + s[2][1]},
            {s[0][1] - s[1][0], s[2][0] + s[0][2], s[1][2] + s[2][1], -s[0][0] - s[1][1] + s[2][2]},
        }};
        const auto q = dominantEigenvector(horn);
        const double w = q[0];
        const double x = q[1];
        const double y = q[2];
        const double z = q[3];
        const double rotation[3][3] = {
            {w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)},
            {2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)},
            {2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z},
        };
        for (std::size_t atom = 0; atom < atom_count; ++atom) {
            std::array<double, 3> centered{};
            for (std::size_t axis = 0; axis < 3; ++axis)
                centered[axis] = static_cast<double>(source[atom * 3U + axis]) - source_mean[axis];
            for (std::size_t row = 0; row < 3; ++row) {
                double value = goal_mean[row];
                for (std::size_t column = 0; column < 3; ++column)
                    value += rotation[row][column] * centered[column];
                source[atom * 3U + row] = static_cast<float>(value);
            }
        }
    }
}

void step_rigid_draws(const RandomStreams& random, std::int32_t step, std::int32_t samples,
                      std::vector<Mat3>& rotations, std::vector<float>& translations) {
    rotations.resize(static_cast<std::size_t>(samples));
    translations.resize(static_cast<std::size_t>(samples) * 3U);
    for (std::int32_t sample = 0; sample < samples; ++sample) {
        const auto slot = static_cast<std::uint64_t>(step) * static_cast<std::uint64_t>(samples) +
                          static_cast<std::uint64_t>(sample);
        std::array<float, 4> quaternion{};
        for (std::uint64_t component = 0; component < 4; ++component)
            quaternion[component] = random.normal(Stream::kRotation, slot * 4U + component);
        rotations[static_cast<std::size_t>(sample)] = rotation_from_quaternion(quaternion);
        for (std::uint64_t axis = 0; axis < 3; ++axis)
            translations[static_cast<std::size_t>(sample) * 3U + axis] =
                random.normal(Stream::kTranslation, slot * 3U + axis);
    }
}

} // namespace trtmc::esmfold2
