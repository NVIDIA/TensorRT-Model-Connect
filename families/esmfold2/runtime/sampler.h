/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/esmfold2/runtime/random.h"

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::esmfold2 {

// Sampler constants of the checkpoint (``structure_head``), as the JSON doubles the reference
// reads.
struct SamplerSettings {
    double sigma_data{16.0};
    double gamma_0{0.0};
    double gamma_min{0.0};
    double noise_scale{0.0};
    double step_scale{1.0};
    double sigma_max_ratio{160.0};
    double sigma_min_ratio{4.0e-4};
    double exponent{7.0};
    double sigma_cap{256.0};
};

// The capped Karras schedule the reference samples on (float32 values), and the churn factor of
// each entry.
struct NoiseSchedule {
    std::vector<float> sigmas; // descending, ending at 0
    std::vector<float> gammas; // gamma_0 where the entry is above gamma_min, else 0
    std::int32_t steps() const { return static_cast<std::int32_t>(sigmas.size()) - 1; }
};

// ``configured_steps`` is the schedule length before the cap truncates it (the reference's
// num_sampling_steps).
NoiseSchedule noise_schedule(const SamplerSettings& settings, std::int32_t configured_steps);

// The scalars of one step, in the reference's double arithmetic.
struct StepLevels {
    double noise_level{0.0}; // sigma_prev * (1 + gamma of the next entry)
    double churn_std{0.0};
    double sigma_next{0.0};
};

StepLevels step_levels(const SamplerSettings& settings, const NoiseSchedule& schedule,
                       std::int32_t step);

using Mat3 = std::array<std::array<float, 3>, 3>;

// Rotation matrix of four normal draws, as the reference builds it (float32).
Mat3 rotation_from_quaternion(const std::array<float, 4>& quaternion);

// Center the valid atoms of every sample, multiply the row vectors by ``rotations[s]`` and add
// ``translations[s]``.
void center_random_augmentation(std::vector<float>& coords, std::int32_t samples,
                                std::int32_t atoms, const std::vector<float>& atom_mask,
                                const std::vector<Mat3>& rotations,
                                const std::vector<float>& translations);

// Align ``coords`` onto ``target`` per sample with the optimal proper rotation (weights = atom
// mask), in place.
void weighted_rigid_align(std::vector<float>& coords, const std::vector<float>& target,
                          std::int32_t samples, std::int32_t atoms,
                          const std::vector<float>& atom_mask);

// Rotations and translations of one sampling step for every sample.
void step_rigid_draws(const RandomStreams& random, std::int32_t step, std::int32_t samples,
                      std::vector<Mat3>& rotations, std::vector<float>& translations);

} // namespace trtmc::esmfold2
