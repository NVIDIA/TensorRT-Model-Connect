/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/esmfold2/runtime/sampler.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace trtmc::esmfold2 {

inline constexpr std::int32_t kMinTokens = 2;

// ``runtime.json`` of an ESMFold2 bundle, validated against the plans' fixed widths.
struct BundleConfig {
    bool bf16{true};
    std::int32_t max_tokens{0};
    std::int32_t max_atoms{0};
    std::int32_t max_samples{0};
    std::int32_t pair_size{0};
    std::int32_t single_inputs_size{0};
    std::int32_t diffusion_hidden_size{0};
    std::int32_t diffusion_layers{0};
    std::int32_t diffusion_heads{0};
    std::int32_t atom_hidden_size{0};
    std::int32_t atom_head_dim{0};
    std::int32_t max_atoms_per_token{0};
    std::int32_t pae_bins{0};
    bool msa_encoder{false};
    double lm_dropout{0.0};
    double trunk_init_std{0.0};
    std::int32_t esmc_bos_id{0};
    std::int32_t esmc_eos_id{0};
    std::int32_t num_diffusion_samples{0};
    std::int32_t num_sampling_steps{0};
    double confidence_eps{0.0};
    SamplerSettings sampler;
};

BundleConfig parse_bundle_config(std::string_view text);

} // namespace trtmc::esmfold2
