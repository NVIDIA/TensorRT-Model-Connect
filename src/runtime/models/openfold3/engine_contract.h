/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <string_view>

namespace trtmc::openfold3 {

inline constexpr std::string_view kStrategy = "openfold3_structure_prediction";
inline constexpr int kPairformerSegments = 8;
inline constexpr int kTokenSegments = 4;
inline constexpr int kMaxTokenCount = 128;
inline constexpr int kMsaDepth = 1;
inline constexpr int kAtomWindowQueries = 32;

inline constexpr std::array<std::string_view, kPairformerSegments> kPairformerSections{
    "openfold3_pairformer_00_06_plan", "openfold3_pairformer_06_12_plan",
    "openfold3_pairformer_12_18_plan", "openfold3_pairformer_18_24_plan",
    "openfold3_pairformer_24_30_plan", "openfold3_pairformer_30_36_plan",
    "openfold3_pairformer_36_42_plan", "openfold3_pairformer_42_48_plan",
};

inline constexpr std::array<std::string_view, kTokenSegments> kTokenSections{
    "openfold3_diffusion_token_00_06_plan",
    "openfold3_diffusion_token_06_12_plan",
    "openfold3_diffusion_token_12_18_plan",
    "openfold3_diffusion_token_18_24_plan",
};

inline constexpr std::array<std::string_view, 20> kFeatureNames{
    "ref_pos",
    "ref_mask",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_mask",
    "atom_to_token_index",
    "token_mask",
    "restype",
    "profile",
    "deletion_mean",
    "relpos",
    "token_bonds",
    "msa",
    "has_deletion",
    "deletion_value",
    "msa_mask",
    "representative_atom_map",
    "atom_head_index",
};

} // namespace trtmc::openfold3
