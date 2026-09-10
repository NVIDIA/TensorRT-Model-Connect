/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <string_view>

namespace trtmc::boltz2 {

inline constexpr std::string_view kModelId = "boltz2";
inline constexpr std::string_view kStrategy = "boltz2_structure_prediction";
inline constexpr int kPairformerSegments = 8;
inline constexpr int kTokenSegments = 4;
inline constexpr int kMaxTokenCount = 117;
inline constexpr int kMaxAtomCount = 928;
inline constexpr int kMsaDepth = 1;
inline constexpr int kAtomWindowQueries = 32;

inline constexpr std::array<std::string_view, kPairformerSegments> kPairformerSections{
    "boltz2_pairformer_00_08_plan", "boltz2_pairformer_08_16_plan", "boltz2_pairformer_16_24_plan",
    "boltz2_pairformer_24_32_plan", "boltz2_pairformer_32_40_plan", "boltz2_pairformer_40_48_plan",
    "boltz2_pairformer_48_56_plan", "boltz2_pairformer_56_64_plan",
};

inline constexpr std::array<std::string_view, kTokenSegments> kTokenSections{
    "boltz2_diffusion_token_00_06_plan",
    "boltz2_diffusion_token_06_12_plan",
    "boltz2_diffusion_token_12_18_plan",
    "boltz2_diffusion_token_18_24_plan",
};

inline constexpr std::array<std::string_view, 31> kFeatureNames{
    "ref_pos",
    "ref_space_uid",
    "ref_charge",
    "ref_element",
    "ref_atom_name_chars",
    "atom_to_token",
    "atom_pad_mask",
    "res_type",
    "profile",
    "deletion_mean",
    "method_feature",
    "modified",
    "cyclic_period",
    "mol_type",
    "asym_id",
    "residue_index",
    "entity_id",
    "token_index",
    "sym_id",
    "token_bonds",
    "type_bonds",
    "contact_conditioning",
    "contact_threshold",
    "msa",
    "has_deletion",
    "deletion_value",
    "msa_paired",
    "msa_mask",
    "token_pad_mask",
    "token_to_rep_atom",
    "frames_idx",
};

} // namespace trtmc::boltz2
