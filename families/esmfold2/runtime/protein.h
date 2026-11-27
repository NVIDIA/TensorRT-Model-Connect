/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::esmfold2 {

inline constexpr std::int32_t kAtomBlock = 32;
inline constexpr std::size_t kMaxDocumentBytes = 1U << 20U;

// One single-chain protein request: the parsed sequence of a FASTA record or of a bare sequence
// document.
struct ProteinRequest {
    std::string name;
    std::string sequence;
};

// Parse a document with exactly one protein chain: a FASTA record (">" header line, sequence lines)
// or a bare sequence. Letters must be the 20 standard one-letter codes or X; whitespace between
// lines is ignored.
ProteinRequest parse_protein_document(std::string_view document, std::int32_t min_tokens,
                                      std::int32_t max_tokens);

// The featurized inputs of a single-chain protein, as the reference featurizer builds them.
struct ProteinFeatures {
    std::string sequence;
    std::int32_t tokens{0};
    std::int32_t atoms{0};               // padded to a multiple of kAtomBlock
    std::int32_t real_atoms{0};          // heavy atoms of the sequence
    std::vector<std::int32_t> input_ids; // [BOS] sequence [EOS], tokens + 2
    std::vector<std::int32_t> res_type;
    std::vector<std::int32_t> residue_index;
    std::vector<std::int32_t> token_index;
    std::vector<std::int32_t> asym_id;
    std::vector<std::int32_t> sym_id;
    std::vector<std::int32_t> entity_id;
    std::vector<std::int32_t> rep_atom; // CB, or CA when the residue has no CB
    std::vector<float> ref_pos;         // atoms * 3
    std::vector<float> ref_charge;
    std::vector<float> atom_mask;
    std::vector<std::int32_t> ref_element;
    std::vector<std::int32_t> ref_name_chars; // atoms * 4
    std::vector<std::int32_t> ref_space_uid;
    std::vector<std::int32_t> atom_to_token;
    std::vector<std::int32_t> intra_index;  // atom slot within its residue (0 for padding)
    std::vector<std::string> residue_names; // three-letter code per token
    std::vector<std::string> atom_names;    // per real atom
};

ProteinFeatures featurize_protein(const std::string& sequence, std::int32_t bos_id,
                                  std::int32_t eos_id);

} // namespace trtmc::esmfold2
