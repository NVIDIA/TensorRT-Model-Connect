/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmfold2/runtime/protein.h"

#include "families/esmfold2/runtime/residue_table.h"

#include <cstring>
#include <stdexcept>

namespace trtmc::esmfold2 {
namespace {

const ResidueDefinition* findResidue(char letter) {
    for (const auto& residue : kResidues) {
        if (residue.letter == letter)
            return &residue;
    }
    return nullptr;
}

std::int32_t atomicNumber(char element) {
    switch (element) {
    case 'C':
        return 6;
    case 'N':
        return 7;
    case 'O':
        return 8;
    case 'S':
        return 16;
    default:
        throw std::logic_error("ESMFold2 residue table holds an unexpected element");
    }
}

bool isSpace(char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n' || value == '\v' ||
           value == '\f';
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && isSpace(text.front()))
        text.remove_prefix(1);
    while (!text.empty() && isSpace(text.back()))
        text.remove_suffix(1);
    return text;
}

void appendSequence(std::string& sequence, std::string_view line, std::int32_t max_tokens) {
    for (const char letter : line) {
        if (isSpace(letter))
            continue;
        if (findResidue(letter) == nullptr) {
            const auto code = static_cast<unsigned>(static_cast<unsigned char>(letter));
            throw std::invalid_argument("ESMFold2 sequences take the 20 standard one-letter "
                                        "amino-acid codes and X; got character code " +
                                        std::to_string(code));
        }
        if (static_cast<std::int64_t>(sequence.size()) >= max_tokens)
            throw std::invalid_argument("ESMFold2 sequence is longer than this bundle's " +
                                        std::to_string(max_tokens) + " residues");
        sequence.push_back(letter);
    }
}

} // namespace

ProteinRequest parse_protein_document(std::string_view document, std::int32_t min_tokens,
                                      std::int32_t max_tokens) {
    if (document.empty())
        throw std::invalid_argument("ESMFold2 request document is empty");
    if (document.size() > kMaxDocumentBytes)
        throw std::invalid_argument("ESMFold2 request document exceeds 1 MiB");
    if (std::memchr(document.data(), '\0', document.size()) != nullptr)
        throw std::invalid_argument("ESMFold2 request document contains a NUL byte");
    ProteinRequest request;
    const std::string_view body = trim(document);
    const bool fasta = !body.empty() && body.front() == '>';
    std::size_t position = 0;
    bool header_seen = false;
    while (position <= body.size()) {
        const std::size_t end = body.find('\n', position);
        const std::string_view line = trim(body.substr(
            position, end == std::string_view::npos ? std::string_view::npos : end - position));
        position = end == std::string_view::npos ? body.size() + 1 : end + 1;
        if (fasta && !line.empty() && line.front() == '>') {
            if (header_seen)
                throw std::invalid_argument(
                    "ESMFold2 supports one protein chain per request; the FASTA "
                    "document has more than one record");
            header_seen = true;
            const std::string_view header = trim(line.substr(1));
            const std::size_t space = header.find_first_of(" \t");
            request.name = std::string(header.substr(0, space));
            continue;
        }
        appendSequence(request.sequence, line, max_tokens);
    }
    if (static_cast<std::int64_t>(request.sequence.size()) < min_tokens)
        throw std::invalid_argument("ESMFold2 needs at least " + std::to_string(min_tokens) +
                                    " residues");
    return request;
}

ProteinFeatures featurize_protein(const std::string& sequence, std::int32_t bos_id,
                                  std::int32_t eos_id) {
    ProteinFeatures features;
    features.sequence = sequence;
    features.tokens = static_cast<std::int32_t>(sequence.size());
    std::int32_t real_atoms = 0;
    for (const char letter : sequence) {
        const ResidueDefinition* residue = findResidue(letter);
        if (residue == nullptr)
            throw std::invalid_argument("ESMFold2 featurizer received an unsupported residue");
        real_atoms += residue->atom_count;
    }
    features.real_atoms = real_atoms;
    features.atoms = ((real_atoms + kAtomBlock - 1) / kAtomBlock) * kAtomBlock;
    const auto atoms = static_cast<std::size_t>(features.atoms);
    const auto tokens = static_cast<std::size_t>(features.tokens);
    features.ref_pos.assign(atoms * 3U, 0.0F);
    features.ref_charge.assign(atoms, 0.0F);
    features.atom_mask.assign(atoms, 0.0F);
    features.ref_element.assign(atoms, 0);
    features.ref_name_chars.assign(atoms * 4U, 0);
    features.ref_space_uid.assign(atoms, 0);
    features.atom_to_token.assign(atoms, 0);
    features.intra_index.assign(atoms, 0);
    features.input_ids.reserve(tokens + 2U);
    features.input_ids.push_back(bos_id);
    std::size_t atom = 0;
    for (std::size_t token = 0; token < tokens; ++token) {
        const ResidueDefinition& residue = *findResidue(sequence[token]);
        const auto token_id = static_cast<std::int32_t>(token);
        features.input_ids.push_back(residue.esm_token);
        features.res_type.push_back(residue.res_type);
        features.residue_index.push_back(token_id);
        features.token_index.push_back(token_id);
        features.asym_id.push_back(0);
        features.sym_id.push_back(0);
        features.entity_id.push_back(1);
        features.residue_names.emplace_back(residue.name);
        std::int32_t representative = -1;
        std::int32_t alpha = -1;
        for (std::int32_t slot = 0; slot < residue.atom_count; ++slot, ++atom) {
            const ResidueAtom& entry = residue.atoms[static_cast<std::size_t>(slot)];
            const std::string_view name(entry.name);
            features.ref_pos[atom * 3U] = entry.x;
            features.ref_pos[atom * 3U + 1U] = entry.y;
            features.ref_pos[atom * 3U + 2U] = entry.z;
            features.ref_charge[atom] = static_cast<float>(entry.charge);
            features.atom_mask[atom] = 1.0F;
            features.ref_element[atom] = atomicNumber(name.front());
            for (std::size_t character = 0; character < 4U; ++character) {
                const char value = character < name.size() ? name[character] : ' ';
                features.ref_name_chars[atom * 4U + character] =
                    value == ' ' ? 0 : static_cast<std::int32_t>(value) - 32;
            }
            features.ref_space_uid[atom] = token_id;
            features.atom_to_token[atom] = token_id;
            features.intra_index[atom] = slot;
            features.atom_names.emplace_back(name);
            if (name == "CB")
                representative = static_cast<std::int32_t>(atom);
            if (name == "CA")
                alpha = static_cast<std::int32_t>(atom);
        }
        features.rep_atom.push_back(representative >= 0 ? representative : alpha);
    }
    features.input_ids.push_back(eos_id);
    return features;
}

} // namespace trtmc::esmfold2
