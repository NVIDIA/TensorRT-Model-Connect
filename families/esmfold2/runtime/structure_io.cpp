/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmfold2/runtime/structure_io.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace trtmc::esmfold2 {
namespace {

template <typename... Values>
std::string formatted(const char* pattern, Values... values) {
    const int size = std::snprintf(nullptr, 0, pattern, values...);
    if (size < 0)
        throw std::runtime_error("ESMFold2 structure writer failed to format a record");
    std::string text(static_cast<std::size_t>(size) + 1U, '\0');
    std::snprintf(text.data(), text.size(), pattern, values...);
    text.resize(static_cast<std::size_t>(size));
    return text;
}

void requireFinite(const float* coords, std::size_t atom) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(coords[atom * 3U + axis]))
            throw std::runtime_error("ESMFold2 predicted a non-finite coordinate");
    }
}

void requireSizes(const ProteinFeatures& features, const std::vector<float>& plddt) {
    if (plddt.size() != static_cast<std::size_t>(features.tokens))
        throw std::invalid_argument("ESMFold2 writer needs one pLDDT per residue");
    if (features.atom_names.size() != static_cast<std::size_t>(features.real_atoms))
        throw std::invalid_argument("ESMFold2 writer needs one name per real atom");
}

} // namespace

std::string write_pdb(const ProteinFeatures& features, const float* coords,
                      const std::vector<float>& plddt) {
    requireSizes(features, plddt);
    std::string text = "PARENT N/A\n";
    std::size_t serial = 1;
    std::string residue_name;
    std::int32_t residue_number = 0;
    for (std::size_t atom = 0; atom < static_cast<std::size_t>(features.real_atoms); ++atom) {
        requireFinite(coords, atom);
        const auto token = static_cast<std::size_t>(features.atom_to_token[atom]);
        residue_name = features.residue_names[token];
        residue_number = features.residue_index[token] + 1;
        const std::string& name = features.atom_names[atom];
        const std::string padded = name.size() == 4 ? name : " " + name;
        text += formatted(
            "ATOM  %5zu %-4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n", serial,
            padded.c_str(), residue_name.c_str(), "A", residue_number,
            static_cast<double>(coords[atom * 3U]), static_cast<double>(coords[atom * 3U + 1U]),
            static_cast<double>(coords[atom * 3U + 2U]), 1.0, static_cast<double>(plddt[token]),
            name.substr(0, 1).c_str());
        ++serial;
    }
    if (features.real_atoms > 0)
        text += formatted("TER   %5zu      %3s %1s%4d\n", serial, residue_name.c_str(), "A",
                          residue_number);
    text += "END\n";
    return text;
}

std::string write_mmcif(const ProteinFeatures& features, const float* coords,
                        const std::vector<float>& plddt) {
    requireSizes(features, plddt);
    std::string text = "data_esmfold2\n#\n_entry.id esmfold2\n#\n";
    text += "loop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n"
            "_entity_poly_seq.hetero\n";
    for (std::size_t token = 0; token < static_cast<std::size_t>(features.tokens); ++token) {
        text += formatted("1 %d %s n\n", features.residue_index[token] + 1,
                          features.residue_names[token].c_str());
    }
    text += "#\nloop_\n";
    for (const char* field : {"group_PDB",      "id",
                              "type_symbol",    "label_atom_id",
                              "label_alt_id",   "label_comp_id",
                              "label_asym_id",  "label_entity_id",
                              "label_seq_id",   "pdbx_PDB_ins_code",
                              "Cartn_x",        "Cartn_y",
                              "Cartn_z",        "occupancy",
                              "B_iso_or_equiv", "auth_seq_id",
                              "auth_comp_id",   "auth_asym_id",
                              "auth_atom_id",   "pdbx_PDB_model_num"}) {
        text += "_atom_site.";
        text += field;
        text += '\n';
    }
    for (std::size_t atom = 0; atom < static_cast<std::size_t>(features.real_atoms); ++atom) {
        requireFinite(coords, atom);
        const auto token = static_cast<std::size_t>(features.atom_to_token[atom]);
        const std::string& name = features.atom_names[atom];
        const std::string& residue = features.residue_names[token];
        const std::int32_t number = features.residue_index[token] + 1;
        text += formatted("ATOM %zu %s %s . %s A 1 %d ? %.3f %.3f %.3f 1.00 %.4f %d %s A %s 1\n",
                          atom + 1U, name.substr(0, 1).c_str(), name.c_str(), residue.c_str(),
                          number, static_cast<double>(coords[atom * 3U]),
                          static_cast<double>(coords[atom * 3U + 1U]),
                          static_cast<double>(coords[atom * 3U + 2U]),
                          static_cast<double>(plddt[token]), number, residue.c_str(), name.c_str());
    }
    text += "#\n";
    return text;
}

} // namespace trtmc::esmfold2
