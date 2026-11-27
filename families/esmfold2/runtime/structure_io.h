/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/esmfold2/runtime/protein.h"

#include <string>
#include <vector>

namespace trtmc::esmfold2 {

// PDB text of one predicted sample: every real atom under its name, per-residue pLDDT (0-1) as the
// B-factor, chain A, in the layout of the reference's ``output_to_pdb``.
std::string write_pdb(const ProteinFeatures& features, const float* coords,
                      const std::vector<float>& plddt);

// mmCIF text of the same sample: ``atom_site`` with label and author ids, and the chain's
// ``entity_poly_seq``.
std::string write_mmcif(const ProteinFeatures& features, const float* coords,
                        const std::vector<float>& plddt);

} // namespace trtmc::esmfold2
