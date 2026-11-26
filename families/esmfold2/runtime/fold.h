/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/esmfold2/runtime/bundle_config.h"
#include "families/esmfold2/runtime/protein.h"
#include "trtmc/task.h"

#include <cstdint>
#include <vector>

namespace trtmc::esmfold2 {

// Resolved controls of one request.
struct FoldControls {
    std::int32_t loops{3};
    std::int32_t configured_steps{0};
    std::int32_t samples{0};
    std::uint64_t seed{0};
    StructureFormat format{StructureFormat::kMmcif};
};

inline constexpr std::int32_t kMaxRecyclingSteps = 64;
inline constexpr std::int32_t kMaxSamplingSteps = 1000;

// recycling_steps = trunk loops (num_loops); 0 sampling steps or samples select the checkpoint's
// values; seed -1 selects 0.
FoldControls resolve_controls(const StructurePredictionConfig& requested,
                              const BundleConfig& bundle);

// Confidence of one sample, from the confidence plan's per-atom pLDDT and per-row pTM terms.
struct SampleConfidence {
    std::vector<float> plddt; // per residue, 0-1
    float complex_plddt{0.0F};
    float ptm{0.0F};
};

SampleConfidence aggregate_confidence(const ProteinFeatures& features,
                                      const std::vector<float>& plddt_atom,
                                      const std::vector<float>& ptm_rows, double eps);

// TM-score weight of each PAE bin (0-32 A) for a chain of ``tokens`` residues, in the reference's
// float32 steps.
std::vector<float> tm_per_bin(std::int32_t tokens, std::int32_t bins);

// Order of the samples, best first: descending pTM, ties by index.
std::vector<std::int32_t> rank_samples(const std::vector<SampleConfidence>& confidences);

} // namespace trtmc::esmfold2
