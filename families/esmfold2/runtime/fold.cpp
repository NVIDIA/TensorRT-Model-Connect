/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmfold2/runtime/fold.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace trtmc::esmfold2 {

FoldControls resolve_controls(const StructurePredictionConfig& requested,
                              const BundleConfig& bundle) {
    FoldControls controls;
    if (requested.recycling_steps < 0 || requested.recycling_steps > kMaxRecyclingSteps)
        throw std::invalid_argument("ESMFold2 recycling_steps must lie in [0, " +
                                    std::to_string(kMaxRecyclingSteps) + "]");
    controls.loops = requested.recycling_steps;
    controls.configured_steps =
        requested.sampling_steps == 0 ? bundle.num_sampling_steps : requested.sampling_steps;
    if (controls.configured_steps < 1 || controls.configured_steps > kMaxSamplingSteps)
        throw std::invalid_argument("ESMFold2 sampling_steps must lie in [1, " +
                                    std::to_string(kMaxSamplingSteps) +
                                    "] (0 selects the checkpoint's schedule)");
    controls.samples = requested.diffusion_samples == 0 ? bundle.num_diffusion_samples
                                                        : requested.diffusion_samples;
    if (controls.samples < 1 || controls.samples > bundle.max_samples)
        throw std::invalid_argument("ESMFold2 diffusion_samples must lie in [1, " +
                                    std::to_string(bundle.max_samples) +
                                    "] (0 selects the checkpoint's count)");
    if (requested.seed < -1)
        throw std::invalid_argument("ESMFold2 seed must be non-negative (-1 selects seed 0)");
    controls.seed = requested.seed == -1 ? 0U : static_cast<std::uint64_t>(requested.seed);
    if (requested.output_format != StructureFormat::kMmcif &&
        requested.output_format != StructureFormat::kPdb)
        throw std::invalid_argument("ESMFold2 writes mmCIF or PDB structures");
    controls.format = requested.output_format;
    return controls;
}

std::vector<float> tm_per_bin(std::int32_t tokens, std::int32_t bins) {
    if (tokens < 1 || bins < 1)
        throw std::invalid_argument("ESMFold2 pTM weights need positive token and bin counts");
    const float residues = static_cast<float>(std::max(tokens, 19));
    const float d0 = 1.24F * std::pow(residues - 15.0F, static_cast<float>(1.0 / 3.0)) - 1.8F;
    const double width = 32.0 / bins;
    std::vector<float> weights;
    weights.reserve(static_cast<std::size_t>(bins));
    for (std::int32_t bin = 0; bin < bins; ++bin) {
        const auto center = static_cast<float>(0.5 * width + bin * width);
        const float ratio = center / d0;
        weights.push_back(1.0F / (1.0F + ratio * ratio));
    }
    return weights;
}

SampleConfidence aggregate_confidence(const ProteinFeatures& features,
                                      const std::vector<float>& plddt_atom,
                                      const std::vector<float>& ptm_rows, double eps) {
    const auto tokens = static_cast<std::size_t>(features.tokens);
    const auto atoms = static_cast<std::size_t>(features.atoms);
    if (plddt_atom.size() != atoms || ptm_rows.size() != tokens || tokens == 0)
        throw std::invalid_argument("ESMFold2 confidence outputs do not match the request");
    for (const float value : plddt_atom)
        if (!std::isfinite(value))
            throw std::runtime_error("ESMFold2 confidence head produced a non-finite pLDDT");
    for (const float value : ptm_rows)
        if (!std::isfinite(value))
            throw std::runtime_error("ESMFold2 confidence head produced a non-finite pTM term");
    SampleConfidence result;
    std::vector<double> sums(tokens, 0.0);
    std::vector<double> counts(tokens, 0.0);
    double total = 0.0;
    double valid = 0.0;
    for (std::size_t atom = 0; atom < atoms; ++atom) {
        const double mask = features.atom_mask[atom];
        const auto token = static_cast<std::size_t>(features.atom_to_token[atom]);
        sums[token] += static_cast<double>(plddt_atom[atom]) * mask;
        counts[token] += mask;
        total += static_cast<double>(plddt_atom[atom]) * mask;
        valid += mask;
    }
    result.plddt.resize(tokens);
    for (std::size_t token = 0; token < tokens; ++token)
        result.plddt[token] =
            static_cast<float>(sums[token]) / std::max(static_cast<float>(counts[token]), 1.0e-6F);
    result.complex_plddt =
        static_cast<float>(total) / (static_cast<float>(valid) + static_cast<float>(eps));
    result.ptm = *std::max_element(ptm_rows.begin(), ptm_rows.end());
    return result;
}

std::vector<std::int32_t> rank_samples(const std::vector<SampleConfidence>& confidences) {
    std::vector<std::int32_t> order(confidences.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::int32_t left, std::int32_t right) {
        return confidences[static_cast<std::size_t>(left)].ptm >
               confidences[static_cast<std::size_t>(right)].ptm;
    });
    return order;
}

} // namespace trtmc::esmfold2
