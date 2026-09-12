/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/structure.h"

#include "api_internal.h"
#include "trtmc/structure.h"

#include <cmath>

namespace trtmc::api {
namespace {

struct MolecularStructureStorage final : ResultStorage {
    explicit MolecularStructureStorage(internal::MolecularStructureResult result)
        : value(std::move(result)) {
        if (value.structure.empty())
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned an empty molecular structure"};
        switch (value.format) {
        case StructureFormat::kMmcif:
            view.format = TRTMC_STRUCTURE_MMCIF;
            break;
        case StructureFormat::kPdb:
            view.format = TRTMC_STRUCTURE_PDB;
            break;
        default:
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned an unknown structure format"};
        }
        view.structure = borrowed_string(value.structure);
        view.metadata_json = borrowed_string(value.metadata_json);
        if (value.confidence) {
            const auto& score = *value.confidence;
            const float values[] = {score.confidence_score, score.ptm,          score.iptm,
                                    score.ligand_iptm,      score.protein_iptm, score.complex_plddt,
                                    score.complex_iplddt};
            for (const float item : values)
                require_finite_score(item);
            for (const float item : score.plddt)
                require_finite_score(item);
            view.has_confidence = 1;
            view.confidence = {score.confidence_score,
                               score.ptm,
                               score.iptm,
                               score.ligand_iptm,
                               score.protein_iptm,
                               score.complex_plddt,
                               score.complex_iplddt,
                               {score.plddt.data(), score.plddt.size()}};
        }
    }
    static void require_finite_score(float value) {
        if (!std::isfinite(value))
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "family returned nonfinite structure confidence"};
    }
    internal::MolecularStructureResult value;
    trtmc_molecular_structure_view_v1 view{};
};

trtmc_status TRTMC_CALL predict_structure(
    trtmc_model* model, const trtmc_molecular_document_to_structure_request_v1* request,
    const trtmc_config_view_v1* config, trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request && out, "molecular document request or result output is null");
        const std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::IMolecularDocumentToStructure>(
            model, internal::IMolecularDocumentToStructure::kTask);
        const auto document = checked_span(request->document, request->document_size);
        const auto encoding = string_view(request->encoding);
        const auto source_path = string_view(request->source_path);
        require(!document.empty(), "molecular document is required");
        require(!encoding.empty(), "molecular document encoding is required");
        const ConvertedConfig options(config);
        validate_task_config(model_owner(model),
                             internal::contract_key<internal::IMolecularDocumentToStructure>(),
                             options.view());
        *out = make_result<MolecularStructureStorage>(
            family.run({document, encoding, source_path}, options.view()));
    });
}

trtmc_status TRTMC_CALL structure_result_view(const trtmc_result* result,
                                              trtmc_molecular_structure_view_v1* out,
                                              trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "molecular structure view output is null");
        *out = require_result<MolecularStructureStorage>(result).view;
    });
}

const trtmc_molecular_document_to_structure_api_v1 structure_api = {
    {1, 0, sizeof(trtmc_molecular_document_to_structure_api_v1)},
    predict_structure,
    structure_result_view};
} // namespace

Span<const TaskBinding> structure_task_bindings() noexcept {
    static const TaskBinding bindings[] = {
        {internal::IMolecularDocumentToStructure::kTask, 1, 0, &structure_api.header}};
    return {bindings};
}
} // namespace trtmc::api
