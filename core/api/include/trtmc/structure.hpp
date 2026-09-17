/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/core.hpp"
#include "trtmc/structure.h"

namespace trtmc {

enum class MolecularStructureFormat : std::uint32_t {
    Mmcif = TRTMC_STRUCTURE_MMCIF,
    Pdb = TRTMC_STRUCTURE_PDB,
};

struct MolecularDocumentToStructureRequest {
    Span<const std::uint8_t> document;
    std::string_view encoding;
    std::string_view source_path{};
};

using MolecularStructureConfidenceView = trtmc_structure_confidence_view_v1;

class MolecularStructureResult : public detail::ViewResult<trtmc_molecular_structure_view_v1> {
  public:
    using ViewResult::ViewResult;
    std::string_view structure() const { return detail::string_view(view().structure); }
    MolecularStructureFormat format() const {
        return static_cast<MolecularStructureFormat>(view().format);
    }
    std::string_view metadata_json() const { return detail::string_view(view().metadata_json); }
    std::optional<MolecularStructureConfidenceView> confidence() const {
        const auto value = view();
        if (!value.has_confidence)
            return std::nullopt;
        return value.confidence;
    }
};

class MolecularDocumentToStructure {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_MOLECULAR_DOCUMENT_TO_STRUCTURE;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != kMajor || table->minor != kMinor ||
            table->byte_size < sizeof(trtmc_molecular_document_to_structure_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible molecular structure Task table");
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    MolecularStructureResult run(const MolecularDocumentToStructureRequest& input,
                                 const Config& config = {}) const {
        const trtmc_molecular_document_to_structure_request_v1 request{
            input.document.data(), input.document.size(), detail::c_string(input.encoding),
            detail::c_string(input.source_path)};
        const auto entries = config.c_entries();
        const auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run(model_->handle, &request, &options, &raw, &error);
        detail::ResultOwner owner(model_, raw);
        detail::check(model_->api, status, error);
        return MolecularStructureResult(std::move(owner), api_->result_view);
    }

  private:
    friend class Model;
    MolecularDocumentToStructure(std::shared_ptr<detail::ModelState> model,
                                 const trtmc_api_header* table) noexcept
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_molecular_document_to_structure_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_molecular_document_to_structure_api_v1* api_;
};

} // namespace trtmc
