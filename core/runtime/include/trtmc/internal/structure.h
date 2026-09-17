/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/config.h"
#include "trtmc/task.h"

#include <optional>

namespace trtmc::internal {

struct MolecularDocumentToStructureRequest {
    Span<const std::uint8_t> document;
    // Explicit family-supported encoding, for example yaml, json or b2rq.
    // Prepared documents may contain NUL bytes. Core never parses or converts
    // documents and never prepares model features or resolves source paths.
    std::string_view encoding;
    // Original document location, when relevant to family-owned references.
    // Empty means unspecified; Core does not read this path.
    std::string_view source_path;
};

struct MolecularStructureResult {
    std::string structure;
    trtmc::StructureFormat format{trtmc::StructureFormat::kMmcif};
    std::string metadata_json;
    // Missing is not a collection of zero scores. The family supplies this
    // only when the complete existing confidence contract is available.
    std::optional<trtmc::StructureConfidence> confidence;
};

class IMolecularDocumentToStructure {
  public:
    using TaskInterface = IMolecularDocumentToStructure;
    static constexpr std::string_view kTask = "molecular_document_to_structure";
    virtual ~IMolecularDocumentToStructure() = default;
    // All input views borrow through synchronous return. Family owns request
    // preparation, supported encodings, profile constraints and Config policy.
    virtual MolecularStructureResult run(const MolecularDocumentToStructureRequest&,
                                         ConfigView) = 0;
};

} // namespace trtmc::internal
