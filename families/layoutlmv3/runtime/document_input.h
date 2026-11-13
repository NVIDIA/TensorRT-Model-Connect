/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/layoutlmv3/runtime/tokenizer.h"
#include "trtmc/internal/features.h"

#include <nlohmann/json.hpp>
namespace trtmc::layoutlmv3 {
struct DocumentInput {
    std::vector<int32_t> ids, valid, positions, boxes, layout, relative_x, relative_y;
    std::vector<internal::DocumentFeatureToken> mapping;
};
DocumentInput encode_document(const ITokenizer&,
                              const internal::DocumentWordsToTokenFeaturesRequest&,
                              const nlohmann::json&);
} // namespace trtmc::layoutlmv3
