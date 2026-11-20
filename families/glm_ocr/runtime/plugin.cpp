/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/glm_ocr/runtime/model.h"
#include "trtmc/runtime/family_factory.h"

#include <stdexcept>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    // The key/value cache is sized at build time by max_sequence_length.
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("glm_ocr does not support --kv-cache-size");
    return new trtmc::glm_ocr::GlmOcrModel(context.reader, context.backend);
}
