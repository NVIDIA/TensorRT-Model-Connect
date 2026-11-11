/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_next/runtime/pipeline.h"

#include <stdexcept>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen3_next does not support --kv-cache-size");
    if (context.reader.info().family != "qwen3_next" ||
        context.reader.info().task != "text_generation")
        throw std::invalid_argument(
            "Qwen3Next factory requires a Qwen3Next text_generation bundle");
    return trtmc::qwen3_next::create_pipeline(context);
}
