/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/flint/runtime/pipeline.h"

#include <stdexcept>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("flint does not support --kv-cache-size");
    if (context.reader.info().family != "flint" || context.reader.info().task != "text_generation")
        throw std::invalid_argument("Flint factory requires a Flint text_generation bundle");
    return trtmc::flint::create_pipeline(context);
}
