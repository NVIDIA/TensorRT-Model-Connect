/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_5_moe/runtime/edge_llm/adapter.h"
#include "trtmc/runtime/family_factory.h"

#include <stdexcept>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (!context.reader.find_section("edge_llm.json"))
        throw std::runtime_error(
            "Qwen3.5-MoE has no native implementation; an Edge bundle is required");
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen3_5_moe does not support --kv-cache-size");
#ifdef TRTMC_HAS_EDGE_LLM
    return trtmc::qwen3_5_moe::edge_llm::create(context.reader);
#else
    throw std::runtime_error(
        "Qwen3.5-MoE Edge runtime is unavailable; enable the CMake Edge dependency");
#endif
}
