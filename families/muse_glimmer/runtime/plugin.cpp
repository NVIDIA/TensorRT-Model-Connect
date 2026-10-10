/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/runtime/family_factory.h"
#ifdef TRTMC_HAS_EDGE_LLM
#include "families/muse_glimmer/runtime/edge_llm/adapter.h"
#endif

#include <stdexcept>

namespace trtmc::muse_glimmer {
ITask* create(const FamilyContext& context) {
    if (!context.reader.find_section("edge_llm.json"))
        throw std::runtime_error("Muse-Glimmer supports only family-owned Edge-LLM bundles");
#ifdef TRTMC_HAS_EDGE_LLM
    return edge_llm::create(context.reader);
#else
    throw std::runtime_error(
        "Muse-Glimmer bundle requires a runtime configured with -DTRTMC_ENABLE_EDGELLM=ON");
#endif
}
} // namespace trtmc::muse_glimmer

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("muse_glimmer does not support --kv-cache-size");
    return trtmc::muse_glimmer::create(context);
}
