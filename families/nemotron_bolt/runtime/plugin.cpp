/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "native_task.h"
#ifdef TRTMC_BOLT_EDGE
#include "edge_llm/adapter.h"
#endif
#include <stdexcept>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("nemotron_bolt does not support --kv-cache-size");
    if (context.reader.find_section("edge_llm.json")) {
#ifdef TRTMC_BOLT_EDGE
        return trtmc::nemotron_bolt::edge_llm::create(context.reader);
#else
        throw std::runtime_error("This Bolt runtime was built without the Edge 0.12 SDK");
#endif
    }
    return trtmc::nemotron_bolt::create_native(context);
}
