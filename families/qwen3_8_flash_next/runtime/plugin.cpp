/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "native_task.h"
#ifdef TRTMC_FLASH_NEXT_EDGE
#include "edge_llm/adapter.h"
#endif
#include <stdexcept>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen3_8_flash_next does not support --kv-cache-size");
    if (context.reader.find_section("edge_llm.json")) {
#ifdef TRTMC_FLASH_NEXT_EDGE
        return trtmc::qwen3_8_flash_next::edge_llm::create(context.reader);
#else
        throw std::runtime_error("This Flash-Next runtime was built without the Edge 0.12 SDK");
#endif
    }
    return trtmc::qwen3_8_flash_next::create_native(context);
}
