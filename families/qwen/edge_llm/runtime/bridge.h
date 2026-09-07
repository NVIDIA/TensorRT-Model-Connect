/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace trtmc::qwen::edge_llm {

inline constexpr std::uint32_t kBridgeAbiVersion = 1;
inline constexpr const char* kBridgeSymbol = "trtmc_qwen_edge_llm_bridge_v1";

struct BridgeRequest {
    const char* prompt;
    std::size_t prompt_size;
    std::int32_t max_new_tokens;
    float temperature;
    std::int32_t top_k;
    float top_p;
    bool use_chat_template;
    bool enable_thinking;
};

struct BridgeResult {
    const char* text;
    const std::int32_t* token_ids;
    std::size_t token_count;
};

struct BridgeApi {
    std::uint32_t abi_version;
    std::size_t struct_size;
    void* (*create)(const char* engine_directory, char* error, std::size_t error_capacity) noexcept;
    void (*destroy)(void* handle) noexcept;
    bool (*generate)(void* handle, const BridgeRequest* request, BridgeResult* result, char* error,
                     std::size_t error_capacity) noexcept;
};

} // namespace trtmc::qwen::edge_llm

extern "C" const trtmc::qwen::edge_llm::BridgeApi* trtmc_qwen_edge_llm_bridge_v1() noexcept;
