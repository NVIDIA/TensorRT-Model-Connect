/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/edge_llm/runtime/bridge.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::qwen::edge_llm {

namespace {

struct Handle {
    std::string text;
    std::vector<std::int32_t> token_ids;
};

void set_error(char* output, std::size_t capacity, const char* message) noexcept {
    if (output != nullptr && capacity != 0)
        std::snprintf(output, capacity, "%s", message);
}

void* create(const char* engine_directory, char* error, std::size_t error_capacity) noexcept {
    if (engine_directory == nullptr || !std::filesystem::is_directory(engine_directory)) {
        set_error(error, error_capacity, "fake bridge requires an engine directory");
        return nullptr;
    }
    return new Handle();
}

void destroy(void* handle) noexcept {
    delete static_cast<Handle*>(handle);
}

bool generate(void* opaque, const BridgeRequest* request, BridgeResult* result, char* error,
              std::size_t error_capacity) noexcept {
    if (opaque == nullptr || request == nullptr || result == nullptr ||
        request->prompt == nullptr) {
        set_error(error, error_capacity, "invalid fake bridge request");
        return false;
    }
    auto& handle = *static_cast<Handle*>(opaque);
    const std::string prompt(request->prompt, request->prompt_size);
    handle.text = "fake:" + prompt;
    const std::size_t count =
        std::min<std::size_t>(prompt.size(), static_cast<std::size_t>(request->max_new_tokens));
    handle.token_ids.assign(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(count));
    *result = {handle.text.c_str(), handle.token_ids.data(), handle.token_ids.size()};
    return true;
}

const BridgeApi kApi{kBridgeAbiVersion, sizeof(BridgeApi), &create, &destroy, &generate};

} // namespace

} // namespace trtmc::qwen::edge_llm

extern "C" const trtmc::qwen::edge_llm::BridgeApi* trtmc_qwen_edge_llm_bridge_v1() noexcept {
    return &trtmc::qwen::edge_llm::kApi;
}
