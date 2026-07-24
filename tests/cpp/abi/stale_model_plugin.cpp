/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Independently compiled model DSO using only the frozen pre-handshake ABI.
// The current loader must reject it before model-id, registration, or create.

#include "frozen_github_main_abi.h"

#include <cstdint>
#include <memory>

namespace {

std::uint32_t g_model_id_calls = 0;
std::uint32_t g_registration_calls = 0;
std::uint32_t g_factory_calls = 0;

class StaleModelPlugin final : public trtmc::IPipelinePlugin {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext&) override {
        ++g_factory_calls;
        return nullptr;
    }
};

StaleModelPlugin g_plugin;

} // namespace

extern "C" const char* trtmc_model_plugin_id() {
    ++g_model_id_calls;
    return "qwen";
}

extern "C" void trtmc_register_model_plugin(trtmc::PipelineRegistry* registry) {
    ++g_registration_calls;
    registry->register_plugin("qwen_decoder_kv_cache", &g_plugin);
}

extern "C" std::uint32_t trtmc_test_stale_model_id_calls() {
    return g_model_id_calls;
}

extern "C" std::uint32_t trtmc_test_stale_model_registration_calls() {
    return g_registration_calls;
}

extern "C" std::uint32_t trtmc_test_stale_model_factory_calls() {
    return g_factory_calls;
}
