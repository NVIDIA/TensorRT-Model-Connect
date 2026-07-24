/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "frozen_github_main_abi.h"

#include <cstdint>
#include <memory>

namespace {

std::uint32_t create_calls = 0;

class LegacyQwenPlugin final : public trtmc::IPipelinePlugin {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext&) override {
        ++create_calls;
        return nullptr;
    }
};

LegacyQwenPlugin plugin;

} // namespace

extern "C" const char* trtmc_model_plugin_id() {
    return "qwen";
}

extern "C" void trtmc_register_model_plugin(trtmc::PipelineRegistry* registry) {
    registry->register_plugin("qwen_decoder_kv_cache", &plugin);
}

extern "C" std::uint32_t trtmc_test_legacy_plugin_create_calls() {
    return create_calls;
}

extern "C" void trtmc_test_legacy_plugin_reset() {
    create_calls = 0;
}
