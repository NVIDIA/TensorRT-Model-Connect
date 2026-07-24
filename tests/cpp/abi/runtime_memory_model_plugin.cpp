/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/registry/model_plugin_abi.h"

#include <cstdint>
#include <memory>

#ifndef TRTMC_TEST_RUNTIME_PLUGIN_MODE
#error "TRTMC_TEST_RUNTIME_PLUGIN_MODE must select the fixture behavior"
#endif

namespace {

std::uint32_t g_legacy_create_calls = 0;
std::uint32_t g_runtime_create_calls = 0;
std::uint64_t g_captured_context_kv_cache_size_bytes = 0;
trtmc::RuntimeMemoryPluginOptionsV1 g_captured_options;

class FixturePipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "runtime-memory-abi-fixture"; }
    const char* pipeline_type() const override { return "runtime-memory-abi-fixture"; }
};

#if TRTMC_TEST_RUNTIME_PLUGIN_MODE == 1

class RuntimeMemoryFixturePlugin final : public trtmc::IPipelinePlugin {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext& context) override {
        ++g_legacy_create_calls;
        g_captured_context_kv_cache_size_bytes = context.kv_cache_size_bytes;
        return nullptr;
    }
};

#elif TRTMC_TEST_RUNTIME_PLUGIN_MODE == 2

class RuntimeMemoryFixturePlugin final : public trtmc::IPipelinePlugin,
                                         public trtmc::IRuntimeMemoryPipelinePluginV1 {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext& context) override {
        ++g_legacy_create_calls;
        g_captured_context_kv_cache_size_bytes = context.kv_cache_size_bytes;
        return nullptr;
    }

    std::uint32_t runtime_memory_plugin_api_version() const override {
        return trtmc::kRuntimeMemoryPluginApiVersionV1 + 1U;
    }

    std::unique_ptr<trtmc::IPipeline>
    create_runtime_memory(const trtmc::PipelineContext& context,
                          const trtmc::RuntimeMemoryPluginOptionsV1&) override {
        ++g_runtime_create_calls;
        g_captured_context_kv_cache_size_bytes = context.kv_cache_size_bytes;
        return nullptr;
    }
};

#elif TRTMC_TEST_RUNTIME_PLUGIN_MODE == 3

class RuntimeMemoryFixturePlugin final : public trtmc::IPipelinePlugin,
                                         public trtmc::IRuntimeMemoryPipelinePluginV1 {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext& context) override {
        ++g_legacy_create_calls;
        g_captured_context_kv_cache_size_bytes = context.kv_cache_size_bytes;
        return std::make_unique<FixturePipeline>();
    }

    std::unique_ptr<trtmc::IPipeline>
    create_runtime_memory(const trtmc::PipelineContext& context,
                          const trtmc::RuntimeMemoryPluginOptionsV1& options) override {
        ++g_runtime_create_calls;
        g_captured_context_kv_cache_size_bytes = context.kv_cache_size_bytes;
        g_captured_options = options;
        return std::make_unique<FixturePipeline>();
    }
};

#else
#error "unsupported TRTMC_TEST_RUNTIME_PLUGIN_MODE"
#endif

RuntimeMemoryFixturePlugin g_plugin;

} // namespace

extern "C" std::int32_t
trtmc_model_plugin_query_abi_contract_v2(trtmc::ModelPluginDsoAbiContractV2* contract,
                                         std::size_t contract_size) noexcept {
    if (contract == nullptr || contract_size < sizeof(*contract))
        return -1;
    *contract = trtmc::make_model_plugin_dso_abi_contract_v2();
    return 0;
}

extern "C" const char* trtmc_model_plugin_id() {
    return "qwen";
}

extern "C" void trtmc_register_model_plugin(trtmc::PipelineRegistry* registry) {
    if (registry != nullptr)
        registry->register_plugin("qwen_decoder_kv_cache", &g_plugin);
}

extern "C" void trtmc_test_runtime_plugin_reset() {
    g_legacy_create_calls = 0;
    g_runtime_create_calls = 0;
    g_captured_context_kv_cache_size_bytes = 0;
    g_captured_options = {};
}

extern "C" std::uint32_t trtmc_test_runtime_plugin_legacy_create_calls() {
    return g_legacy_create_calls;
}

extern "C" std::uint32_t trtmc_test_runtime_plugin_runtime_create_calls() {
    return g_runtime_create_calls;
}

extern "C" std::uint32_t trtmc_test_runtime_plugin_captured_policy() {
    return static_cast<std::uint32_t>(g_captured_options.kv_cache_memory_policy);
}

extern "C" double trtmc_test_runtime_plugin_captured_fraction() {
    return g_captured_options.kv_cache_memory_fraction;
}

extern "C" std::uint64_t trtmc_test_runtime_plugin_captured_bytes() {
    return g_captured_options.kv_cache_memory_bytes;
}

extern "C" std::uint64_t trtmc_test_runtime_plugin_captured_context_kv_cache_size_bytes() {
    return g_captured_context_kv_cache_size_bytes;
}

extern "C" std::uint64_t trtmc_test_runtime_plugin_captured_max_sequence_length() {
    return g_captured_options.max_sequence_length;
}

extern "C" std::uint32_t trtmc_test_runtime_plugin_captured_max_sequence_length_explicit() {
    return g_captured_options.max_sequence_length_explicit;
}
