/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/registry/model_plugin_abi.h"

#include <cstdint>
#include <memory>

namespace {

std::uint32_t g_query_calls = 0;
std::uint32_t g_model_id_calls = 0;
std::uint32_t g_registration_calls = 0;
std::uint32_t g_factory_calls = 0;

class PairedModelPlugin final : public trtmc::IPipelinePlugin {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext&) override {
        ++g_factory_calls;
        return nullptr;
    }
};

} // namespace

extern "C" std::int32_t
trtmc_model_plugin_query_abi_contract_v2(trtmc::ModelPluginDsoAbiContractV2* contract,
                                         std::size_t contract_size) noexcept {
    ++g_query_calls;
    if (contract == nullptr || contract_size < sizeof(*contract))
        return -1;
    *contract = trtmc::make_model_plugin_dso_abi_contract_v2();
#if defined(TRTMC_TEST_MODEL_PLUGIN_BAD_FINGERPRINT)
    contract->interface_fingerprint ^= 1;
#endif
    return 0;
}

extern "C" const char* trtmc_model_plugin_id() {
    ++g_model_id_calls;
    return "qwen";
}

extern "C" void trtmc_register_model_plugin(trtmc::PipelineRegistry* registry) {
    ++g_registration_calls;
    if (registry == nullptr)
        return;
    static PairedModelPlugin plugin;
    registry->register_plugin("qwen_decoder_kv_cache", &plugin);
}

extern "C" std::uint32_t trtmc_test_paired_model_query_calls() {
    return g_query_calls;
}

extern "C" std::uint32_t trtmc_test_paired_model_id_calls() {
    return g_model_id_calls;
}

extern "C" std::uint32_t trtmc_test_paired_model_registration_calls() {
    return g_registration_calls;
}

extern "C" std::uint32_t trtmc_test_paired_model_factory_calls() {
    return g_factory_calls;
}
