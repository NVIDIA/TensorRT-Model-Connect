/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Private C ABI handshake for model-plugin DSOs.  Registration and pipeline
// creation use C++ vtables and STL-bearing values, so the loader must validate
// this POD contract before calling even the model-id or registration entrypoint.

#include "bundle/bundle_format.h"
#include "runtime/backend/runtime_memory_backend.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace trtmc {

inline constexpr std::uint32_t kModelPluginDsoAbiContractVersionV2 = 2;
inline constexpr std::uint64_t kModelPluginDsoInterfaceFingerprintV2 =
    0x4d4f44504c414232ULL; // "MODPLAB2"
inline constexpr const char kModelPluginDsoAbiQuerySymbolV2[] =
    "trtmc_model_plugin_query_abi_contract_v2";

struct ModelPluginDsoAbiContractV2 {
    std::uint32_t struct_size{sizeof(ModelPluginDsoAbiContractV2)};
    std::uint32_t contract_version{kModelPluginDsoAbiContractVersionV2};
    std::uint64_t interface_fingerprint{kModelPluginDsoInterfaceFingerprintV2};

    // Reuse the exact compiler/stdlib, tensor, backend-vtable, and private
    // runtime-memory layout contract already required at the backend boundary.
    BackendDsoAbiContractV2 shared_cpp_contract{};

    std::uint32_t runtime_memory_plugin_api_version{kRuntimeMemoryPluginApiVersionV1};
    std::uint32_t io_map_size{sizeof(IoMap)};
    std::uint32_t io_map_alignment{alignof(IoMap)};
    std::uint32_t base_config_size{sizeof(BaseConfig)};
    std::uint32_t base_config_alignment{alignof(BaseConfig)};
    std::uint32_t pipeline_context_size{sizeof(PipelineContext)};
    std::uint32_t pipeline_context_alignment{alignof(PipelineContext)};
    std::uint32_t runtime_memory_plugin_options_size{sizeof(RuntimeMemoryPluginOptionsV1)};
    std::uint32_t runtime_memory_plugin_options_alignment{alignof(RuntimeMemoryPluginOptionsV1)};
    std::uint32_t pipeline_interface_size{sizeof(IPipeline)};
    std::uint32_t pipeline_interface_alignment{alignof(IPipeline)};
    std::uint32_t pipeline_plugin_interface_size{sizeof(IPipelinePlugin)};
    std::uint32_t pipeline_plugin_interface_alignment{alignof(IPipelinePlugin)};
    std::uint32_t runtime_memory_plugin_interface_size{sizeof(IRuntimeMemoryPipelinePluginV1)};
    std::uint32_t runtime_memory_plugin_interface_alignment{
        alignof(IRuntimeMemoryPipelinePluginV1)};
    std::uint32_t pipeline_registry_size{sizeof(PipelineRegistry)};
    std::uint32_t pipeline_registry_alignment{alignof(PipelineRegistry)};
    std::uint32_t bundle_info_size{sizeof(BundleInfo)};
    std::uint32_t bundle_info_alignment{alignof(BundleInfo)};
    std::uint32_t bundle_file_size{sizeof(BundleFile)};
    std::uint32_t bundle_file_alignment{alignof(BundleFile)};
    std::uint32_t config_bundle_size{sizeof(config::ConfigBundle)};
    std::uint32_t config_bundle_alignment{alignof(config::ConfigBundle)};
    std::uint32_t schema_registry_size{sizeof(config::SchemaRegistry)};
    std::uint32_t schema_registry_alignment{alignof(config::SchemaRegistry)};
};

static_assert(std::is_standard_layout_v<ModelPluginDsoAbiContractV2>);
static_assert(std::is_trivially_copyable_v<ModelPluginDsoAbiContractV2>);

inline ModelPluginDsoAbiContractV2 make_model_plugin_dso_abi_contract_v2() noexcept {
    ModelPluginDsoAbiContractV2 contract;
    contract.shared_cpp_contract = make_runtime_memory_backend_dso_abi_contract_v2(0);
    return contract;
}

using ModelPluginDsoAbiQueryFnV2 = std::int32_t (*)(ModelPluginDsoAbiContractV2* contract,
                                                    std::size_t contract_size) noexcept;

} // namespace trtmc

extern "C" std::int32_t
trtmc_model_plugin_query_abi_contract_v2(trtmc::ModelPluginDsoAbiContractV2* contract,
                                         std::size_t contract_size) noexcept;
