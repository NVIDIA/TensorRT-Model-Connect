/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::wan2_2_ti2v {

struct PluginRuntimeAbi {
    int32_t tensorrt_major{0};
    int32_t tensorrt_minor{0};
    int32_t cuda_major{0};
    int32_t cudnn_major{0};

    bool operator==(const PluginRuntimeAbi& other) const {
        return tensorrt_major == other.tensorrt_major && tensorrt_minor == other.tensorrt_minor &&
               cuda_major == other.cuda_major && cudnn_major == other.cudnn_major;
    }
};

struct PluginContract {
    int32_t schema{0};
    std::string family;
    std::string semantic_abi;
    std::string source_digest;
    std::string creator_set;
    PluginRuntimeAbi runtime_abi;
    std::vector<int32_t> cuda_architectures;

    bool operator==(const PluginContract& other) const {
        return schema == other.schema && family == other.family &&
               semantic_abi == other.semantic_abi && source_digest == other.source_digest &&
               creator_set == other.creator_set && runtime_abi == other.runtime_abi &&
               cuda_architectures == other.cuda_architectures;
    }
};

// Parse the required top-level _trtmc_wan22_plugin_contract object from the
// bundle's config.json. The v1 contract is exact: missing or unknown fields
// fail closed so a semantic change must increment the schema explicitly.
PluginContract parse_bundle_plugin_contract(const std::string& config_json);

// Parse the contract object exported by the installed, model-local companion
// DSO. Unlike the bundle form, this JSON is the object itself (not wrapped in
// config.json).
PluginContract parse_companion_plugin_contract(const std::string& manifest_json);

// Return the only accepted canonical encoding for the companion's runtime ABI
// getter. It deliberately includes the TensorRT minor: serialized plans and
// plugin registration are qualified as one TRT major.minor unit.
std::string canonical_runtime_abi(const PluginRuntimeAbi& abi);

// Fail closed unless bundle provenance, installed companion provenance,
// actually loaded TRT/CUDA/cuDNN ABI, and current GPU architecture all agree.
void validate_plugin_contract(const PluginContract& expected, const PluginContract& installed,
                              const std::string& loaded_runtime_abi, int32_t current_sm);

} // namespace trtmc::wan2_2_ti2v
