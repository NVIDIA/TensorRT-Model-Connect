/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/pipeline_plugin.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

namespace trtmc {
// This fixture intentionally does not link libtrtmc_core. It owns the optional
// interface key function so an independently built legacy harness can dlopen it.
IRuntimeMemoryPipelinePluginV1::~IRuntimeMemoryPipelinePluginV1() = default;
} // namespace trtmc

namespace {

std::uint32_t legacy_create_calls = 0;
std::uint32_t runtime_create_calls = 0;

bool bundle_header_declares_runtime_memory(const std::string& bundle_path) {
    std::ifstream input(bundle_path, std::ios::binary);
    if (!input)
        throw std::runtime_error("current plugin fixture could not read bundle header");
    const std::string bytes{std::istreambuf_iterator<char>(input),
                            std::istreambuf_iterator<char>()};
    return bytes.find("\"runtime_memory\"") != std::string::npos;
}

class CurrentPlugin final : public trtmc::IPipelinePlugin,
                            public trtmc::IRuntimeMemoryPipelinePluginV1 {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext& context) override {
        // A legacy core can only reach the stable base create slot. A new plugin
        // must therefore reject a dynamic bundle explicitly instead of silently
        // treating it as a static request. Read the immutable bundle header by
        // path so this check does not dereference a newer BundleFile layout
        // supplied by an older core.
        if (bundle_header_declares_runtime_memory(context.bundle_path)) {
            throw std::runtime_error(
                "runtime_memory bundle requires the versioned plugin entry point; "
                "legacy core fail closed");
        }
        ++legacy_create_calls;
        return nullptr;
    }

    std::unique_ptr<trtmc::IPipeline>
    create_runtime_memory(const trtmc::PipelineContext&,
                          const trtmc::RuntimeMemoryPluginOptionsV1&) override {
        ++runtime_create_calls;
        return nullptr;
    }
};

CurrentPlugin plugin;

} // namespace

extern "C" trtmc::IPipelinePlugin* trtmc_test_current_plugin_base_v1() {
    return &plugin;
}

extern "C" std::uint32_t trtmc_test_current_plugin_legacy_create_calls() {
    return legacy_create_calls;
}

extern "C" std::uint32_t trtmc_test_current_plugin_runtime_create_calls() {
    return runtime_create_calls;
}

extern "C" void trtmc_test_current_plugin_reset() {
    legacy_create_calls = 0;
    runtime_create_calls = 0;
}
