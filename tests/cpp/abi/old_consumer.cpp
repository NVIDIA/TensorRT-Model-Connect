/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "frozen_github_main_abi.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

namespace trtmc {
// PipelineContext carries this internal type only by reference. Completing it
// locally lets the old consumer instantiate and measure the frozen aggregate
// without importing a current internal header.
struct BundleFile {};
} // namespace trtmc

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(trtmc::LoadOptions) == 184);
static_assert(alignof(trtmc::LoadOptions) == 8);
static_assert(offsetof(trtmc::LoadOptions, hf_python) == 0);
static_assert(offsetof(trtmc::LoadOptions, runtime_cache_path) == 32);
static_assert(offsetof(trtmc::LoadOptions, cuda_graphs) == 64);
static_assert(offsetof(trtmc::LoadOptions, kv_cache_size_bytes) == 72);
static_assert(offsetof(trtmc::LoadOptions, config_path) == 80);
static_assert(offsetof(trtmc::LoadOptions, set_tokens) == 112);
static_assert(offsetof(trtmc::LoadOptions, backend_search_paths) == 136);
static_assert(offsetof(trtmc::LoadOptions, model_plugin_search_paths) == 160);

static_assert(sizeof(trtmc::PipelineContext) == 80);
static_assert(alignof(trtmc::PipelineContext) == 8);

static_assert(sizeof(TrtmcPipelineOptions) == 40);
static_assert(alignof(TrtmcPipelineOptions) == 8);
static_assert(offsetof(TrtmcPipelineOptions, max_new_tokens) == 0);
static_assert(offsetof(TrtmcPipelineOptions, hf_python) == 8);
static_assert(offsetof(TrtmcPipelineOptions, image_path) == 16);
static_assert(offsetof(TrtmcPipelineOptions, runtime_cache) == 24);
static_assert(offsetof(TrtmcPipelineOptions, cuda_graphs) == 32);
#endif

std::uintptr_t pointer_slot(const trtmc::PipelineContext& context, std::size_t offset) {
    std::uintptr_t value = 0;
    std::memcpy(&value, reinterpret_cast<const unsigned char*>(&context) + offset, sizeof(value));
    return value;
}

void validate_frozen_pipeline_context_offsets() {
    trtmc::BundleFile bundle;
    trtmc::BaseConfig config;
    const std::string config_json;
    const std::string hf_python;
    const std::string bundle_path;
    const std::string runtime_cache_path;
    int backend_marker = 0;
    int runtime_config_marker = 0;
    auto* backend = reinterpret_cast<trtmc::IBackend*>(std::addressof(backend_marker));
    auto* runtime_config =
        reinterpret_cast<const trtmc::config::ConfigBundle*>(std::addressof(runtime_config_marker));
    constexpr std::uint64_t cache_bytes = 0x0123456789abcdefULL;
    const trtmc::PipelineContext context{bundle,      config,        config_json,        hf_python,
                                         bundle_path, backend,       runtime_cache_path, true,
                                         cache_bytes, runtime_config};

    check(pointer_slot(context, 0) == reinterpret_cast<std::uintptr_t>(&bundle),
          "PipelineContext.bundle offset");
    check(pointer_slot(context, 8) == reinterpret_cast<std::uintptr_t>(&config),
          "PipelineContext.config offset");
    check(pointer_slot(context, 16) == reinterpret_cast<std::uintptr_t>(&config_json),
          "PipelineContext.config_json offset");
    check(pointer_slot(context, 24) == reinterpret_cast<std::uintptr_t>(&hf_python),
          "PipelineContext.hf_python offset");
    check(pointer_slot(context, 32) == reinterpret_cast<std::uintptr_t>(&bundle_path),
          "PipelineContext.bundle_path offset");
    check(pointer_slot(context, 40) == reinterpret_cast<std::uintptr_t>(backend),
          "PipelineContext.backend offset");
    check(pointer_slot(context, 48) == reinterpret_cast<std::uintptr_t>(&runtime_cache_path),
          "PipelineContext.runtime_cache_path offset");

    bool cuda_graphs = false;
    std::memcpy(&cuda_graphs, reinterpret_cast<const unsigned char*>(&context) + 56,
                sizeof(cuda_graphs));
    check(cuda_graphs, "PipelineContext.cuda_graphs offset");
    std::uint64_t observed_cache_bytes = 0;
    std::memcpy(&observed_cache_bytes, reinterpret_cast<const unsigned char*>(&context) + 64,
                sizeof(observed_cache_bytes));
    check(observed_cache_bytes == cache_bytes, "PipelineContext.kv_cache_size_bytes offset");
    check(pointer_slot(context, 72) == reinterpret_cast<std::uintptr_t>(runtime_config),
          "PipelineContext.runtime_config offset");
}

void call_frozen_cpp_surface() {
    trtmc::LoadOptions options{"python", "cache", true,        123,
                               "config", {"set"}, {"backend"}, {"model"}};
    check(options.kv_cache_size_bytes == 123, "old LoadOptions aggregate remains usable");

    bool rejected = false;
    try {
        (void)trtmc::load("/definitely/not/a/trtmc/bundle.trtfb", options);
    } catch (const std::exception& error) {
        rejected = std::string(error.what()).find("bundle") != std::string::npos ||
                   std::string(error.what()).find("Bundle") != std::string::npos ||
                   !std::string(error.what()).empty();
    }
    check(rejected, "old C++ LoadOptions overload links to and executes current core");
}

void call_frozen_c_surface() {
    TrtmcPipelineOptions options{7, "python", nullptr, "cache", 1};
    trtmc::IPipeline* pipeline =
        trtmc_create_pipeline_ex("/definitely/not/a/trtmc/bundle.trtfb", &options);
    check(pipeline == nullptr, "old C create_ex returns null for an invalid bundle");
    const char* error = trtmc_last_error();
    check(error != nullptr && error[0] != '\0', "old C create_ex reports current-core error");
    check(trtmc_version() != nullptr, "old C version symbol remains callable");
}

} // namespace

int main() {
    validate_frozen_pipeline_context_offsets();
    call_frozen_cpp_surface();
    call_frozen_c_surface();
    if (failures == 0)
        std::cout << "old consumer/current core ABI: PASS\n";
    return failures == 0 ? 0 : 1;
}
