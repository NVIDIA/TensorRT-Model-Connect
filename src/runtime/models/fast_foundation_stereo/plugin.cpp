/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/models/fast_foundation_stereo/stereo_pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

std::filesystem::path native_plugin_cache_path(const std::vector<char>& bytes) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char value : bytes) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    }
    const char* configured = std::getenv("TRTMC_FAST_FOUNDATION_STEREO_NATIVE_PLUGIN_CACHE_DIR");
    const auto directory =
        configured != nullptr && configured[0] != '\0'
            ? std::filesystem::path(configured)
            : std::filesystem::temp_directory_path() / "trtmc-fast-foundation-stereo";
    std::ostringstream name;
    name << "libtrtmc_fast_foundation_stereo_native_plugin_" << std::hex << hash << ".so";
    return directory / name.str();
}

void write_native_plugin_cache_file(const std::filesystem::path& output,
                                    const std::vector<char>& bytes) {
    std::filesystem::create_directories(output.parent_path());
    if (std::filesystem::is_regular_file(output) &&
        std::filesystem::file_size(output) == bytes.size()) {
        return;
    }

    const auto temporary = output.string() + ".tmp";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error(
            "Unable to create Fast Foundation Stereo native plugin cache file");
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream)
        throw std::runtime_error("Unable to write Fast Foundation Stereo native plugin cache file");
    std::filesystem::rename(temporary, output);
}

std::string resolve_native_plugin_path(const PipelineContext& ctx) {
    const auto* bytes = find_section(ctx.bundle, "fast_foundation_stereo_native_plugin_so");
    if (bytes != nullptr && !bytes->empty()) {
        const auto path = native_plugin_cache_path(*bytes);
        write_native_plugin_cache_file(path, *bytes);
        return path.string();
    }

    const char* configured = std::getenv("TRTMC_FAST_FOUNDATION_STEREO_NATIVE_PLUGIN_LIBRARY");
    return configured != nullptr ? configured : std::string{};
}

void load_native_plugin(const PipelineContext& ctx) {
    const auto path = resolve_native_plugin_path(ctx);
    if (path.empty()) {
        throw std::runtime_error(
            "Fast Foundation Stereo bundle is missing its native TensorRT plugin section");
    }

    dlerror();
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* message = dlerror();
        throw std::runtime_error(
            std::string("Unable to load Fast Foundation Stereo native plugin: ") +
            (message != nullptr ? message : path));
    }
    dlerror();
    auto* symbol = dlsym(handle, "fast_foundation_stereo_gwc_plugin_force_link");
    const char* message = dlerror();
    if (message != nullptr || symbol == nullptr) {
        throw std::runtime_error(
            "Fast Foundation Stereo native plugin is missing its identity symbol");
    }
    reinterpret_cast<void (*)()>(symbol)();
}

std::unique_ptr<ITrtModule> load_module(IBackend* backend, const std::vector<char>* plan,
                                        const ModuleCreateOptions& options, const char* label) {
    if (backend == nullptr || plan == nullptr || plan->empty())
        throw std::runtime_error(std::string("Fast Foundation Stereo missing ") + label);
    auto module = backend->create_module(plan->data(), plan->size(), options);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("Fast Foundation Stereo failed to load ") + label);
    return module;
}

} // namespace

class FastFoundationStereoPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_native_plugin(ctx);
        ModuleCreateOptions feature_options;
        feature_options.runtime_cache_path = ctx.runtime_cache_path.c_str();
        feature_options.cuda_graphs = ctx.cuda_graphs;
        auto feature = load_module(ctx.backend, find_section(ctx.bundle, "engine_plan"),
                                   feature_options, "feature engine_plan");

        ModuleCreateOptions post_options = feature_options;
        post_options.stream = feature->stream();
        auto post = load_module(ctx.backend,
                                find_section(ctx.bundle, "fast_foundation_stereo_post_engine_plan"),
                                post_options, "post engine plan");
        return std::make_unique<FastFoundationStereoPipeline>(std::move(feature), std::move(post),
                                                              ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_fast_foundation_stereo_plugin,
                                       FastFoundationStereoPlugin,
                                       "fast_foundation_stereo_disparity");

} // namespace trtmc
