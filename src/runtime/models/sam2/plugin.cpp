/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "runtime/backend/trt_version.h"
#include "runtime/models/sam2/sam2_pipeline.h"
#include "runtime/models/sam2/sam2_video_session.h"
#include "trtmc/models/sam2_video.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

namespace {

sam2::NativeBundleRuntimeTarget observeRuntimeTarget() {
    const auto version = detect_installed_trt_version();
    if (!version) {
        throw std::runtime_error(
            "SAM2 production runtime could not observe the loaded TensorRT version");
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
        throw std::runtime_error("SAM2 production runtime could not query the CUDA device");
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        throw std::runtime_error("SAM2 production runtime could not query CUDA device properties");
    }

    sam2::NativeBundleRuntimeTarget target;
    target.tensorrt_version = format_trt_version(*version);
    target.tensorrt_abi = trt_abi_string(*version);
    target.gpu_name = properties.name;
    target.compute_capability =
        std::to_string(properties.major) + "." + std::to_string(properties.minor);
    return target;
}

struct ModuleFactoryState {
    IBackend* backend{nullptr};
    std::string runtime_cache_path;
    cudaStream_t stream{nullptr};
};

sam2::NativePlanModuleFactory makeModuleFactory(IBackend* backend, std::string runtime_cache_path) {
    if (backend == nullptr)
        throw std::invalid_argument("SAM2 production runtime requires a TensorRT backend");
    if (std::string(backend->name() == nullptr ? "" : backend->name()) != "trt") {
        throw std::invalid_argument(
            "SAM2 production runtime requires the standard TensorRT backend");
    }

    auto state = std::make_shared<ModuleFactoryState>();
    state->backend = backend;
    state->runtime_cache_path = std::move(runtime_cache_path);
    return [state](std::string_view section, const void* plan_data,
                   std::size_t plan_size) -> std::unique_ptr<ITrtModule> {
        ModuleCreateOptions options;
        options.stream = state->stream;
        options.runtime_cache_path = state->runtime_cache_path.c_str();
        options.cuda_graphs = false;
        auto module = state->backend->create_module(plan_data, plan_size, options);
        if (module == nullptr || !module->ok()) {
            throw std::runtime_error("SAM2 failed to create TensorRT module for " +
                                     std::string(section));
        }
        if (module->stream() == nullptr) {
            throw std::runtime_error("SAM2 TensorRT module has no CUDA stream for " +
                                     std::string(section));
        }
        if (state->stream == nullptr)
            state->stream = module->stream();
        else if (module->stream() != state->stream) {
            throw std::runtime_error(
                "SAM2 TensorRT modules were not created on one shared CUDA stream");
        }
        module->set_timing_label("sam2 " + std::string(section));
        return module;
    };
}

class Sam2Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& context) override {
        if (context.qualification_record_path.empty()) {
            throw std::invalid_argument(
                "SAM2 production runtime requires an explicit qualification-record path");
        }
        if (context.cuda_graphs) {
            throw std::invalid_argument(
                "SAM2 production runtime does not admit unqualified CUDA graph execution");
        }
        return sam2::Sam2Pipeline::createProductionQualified(
            context.bundle_path, context.qualification_record_path, observeRuntimeTarget(),
            makeModuleFactory(context.backend, context.runtime_cache_path),
            context.bundle.info.model_id);
    }
};

} // namespace

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_sam2_plugin, Sam2Plugin,
                                       "sam2_bbox_video_tracking");

} // namespace trtmc

extern "C" TrtmcSam2VideoSession* trtmc_sam2_video_create_from_qualified_bundle_v1(
    const char* bundle_path, const char* qualification_record_path, const char* plugin_dir,
    const char* backend_dir) noexcept {
    trtmc::sam2::c_api_internal::clearLastError();
    try {
        if (bundle_path == nullptr || qualification_record_path == nullptr ||
            plugin_dir == nullptr || backend_dir == nullptr || *bundle_path == '\0' ||
            *qualification_record_path == '\0' || *plugin_dir == '\0' || *backend_dir == '\0') {
            throw std::invalid_argument(
                "SAM2 bundle, qualification-record, plugin, and backend paths are required");
        }

        trtmc::LoadOptions options;
        options.qualification_record_path = qualification_record_path;
        options.model_plugin_search_paths.emplace_back(plugin_dir);
        options.backend_search_paths.emplace_back(backend_dir);
        auto pipeline = trtmc::load(bundle_path, options);
        auto* sam2_pipeline = dynamic_cast<trtmc::sam2::Sam2Pipeline*>(pipeline.get());
        if (sam2_pipeline == nullptr) {
            throw std::runtime_error(
                "loaded bundle did not create a qualified SAM2 native-video pipeline");
        }
        return trtmc::make_sam2_video_session_handle(sam2_pipeline->releaseVideoProcessor());
    } catch (const std::exception& error) {
        trtmc::sam2::c_api_internal::setLastError(error.what());
    } catch (...) {
        trtmc::sam2::c_api_internal::setLastError("unknown native exception");
    }
    return nullptr;
}
