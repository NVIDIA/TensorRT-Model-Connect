/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_pipeline.h"
#include "sam2_video_session.h"
#include "trtmc/models/sam2_video.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

namespace {

struct ModuleFactoryState {
    IBackend* backend{nullptr};
    cudaStream_t stream{nullptr};
};

sam2::NativePlanModuleFactory makeModuleFactory(IBackend* backend) {
    if (backend == nullptr)
        throw std::invalid_argument("SAM2 runtime requires a TensorRT backend");
    if (std::string(backend->name() == nullptr ? "" : backend->name()) != "trt") {
        throw std::invalid_argument("SAM2 runtime requires the standard TensorRT backend");
    }

    auto state = std::make_shared<ModuleFactoryState>();
    state->backend = backend;
    return [state](std::string_view section, const void* plan_data,
                   std::size_t plan_size) -> std::unique_ptr<ITrtModule> {
        ModuleCreateOptions options;
        options.stream = state->stream;
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
        if (context.cuda_graphs) {
            throw std::invalid_argument(
                "SAM2 user-built runtime does not admit CUDA graph execution");
        }
        return sam2::Sam2Pipeline::create(context, makeModuleFactory(context.backend));
    }
};

} // namespace

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_sam2_plugin, Sam2Plugin,
                                       "sam2_bbox_video_tracking");

} // namespace trtmc

extern "C" TrtmcSam2VideoSession*
trtmc_sam2_video_create_from_bundle_v1(const char* bundle_path, const char* plugin_dir,
                                       const char* backend_dir) noexcept {
    trtmc::sam2::c_api_internal::clearLastError();
    try {
        if (bundle_path == nullptr || plugin_dir == nullptr || backend_dir == nullptr ||
            *bundle_path == '\0' || *plugin_dir == '\0' || *backend_dir == '\0') {
            throw std::invalid_argument("SAM2 bundle, plugin, and backend paths are required");
        }

        trtmc::LoadOptions options;
        options.model_plugin_search_paths.emplace_back(plugin_dir);
        options.backend_search_paths.emplace_back(backend_dir);
        auto pipeline = trtmc::load(bundle_path, options);
        auto* sam2_pipeline = dynamic_cast<trtmc::sam2::Sam2Pipeline*>(pipeline.get());
        if (sam2_pipeline == nullptr) {
            throw std::runtime_error("loaded bundle did not create a SAM2 native-video pipeline");
        }
        return trtmc::sam2::makeVideoSessionHandle(sam2_pipeline->releaseVideoProcessor());
    } catch (const std::exception& error) {
        trtmc::sam2::c_api_internal::setLastError(error.what());
    } catch (...) {
        trtmc::sam2::c_api_internal::setLastError("unknown native exception");
    }
    return nullptr;
}
