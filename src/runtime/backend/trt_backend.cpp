/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// TrtBackend: IBackend implementation for standard TensorRT.
// Compiled into libtrtmc_backend_trt.so. Links libnvinfer.so.

#include "trtmc/runtime/trt_backend.h"

#include "runtime/backend/trt_logger.h"
#include "runtime/core/cuda_common.h"
#include "trt_module_impl.h"

#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#ifndef TRTMC_TRT_BACKEND_ABI_STRING
#define TRTMC_TRT_BACKEND_ABI_STRING ""
#endif

namespace trtmc {

namespace {

std::string trt_runtime_version_string() {
    std::ostringstream oss;
    oss << getInferLibMajorVersion() << "." << getInferLibMinorVersion() << "."
        << getInferLibPatchVersion() << "." << getInferLibBuildVersion();
    return oss.str();
}

std::string trt_backend_abi_string() {
    const std::string configured = TRTMC_TRT_BACKEND_ABI_STRING;
    if (!configured.empty())
        return configured;
    std::ostringstream oss;
    oss << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR;
    return oss.str();
}

void keep_backend_resources(ITrtModule& module,
                            const std::shared_ptr<nvinfer1::ICudaEngine>& engine,
                            const std::shared_ptr<void>& stream_owner,
                            const std::shared_ptr<void>& distributed_owner) {
    module.keep_alive(engine);
    if (stream_owner)
        module.keep_alive(stream_owner);
    if (distributed_owner)
        module.keep_alive(distributed_owner);
}

} // namespace

class TrtBackend final : public IBackend {
  public:
    TrtBackend() : runtime_(create_trt_runtime()) {
        if (!runtime_)
            throw std::runtime_error("[trtmc] Failed to create TRT runtime");
    }

    std::unique_ptr<ITrtModule> create_module(const void* plan_data, size_t plan_size,
                                              const ModuleCreateOptions& options) override {
        auto* engine = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (TRT)");

        auto* ctx = engine->createExecutionContext();
        if (!ctx) {
            delete engine;
            throw std::runtime_error("[trtmc] Failed to create TRT execution context");
        }

        cudaStream_t stream = options.stream;
        std::shared_ptr<void> stream_owner;
        if (!stream) {
            auto owned = std::make_shared<CudaStream>();
            if (!owned->ok()) {
                delete ctx;
                delete engine;
                throw std::runtime_error("[trtmc] Failed to create CUDA stream");
            }
            stream = owned->get();
            stream_owner = owned;
        }

        auto module = std::make_unique<TrtModuleImpl>(
            engine, ctx, stream, 0, options.distributed_communicator, options.external_bindings);
        if (!module->ok()) {
            delete engine;
            throw std::runtime_error("[trtmc] TrtModuleImpl creation failed");
        }

        keep_backend_resources(*module,
                               std::shared_ptr<nvinfer1::ICudaEngine>(
                                   engine, [](nvinfer1::ICudaEngine* p) { delete p; }),
                               stream_owner, options.distributed_owner);

        return module;
    }

    BackendDualProfileModules
    create_dual_profile_modules(const void* plan_data, size_t plan_size,
                                const ModuleCreateOptions& options) override {
        auto* engine_raw = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine_raw)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (TRT)");
        std::shared_ptr<nvinfer1::ICudaEngine> engine(engine_raw,
                                                      [](nvinfer1::ICudaEngine* p) { delete p; });

        cudaStream_t stream = options.stream;
        std::shared_ptr<void> stream_owner;
        if (!stream) {
            auto owned = std::make_shared<CudaStream>();
            if (!owned->ok())
                throw std::runtime_error("[trtmc] Failed to create CUDA stream");
            stream = owned->get();
            stream_owner = owned;
        }

        const int32_t nprofiles = engine->getNbOptimizationProfiles();
        if (!options.external_bindings.empty() && nprofiles > 1) {
            throw std::invalid_argument(
                "[trtmc] One external binding set cannot be shared by multiple live TRT "
                "profile modules; use create_module or provide a per-profile binding API");
        }
        auto make_ctx_module = [&](int32_t profile_idx) -> std::unique_ptr<ITrtModule> {
            auto* ctx = engine->createExecutionContext();
            if (!ctx)
                throw std::runtime_error("[trtmc] Failed to create TRT execution context");
            auto mod = std::make_unique<TrtModuleImpl>(engine.get(), ctx, stream, profile_idx,
                                                       options.distributed_communicator,
                                                       options.external_bindings);
            if (!mod->ok())
                throw std::runtime_error("[trtmc] TrtModuleImpl creation failed");
            keep_backend_resources(*mod, engine, stream_owner, options.distributed_owner);
            return mod;
        };

        BackendDualProfileModules out;
        if (nprofiles < 2) {
            out.decode = make_ctx_module(0);
            return out;
        }
        out.prefill = make_ctx_module(0);
        out.decode = make_ctx_module(1);
        return out;
    }

    BackendProfileModules
    create_profile_modules(const void* plan_data, size_t plan_size,
                           const ModuleCreateOptions& options,
                           const std::vector<int32_t>& profile_indices) override {
        auto* engine_raw = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine_raw)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (TRT)");
        std::shared_ptr<nvinfer1::ICudaEngine> engine(engine_raw,
                                                      [](nvinfer1::ICudaEngine* p) { delete p; });

        cudaStream_t stream = options.stream;
        std::shared_ptr<void> stream_owner;
        if (!stream) {
            auto owned = std::make_shared<CudaStream>();
            if (!owned->ok())
                throw std::runtime_error("[trtmc] Failed to create CUDA stream");
            stream = owned->get();
            stream_owner = owned;
        }

        const int32_t nprofiles = engine->getNbOptimizationProfiles();
        int32_t requested_profile_count = 0;
        for (const int32_t profile_idx : profile_indices) {
            if (profile_idx >= 0 && profile_idx < nprofiles)
                ++requested_profile_count;
        }
        if (!options.external_bindings.empty() && requested_profile_count > 1) {
            throw std::invalid_argument(
                "[trtmc] One external binding set cannot be shared by multiple live TRT "
                "profile modules; use create_module or provide a per-profile binding API");
        }
        BackendProfileModules out;
        out.modules.reserve(profile_indices.size());
        for (int32_t profile_idx : profile_indices) {
            if (profile_idx < 0 || profile_idx >= nprofiles)
                continue;
            auto* ctx = engine->createExecutionContext();
            if (!ctx)
                throw std::runtime_error("[trtmc] Failed to create TRT execution context");
            auto mod = std::make_unique<TrtModuleImpl>(engine.get(), ctx, stream, profile_idx,
                                                       options.distributed_communicator,
                                                       options.external_bindings);
            if (!mod->ok())
                throw std::runtime_error("[trtmc] TrtModuleImpl creation failed");
            keep_backend_resources(*mod, engine, stream_owner, options.distributed_owner);
            out.modules.push_back(BackendProfileModule{profile_idx, std::move(mod)});
        }
        return out;
    }

    BackendContextModules
    create_context_modules(const void* plan_data, size_t plan_size,
                           const std::vector<ModuleCreateOptions>& options) override {
        if (options.empty())
            throw std::invalid_argument("[trtmc] Context module options must not be empty");
        auto* engine_raw = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine_raw)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (TRT)");
        std::shared_ptr<nvinfer1::ICudaEngine> engine(engine_raw,
                                                      [](nvinfer1::ICudaEngine* p) { delete p; });

        BackendContextModules out;
        out.modules.reserve(options.size());
        for (const auto& lane_options : options) {
            const int32_t profile_idx = lane_options.optimization_profile;
            if (profile_idx < 0 || profile_idx >= engine->getNbOptimizationProfiles())
                throw std::invalid_argument("[trtmc] Invalid optimization profile index");
            auto* ctx = engine->createExecutionContext();
            if (!ctx)
                throw std::runtime_error("[trtmc] Failed to create TRT execution context");

            cudaStream_t stream = lane_options.stream;
            std::shared_ptr<void> stream_owner;
            if (!stream) {
                auto owned = std::make_shared<CudaStream>();
                if (!owned->ok()) {
                    delete ctx;
                    throw std::runtime_error("[trtmc] Failed to create CUDA stream");
                }
                stream = owned->get();
                stream_owner = owned;
            }

            auto module = std::make_unique<TrtModuleImpl>(engine.get(), ctx, stream, profile_idx,
                                                          lane_options.distributed_communicator,
                                                          lane_options.external_bindings);
            if (!module->ok())
                throw std::runtime_error("[trtmc] TrtModuleImpl creation failed");
            keep_backend_resources(*module, engine, stream_owner, lane_options.distributed_owner);
            out.modules.push_back(std::move(module));
        }
        return out;
    }

    const char* name() const override { return "trt"; }

  private:
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
};

} // namespace trtmc

extern "C" trtmc::IBackend* trtmc_create_backend_v2() {
    try {
        return new trtmc::TrtBackend();
    } catch (const std::exception& e) {
        std::cerr << "[trtmc] TRT backend init failed: " << e.what() << std::endl;
        return nullptr;
    }
}

extern "C" std::uint32_t trtmc_backend_api_abi_version() {
    return trtmc::kTrtmcBackendApiAbiVersion;
}

extern "C" void trtmc_destroy_backend_v2(trtmc::IBackend* b) {
    delete b;
}

extern "C" const char* trtmc_backend_abi() {
    static const std::string abi = trtmc::trt_backend_abi_string();
    return abi.c_str();
}

extern "C" const char* trtmc_backend_runtime_version() {
    static const std::string version = trtmc::trt_runtime_version_string();
    return version.c_str();
}
