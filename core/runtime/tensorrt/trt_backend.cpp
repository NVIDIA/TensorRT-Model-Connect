/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// TrtBackend: IBackend implementation for standard TensorRT.
// Compiled into libtrtmc_backend_trt.so. Links libnvinfer.so.

#include "trtmc/runtime/trt_backend.h"

#include "runtime/primitives/cuda_common.h"
#include "runtime/tensorrt/trt_logger.h"
#include "trt_module_impl.h"

#include <NvInfer.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>

namespace trtmc {

namespace {

// Shared by both contexts of a dual-profile module. Destruction order is
// explicit, including failed loading/deserialization; no vector-entry ordering
// in ITrtModule::keep_alive is relied upon.
struct PluginEngineOwner {
    std::vector<std::shared_ptr<void>> file_owners;
    TrtUniquePtr<nvinfer1::IRuntime> runtime;
    std::vector<nvinfer1::IPluginRegistry::PluginLibraryHandle> libraries;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;

    ~PluginEngineOwner() {
        engine.reset();
        if (runtime) {
            auto& registry = runtime->getPluginRegistry();
            for (auto handle = libraries.rbegin(); handle != libraries.rend(); ++handle)
                registry.deregisterLibrary(*handle);
        }
        libraries.clear();
        runtime.reset();
        file_owners.clear();
    }
};

void validate_plugin_libraries(const std::vector<ModulePluginLibrary>& libraries) {
    std::set<std::string> paths;
    for (const auto& library : libraries) {
        if (library.path.empty() || library.path.find('\0') != std::string::npos ||
            !std::filesystem::path(library.path).is_absolute())
            throw std::invalid_argument("[trtmc] Plugin library paths must be absolute");
        if (!paths.insert(library.path).second)
            throw std::invalid_argument("[trtmc] Duplicate plugin library path: " + library.path);
    }
}

std::shared_ptr<nvinfer1::ICudaEngine> deserialize_engine(nvinfer1::IRuntime& default_runtime,
                                                          const void* data, std::size_t bytes,
                                                          const ModuleCreateOptions& options) {
    if (options.plugin_libraries.empty()) {
        std::shared_ptr<nvinfer1::ICudaEngine> engine(
            default_runtime.deserializeCudaEngine(data, bytes));
        if (!engine)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (TRT)");
        return engine;
    }
    validate_plugin_libraries(options.plugin_libraries);
    auto owner = std::make_shared<PluginEngineOwner>();
    // Reserve before loading: an allocation failure cannot lose a library handle.
    owner->file_owners.reserve(options.plugin_libraries.size());
    owner->libraries.reserve(options.plugin_libraries.size());
    for (const auto& library : options.plugin_libraries)
        owner->file_owners.push_back(library.owner);
    owner->runtime = create_trt_runtime();
    if (!owner->runtime)
        throw std::runtime_error("[trtmc] Failed to create isolated TRT runtime");
    auto& registry = owner->runtime->getPluginRegistry();
    registry.setParentSearchEnabled(false);
    for (const auto& library : options.plugin_libraries) {
        auto handle = registry.loadLibrary(library.path.c_str());
        if (!handle)
            throw std::runtime_error("[trtmc] Failed to load TRT plugin library: " + library.path);
        owner->libraries.push_back(handle);
    }
    owner->engine.reset(owner->runtime->deserializeCudaEngine(data, bytes));
    if (!owner->engine)
        throw std::runtime_error("[trtmc] Failed to deserialize engine with native plugins");
    return std::shared_ptr<nvinfer1::ICudaEngine>(owner, owner->engine.get());
}

struct ModuleResources {
    std::shared_ptr<void> stream;
    std::shared_ptr<void> distributed;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    // Reverse member destruction keeps stream/communicator alive through engine
    // and its complete plugin/runtime/file-owner chain.
};

void keep_backend_resources(ITrtModule& module,
                            const std::shared_ptr<nvinfer1::ICudaEngine>& engine,
                            const std::shared_ptr<void>& stream_owner,
                            const std::shared_ptr<void>& distributed_owner) {
    module.keep_alive(std::make_shared<ModuleResources>(
        ModuleResources{stream_owner, distributed_owner, engine}));
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
        return create_module_impl(plan_data, plan_size, options, {});
    }

    std::unique_ptr<ITrtModule>
    create_module_prebound(const void* plan_data, size_t plan_size,
                           const ModuleCreateOptions& options,
                           const std::vector<ModuleExternalBinding>& external_bindings) override {
        if (external_bindings.empty())
            throw std::invalid_argument("[trtmc] External I/O prebindings must not be empty");
        return create_module_impl(plan_data, plan_size, options, external_bindings);
    }

    BackendDualProfileModules
    create_dual_profile_modules(const void* plan_data, size_t plan_size,
                                const ModuleCreateOptions& options) override {
        auto engine = deserialize_engine(*runtime_, plan_data, plan_size, options);
        if (options.cuda_graphs)
            TrtModuleImpl::validate_automatic_cuda_graph_engine(engine.get());

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
        auto make_ctx_module = [&](int32_t profile_idx) -> std::unique_ptr<ITrtModule> {
            TrtUniquePtr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext());
            if (!ctx)
                throw std::runtime_error("[trtmc] Failed to create TRT execution context");
            auto mod = std::make_unique<TrtModuleImpl>(
                engine.get(), std::move(ctx), stream, profile_idx, options.distributed_communicator,
                std::vector<ModuleExternalBinding>{}, false, !options.plugin_libraries.empty(),
                options.collect_timing);
            if (!mod->ok())
                throw std::runtime_error("[trtmc] TrtModuleImpl creation failed");
            if (options.cuda_graphs)
                mod->enable_automatic_cuda_graph();
            keep_backend_resources(*mod, engine, stream_owner, options.distributed_owner);
            return mod;
        };

        BackendDualProfileModules out;
        if (nprofiles < 2)
            throw std::runtime_error("[trtmc] Dual-profile engine requires two profiles");
        out.prefill = make_ctx_module(0);
        out.decode = make_ctx_module(1);
        return out;
    }

    const char* name() const override { return "trt"; }

  private:
    std::unique_ptr<ITrtModule>
    create_module_impl(const void* plan_data, size_t plan_size, const ModuleCreateOptions& options,
                       const std::vector<ModuleExternalBinding>& external_bindings) {
        auto engine = deserialize_engine(*runtime_, plan_data, plan_size, options);
        if (options.cuda_graphs)
            TrtModuleImpl::validate_automatic_cuda_graph_engine(engine.get());
        TrtUniquePtr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext());
        if (!ctx)
            throw std::runtime_error("[trtmc] Failed to create TRT execution context");

        cudaStream_t stream = options.stream;
        std::shared_ptr<void> stream_owner;
        if (!stream) {
            auto owned = std::make_shared<CudaStream>();
            if (!owned->ok())
                throw std::runtime_error("[trtmc] Failed to create CUDA stream");
            stream = owned->get();
            stream_owner = owned;
        }

        auto module = std::make_unique<TrtModuleImpl>(
            engine.get(), std::move(ctx), stream, 0, options.distributed_communicator,
            external_bindings, false, !options.plugin_libraries.empty(), options.collect_timing);
        if (!module->ok())
            throw std::runtime_error("[trtmc] TrtModuleImpl creation failed");
        if (options.cuda_graphs)
            module->enable_automatic_cuda_graph();

        keep_backend_resources(*module, engine, stream_owner, options.distributed_owner);

        return module;
    }
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
};

} // namespace trtmc

extern "C" trtmc::IBackend* trtmc_create_backend() {
    try {
        return new trtmc::TrtBackend();
    } catch (const std::exception& e) {
        std::cerr << "[trtmc] TRT backend init failed: " << e.what() << std::endl;
        return nullptr;
    }
}

extern "C" void trtmc_destroy_backend(trtmc::IBackend* b) {
    delete b;
}
