/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// RtxBackend: IBackend implementation for TensorRT-RTX.
// Compiled into libtrtmc_backend_trt_rtx.so. Links libtensorrt_rtx.so.

#include "runtime/primitives/cuda_common.h"
#include "runtime/tensorrt/trt_logger.h"
#include "trt_module_impl.h"
#include "trtmc/runtime/trt_backend.h"

#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

struct StreamSetup {
    cudaStream_t stream{nullptr};
    std::shared_ptr<void> owner;
};

StreamSetup resolve_stream(cudaStream_t requested_stream) {
    if (requested_stream)
        return {requested_stream, {}};

    auto owned = std::make_shared<CudaStream>();
    if (!owned->ok())
        throw std::runtime_error("[trtmc] Failed to create CUDA stream");
    return {owned->get(), owned};
}

std::shared_ptr<nvinfer1::ICudaEngine> deserialize_engine(nvinfer1::IRuntime& runtime,
                                                          const void* plan_data, size_t plan_size) {
    auto* engine = runtime.deserializeCudaEngine(plan_data, plan_size);
    if (!engine)
        throw std::runtime_error("[trtmc] Failed to deserialize engine (RTX)");
    return {engine, [](nvinfer1::ICudaEngine* value) { delete value; }};
}

class RuntimeCacheState {
  public:
    RuntimeCacheState(nvinfer1::IRuntimeConfig& config, std::string path)
        : cache_(config.createRuntimeCache()), path_(std::move(path)) {
        if (cache_ == nullptr)
            throw std::runtime_error("[trtmc] Failed to create TensorRT-RTX runtime cache");

        std::ifstream input(path_, std::ios::binary | std::ios::ate);
        if (!input)
            return;
        const auto end = input.tellg();
        if (end <= 0)
            return;
        std::vector<char> bytes(static_cast<std::size_t>(end));
        input.seekg(0);
        input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!input)
            throw std::runtime_error("[trtmc] Failed to read TensorRT-RTX runtime cache");
        if (!cache_->deserialize(bytes.data(), bytes.size()))
            throw std::runtime_error("[trtmc] TensorRT-RTX rejected the runtime cache");
    }

    RuntimeCacheState(const RuntimeCacheState&) = delete;
    RuntimeCacheState& operator=(const RuntimeCacheState&) = delete;

    ~RuntimeCacheState() {
        auto* serialized = cache_ != nullptr ? cache_->serialize() : nullptr;
        if (serialized != nullptr && serialized->size() > 0) {
            std::ofstream output(path_, std::ios::binary | std::ios::trunc);
            if (output) {
                output.write(static_cast<const char*>(serialized->data()),
                             static_cast<std::streamsize>(serialized->size()));
            } else {
                std::cerr << "[trtmc] Failed to write TensorRT-RTX runtime cache: " << path_
                          << '\n';
            }
        }
        delete serialized;
        delete cache_;
    }

    nvinfer1::IRuntimeCache& cache() const { return *cache_; }
    const std::string& path() const { return path_; }

  private:
    nvinfer1::IRuntimeCache* cache_{nullptr};
    std::string path_;
};

void keep_backend_resources(ITrtModule& module,
                            const std::shared_ptr<nvinfer1::ICudaEngine>& engine,
                            const StreamSetup& stream_setup,
                            const std::shared_ptr<void>& distributed_owner) {
    module.keep_alive(engine);
    if (stream_setup.owner)
        module.keep_alive(stream_setup.owner);
    if (distributed_owner)
        module.keep_alive(distributed_owner);
}

} // namespace

class RtxBackend final : public IBackend {
  public:
    RtxBackend() : runtime_(create_trt_runtime()) {
        if (!runtime_)
            throw std::runtime_error("[trtmc] Failed to create TRT-RTX runtime");
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
        auto engine = deserialize_engine(*runtime_, plan_data, plan_size);
        const auto stream_setup = resolve_stream(options.stream);
        if (engine->getNbOptimizationProfiles() < 2)
            throw std::runtime_error("[trtmc] Dual-profile engine requires two profiles");

        BackendDualProfileModules modules;
        modules.prefill = create_context_module(engine, stream_setup, options, 0, {});
        modules.decode = create_context_module(engine, stream_setup, options, 1, {});
        return modules;
    }

    const char* name() const override { return "trt_rtx"; }

  private:
    std::unique_ptr<ITrtModule>
    create_module_impl(const void* plan_data, size_t plan_size, const ModuleCreateOptions& options,
                       const std::vector<ModuleExternalBinding>& external_bindings) {
        auto engine = deserialize_engine(*runtime_, plan_data, plan_size);
        return create_context_module(engine, resolve_stream(options.stream), options, 0,
                                     external_bindings);
    }

    std::unique_ptr<ITrtModule>
    create_context_module(const std::shared_ptr<nvinfer1::ICudaEngine>& engine,
                          const StreamSetup& stream_setup, const ModuleCreateOptions& options,
                          int32_t profile_idx,
                          const std::vector<ModuleExternalBinding>& external_bindings) {
        TrtUniquePtr<nvinfer1::IRuntimeConfig> runtime_config(engine->createRuntimeConfig());
        if (!runtime_config)
            throw std::runtime_error("[trtmc] Failed to create RTX runtime config");

        auto runtime_cache = resolve_runtime_cache(*runtime_config, options.runtime_cache_path);
        if (runtime_cache && !runtime_config->setRuntimeCache(runtime_cache->cache()))
            throw std::runtime_error("[trtmc] TensorRT-RTX rejected the runtime cache");
        if (options.cuda_graphs) {
            if (!runtime_config->setCudaGraphStrategy(
                    nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE)) {
                throw std::runtime_error("[trtmc] TensorRT-RTX rejected whole-graph capture");
            }
        }

        TrtUniquePtr<nvinfer1::IExecutionContext> context(
            engine->createExecutionContext(runtime_config.get()));
        if (!context)
            throw std::runtime_error("[trtmc] Failed to create RTX execution context");

        auto module = std::make_unique<TrtModuleImpl>(
            engine.get(), context.release(), stream_setup.stream, profile_idx,
            options.distributed_communicator, external_bindings, true);
        if (!module->ok())
            throw std::runtime_error("[trtmc] TrtModuleImpl creation failed (RTX)");

        keep_backend_resources(*module, engine, stream_setup, options.distributed_owner);
        if (runtime_cache)
            module->keep_alive(runtime_cache);
        return module;
    }

    std::shared_ptr<RuntimeCacheState> resolve_runtime_cache(nvinfer1::IRuntimeConfig& config,
                                                             const char* raw_path) {
        if (raw_path == nullptr || *raw_path == '\0')
            return {};
        const std::string path(raw_path);
        std::lock_guard<std::mutex> lock(runtime_cache_mutex_);
        if (auto existing = runtime_cache_.lock()) {
            if (existing->path() != path) {
                throw std::invalid_argument(
                    "one TensorRT-RTX backend cannot use two runtime cache paths concurrently");
            }
            return existing;
        }
        auto created = std::make_shared<RuntimeCacheState>(config, path);
        runtime_cache_ = created;
        return created;
    }

    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
    std::mutex runtime_cache_mutex_;
    std::weak_ptr<RuntimeCacheState> runtime_cache_;
};

} // namespace trtmc

extern "C" trtmc::IBackend* trtmc_create_backend() {
    try {
        return new trtmc::RtxBackend();
    } catch (const std::exception& error) {
        std::cerr << "[trtmc] RTX backend init failed: " << error.what() << std::endl;
        return nullptr;
    }
}

extern "C" void trtmc_destroy_backend(trtmc::IBackend* backend) {
    delete backend;
}
