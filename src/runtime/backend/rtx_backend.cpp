/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// RtxBackend: IBackend implementation for TensorRT-RTX.
// Compiled into libtrtmc_backend_rtx.so. Links libtensorrt_rtx.so.
//
// Uses the RTX-specific NvInfer.h headers which declare IRuntimeCache,
// CudaGraphStrategy, and DynamicShapesKernelSpecializationStrategy.

#include "runtime/backend/trt_logger.h"
#include "runtime/core/cuda_common.h"
#include "trt_module_impl.h"
#include "trtmc/runtime/trt_backend.h"

#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

struct StreamSetup {
    cudaStream_t stream{nullptr};
    std::shared_ptr<void> owner;
};

StreamSetup resolve_stream(cudaStream_t requested_stream) {
    if (requested_stream) {
        return StreamSetup{requested_stream, {}};
    }

    auto owned = std::make_shared<CudaStream>();
    if (!owned->ok()) {
        throw std::runtime_error("[trtmc] Failed to create CUDA stream");
    }

    return StreamSetup{owned->get(), owned};
}

} // namespace

class RtxBackend final : public IBackend {
  public:
    RtxBackend() : runtime_(create_trt_runtime()) {
        if (!runtime_)
            throw std::runtime_error("[trtmc] Failed to create TRT-RTX runtime");
    }

    ~RtxBackend() override {
        flush_runtime_cache();
        delete runtime_cache_;
    }

    std::unique_ptr<ITrtModule> create_module(const void* plan_data, size_t plan_size,
                                              const ModuleCreateOptions& options) override {
        auto* engine = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (RTX)");

        const int32_t profile_idx = options.optimization_profile;
        if (profile_idx < 0 || profile_idx >= engine->getNbOptimizationProfiles()) {
            delete engine;
            throw std::invalid_argument("[trtmc] Invalid optimization profile index");
        }

        // Create IRuntimeConfig with RTX-specific features
        auto* rt_config = engine->createRuntimeConfig();
        if (!rt_config) {
            delete engine;
            throw std::runtime_error("[trtmc] Failed to create RTX runtime config");
        }

        // JIT kernel cache
        if (options.runtime_cache_path && options.runtime_cache_path[0] != '\0') {
            ensure_runtime_cache(rt_config, options.runtime_cache_path);
        }

        // CUDA graph capture
        if (options.cuda_graphs) {
            rt_config->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
            std::cerr << "[trtmc] CUDA graphs enabled (whole-graph capture)\n";
        }

        auto* ctx = engine->createExecutionContext(rt_config);
        delete rt_config;
        if (!ctx) {
            delete engine;
            throw std::runtime_error("[trtmc] Failed to create RTX execution context");
        }

        StreamSetup stream_setup;
        try {
            stream_setup = resolve_stream(options.stream);
        } catch (...) {
            delete ctx;
            delete engine;
            throw;
        }

        auto module =
            std::make_unique<TrtModuleImpl>(engine, ctx, stream_setup.stream, profile_idx);
        if (!module->ok()) {
            delete engine;
            throw std::runtime_error("[trtmc] TrtModuleImpl creation failed (RTX)");
        }

        module->keep_alive(std::shared_ptr<nvinfer1::ICudaEngine>(
            engine, [](nvinfer1::ICudaEngine* p) { delete p; }));
        if (stream_setup.owner)
            module->keep_alive(stream_setup.owner);

        return module;
    }

    BackendDualProfileModules
    create_dual_profile_modules(const void* plan_data, size_t plan_size,
                                const ModuleCreateOptions& options) override {
        auto* engine_raw = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine_raw)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (RTX)");
        std::shared_ptr<nvinfer1::ICudaEngine> engine(engine_raw,
                                                      [](nvinfer1::ICudaEngine* p) { delete p; });

        StreamSetup stream_setup = resolve_stream(options.stream);

        const int32_t nprofiles = engine->getNbOptimizationProfiles();
        auto make_ctx_module = [&](int32_t profile_idx) -> std::unique_ptr<ITrtModule> {
            return create_profile_module(engine, stream_setup, options, profile_idx);
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
            throw std::runtime_error("[trtmc] Failed to deserialize engine (RTX)");
        std::shared_ptr<nvinfer1::ICudaEngine> engine(engine_raw,
                                                      [](nvinfer1::ICudaEngine* p) { delete p; });

        StreamSetup stream_setup = resolve_stream(options.stream);
        const int32_t nprofiles = engine->getNbOptimizationProfiles();
        BackendProfileModules out;
        out.modules.reserve(profile_indices.size());
        for (int32_t profile_idx : profile_indices) {
            if (profile_idx < 0 || profile_idx >= nprofiles)
                continue;
            out.modules.push_back(BackendProfileModule{
                profile_idx, create_profile_module(engine, stream_setup, options, profile_idx)});
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
            throw std::runtime_error("[trtmc] Failed to deserialize engine (RTX)");
        std::shared_ptr<nvinfer1::ICudaEngine> engine(engine_raw,
                                                      [](nvinfer1::ICudaEngine* p) { delete p; });

        BackendContextModules out;
        out.modules.reserve(options.size());
        for (const auto& lane_options : options) {
            const int32_t profile_idx = lane_options.optimization_profile;
            if (profile_idx < 0 || profile_idx >= engine->getNbOptimizationProfiles())
                throw std::invalid_argument("[trtmc] Invalid optimization profile index");
            StreamSetup stream_setup = resolve_stream(lane_options.stream);
            out.modules.push_back(
                create_profile_module(engine, stream_setup, lane_options, profile_idx));
        }
        return out;
    }

    const char* name() const override { return "trt_rtx"; }

  private:
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
    nvinfer1::IRuntimeCache* runtime_cache_{nullptr};
    std::string cache_path_;

    std::unique_ptr<ITrtModule>
    create_profile_module(const std::shared_ptr<nvinfer1::ICudaEngine>& engine,
                          const StreamSetup& stream_setup, const ModuleCreateOptions& options,
                          int32_t profile_idx) {
        auto* rt_config = engine->createRuntimeConfig();
        if (!rt_config)
            throw std::runtime_error("[trtmc] Failed to create RTX runtime config");
        if (options.runtime_cache_path && options.runtime_cache_path[0] != '\0')
            ensure_runtime_cache(rt_config, options.runtime_cache_path);
        if (options.cuda_graphs)
            rt_config->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);

        auto* ctx = engine->createExecutionContext(rt_config);
        delete rt_config;
        if (!ctx)
            throw std::runtime_error("[trtmc] Failed to create RTX execution context");

        auto mod =
            std::make_unique<TrtModuleImpl>(engine.get(), ctx, stream_setup.stream, profile_idx);
        if (!mod->ok())
            throw std::runtime_error("[trtmc] TrtModuleImpl creation failed (RTX)");
        mod->keep_alive(engine);
        if (stream_setup.owner)
            mod->keep_alive(stream_setup.owner);
        return mod;
    }

    void ensure_runtime_cache(nvinfer1::IRuntimeConfig* cfg, const char* path) {
        if (!runtime_cache_) {
            runtime_cache_ = cfg->createRuntimeCache();
            cache_path_ = path;
            std::ifstream ifs(path, std::ios::binary | std::ios::ate);
            if (ifs) {
                auto sz = ifs.tellg();
                if (sz > 0) {
                    std::vector<char> buf(static_cast<size_t>(sz));
                    ifs.seekg(0);
                    ifs.read(buf.data(), sz);
                    runtime_cache_->deserialize(buf.data(), buf.size());
                    std::cerr << "[trtmc] RTX runtime cache loaded: " << path << " (" << sz
                              << " bytes)\n";
                }
            }
        }
        cfg->setRuntimeCache(*runtime_cache_);
    }

    void flush_runtime_cache() {
        if (!runtime_cache_ || cache_path_.empty())
            return;
        auto* mem = runtime_cache_->serialize();
        if (mem && mem->size() > 0) {
            std::ofstream ofs(cache_path_, std::ios::binary | std::ios::trunc);
            if (ofs) {
                ofs.write(static_cast<const char*>(mem->data()),
                          static_cast<std::streamsize>(mem->size()));
                std::cerr << "[trtmc] RTX runtime cache saved: " << cache_path_ << " ("
                          << mem->size() << " bytes)\n";
            }
            delete mem;
        }
    }
};

} // namespace trtmc

extern "C" trtmc::IBackend* trtmc_create_backend() {
    try {
        return new trtmc::RtxBackend();
    } catch (const std::exception& e) {
        std::cerr << "[trtmc] RTX backend init failed: " << e.what() << std::endl;
        return nullptr;
    }
}

extern "C" void trtmc_destroy_backend(trtmc::IBackend* b) {
    delete b;
}
