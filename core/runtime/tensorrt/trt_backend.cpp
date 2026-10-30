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
#include <iostream>
#include <memory>
#include <stdexcept>

namespace trtmc {

namespace {

nvinfer1::DataType to_trt_dtype(DType dtype) {
    switch (dtype) {
    case DType::kFloat32: return nvinfer1::DataType::kFLOAT;
    case DType::kFloat16: return nvinfer1::DataType::kHALF;
    case DType::kBFloat16: return nvinfer1::DataType::kBF16;
    case DType::kInt32: return nvinfer1::DataType::kINT32;
    case DType::kInt8: return nvinfer1::DataType::kINT8;
    }
    throw std::runtime_error("[trtmc] refit: unsupported weight dtype");
}

// Supply weights to an engine built with kSTRIP_PLAN. Must run before any
// execution context is created: a placeholder engine produces garbage until
// refit, and ensureSessionWeightsFullyBacked fires inside CUDA-graph capture
// while the session weights are unbacked.
void apply_refit_weights(nvinfer1::ICudaEngine& engine,
                         const ModuleCreateOptions& options) {
    if (options.refit_weights == nullptr || options.refit_weights->empty())
        return;

    std::unique_ptr<nvinfer1::IRefitter> refitter(
        nvinfer1::createInferRefitter(engine, trt_shared_logger()));
    if (!refitter)
        throw std::runtime_error(
            "[trtmc] refit: engine is not refittable (was it built with "
            "kSTRIP_PLAN + kREFIT_INDIVIDUAL?)");

    for (const auto& [name, view] : *options.refit_weights) {
        nvinfer1::Weights weights{to_trt_dtype(view.dtype), view.data,
                                  static_cast<int64_t>(view.count)};
        if (!refitter->setNamedWeights(name.c_str(), weights))
            throw std::runtime_error("[trtmc] refit: setNamedWeights failed for " + name);
    }

    // getMissingWeights() is empty on a fresh stripped engine even before any
    // weight is supplied, so it is only meaningful as a post-set check.
    const int32_t missing = refitter->getMissingWeights(0, nullptr);
    if (missing > 0)
        throw std::runtime_error("[trtmc] refit: " + std::to_string(missing) +
                                 " weight(s) still missing after setNamedWeights");
    if (!refitter->refitCudaEngine())
        throw std::runtime_error("[trtmc] refit: refitCudaEngine() failed");
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
        auto* engine_raw = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine_raw)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (TRT)");
        apply_refit_weights(*engine_raw, options);
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
        auto make_ctx_module = [&](int32_t profile_idx) -> std::unique_ptr<ITrtModule> {
            auto* ctx = engine->createExecutionContext();
            if (!ctx)
                throw std::runtime_error("[trtmc] Failed to create TRT execution context");
            auto mod = std::make_unique<TrtModuleImpl>(engine.get(), ctx, stream, profile_idx,
                                                       options.distributed_communicator);
            if (!mod->ok())
                throw std::runtime_error("[trtmc] TrtModuleImpl creation failed");
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
        auto* engine = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (TRT)");

        try {
            apply_refit_weights(*engine, options);
        } catch (...) {
            delete engine;
            throw;
        }

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
            engine, ctx, stream, 0, options.distributed_communicator, external_bindings);
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
