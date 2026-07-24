/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// TrtBackend: IBackend implementation for standard TensorRT.
// Compiled into libtrtmc_backend_trt.so. Links libnvinfer.so.

#include "trtmc/runtime/trt_backend.h"

#include "plugins/runtime_kv/runtime_kv_plugin_api.h"
#include "runtime/backend/prebound_backend.h"
#include "runtime/backend/runtime_memory_backend.h"
#include "runtime/backend/trt_logger.h"
#include "runtime/core/cuda_common.h"
#include "trt_module_impl.h"

#include <NvInfer.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

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

class TrtBackend final : public IBackend, public IPreboundBackend, public IRuntimeMemoryBackendV1 {
  public:
    TrtBackend() : runtime_(create_trt_runtime()) {
        // This is an intentional, real symbol reference to the common plugin
        // DSO. It prevents --as-needed from dropping the dependency, so the
        // static TensorRT plugin registrars run before engine deserialization.
        trtmc_runtime_kv_plugin_force_link();
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
                                                       options.distributed_communicator,
                                                       std::vector<ModuleExternalBinding>{});
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
                                                       std::vector<ModuleExternalBinding>{});
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
                                                          std::vector<ModuleExternalBinding>{});
            if (!module->ok())
                throw std::runtime_error("[trtmc] TrtModuleImpl creation failed");
            keep_backend_resources(*module, engine, stream_owner, lane_options.distributed_owner);
            out.modules.push_back(std::move(module));
        }
        return out;
    }

    std::unique_ptr<ITrtModule>
    create_module_runtime_memory(const void* plan_data, size_t plan_size,
                                 const ModuleCreateOptions& options,
                                 const RuntimeMemoryModuleOptionsV1& memory_options) override {
        validate_runtime_memory_options(memory_options);
        return create_module_impl(plan_data, plan_size, options, {}, true,
                                  memory_options.deferred_tensor_names,
                                  options.optimization_profile, memory_options.alias_pairs);
    }

    BackendProfileModules create_profile_modules_runtime_memory(
        const void* plan_data, size_t plan_size, const ModuleCreateOptions& options,
        const std::vector<int32_t>& profile_indices,
        const RuntimeMemoryModuleOptionsV1& memory_options) override {
        validate_runtime_memory_options(memory_options);
        if (profile_indices.empty())
            throw std::invalid_argument("[trtmc] Runtime-memory profile list must not be empty");

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
        BackendProfileModules out;
        out.modules.reserve(profile_indices.size());
        for (int32_t profile_idx : profile_indices) {
            if (profile_idx < 0 || profile_idx >= nprofiles)
                throw std::invalid_argument("[trtmc] Invalid runtime-memory profile index");
            auto* ctx = engine->createExecutionContext(
                nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
            if (!ctx)
                throw std::runtime_error(
                    "[trtmc] Failed to create USER_MANAGED TRT execution context");
            auto module = std::make_unique<RuntimeMemoryTrtModuleImpl>(
                engine.get(), ctx, stream, profile_idx, options.distributed_communicator,
                std::vector<ModuleExternalBinding>{}, true, memory_options.deferred_tensor_names,
                memory_options.alias_pairs);
            if (!module->ok())
                throw std::runtime_error("[trtmc] Runtime-memory TrtModuleImpl creation failed");
            keep_backend_resources(*module, engine, stream_owner, options.distributed_owner);
            out.modules.push_back(BackendProfileModule{profile_idx, std::move(module)});
        }
        return out;
    }

    RuntimeMemoryContextRequirementV1
    shared_context_memory_requirement(const std::vector<ITrtModule*>& modules) override {
        if (modules.empty())
            throw std::invalid_argument("[trtmc] Shared context list must not be empty");

        RuntimeMemoryContextRequirementV1 shared;
        bool have_device = false;
        for (auto* module : modules) {
            auto* runtime_module = dynamic_cast<IRuntimeMemoryModuleV1*>(module);
            if (!runtime_module)
                throw std::invalid_argument(
                    "[trtmc] Module does not implement runtime-memory API v1");
            const auto requirement = runtime_module->context_memory_requirement();
            if (!have_device) {
                shared.device = requirement.device;
                have_device = true;
            } else if (shared.device != requirement.device) {
                throw std::invalid_argument(
                    "[trtmc] Shared contexts must reside on the same CUDA device");
            }
            shared.capacity_bytes = std::max(shared.capacity_bytes, requirement.capacity_bytes);
            shared.alignment = std::max(shared.alignment, requirement.alignment);
        }
        return shared;
    }

    void bind_shared_context_memory(const std::vector<ITrtModule*>& modules,
                                    const RuntimeMemoryContextBlockV1& block) override {
        if (modules.empty())
            throw std::invalid_argument("[trtmc] Shared context list must not be empty");

        std::vector<IRuntimeMemoryModuleV1*> runtime_modules;
        runtime_modules.reserve(modules.size());
        std::size_t required_bytes = 0;
        int32_t required_device = -1;
        for (auto* module : modules) {
            auto* runtime_module = dynamic_cast<IRuntimeMemoryModuleV1*>(module);
            if (!runtime_module)
                throw std::invalid_argument(
                    "[trtmc] Module does not implement runtime-memory API v1");
            const auto requirement = runtime_module->context_memory_requirement();
            required_bytes = std::max(required_bytes, requirement.capacity_bytes);
            if (required_device < 0)
                required_device = requirement.device;
            else if (required_device != requirement.device)
                throw std::invalid_argument(
                    "[trtmc] Shared contexts must reside on the same CUDA device");
            runtime_modules.push_back(runtime_module);
        }
        if (block.capacity_bytes < required_bytes)
            throw std::invalid_argument("[trtmc] Shared context block is too small");
        if (block.device != required_device)
            throw std::invalid_argument("[trtmc] Shared context block is on the wrong CUDA device");

        for (auto* runtime_module : runtime_modules)
            runtime_module->bind_context_memory(block);
    }

    const char* name() const override { return "trt"; }

  private:
    static void
    validate_runtime_memory_options(const RuntimeMemoryModuleOptionsV1& memory_options) {
        if (memory_options.api_version != kRuntimeMemoryBackendApiVersionCurrent) {
            throw std::invalid_argument("[trtmc] Unsupported runtime-memory module api_version");
        }
        if (memory_options.struct_size < sizeof(RuntimeMemoryModuleOptionsV1)) {
            throw std::invalid_argument("[trtmc] Runtime-memory module options are truncated");
        }
        if (memory_options.deferred_tensor_names.empty() && memory_options.alias_pairs.empty()) {
            throw std::invalid_argument(
                "[trtmc] Runtime-memory tensor declarations must not be empty");
        }
        std::unordered_set<std::string> names;
        for (const auto& name : memory_options.deferred_tensor_names) {
            if (name.empty())
                throw std::invalid_argument(
                    "[trtmc] Runtime-memory deferred tensor name must not be empty");
            if (!names.insert(name).second)
                throw std::invalid_argument("[trtmc] Duplicate runtime-memory tensor '" + name +
                                            "'");
        }
        std::unordered_set<std::string> alias_endpoints;
        for (const auto& pair : memory_options.alias_pairs) {
            if (pair.api_version != kRuntimeMemoryBackendApiVersionCurrent ||
                pair.struct_size < sizeof(RuntimeMemoryAliasPairV1)) {
                throw std::invalid_argument("[trtmc] Invalid runtime-memory alias-pair descriptor");
            }
            if (pair.input_name.empty() || pair.output_name.empty() ||
                pair.input_name == pair.output_name) {
                throw std::invalid_argument(
                    "[trtmc] Runtime-memory alias pair requires distinct names");
            }
            if (!alias_endpoints.insert(pair.input_name).second ||
                !alias_endpoints.insert(pair.output_name).second) {
                throw std::invalid_argument(
                    "[trtmc] Runtime-memory tensor appears in multiple alias pairs");
            }
        }
    }

    std::unique_ptr<ITrtModule>
    create_module_impl(const void* plan_data, size_t plan_size, const ModuleCreateOptions& options,
                       const std::vector<ModuleExternalBinding>& external_bindings,
                       bool runtime_managed_context = false,
                       const std::vector<std::string>& deferred_runtime_tensors = {},
                       int32_t profile_idx = 0,
                       const std::vector<RuntimeMemoryAliasPairV1>& runtime_alias_pairs = {}) {
        auto* engine = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (TRT)");

        if (profile_idx < 0 || profile_idx >= engine->getNbOptimizationProfiles()) {
            delete engine;
            throw std::invalid_argument("[trtmc] Invalid optimization profile index");
        }
        auto* ctx = runtime_managed_context
                        ? engine->createExecutionContext(
                              nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED)
                        : engine->createExecutionContext();
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

        std::unique_ptr<TrtModuleImpl> module;
        if (runtime_managed_context) {
            module = std::make_unique<RuntimeMemoryTrtModuleImpl>(
                engine, ctx, stream, profile_idx, options.distributed_communicator,
                external_bindings, true, deferred_runtime_tensors, runtime_alias_pairs);
        } else {
            module = std::make_unique<TrtModuleImpl>(engine, ctx, stream, profile_idx,
                                                     options.distributed_communicator,
                                                     external_bindings);
        }
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

extern "C" std::int32_t
trtmc_backend_query_abi_contract_v2(trtmc::BackendDsoAbiContractV2* contract,
                                    std::size_t contract_size) noexcept {
    if (contract == nullptr || contract_size < sizeof(*contract))
        return -1;
    *contract = trtmc::make_runtime_memory_backend_dso_abi_contract_v2(
        trtmc::kBackendDsoCapabilityRuntimeMemoryV2);
    return 0;
}

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

extern "C" const char* trtmc_backend_abi() {
    static const std::string abi = trtmc::trt_backend_abi_string();
    return abi.c_str();
}

extern "C" const char* trtmc_backend_runtime_version() {
    static const std::string version = trtmc::trt_runtime_version_string();
    return version.c_str();
}

extern "C" const char* trtmc_backend_runtime_memory_stack_json_v1() {
    // This call resolves through the backend's DT_NEEDED edge to the exact
    // common plugin DSO that will register creators for deserialization.
    return trtmc_runtime_kv_plugin_runtime_stack_json_v1();
}
