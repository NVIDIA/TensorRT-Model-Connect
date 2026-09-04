/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam2/runtime/sam2_pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::sam2 {
namespace {

NativePlanModuleFactory makeModuleFactory(IBackend& backend) {
    struct State {
        explicit State(IBackend& implementation) : backend(implementation) {}
        IBackend& backend;
        cudaStream_t stream{nullptr};
    };
    auto state = std::make_shared<State>(backend);
    return [state](std::string_view section, const void* plan_data,
                   std::size_t plan_size) -> std::unique_ptr<ITrtModule> {
        ModuleCreateOptions options{};
        options.stream = state->stream;
        auto module = state->backend.create_module(plan_data, plan_size, options);
        if (!module || !module->ok())
            throw std::runtime_error("SAM2 failed to load " + std::string(section));
        if (state->stream == nullptr)
            state->stream = module->stream();
        else if (module->stream() != state->stream)
            throw std::runtime_error("SAM2 modules do not share one CUDA stream");
        return module;
    };
}

} // namespace
} // namespace trtmc::sam2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    auto engines = trtmc::sam2::makeNativeVideoEngineSet(
        context.reader, trtmc::sam2::makeModuleFactory(context.backend));
    auto processor = std::make_unique<trtmc::sam2::NativeVideoProcessor>(std::move(engines));
    return new trtmc::sam2::Sam2Pipeline(std::move(processor));
}
