/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmfold2/runtime/bundle_config.h"
#include "families/esmfold2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

std::vector<char> requireSection(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::invalid_argument("ESMFold2 bundle is missing nonempty section: " +
                                    std::string(name));
    return bundle.read_section(name);
}

class ModuleLoader {
  public:
    explicit ModuleLoader(const FamilyContext& context) : context_(context) {
        const char* backend = context.backend.name();
        if (backend == nullptr || std::string(backend) != "trt")
            throw std::invalid_argument("ESMFold2 requires the standard TensorRT backend");
    }

    std::unique_ptr<ITrtModule> load(std::string_view name) {
        const std::string section = "esmfold2_" + std::string(name) + "_plan";
        auto plan = requireSection(context_.reader, section);
        ModuleCreateOptions options;
        options.stream = stream_;
        auto module = context_.backend.create_module(plan.data(), plan.size(), options);
        if (module == nullptr || !module->ok())
            throw std::runtime_error("ESMFold2 failed to load TensorRT section: " + section);
        if (stream_ == nullptr)
            stream_ = module->stream();
        if (module->stream() == nullptr || module->stream() != stream_)
            throw std::runtime_error("ESMFold2 plans must share one CUDA stream");
        module->set_timing_label("esmfold2 " + std::string(name));
        return module;
    }

  private:
    const FamilyContext& context_;
    cudaStream_t stream_{nullptr};
};

} // namespace
} // namespace trtmc

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("esmfold2 does not support --kv-cache-size");
    if (context.reader.info().task != trtmc::IStructurePrediction::kTask)
        throw std::invalid_argument("ESMFold2 requires task=structure_prediction");
    const auto runtime = trtmc::requireSection(context.reader, "runtime.json");
    auto config =
        trtmc::esmfold2::parse_bundle_config(std::string_view(runtime.data(), runtime.size()));
    trtmc::ModuleLoader loader(context);
    trtmc::esmfold2::EngineSet engines;
    engines.esmc = loader.load("esmc");
    engines.embed = loader.load("embed");
    if (config.msa_encoder)
        engines.msa = loader.load("msa");
    else if (context.reader.find_section("esmfold2_msa_plan") != nullptr)
        throw std::invalid_argument(
            "ESMFold2 bundle carries an MSA plan that runtime.json disables");
    engines.loop = loader.load("loop");
    engines.decode = loader.load("decode");
    engines.conditioning = loader.load("conditioning");
    engines.denoise = loader.load("denoise");
    engines.confidence = loader.load("confidence");
    return new trtmc::esmfold2::EsmFold2Pipeline(std::move(engines), std::move(config));
}
