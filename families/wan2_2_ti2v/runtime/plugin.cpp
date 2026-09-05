/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wan2_2_ti2v/runtime/pipeline.h"
#include "families/wan2_2_ti2v/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace trtmc::wan22_factory {
namespace {

using PlanMap = std::unordered_map<std::string, std::vector<char>>;

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

PlanMap index_plans(const BundleReader& bundle) {
    constexpr std::array<const char*, 4> names = {"text_encoder.0.plan", "denoiser.plan",
                                                  "vae.plan", "vae.first_frame.plan"};
    PlanMap plans;
    for (const char* name : names) {
        plans.emplace(name, require_section(bundle, name));
    }
    return plans;
}

Wan22ModuleLoader make_loader(IBackend& backend, PlanMap plans) {
    return
        [&backend, plans = std::move(plans)](const std::string& section_name, cudaStream_t stream,
                                             const std::vector<ModuleExternalBinding>& bindings) {
            const auto found = plans.find(section_name);
            if (found == plans.end())
                throw std::runtime_error("Wan2.2 requested undeclared plan: " + section_name);
            const auto& plan = found->second;
            ModuleCreateOptions options{};
            options.stream = stream;
            if (bindings.empty())
                return backend.create_module(plan.data(), plan.size(), options);
            return backend.create_module_prebound(plan.data(), plan.size(), options, bindings);
        };
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleReader& bundle) {
    const auto& data = require_section(bundle, "tokenizer.json");
    auto tokenizer = CreateUnigramTokenizer(data.data(), data.size(), false);
    if (!tokenizer)
        throw std::runtime_error("Wan2.2 tokenizer.json is not its required Unigram tokenizer");
    return tokenizer;
}

} // namespace
} // namespace trtmc::wan22_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("wan2_2_ti2v does not support --kv-cache-size");
    const auto& runtime = trtmc::wan22_factory::require_section(context.reader, "runtime.json");
    const std::string runtime_text(runtime.begin(), runtime.end());
    return new trtmc::Wan22TI2VPipeline(
        trtmc::wan22_factory::make_loader(context.backend,
                                          trtmc::wan22_factory::index_plans(context.reader)),
        trtmc::wan22_factory::load_tokenizer(context.reader),
        trtmc::parse_wan22_options(runtime_text),
        trtmc::wan2_2_ti2v::parse_runtime_config(runtime_text), "");
}
