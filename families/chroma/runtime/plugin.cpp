/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/chroma/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>
#include <string>

namespace {

std::vector<char> require_section(const trtmc::BundleReader& bundle, const std::string& name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("Chroma bundle section is missing or empty: " + name);
    return bundle.read_section(name);
}

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("chroma does not support --kv-cache-size");
    if (context.reader.info().task != trtmc::internal::ITextToImage::kTask)
        throw std::invalid_argument("chroma bundles serve only text_to_image");
    const auto runtime = require_section(context.reader, "runtime.json");
    const auto tokenizer = require_section(context.reader, "tokenizer.json");
    auto& backend = context.backend;
    const auto& reader = context.reader;
    const trtmc::chroma::ModuleLoader loader = [&backend, &reader](const std::string& name) {
        const auto plan = require_section(reader, name);
        auto module = backend.create_module(plan.data(), plan.size(), trtmc::ModuleCreateOptions{});
        if (!module || !module->ok())
            throw std::runtime_error("Chroma could not deserialize " + name);
        module->set_timing_label("chroma." + name);
        return module;
    };
    return new trtmc::chroma::ChromaPipeline(
        loader, trtmc::chroma::T5Tokenizer(tokenizer.data(), tokenizer.size()),
        trtmc::chroma::parse_runtime_config(std::string_view(runtime.data(), runtime.size())));
}
