/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/videomae/runtime/bundle_config.h"
#include "families/videomae/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trtmc::videomae {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

} // namespace
} // namespace trtmc::videomae

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc::videomae;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("videomae does not support --kv-cache-size");
    const auto text = require_section(context.reader, "runtime.json");
    const auto config = parse_runtime_config(std::string_view(text.data(), text.size()));
    const auto plan = require_section(context.reader, "engine.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("videomae engine failed to load");
    return new MaskedVideoReconstruction(std::move(engine), config.geometry, config.normalization);
}
