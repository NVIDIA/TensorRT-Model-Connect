/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_pipeline.h"
#include "trtmc/models/sam2_video.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace trtmc {
void register_sam2_plugin(PipelineRegistry& registry);
}

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

trtmc::BundleFile valid_bundle() {
    trtmc::BundleFile bundle;
    bundle.info.family = "sam2";
    bundle.info.runtime_strategy = "sam2_bbox_video_tracking";
    for (const auto name : trtmc::sam2::kRequiredPlanSections)
        bundle.sections.push_back({std::string(name), {'p'}});
    bundle.sections.push_back({std::string(trtmc::sam2::kConfigSection), {'{', '}'}});
    return bundle;
}

void test_materialized_bundle_fails_before_deserialization() {
    int factory_calls = 0;
    trtmc::sam2::NativePlanModuleFactory factory =
        [&](std::string_view, const void*, std::size_t) -> std::unique_ptr<trtmc::ITrtModule> {
        ++factory_calls;
        return nullptr;
    };

    auto bundle = valid_bundle();
    try {
        (void)trtmc::sam2::makeNativeVideoEngineSet(bundle, factory);
    } catch (const std::runtime_error&) {
    }
    check(factory_calls == 1, "valid bundle must reach TensorRT deserialization");

    for (int mutation = 0; mutation < 5; ++mutation) {
        bundle = valid_bundle();
        if (mutation == 0)
            bundle.info.family = "other";
        else if (mutation == 1)
            bundle.sections.front().data.clear();
        else if (mutation == 2)
            bundle.sections.push_back({"unexpected", {'x'}});
        else if (mutation == 3)
            bundle.sections[5].data.clear();
        else
            bundle.sections[4].name = bundle.sections[5].name;
        factory_calls = 0;
        try {
            (void)trtmc::sam2::makeNativeVideoEngineSet(bundle, factory);
        } catch (const std::exception&) {
        }
        check(factory_calls == 0, "invalid bundle must fail before TensorRT deserialization");
    }
}

} // namespace

int main() {
    auto& registry = trtmc::PipelineRegistry::instance();
    trtmc::register_sam2_plugin(registry);
    check(registry.lookup("sam2_bbox_video_tracking") != nullptr,
          "SAM2 plugin must register its runtime strategy");

    auto* session = trtmc_sam2_video_create_from_bundle_v1(nullptr, "plugin", "backend");
    check(session == nullptr, "SAM2 C API must reject missing paths");
    check(std::string(trtmc_sam2_video_last_error()).find("paths are required") !=
              std::string::npos,
          "SAM2 C API must report its path contract");

    session = trtmc_sam2_video_create_from_bundle_v1("/missing-user-built.bundle", ".", ".");
    check(session == nullptr, "SAM2 C API must enter the bundle loader");
    check(std::string(trtmc_sam2_video_last_error()).find("Failed to open bundle file") !=
              std::string::npos,
          "SAM2 C API must preserve the loader diagnostic");
    test_materialized_bundle_fails_before_deserialization();
    return 0;
}
