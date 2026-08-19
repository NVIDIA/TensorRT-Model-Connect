/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_helpers.h"
#include "kv_cache.h"
#include "pipeline.h"
#include "runtime/backend/trt_module_impl.h"
#include "speech_config.h"
#include "support/mock_trt_engines.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_speech_pipeline_construction() {
    const std::vector<float> step_logits = {0.1F, 0.9F, 0.0F};
    auto temporal_engine = trtmc::test::build_mock_step_engine(9, 3, step_logits);
    if (!temporal_engine) {
        std::cerr << "WARNING: Could not build temporal engine for SpeechPipeline, skipping\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto temporal = std::make_unique<trtmc::TrtModuleImpl>(
        temporal_engine.get(), temporal_engine->createExecutionContext(), stream);
    auto temporal_cache = std::make_unique<trtmc::PersonaplexKvCache>(0, 8, 0, stream);

    check(temporal->ok(), "speech temporal module ok");
    check(temporal_cache->ok(), "speech temporal cache ok");

    trtmc::SpeechConfig cfg;
    trtmc::SpeechPipeline pipeline(nullptr, std::move(temporal), std::move(temporal_cache), {},
                                   nullptr, nullptr, cfg, stream, nullptr, "test-speech");

    check(std::string(pipeline.pipeline_type()) == "SpeechPipeline",
          "SpeechPipeline: pipeline_type");
    check(std::string(pipeline.model_id()) == "test-speech", "SpeechPipeline: model_id");

    cudaStreamDestroy(stream);
}

void test_speech_validates_temporal() {
    bool threw = false;
    try {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        trtmc::SpeechConfig cfg;
        trtmc::SpeechPipeline p(nullptr, nullptr, nullptr, {}, nullptr, nullptr, cfg, stream,
                                nullptr, "x");
        check(false, "null temporal should throw");
        cudaStreamDestroy(stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "speech: null temporal throws");
}

void test_depth_engines_are_ordered_by_numeric_codebook() {
    trtmc::BundleFile bundle;
    const std::vector<int> insertion_order = {10, 2, 0, 7, 1, 9, 5, 3, 8, 4, 6};
    for (const int codebook : insertion_order) {
        trtmc::BundleSection section;
        section.name = "depth_engine_plan_" + std::to_string(codebook);
        const std::string contents = "plan-" + std::to_string(codebook);
        section.data.assign(contents.begin(), contents.end());
        bundle.sections.push_back(std::move(section));
    }

    const auto plans = trtmc::find_depth_engine_plans_in_codebook_order(bundle);
    check(plans.size() == insertion_order.size(), "all depth engine plans found");
    for (std::size_t codebook = 0; codebook < plans.size(); ++codebook) {
        const std::string contents(plans[codebook]->begin(), plans[codebook]->end());
        check(contents == "plan-" + std::to_string(codebook),
              "depth engine plan uses numeric codebook order");
    }
}

} // namespace

int main() {
    test_speech_pipeline_construction();
    test_speech_validates_temporal();
    test_depth_engines_are_ordered_by_numeric_codebook();
    if (failures > 0) {
        std::cerr << failures << " speech pipeline test(s) FAILED\n";
    }
    return failures;
}
