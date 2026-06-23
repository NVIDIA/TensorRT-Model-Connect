#include "../../support/mock_trt_engines.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/models/speech/pipeline.h"
#include "runtime/models/speech/speech_config.h"
#include "trtmc/runtime/kv_cache.h"

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
    auto temporal_cache = std::make_unique<trtmc::KvCache>(0, 8, 0, stream);

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

} // namespace

int main() {
    test_speech_pipeline_construction();
    test_speech_validates_temporal();
    if (failures > 0) {
        std::cerr << failures << " speech pipeline test(s) FAILED\n";
    }
    return failures;
}
