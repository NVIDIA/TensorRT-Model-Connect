#include "../../support/mock_trt_engines.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/models/omni/omni_config.h"
#include "runtime/models/omni/pipeline.h"
#include "trtmc/runtime/kv_cache.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class OmniFixedTokenizer : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {1, 2}; }
    std::string decode(const std::vector<int32_t>&) const override { return ""; }
    int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(int32_t) const override { return ""; }
};

void test_omni_pipeline_construction() {
    const std::vector<float> thinker_logits = {1.0F, 0.1F, 0.1F, 0.1F};
    auto thinker_engine = trtmc::test::build_mock_step_engine(9, 4, thinker_logits);
    if (!thinker_engine) {
        std::cerr << "WARNING: Could not build thinker engine for OmniPipeline, skipping\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto thinker = std::make_unique<trtmc::TrtModuleImpl>(
        thinker_engine.get(), thinker_engine->createExecutionContext(), stream);
    auto thinker_cache = std::make_unique<trtmc::KvCache>(0, 8, 0, stream);

    check(thinker->ok(), "omni thinker module ok");
    check(thinker_cache->ok(), "omni thinker cache ok");

    trtmc::OmniConfig cfg;
    trtmc::OmniPipeline pipeline(std::move(thinker), std::move(thinker_cache), nullptr, nullptr,
                                 nullptr, cfg, stream, nullptr, "test-omni");

    check(std::string(pipeline.pipeline_type()) == "OmniPipeline", "OmniPipeline: pipeline_type");
    check(std::string(pipeline.model_id()) == "test-omni", "OmniPipeline: model_id");

    cudaStreamDestroy(stream);
}

void test_omni_generate_audio() {
    const std::vector<float> thinker_logits = {1.0F, 0.1F, 0.1F, 0.1F};
    auto thinker_engine = trtmc::test::build_mock_step_engine(9, 4, thinker_logits);
    if (!thinker_engine) {
        std::cerr << "WARNING: Could not build thinker engine for omni_generate, skipping\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto thinker = std::make_unique<trtmc::TrtModuleImpl>(
        thinker_engine.get(), thinker_engine->createExecutionContext(), stream);
    auto thinker_cache = std::make_unique<trtmc::KvCache>(0, 8, 0, stream);

    trtmc::OmniConfig cfg;
    trtmc::OmniPipeline pipeline(std::move(thinker), std::move(thinker_cache), nullptr, nullptr,
                                 nullptr, cfg, stream, std::make_shared<OmniFixedTokenizer>(),
                                 "test-omni-gen");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;

    auto result = pipeline.generate_audio("hello", gen_cfg);
    check(result.num_samples == 0,
          "omni generate_audio: no audio when thinker returns empty text tokens");
    check(result.sample_rate == 24000, "omni generate_audio: sample_rate = 24000");

    cudaStreamDestroy(stream);
}

void test_omni_validates_thinker() {
    bool threw = false;
    try {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        trtmc::OmniConfig cfg;
        trtmc::OmniPipeline p(nullptr, nullptr, nullptr, nullptr, nullptr, cfg, stream, nullptr,
                              "x");
        check(false, "null thinker should throw");
        cudaStreamDestroy(stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "omni: null thinker throws");
}

} // namespace

int main() {
    test_omni_pipeline_construction();
    test_omni_generate_audio();
    test_omni_validates_thinker();
    if (failures > 0) {
        std::cerr << failures << " omni pipeline test(s) FAILED\n";
    }
    return failures;
}
