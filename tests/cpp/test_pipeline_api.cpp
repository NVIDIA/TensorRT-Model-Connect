// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-PIP-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FAC-01
// Intent:         C API pipeline creation via trtmc_create_pipeline_ex
// Preconditions:  TRT runtime available
// Postconditions: Pipeline created or appropriate error returned
// =============================================================================

// =============================================================================
// Test suite: Pipeline C ABI -- IPipeline virtual interface via trtmc_create_pipeline
// =============================================================================
//
// Purpose:
//   Validates the public C ABI entry point trtmc_create_pipeline() and the
//   IPipeline virtual interface it returns. Tests cover null/invalid input
//   handling, version queries, and ABI stability guarantees.
//
// Dependencies:
//   - trtmc/pipeline.h (IPipeline, trtmc_create_pipeline, trtmc_last_error,
//     trtmc_version, trtmc_has_trt)
//   - No TRT, GPU, or model files required.
// =============================================================================

#include "trtmc/pipeline.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// Minimal DummyPipeline implementing only the two required pure virtuals.
class DummyPipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "dummy-model"; }
    const char* pipeline_type() const override { return "DummyPipeline"; }
};

class DummyImagePipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "dummy-image-model"; }
    const char* pipeline_type() const override { return "DummyImagePipeline"; }

    trtmc::ImageResult generate_image(const std::string& prompt,
                                      const trtmc::GenerateConfig& cfg = {}) override {
        prompts += prompt + ";";
        seeds.push_back(cfg.seed);
        trtmc::ImageResult result;
        result.width = 1;
        result.height = 1;
        result.channels = 3;
        result.pixels = {static_cast<float>(cfg.seed), 0.0F, 0.0F};
        return result;
    }

    std::string prompts;
    std::vector<int32_t> seeds;
};

static void test_null_input_returns_null() {
    auto* p = trtmc_create_pipeline(nullptr, 0);
    check(p == nullptr, "null input returns nullptr");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set after null input");
}

static void test_invalid_path_returns_null() {
    auto* p = trtmc_create_pipeline("/nonexistent/path/to/bundle.trtfb", 0);
    check(p == nullptr, "invalid path returns nullptr");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set after invalid path");
}

static void test_version_available() {
    const char* ver = trtmc_version();
    check(ver != nullptr, "version is non-null");
    check(std::strlen(ver) > 0, "version is non-empty");
}

static void test_has_trt_returns_bool() {
    const int val = trtmc_has_trt();
    check(val == 0 || val == 1, "trtmc_has_trt returns 0 or 1");
}

static void test_sizeof_ipipeline_is_vtable() {
    check(sizeof(trtmc::IPipeline) == sizeof(void*), "sizeof(IPipeline) equals vtable pointer size");
}

static void test_delete_null_safe() {
    trtmc::IPipeline* p = nullptr;
    delete p;
    check(true, "delete null IPipeline is safe");
}

// Exercise default virtual methods -- they should throw with descriptive messages.
static void test_ipipeline_default_virtuals() {
    DummyPipeline pipeline;

    check(std::string(pipeline.model_id()) == "dummy-model", "model_id");
    check(std::string(pipeline.pipeline_type()) == "DummyPipeline", "pipeline_type");

    // Default generate(string) should throw
    bool threw = false;
    try {
        pipeline.generate("hello");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default generate(string) throws");

    // Default generate(string, image) should throw
    threw = false;
    try {
        pipeline.generate("hello", nullptr, 0, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default generate(string,image) throws");

    // Default generate_image should throw
    threw = false;
    try {
        pipeline.generate_image("prompt");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default generate_image throws");

    // Default generate_audio should throw
    threw = false;
    try {
        pipeline.generate_audio("hello");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default generate_audio throws");

    // Default transcribe should throw
    threw = false;
    try {
        pipeline.transcribe(nullptr, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default transcribe throws");

    // Default streaming transcription should throw
    threw = false;
    try {
        pipeline.create_transcription_stream();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default create_transcription_stream throws");

    threw = false;
    try {
        trtmc::TranscriptionStreamConfig cfg;
        pipeline.transcribe_streaming(nullptr, 0, cfg);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default transcribe_streaming throws");

    // Default speak should throw
    threw = false;
    try {
        pipeline.speak(nullptr, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default speak throws");

    // Default embed should throw
    threw = false;
    try {
        pipeline.embed("text");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default embed throws");

    // Default rerank should throw
    threw = false;
    try {
        pipeline.rerank("q", "doc");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default rerank throws");

    // Default segment should throw
    threw = false;
    try {
        pipeline.segment(nullptr, 0, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default segment throws");

    // Default encode should throw
    threw = false;
    try {
        pipeline.encode("text");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default encode throws");

    // Default solve should throw
    threw = false;
    try {
        pipeline.solve(nullptr, 0, nullptr, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default solve throws");

    // Default detect should throw
    threw = false;
    try {
        pipeline.detect(nullptr, 0, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default detect throws");
}

static void test_generate_images_default_loops_generate_image() {
    DummyImagePipeline pipeline;
    trtmc::GenerateConfig cfg;
    cfg.seed = 999;
    const auto results = pipeline.generate_images({"a", "b"}, {7, 8}, cfg);

    check(results.size() == 2U, "generate_images result count");
    check(pipeline.prompts == "a;b;", "generate_images prompt order");
    check(pipeline.seeds == std::vector<int32_t>({7, 8}), "generate_images seeds");
    check(results[0].pixels[0] == 7.0F, "generate_images first result");
    check(results[1].pixels[0] == 8.0F, "generate_images second result");
}

int main() {
    test_null_input_returns_null();
    test_invalid_path_returns_null();
    test_version_available();
    test_has_trt_returns_bool();
    test_sizeof_ipipeline_is_vtable();
    test_delete_null_safe();
    test_ipipeline_default_virtuals();
    test_generate_images_default_loops_generate_image();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All pipeline_api tests passed.\n";
    return 0;
}
