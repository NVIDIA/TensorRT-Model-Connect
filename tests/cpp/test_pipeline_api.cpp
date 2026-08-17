/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include "trtmc/runtime/pipeline_pool.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

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

class RecordingTranscriptionPipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "recording"; }
    const char* pipeline_type() const override { return "RecordingTranscriptionPipeline"; }

    trtmc::TextResult transcribe(const float* audio, int32_t count,
                                 const trtmc::TranscriptionConfig& cfg) override {
        observed.push_back(cfg);
        return {std::to_string(count) + ":" + std::to_string(audio == nullptr ? -1 : audio[0]),
                {cfg.beam_size}};
    }

    std::vector<trtmc::TranscriptionConfig> observed;
};

class PoolTestPipeline final : public trtmc::IPipeline {
  public:
    explicit PoolTestPipeline(std::string id) : id_(std::move(id)) {}

    const char* model_id() const override { return id_.c_str(); }
    const char* pipeline_type() const override { return "PoolTestPipeline"; }
    bool supports_lora_adapters() const override { return true; }

    void load_lora_adapter(const std::string& adapter_id, const std::string&) override {
        adapters_.insert(adapter_id);
    }

    void unload_lora_adapter(const std::string& adapter_id) override {
        if (adapters_.erase(adapter_id) == 0)
            throw std::invalid_argument("unknown adapter");
    }

    std::vector<std::string> loaded_lora_adapters() const override {
        return {adapters_.begin(), adapters_.end()};
    }

  private:
    std::string id_;
    std::set<std::string> adapters_;
};

static void test_null_input_returns_null() {
    auto* p = trtmc_create_pipeline(nullptr, 0);
    check(p == nullptr, "null input returns nullptr");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set after null input");
}

static void test_invalid_path_returns_null() {
    auto* p = trtmc_create_pipeline("/nonexistent/path/to/bundle.bundle", 0);
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
    check(sizeof(trtmc::IPipeline) == sizeof(void*),
          "sizeof(IPipeline) equals vtable pointer size");
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

    // Default generate_image(string, image) should throw via the text-only
    // image-generation overload.
    threw = false;
    try {
        pipeline.generate_image("prompt", nullptr, 0, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default generate_image(string,image) throws");

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

    // Default image feature extraction should throw
    threw = false;
    try {
        pipeline.extract_image_features(nullptr, 0, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default extract_image_features throws");

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

static void test_transcription_batch_preserves_per_request_config() {
    RecordingTranscriptionPipeline pipeline;
    trtmc::TranscriptionRequest first;
    first.audio_samples = {1.0F, 2.0F};
    first.config.beam_size = 1;
    first.config.source_language = "en";
    trtmc::TranscriptionRequest second;
    second.audio_samples = {3.0F};
    second.config.beam_size = 4;
    second.config.source_language = "fr";

    const auto results = pipeline.transcribe_batch({first, second});
    check(results.size() == 2, "transcription batch result count");
    check(results[0].token_ids == std::vector<int32_t>({1}) &&
              results[1].token_ids == std::vector<int32_t>({4}),
          "transcription batch preserves request order");
    check(pipeline.observed.size() == 2 && pipeline.observed[0].source_language == "en" &&
              pipeline.observed[1].source_language == "fr",
          "transcription batch preserves per-request config");
}

static void test_pipeline_pool_leases_and_adapter_maintenance() {
    std::vector<std::unique_ptr<trtmc::IPipeline>> pipelines;
    pipelines.push_back(std::make_unique<PoolTestPipeline>("lane-0"));
    pipelines.push_back(std::make_unique<PoolTestPipeline>("lane-1"));
    trtmc::PipelinePool pool(std::move(pipelines));

    check(pool.size() == 2 && pool.available() == 2, "pipeline pool initial capacity");
    auto first = pool.acquire();
    auto second = pool.acquire();
    check(pool.available() == 0, "pipeline pool leases are exclusive");
    check(std::string(first->model_id()) != std::string(second->model_id()),
          "pipeline pool leases distinct lanes");
    check(!pool.try_acquire().has_value(), "pipeline pool reports exhaustion");

    const std::string released_lane = first->model_id();
    auto waiter = std::async(std::launch::async, [&pool] {
        auto lease = pool.acquire();
        return std::string(lease->model_id());
    });
    check(waiter.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "pipeline pool waits when all lanes are busy");
    first = {};
    check(waiter.get() == released_lane, "pipeline pool reuses released lane");
    second = {};

    pool.load_lora_adapter("adapter-a", "/synthetic/adapter");
    check(pool.supports_lora_adapters(), "pipeline pool reports shared adapter capability");
    check(pool.loaded_lora_adapters() == std::vector<std::string>{"adapter-a"},
          "pipeline pool registers adapter across lanes");
    auto lane_a = pool.acquire();
    auto lane_b = pool.acquire();
    check(lane_a->loaded_lora_adapters() == std::vector<std::string>{"adapter-a"} &&
              lane_b->loaded_lora_adapters() == std::vector<std::string>{"adapter-a"},
          "pipeline pool keeps lane adapter registries consistent");
    lane_a = {};
    lane_b = {};
    pool.unload_lora_adapter("adapter-a");
    check(pool.loaded_lora_adapters().empty(), "pipeline pool unloads adapter across lanes");
}

int main() {
    test_null_input_returns_null();
    test_invalid_path_returns_null();
    test_version_available();
    test_has_trt_returns_bool();
    test_sizeof_ipipeline_is_vtable();
    test_delete_null_safe();
    test_ipipeline_default_virtuals();
    test_transcription_batch_preserves_per_request_config();
    test_pipeline_pool_leases_and_adapter_maintenance();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All pipeline_api tests passed.\n";
    return 0;
}
