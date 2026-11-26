/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The pipeline against a stand-in engine: the bindings it accepts at load, the waveform it hands
// the engine, the frame ids it accepts back, the decoded result and every refused request.
#include "families/granite_speech5/runtime/pipeline.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace trtmc;
using trtmc::granite_speech5::Geometry;
using trtmc::granite_speech5::Pipeline;
using trtmc::granite_speech5::Vocabulary;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

template <class F>
void check_throws(F&& call, const char* what) {
    try {
        call();
    } catch (const std::exception&) {
        return;
    }
    std::printf("FAIL: %s did not throw\n", what);
    ++failures;
}

const char* kRuntime =
    R"({"precision": "bf16", "sample_rate": 16000, "n_fft": 512, "hop_length": 160,
  "frame_stacking": 2, "subsampling_layers": 2, "min_samples": 1120, "max_samples": 480000,
  "vocab_size": 10})";

const char* kVocabulary = R"({"blank_id": 0, "entries": [
  [0, ""], [0, ""], [1, " hello"], [1, " world"], [2, 227], [2, 129], [2, 130], [1, ","], [2, 255],
  [1, "  "]]})";

// Returns `ids` (or `count` copies of `fill`) for every waveform and records the waveform's length.
class Engine final : public ITrtModule {
  public:
    std::vector<TensorInfo> inputs{{"waveform", {-1}, DType::kFloat32, true}};
    std::vector<TensorInfo> outputs{{"frame_ids", {-1}, DType::kInt32, false}};
    std::int64_t profile_min{1632};
    std::int64_t profile_max{480513};
    std::vector<std::int32_t> ids;
    std::int64_t seen{-1};

    TensorMap forward(const TensorMap& map) override {
        const auto found = map.find("waveform");
        seen = found == map.end() ? -1 : static_cast<std::int64_t>(found->second.numel());
        return {{"frame_ids",
                 Tensor{ids.data(), {static_cast<std::int64_t>(ids.size())}, DType::kInt32}}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override { return {}; }
    void forward_device_async(const DeviceTensorMap&) override {}
    void forward_async(const TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return inputs; }
    std::vector<TensorInfo> output_info() const override { return outputs; }
    bool has_input(const std::string& name) const override { return find(inputs, name) != nullptr; }
    bool has_output(const std::string& name) const override {
        return find(outputs, name) != nullptr;
    }
    DType tensor_dtype(const std::string& name) const override {
        const auto* info = find(inputs, name);
        info = info != nullptr ? info : find(outputs, name);
        return info != nullptr ? info->dtype : DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        const auto* info = find(inputs, name);
        info = info != nullptr ? info : find(outputs, name);
        return info != nullptr ? info->shape : std::vector<int64_t>{};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector selector) const override {
        return {selector == ProfileShapeSelector::kMin ? profile_min : profile_max};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string& name) const override {
        const auto* info = find(inputs, name);
        return info != nullptr ? static_cast<int32_t>(info->shape.size()) : -1;
    }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    static const TensorInfo* find(const std::vector<TensorInfo>& infos, const std::string& name) {
        for (const auto& info : infos)
            if (info.name == name)
                return &info;
        return nullptr;
    }
};

struct Built {
    Engine* engine;
    std::unique_ptr<Pipeline> pipeline;
};

template <class Change>
Built build(Change&& change) {
    auto engine = std::make_unique<Engine>();
    change(*engine);
    auto* raw = engine.get();
    auto pipeline = std::make_unique<Pipeline>(std::move(engine), Geometry::from_json(kRuntime),
                                               Vocabulary::from_json(kVocabulary, 10));
    return {raw, std::move(pipeline)};
}

Built healthy() {
    return build([](Engine&) {});
}

internal::SpeechTranscriptionRequest request(const std::vector<float>& samples,
                                             std::uint32_t channels = 1) {
    internal::SpeechTranscriptionRequest result;
    result.audio.samples = {samples.data(), samples.size()};
    result.audio.channels = channels;
    return result;
}

void bindings_are_checked_at_load() {
    check_throws(
        [] {
            build([](Engine& e) { e.inputs.push_back({"extra", {-1}, DType::kFloat32, true}); });
        },
        "an extra engine input");
    check_throws([] { build([](Engine& e) { e.outputs.front().shape = {-1, 10}; }); },
                 "rank-2 frame ids");
    check_throws([] { build([](Engine& e) { e.outputs.front().dtype = DType::kFloat32; }); },
                 "float frame ids");
    check_throws(
        [] {
            build([](Engine& e) {
                e.outputs.push_back({"logits", {-1, 10}, DType::kFloat32, false});
            });
        },
        "a second engine output");
    check_throws([] { build([](Engine& e) { e.inputs.front().shape = {1, -1}; }); },
                 "a rank-2 waveform");
    check_throws([] { build([](Engine& e) { e.profile_max = 480512; }); },
                 "a profile without room for the tail sample");
    check_throws([] { build([](Engine& e) { e.profile_min = 1633; }); },
                 "a profile above the shortest clip");
    check_throws(
        [] {
            Pipeline(
                std::make_unique<Engine>(), Geometry::from_json(kRuntime),
                Vocabulary::from_json(R"({"blank_id": 0, "entries": [[0, ""], [1, "a"]]})", 2));
        },
        "a vocabulary narrower than the CTC head");
}

void transcribes_through_the_engine() {
    auto built = healthy();
    // 16000 samples: 100 hops (even, no tail sample), 12 CTC frames.
    built.engine->ids = {0, 2, 2, 0, 2, 3, 0, 4, 1, 5, 6, 7};
    const std::vector<float> clip(16000, 0.25F);
    const auto result = built.pipeline->run(request(clip), {});
    check(built.engine->seen == 16000 + 512, "the engine reads the clip and the reflect padding");
    check(result.token_ids == std::vector<std::int32_t>{2, 2, 3, 4, 1, 5, 6, 7},
          "collapsed ids keep specials");
    check(result.text == "hello hello world\xE3\x81\x82,", "the transcript");
    // Two equal channels average to the mono clip.
    std::vector<float> stereo;
    for (const auto value : clip) {
        stereo.push_back(value);
        stereo.push_back(value);
    }
    const auto both = built.pipeline->run(request(stereo, 2), {});
    check(both.text == result.text && both.token_ids == result.token_ids &&
              built.engine->seen == 16512,
          "a stereo clip of equal channels");
    // 16160 samples are 101 whole hops: the reference appends one zero sample.
    built.engine->ids.resize(12);
    built.pipeline->run(request(std::vector<float>(16160, 0.25F)), {});
    check(built.engine->seen == 16160 + 1 + 512, "the tail sample");
}

void engine_results_are_checked() {
    auto built = healthy();
    const std::vector<float> clip(16000, 0.25F);
    built.engine->ids = std::vector<std::int32_t>(11, 2);
    check_throws([&] { built.pipeline->run(request(clip), {}); },
                 "a frame count the geometry does not give");
    built.engine->ids = std::vector<std::int32_t>(12, 2);
    built.engine->ids[5] = 10;
    check_throws([&] { built.pipeline->run(request(clip), {}); },
                 "an id beyond the vocabulary (NaN logits)");
    built.engine->ids[5] = -1;
    check_throws([&] { built.pipeline->run(request(clip), {}); }, "a negative id");
}

void requests_outside_the_contract_are_refused() {
    auto built = healthy();
    built.engine->ids = std::vector<std::int32_t>(12, 2);
    const std::vector<float> clip(16000, 0.25F);
    auto wrong_rate = request(clip);
    wrong_rate.audio.sample_rate = 48000;
    check_throws([&] { built.pipeline->run(wrong_rate, {}); }, "48 kHz audio");
    check_throws([&] { built.pipeline->run(request(std::vector<float>(1119, 0.25F)), {}); },
                 "1119 samples");
    check_throws([&] { built.pipeline->run(request(std::vector<float>(480001, 0.25F)), {}); },
                 "480001 samples");
    check_throws([&] { built.pipeline->run(request(std::vector<float>(16001, 0.25F), 2), {}); },
                 "a partial stereo frame");
    auto broken = clip;
    broken[100] = std::numeric_limits<float>::quiet_NaN();
    check_throws([&] { built.pipeline->run(request(broken), {}); }, "a NaN sample");
    broken[100] = std::numeric_limits<float>::infinity();
    check_throws([&] { built.pipeline->run(request(broken), {}); }, "an infinite sample");
    auto language = request(clip);
    language.source_language = "en";
    check_throws([&] { built.pipeline->run(language, {}); }, "a source language");
    const internal::ConfigEntry beam[]{{"beam_size", internal::ConfigValue{std::int64_t{2}}}};
    check_throws([&] { built.pipeline->run(request(clip), {beam, 1}); }, "an unknown option");
    check_throws([&] { built.pipeline->run(request({}), {}); }, "no samples");
}

} // namespace

int main() {
    bindings_are_checked_at_load();
    transcribes_through_the_engine();
    engine_results_are_checked();
    requests_outside_the_contract_are_refused();
    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("granite_speech5 pipeline checks passed\n");
    return 0;
}
