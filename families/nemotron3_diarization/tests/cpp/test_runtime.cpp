/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU tests of the host side of the Nemotron 3 Diarization runtime against the
// reference implementation: the log-mel front end, the Arrival-Order Speaker
// Cache over a sequence of steps with compressions, and the speaker segments.
// The golden data comes from tests/generate_runtime_golden.py.

#include "families/nemotron3_diarization/runtime/bundle.h"
#include "families/nemotron3_diarization/runtime/frontend.h"
#include "families/nemotron3_diarization/runtime/pipeline.h"
#include "families/nemotron3_diarization/runtime/speaker_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace trtmc::nemotron3_diarization;
using nlohmann::json;

int failures = 0;

void check(bool passed, const std::string& label) {
    if (!passed) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

std::string data_path(const char* name) {
    return std::string(NEMOTRON3_DIARIZATION_TEST_DATA) + "/" + name;
}

std::vector<float> read_f32(const char* name) {
    std::ifstream file(data_path(name), std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
    if (bytes.size() % sizeof(float) != 0)
        throw std::runtime_error(std::string(name) + " does not hold whole float32 values");
    std::vector<float> values(bytes.size() / sizeof(float));
    std::copy(bytes.begin(), bytes.end(), reinterpret_cast<char*>(values.data()));
    return values;
}

std::vector<float> flat(const json& rows) {
    std::vector<float> values;
    for (const auto& row : rows)
        for (const auto& value : row)
            values.push_back(value.get<float>());
    return values;
}

float max_gap(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        return std::numeric_limits<float>::infinity();
    float worst = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float gap = std::abs(a[i] - b[i]);
        // std::max would skip a NaN, so a nonfinite gap fails every bound.
        if (!std::isfinite(gap))
            return std::numeric_limits<float>::infinity();
        worst = std::max(worst, gap);
    }
    return worst;
}

bool refuses(const std::function<void()>& action) {
    try {
        action();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void test_top_k() {
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> values{1.0F, 3.0F, 3.0F, 3.0F, 2.0F, -inf};
    auto two = top_k_first_seen(values, 2);
    std::sort(two.begin(), two.end());
    check(two == std::vector<std::size_t>{1, 2}, "ties at the k-th value keep the first seen");
    auto all = top_k_first_seen(values, 6);
    check(all.size() == 6 && all.back() == 5, "minus infinity ranks last");
    const std::vector<float> with_nan{0.0F, std::nanf(""), 1.0F};
    check(top_k_first_seen(with_nan, 1) == std::vector<std::size_t>{1},
          "NaN ranks above every number, as in torch.topk");
}

void test_front_end(const json& golden) {
    FrontEndConfig config;
    config.window = read_f32("window.f32");
    config.mel_filters = read_f32("mel_filters.f32");
    const LogMelFrontEnd front_end(config);
    const auto samples = golden.at("samples").get<std::vector<float>>();
    const auto features =
        front_end.compute(samples.data(), static_cast<std::int64_t>(samples.size()));
    check(features.frames == static_cast<std::int64_t>(golden.at("features").size()) &&
              features.valid_frames == golden.at("valid_frames").get<std::int64_t>(),
          "centred framing gives 1 + L / hop frames, L / hop of them valid");
    // The same front end with a float64 transform: what this one must reproduce.
    const float exact = max_gap(features.values, flat(golden.at("features_float64")));
    // The reference transforms in float32 and sits up to 2.4e-4 from the float64
    // features at mel bins near the 2**-24 floor; twice that bounds it, while a
    // wrong window, framing or filter bank moves features by orders more.
    const float reference = max_gap(features.values, flat(golden.at("features")));
    std::cout << "front end: worst log-mel gap " << exact << " to float64, " << reference
              << " to the reference\n";
    check(exact <= 1e-5F, "log-mel features match the float64 front end");
    check(reference <= 5e-4F, "log-mel features match the reference feature extractor");
    if (features.values.size() < static_cast<std::size_t>(config.num_mel_bins)) {
        check(false, "the front end returned fewer values than one frame");
        return;
    }
    const auto last = features.values.end() - config.num_mel_bins;
    check(std::all_of(last, features.values.end(), [](float value) { return value == 0.0F; }),
          "the frame past the audio is zero, as the reference masks it");
    const std::vector<float> stereo{1.0F, 3.0F, -2.0F, 0.0F};
    check(mix_to_mono(stereo.data(), stereo.size(), 2) == std::vector<float>{2.0F, -1.0F},
          "interleaved channels are averaged to mono");
}

SpeakerCachePolicy policy_from(const json& value) {
    SpeakerCachePolicy policy;
    policy.num_speakers = value.at("num_speakers");
    policy.subsampling_factor = value.at("subsampling_factor");
    policy.hidden_size = value.at("hidden_size");
    policy.cache_length = value.at("cache_length");
    policy.silence_frames_per_speaker = value.at("silence_frames_per_speaker");
    policy.fifo_length = value.at("fifo_length");
    policy.update_period = value.at("update_period");
    policy.score_threshold = value.at("score_threshold");
    policy.latest_frames_boost = value.at("latest_frames_boost");
    policy.min_positive_scores = value.at("min_positive_scores");
    policy.strong_boosted_frames = value.at("strong_boosted_frames");
    policy.weak_boosted_frames = value.at("weak_boosted_frames");
    return policy;
}

void test_speaker_cache(const json& golden) {
    const auto policy = policy_from(golden.at("policy"));
    SpeakerCache cache(policy, golden.at("silence").get<std::vector<float>>());
    // The trajectory must keep frames queued, read look-back rows and compress,
    // or the checks below would pass on paths they never reach.
    bool queued = false, looked_back = false, compressed = false;
    for (const auto& step : golden.at("steps")) {
        queued = queued || step.at("fifo_rows").get<std::int64_t>() > 0;
        looked_back = looked_back || step.at("lookback_frames").get<std::int64_t>() > 0;
        compressed = compressed || step.at("compressed").get<bool>();
    }
    check(queued && looked_back && compressed,
          "the golden trajectory queues frames, reads look-back rows and compresses");
    int index = 0;
    for (const auto& step : golden.at("steps")) {
        const auto label = "step " + std::to_string(index++) + ": ";
        const auto input = flat(step.at("step_input"));
        const auto rows = static_cast<std::int64_t>(step.at("step_input").size());
        // The rows the cache puts ahead of the chunk are the reference's own.
        std::vector<float> cached(static_cast<std::size_t>(cache.cached_rows()) *
                                  static_cast<std::size_t>(policy.hidden_size));
        cache.copy_cached_rows(cached.data());
        // Sizes first: a wrong size stops the trajectory before any range is read.
        if (cached.size() > input.size()) {
            check(false, label + "the cache holds more rows than the step input");
            return;
        }
        check(std::equal(cached.begin(), cached.end(), input.begin()),
              label + "cached rows lead the step input");
        const auto logits = flat(step.at("logits"));
        const auto valid = step.at("valid").get<std::vector<std::uint8_t>>();
        cache.update(input.data(), rows, logits.data(), valid,
                     step.at("chunk_frames").get<std::int64_t>(),
                     step.at("lookback_frames").get<std::int64_t>());
        check(cache.cache_rows() == step.at("cache_rows").get<std::int64_t>() &&
                  cache.fifo_rows() == step.at("fifo_rows").get<std::int64_t>() &&
                  cache.compressed() == step.at("compressed").get<bool>(),
              label + "cache and FIFO sizes follow the reference");
        // The embeddings are copies, so the same selection gives equal values.
        check(cache.cache_embeddings() == flat(step.at("cache")),
              label + "the cache keeps the reference's frames in its order");
        check(max_gap(cache.cache_probabilities(), flat(step.at("cache_probs"))) <= 1e-6F,
              label + "the cache keeps the reference's speaker probabilities");
        std::vector<float> all(static_cast<std::size_t>(cache.cached_rows()) *
                               static_cast<std::size_t>(policy.hidden_size));
        cache.copy_cached_rows(all.data());
        const auto fifo = flat(step.at("fifo"));
        if (fifo.size() > all.size()) {
            check(false, label + "the cache and FIFO hold fewer rows than the reference's FIFO");
            return;
        }
        check(std::equal(fifo.begin(), fifo.end(),
                         all.end() - static_cast<std::ptrdiff_t>(fifo.size())),
              label + "the FIFO queue holds the reference's latest frames");
    }
}

void test_intervals(const json& golden) {
    const auto logits = flat(golden.at("logits"));
    const auto frames = static_cast<std::int64_t>(golden.at("logits").size());
    const auto speakers = static_cast<std::int64_t>(golden.at("logits").at(0).size());
    for (const auto& item : golden.at("cases")) {
        const auto threshold = item.at("threshold").get<float>();
        const auto intervals = active_intervals(logits, frames, speakers, threshold, 160, 16000);
        const auto& expected = item.at("segments");
        bool same = intervals.size() == expected.size();
        for (std::size_t i = 0; same && i < intervals.size(); ++i)
            same = std::abs(intervals[i].start_seconds - expected[i][0].get<double>()) < 1e-9 &&
                   std::abs(intervals[i].end_seconds - expected[i][1].get<double>()) < 1e-9 &&
                   intervals[i].label == expected[i][2].get<std::uint32_t>();
        check(same,
              "speaker segments match the reference at threshold " + std::to_string(threshold));
    }
    check(speaker_active(0.0F, 0.5F) == false && speaker_active(1e-6F, 0.5F) == true,
          "activity is strictly above the threshold");
}

void test_max_gap() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    check(max_gap({1.0F, 2.0F}, {1.0F, 2.5F}) == 0.5F, "max_gap reports the largest difference");
    check(max_gap({nan, 2.0F}, {1.0F, 2.0F}) == inf && max_gap({1.0F, 2.0F}, {1.0F, nan}) == inf,
          "a NaN operand fails every bound");
    check(max_gap({inf}, {inf}) == inf, "infinite operands fail every bound");
}

// runtime.json as config.runtime_config() writes it for the released checkpoint.
json released_runtime_json() {
    json labels = json::array();
    for (int index = 0; index < 8; ++index)
        labels.push_back("speaker_" + std::to_string(index));
    return {{"sample_rate", 16000},
            {"n_fft", 512},
            {"win_length", 400},
            {"hop_length", 160},
            {"preemphasis", 0.97},
            {"log_guard", 5.9604644775390625e-08},
            {"num_mel_bins", 128},
            {"subsampling_factor", 8},
            {"hidden_size", 512},
            {"num_speakers", 8},
            {"chunk_length", 340},
            {"chunk_left_context", 0},
            {"chunk_right_context", 40},
            {"fifo_length", 40},
            {"speaker_cache_update_period", 300},
            {"speaker_cache_length", 264},
            {"silence_frames_per_speaker", 1},
            {"prediction_score_threshold", 0.25},
            {"latest_frames_score_boost", 0.05},
            {"min_positive_scores", 16},
            {"strong_boosted_frames", 24},
            {"weak_boosted_frames", 48},
            {"max_step_rows", 684},
            {"labels", labels}};
}

std::vector<char> text_of(const std::string& text) {
    return {text.begin(), text.end()};
}

void test_runtime_json() {
    const auto config = parse_runtime_json(text_of(released_runtime_json().dump()));
    check(config.cache.subsampling_factor == 8 && config.max_step_rows == 684 &&
              config.labels.size() == 8 && config.front_end.preemphasis == 0.97F,
          "runtime.json of the released checkpoint parses");
    const std::vector<std::pair<std::string, json>> unusable{
        {"subsampling_factor", 0},
        {"subsampling_factor", 2.5},
        {"subsampling_factor", true},
        {"subsampling_factor", "8"},
        {"subsampling_factor", nullptr},
        {"subsampling_factor", std::int64_t{2147483648}},
        {"subsampling_factor", std::uint64_t{4294967296}},
        {"silence_frames_per_speaker", -1},
        {"preemphasis", "0.97"},
        {"preemphasis", false},
        {"log_guard", 1e300},
        {"labels", "speaker_0"},
        {"labels", json::array({"speaker_0", ""})},
        {"labels", json::array({"speaker_0", "speaker_0"})},
        {"labels", json::array({"speaker_0", 1})},
    };
    for (const auto& [key, value] : unusable) {
        auto variant = released_runtime_json();
        variant[key] = value;
        check(refuses([&] { (void)parse_runtime_json(text_of(variant.dump())); }),
              "runtime.json refuses " + key + " = " + value.dump());
    }
    auto missing = released_runtime_json();
    missing.erase("chunk_length");
    check(refuses([&] { (void)parse_runtime_json(text_of(missing.dump())); }),
          "runtime.json refuses a missing key");
    // A repeated key, which the parser alone would resolve to its last value.
    const auto repeated = "{\"subsampling_factor\": 0, " + released_runtime_json().dump().substr(1);
    check(refuses([&] { (void)parse_runtime_json(text_of(repeated)); }),
          "runtime.json refuses a repeated key");
    check(refuses([&] { (void)parse_runtime_json(text_of("[1, 2]")); }),
          "runtime.json must be an object");
}

void test_engine_io() {
    const std::vector<TensorSpec> inputs{{"inputs_embeds", {kStepRows, 4}},
                                         {"key_bias", {kStepRows}}};
    const std::vector<TensorSpec> outputs{{"logits", {kStepRows, 2}}};
    const trtmc::TensorInfo embeds{"inputs_embeds", {-1, 4}, trtmc::DType::kFloat32, true};
    const trtmc::TensorInfo bias{"key_bias", {-1}, trtmc::DType::kFloat32, true};
    const trtmc::TensorInfo logits{"logits", {-1, 2}, trtmc::DType::kFloat32, false};
    const auto io = [](std::vector<trtmc::TensorInfo> tensors, std::int64_t low, std::int64_t high,
                       std::int32_t profiles) {
        EngineIo value{std::move(tensors), {}, profiles};
        for (const auto& tensor : value.tensors)
            if (tensor.is_input)
                value.input_rows.push_back({tensor.name, {low, high}});
        return value;
    };
    const auto loads = [&](const EngineIo& value) {
        return !refuses([&] { check_engine_io(value, inputs, outputs, 24, "encoder.plan"); });
    };
    check(loads(io({embeds, bias, logits}, 1, 24, 1)), "an engine with the expected I/O loads");
    // A backend may report the rows its buffers hold instead of -1.
    check(loads(io({{"inputs_embeds", {24, 4}, trtmc::DType::kFloat32, true},
                    {"key_bias", {24}, trtmc::DType::kFloat32, true},
                    {"logits", {192, 2}, trtmc::DType::kFloat32, false}},
                   1, 24, 1)),
          "an engine reporting its buffers' rows loads");
    check(!loads(io({embeds, bias, {"logits", {192, 3}, trtmc::DType::kFloat32, false}}, 1, 24, 1)),
          "a reported speaker count other than the expected one is refused");
    check(!loads(io({embeds, bias, {"logits", {2, -1}, trtmc::DType::kFloat32, false}}, 1, 24, 1)),
          "transposed logits are refused");
    check(!loads(io({embeds, bias, {"logits", {-1, 3}, trtmc::DType::kFloat32, false}}, 1, 24, 1)),
          "logits for another speaker count are refused");
    check(!loads(io({{"inputs_embeds", {-1, 4}, trtmc::DType::kFloat16, true}, bias, logits}, 1, 24,
                    1)),
          "a float16 input is refused");
    check(!loads(io({embeds, logits}, 1, 24, 1)), "a missing input is refused");
    check(!loads(
              io({embeds, bias, logits, {"extra", {-1}, trtmc::DType::kFloat32, false}}, 1, 24, 1)),
          "an extra output is refused");
    check(!loads(io({embeds, bias, logits}, 2, 24, 1)), "a profile that needs two rows is refused");
    check(!loads(io({embeds, bias, logits}, 1, 23, 1)),
          "a profile shorter than the longest step is refused");
    check(!loads(io({embeds, bias, logits}, 1, 24, 0)), "an engine without a profile is refused");
}

// An engine stand-in for host-side tests: forward() answers with what the test
// computes; nothing else of the interface is used by the pipeline.
class FakeEngine final : public trtmc::ITrtModule {
  public:
    using Respond = std::function<trtmc::TensorMap(const trtmc::TensorMap&, std::vector<float>&)>;
    explicit FakeEngine(Respond respond) : respond_(std::move(respond)) {}
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        return respond_(inputs, buffer_);
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_async(const trtmc::TensorMap&) override { throw std::logic_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 0; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    int calls{0};

  private:
    Respond respond_;
    std::vector<float> buffer_;
};

// Two speakers, a 4-wide encoder and chunks of 5 rows with 2 rows of
// look-ahead, so that one second of audio takes three encoder steps.
DiarizationConfig small_config() {
    DiarizationConfig config;
    config.front_end.window = read_f32("window.f32");
    config.front_end.mel_filters = read_f32("mel_filters.f32");
    config.cache = {2, 8, 4, 8, 1, 4, 3, 0.25F, 0.05F, 1, 1, 2};
    config.chunk_length = 5;
    config.chunk_left_context = 0;
    config.chunk_right_context = 2;
    config.max_step_rows = 8 + 4 + 5 + 2;
    config.labels = {"speaker_0", "speaker_1"};
    config.silence_embedding = std::vector<float>(4, 0.0F);
    return config;
}

struct Faults {
    bool nan_embedding{false};
    bool nan_lookahead_logit{false};
    bool transposed_logits{false};
};

struct Rig {
    FakeEngine* embedder{nullptr};
    FakeEngine* encoder{nullptr};
    std::unique_ptr<DiarizationPipeline> pipeline;
};

Rig make_rig(const Faults& faults) {
    auto embedder = std::make_unique<FakeEngine>(
        [faults](const trtmc::TensorMap& inputs, std::vector<float>& buffer) {
            const auto rows = inputs.at("stacked_features").shape.at(0);
            buffer.assign(static_cast<std::size_t>(rows * 4), 0.1F);
            if (faults.nan_embedding)
                buffer[0] = std::numeric_limits<float>::quiet_NaN();
            return trtmc::TensorMap{
                {"inputs_embeds", {buffer.data(), {rows, 4}, trtmc::DType::kFloat32}}};
        });
    auto calls = std::make_shared<int>(0);
    auto encoder = std::make_unique<FakeEngine>([faults, calls](const trtmc::TensorMap& inputs,
                                                                std::vector<float>& buffer) {
        const auto rows = inputs.at("inputs_embeds").shape.at(0);
        buffer.assign(static_cast<std::size_t>(rows * 8 * 2), -3.0F);
        // The first step is the first chunk (rows 0-4) and its look-ahead
        // (rows 5-6); only the chunk's logits are kept for the result.
        if (faults.nan_lookahead_logit && (*calls)++ == 0)
            buffer[static_cast<std::size_t>(6 * 8 * 2)] = std::numeric_limits<float>::quiet_NaN();
        const std::vector<std::int64_t> shape = faults.transposed_logits
                                                    ? std::vector<std::int64_t>{2, rows * 8}
                                                    : std::vector<std::int64_t>{rows * 8, 2};
        return trtmc::TensorMap{{"logits", {buffer.data(), shape, trtmc::DType::kFloat32}}};
    });
    Rig rig;
    rig.embedder = embedder.get();
    rig.encoder = encoder.get();
    rig.pipeline = std::make_unique<DiarizationPipeline>(std::move(embedder), std::move(encoder),
                                                         small_config());
    return rig;
}

void test_pipeline_rejections() {
    std::vector<float> second(16000);
    for (std::size_t i = 0; i < second.size(); ++i)
        second[i] = 0.1F * std::sin(0.05F * static_cast<float>(i));
    const auto run = [&](Rig& rig, const std::vector<float>& samples) {
        trtmc::internal::AudioToTimedLabelsRequest request;
        request.audio = {{samples.data(), samples.size()}, 16000U, 1U};
        return rig.pipeline->run(request, {});
    };
    auto healthy = make_rig({});
    const auto result = run(healthy, second);
    check(healthy.encoder->calls == 3 && result.frame_count == 100 &&
              result.frame_scores.size() == 200 && result.intervals.empty(),
          "one second of audio takes three steps and yields 100 frames of two speakers");
    auto nan_sample = second;
    nan_sample[8000] = std::numeric_limits<float>::quiet_NaN();
    auto clean = make_rig({});
    check(refuses([&] { (void)run(clean, nan_sample); }), "a NaN audio sample is refused");
    auto infinite_sample = second;
    infinite_sample[0] = std::numeric_limits<float>::infinity();
    check(refuses([&] { (void)run(clean, infinite_sample); }),
          "an infinite audio sample is refused");
    auto nan_embedding = make_rig({true, false, false});
    check(refuses([&] { (void)run(nan_embedding, second); }), "a NaN embedding is refused");
    auto nan_lookahead = make_rig({false, true, false});
    check(refuses([&] { (void)run(nan_lookahead, second); }),
          "a NaN logit in the look-ahead, which the cache reads, is refused");
    auto transposed = make_rig({false, false, true});
    check(refuses([&] { (void)run(transposed, second); }),
          "logits of the right size but transposed shape are refused");
    // The threshold is compared in float32: values that round to 0 or 1 are refused.
    for (const double value : {1e-300, 0.999999999}) {
        const trtmc::internal::ConfigEntry entry{"threshold", trtmc::internal::ConfigValue{value}};
        auto rounded = make_rig({});
        check(refuses([&] {
                  trtmc::internal::AudioToTimedLabelsRequest request;
                  request.audio = {{second.data(), second.size()}, 16000U, 1U};
                  (void)rounded.pipeline->run(request, {&entry, 1});
              }),
              "threshold " + std::to_string(value) + " rounds to an endpoint in float32");
    }
    // Finite samples near the float32 limit overflow the pre-emphasis and the
    // power spectrum; the features are refused before any engine runs.
    auto loud = second;
    for (std::size_t i = 0; i < loud.size(); ++i)
        loud[i] = i % 2 == 0 ? 3.0e38F : -3.0e38F;
    auto overflow = make_rig({});
    check(refuses([&] { (void)run(overflow, loud); }) && overflow.embedder->calls == 0 &&
              overflow.encoder->calls == 0,
          "audio that overflows the front end is refused before any engine call");
}

void test_section_lengths() {
    // Section lengths come from the bundle's table and are checked before a read.
    check(!refuses([] { check_float_section(400 * 4, 400, "window.f32"); }),
          "a float section of the expected length passes");
    check(refuses([] { check_float_section(std::uint64_t{1} << 62, 400, "window.f32"); }),
          "an oversized float section is refused before it is read");
    check(refuses([] {
              check_float_section(16, std::numeric_limits<std::size_t>::max(), "window.f32");
          }),
          "a float count whose byte size overflows is refused");
    check(refuses([] { check_section_limit(std::uint64_t{1} << 40, 1 << 20, "runtime.json"); }),
          "an oversized runtime.json is refused before it is read");
}

void test_boost_needs_its_frames() {
    // As torch.topk, a compression never boosts fewer frames than its budget.
    SpeakerCache cache(small_config().cache, std::vector<float>(4, 0.0F));
    std::vector<float> embeddings(4, 0.0F), probs{0.9F, 0.1F};
    std::int64_t rows = 1;
    check(refuses([&] { cache.compress(embeddings, probs, rows); }),
          "a compression with fewer frames than a boost budget is refused");
}

void test_saturated_scores() {
    // Probabilities of exactly 1 and 0: with a positive floor every log score
    // stays finite or is the deliberate minus infinity of an inactive speaker.
    SpeakerCache cache(small_config().cache, std::vector<float>(4, 0.0F));
    const std::vector<float> probs{1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    const auto scores = cache.frame_scores(probs, 3);
    check(std::none_of(scores.begin(), scores.end(), [](float value) { return std::isnan(value); }),
          "saturated probabilities give no NaN score");
}

void test_policy_checked_at_load() {
    const auto load = [](DiarizationConfig config) {
        return !refuses([&] {
            DiarizationPipeline pipeline(std::make_unique<FakeEngine>(FakeEngine::Respond{}),
                                         std::make_unique<FakeEngine>(FakeEngine::Respond{}),
                                         std::move(config));
        });
    };
    check(load(small_config()), "the small configuration loads");
    auto no_factor = small_config();
    no_factor.cache.subsampling_factor = 0;
    check(!load(no_factor), "a zero subsampling factor is refused at load");
    auto huge_silence = small_config();
    huge_silence.cache.silence_frames_per_speaker = std::numeric_limits<std::int32_t>::max();
    check(!load(huge_silence), "a cache budget that would overflow is refused at load");
    auto huge_cache = small_config();
    huge_cache.cache.cache_length = std::numeric_limits<std::int32_t>::max();
    check(!load(huge_cache), "a step longer than max_step_rows is refused without overflow");
    auto nan_threshold = small_config();
    nan_threshold.cache.score_threshold = std::numeric_limits<float>::quiet_NaN();
    check(!load(nan_threshold), "a NaN score threshold is refused at load");
    for (const float floor : {0.0F, -0.25F, 1.0F}) {
        auto config = small_config();
        config.cache.score_threshold = floor;
        check(!load(config), "a score floor of " + std::to_string(floor) + " is refused at load");
    }
    auto big_boost = small_config();
    big_boost.cache.weak_boosted_frames = big_boost.cache.cache_length + 1;
    check(!load(big_boost), "a boost budget past the cache length is refused at load");
    auto loud_preemphasis = small_config();
    loud_preemphasis.front_end.preemphasis = 1.5F;
    check(!load(loud_preemphasis), "a pre-emphasis outside [0, 1) is refused at load");
}

} // namespace

int main() {
    try {
        std::ifstream file(data_path("runtime_golden.json"));
        const auto golden = json::parse(file);
        test_top_k();
        test_front_end(golden.at("front_end"));
        test_speaker_cache(golden.at("speaker_cache"));
        test_intervals(golden.at("intervals"));
        test_max_gap();
        test_runtime_json();
        test_engine_io();
        test_pipeline_rejections();
        test_policy_checked_at_load();
        test_saturated_scores();
        test_boost_needs_its_frames();
        test_section_lengths();
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        ++failures;
    }
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures == 0 ? 0 : 1;
}
