/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU tests of the host side of the Nemotron 3 Diarization runtime against the
// reference implementation: the log-mel front end, the Arrival-Order Speaker
// Cache over a sequence of steps with compressions, and the speaker segments.
// The golden data comes from tests/generate_runtime_golden.py.

#include "families/nemotron3_diarization/runtime/frontend.h"
#include "families/nemotron3_diarization/runtime/pipeline.h"
#include "families/nemotron3_diarization/runtime/speaker_cache.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
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
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
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

} // namespace

int main() {
    try {
        std::ifstream file(data_path("runtime_golden.json"));
        const auto golden = json::parse(file);
        test_top_k();
        test_front_end(golden.at("front_end"));
        test_speaker_cache(golden.at("speaker_cache"));
        test_intervals(golden.at("intervals"));
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        ++failures;
    }
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures == 0 ? 0 : 1;
}
