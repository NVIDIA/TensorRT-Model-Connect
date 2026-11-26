/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Family-owned command line: `trtmc nemotron3_diarization diarize BUNDLE
// --audio WAV`. The handler runs the public audio_to_timed_labels Task, so the
// shared contract checks every result it prints.
#include "trtmc/internal/cli.h"

#include "families/nemotron3_diarization/runtime/wav.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <trtmc/audio.hpp>
#include <trtmc/control.hpp>
#include <trtmc/core.hpp>
#include <vector>

namespace {

constexpr std::string_view kFamily = "nemotron3_diarization";

const char* score_kind(std::uint32_t kind) {
    switch (kind) {
    case 0:
        return "none";
    case TRTMC_SCORE_LOGIT:
        return "logit";
    case TRTMC_SCORE_PROBABILITY:
        return "probability";
    case TRTMC_SCORE_UNBOUNDED:
        return "unbounded";
    default:
        throw std::runtime_error("timed labels have an unknown frame score kind");
    }
}

nlohmann::json timed_labels_json(const trtmc::TimedLabelsResult& result) {
    std::vector<std::string> labels;
    for (const auto label : result.labels())
        labels.emplace_back(label);
    auto intervals = nlohmann::json::array();
    for (const auto& interval : result.intervals())
        intervals.push_back({{"start_seconds", interval.start_seconds},
                             {"end_seconds", interval.end_seconds},
                             {"label", interval.label}});
    // The Task guarantees rows * columns values, every one finite.
    const auto scores = result.frame_scores();
    auto rows = nlohmann::json::array();
    for (std::uint64_t row = 0; row < scores.rows; ++row) {
        const float* first = scores.values.data() + row * scores.columns;
        rows.push_back(std::vector<float>(first, first + scores.columns));
    }
    return {{"task", std::string(trtmc::AudioToTimedLabels::kTask)},
            {"labels", labels},
            {"vocabulary_id", std::string(result.vocabulary_id())},
            {"intervals", std::move(intervals)},
            {"frame_seconds", result.frame_seconds()},
            {"score_kind", score_kind(result.score_kind())},
            {"frame_scores", std::move(rows)},
            {"setup_ms", result.setup_ms()},
            {"inference_ms", result.inference_ms()}};
}

nlohmann::json diarize(const nlohmann::json& values, const char* default_runtime_root) {
    const auto bundle = values.at("bundle").get<std::string>();
    // Check identity before loading: this command must not run another
    // family's bundle just because it serves the same Task.
    if (trtmc::Bundle::open(bundle).info().family != kFamily)
        throw std::invalid_argument("diarize requires a nemotron3_diarization bundle");
    const auto audio =
        trtmc::nemotron3_diarization::read_wav(values.at("audio").get<std::string>());
    trtmc::Config config;
    // Absent, the threshold keeps the default the family declares.
    if (values.contains("threshold"))
        config.add("threshold", values.at("threshold").get<double>());
    trtmc::LoadOptions options;
    options.runtime_root = values.value("runtime_root", std::string(default_runtime_root));
    const auto model = trtmc::Model::load(bundle, options);
    const auto task = model.task<trtmc::AudioToTimedLabels>();
    const trtmc::AudioToTimedLabelsRequest request{
        {{audio.samples.data(), audio.samples.size()}, audio.sample_rate, audio.channels}};
    return timed_labels_json(task.run(request, config));
}

} // namespace

extern "C" int trtmc_family_cli_v1(const char* handler, const char* values_json,
                                   const char* default_runtime_root, void* context,
                                   trtmc_cli_write_v1 output, trtmc_cli_write_v1 error) {
    if (error == nullptr)
        return 1;
    auto fail = [&](const std::string& reason) {
        const auto message = "Error: " + reason + '\n';
        error(context, message.data(), message.size());
        return 1;
    };
    try {
        if (handler == nullptr || values_json == nullptr || default_runtime_root == nullptr ||
            output == nullptr)
            return fail("the CLI host passed an incomplete call");
        if (std::string_view(handler) != "diarize")
            return fail("unknown Nemotron 3 Diarization command: " + std::string(handler));
        const auto text =
            diarize(nlohmann::json::parse(values_json), default_runtime_root).dump() + '\n';
        output(context, text.data(), text.size());
        return 0;
    } catch (const std::exception& exception) {
        return fail(exception.what());
    } catch (...) {
        return fail("unknown failure");
    }
}
