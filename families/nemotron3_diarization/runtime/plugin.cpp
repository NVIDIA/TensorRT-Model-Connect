/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron3_diarization/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstring>
#include <initializer_list>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::nemotron3_diarization {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("nemotron3_diarization bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

std::vector<float> float_section(const BundleReader& bundle, const char* name,
                                 std::size_t expected) {
    const auto raw = require_section(bundle, name);
    if (raw.size() != expected * sizeof(float))
        throw std::runtime_error("nemotron3_diarization bundle section has the wrong length: " +
                                 std::string(name));
    std::vector<float> values(expected);
    std::memcpy(values.data(), raw.data(), raw.size());
    return values;
}

FrontEndConfig front_end(const nlohmann::json& json, const BundleReader& bundle) {
    FrontEndConfig config;
    config.sample_rate = json.at("sample_rate").get<std::int32_t>();
    config.n_fft = json.at("n_fft").get<std::int32_t>();
    config.win_length = json.at("win_length").get<std::int32_t>();
    config.hop_length = json.at("hop_length").get<std::int32_t>();
    config.num_mel_bins = json.at("num_mel_bins").get<std::int32_t>();
    config.preemphasis = json.at("preemphasis").get<float>();
    config.log_guard = json.at("log_guard").get<float>();
    if (config.win_length <= 0 || config.n_fft <= 0 || config.num_mel_bins <= 0)
        throw std::runtime_error("nemotron3_diarization runtime.json front end is unusable");
    config.window =
        float_section(bundle, "window.f32", static_cast<std::size_t>(config.win_length));
    config.mel_filters = float_section(bundle, "mel_filters.f32",
                                       static_cast<std::size_t>(config.num_mel_bins) *
                                           static_cast<std::size_t>(config.n_fft / 2 + 1));
    return config;
}

SpeakerCachePolicy cache_policy(const nlohmann::json& json) {
    SpeakerCachePolicy policy;
    policy.num_speakers = json.at("num_speakers").get<std::int32_t>();
    policy.subsampling_factor = json.at("subsampling_factor").get<std::int32_t>();
    policy.hidden_size = json.at("hidden_size").get<std::int32_t>();
    policy.cache_length = json.at("speaker_cache_length").get<std::int32_t>();
    policy.silence_frames_per_speaker = json.at("silence_frames_per_speaker").get<std::int32_t>();
    policy.fifo_length = json.at("fifo_length").get<std::int32_t>();
    policy.update_period = json.at("speaker_cache_update_period").get<std::int32_t>();
    policy.score_threshold = json.at("prediction_score_threshold").get<float>();
    policy.latest_frames_boost = json.at("latest_frames_score_boost").get<float>();
    policy.min_positive_scores = json.at("min_positive_scores").get<std::int32_t>();
    policy.strong_boosted_frames = json.at("strong_boosted_frames").get<std::int32_t>();
    policy.weak_boosted_frames = json.at("weak_boosted_frames").get<std::int32_t>();
    return policy;
}

DiarizationConfig parse_config(const BundleReader& bundle) {
    const auto data = require_section(bundle, "runtime.json");
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    DiarizationConfig config;
    config.front_end = front_end(json, bundle);
    config.cache = cache_policy(json);
    config.chunk_length = json.at("chunk_length").get<std::int32_t>();
    config.chunk_left_context = json.at("chunk_left_context").get<std::int32_t>();
    config.chunk_right_context = json.at("chunk_right_context").get<std::int32_t>();
    config.max_step_rows = json.at("max_step_rows").get<std::int32_t>();
    config.labels = json.at("labels").get<std::vector<std::string>>();
    for (const auto& label : config.labels)
        if (label.empty())
            throw std::runtime_error("nemotron3_diarization runtime.json has an empty label");
    if (config.cache.hidden_size <= 0)
        throw std::runtime_error("nemotron3_diarization runtime.json encoder width is unusable");
    config.silence_embedding = float_section(bundle, "silence_embeds.f32",
                                             static_cast<std::size_t>(config.cache.hidden_size));
    return config;
}

std::unique_ptr<ITrtModule> load_engine(const FamilyContext& context, const char* section,
                                        std::initializer_list<const char*> inputs,
                                        const char* output) {
    const auto plan = require_section(context.reader, section);
    ModuleCreateOptions options{};
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("nemotron3_diarization engine failed to load: " +
                                 std::string(section));
    for (const auto* name : inputs)
        if (!engine->has_input(name))
            throw std::runtime_error("nemotron3_diarization engine lacks input " +
                                     std::string(name));
    if (!engine->has_output(output))
        throw std::runtime_error("nemotron3_diarization engine lacks output " +
                                 std::string(output));
    return engine;
}

} // namespace
} // namespace trtmc::nemotron3_diarization

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc::nemotron3_diarization;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("nemotron3_diarization does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::IAudioToTimedLabels::kTask)
        throw std::runtime_error("nemotron3_diarization bundle declares an unsupported task: " +
                                 task);
    auto config = parse_config(context.reader);
    auto embedder = load_engine(context, "embedder.plan", {"stacked_features"}, "inputs_embeds");
    auto encoder = load_engine(context, "encoder.plan", {"inputs_embeds", "key_bias"}, "logits");
    return new DiarizationPipeline(std::move(embedder), std::move(encoder), std::move(config));
}
