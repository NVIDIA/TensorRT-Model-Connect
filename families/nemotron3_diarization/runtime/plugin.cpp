/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron3_diarization/runtime/bundle.h"
#include "families/nemotron3_diarization/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::nemotron3_diarization {
namespace {

// runtime.json holds a few hundred bytes of geometry and policy.
constexpr std::uint64_t kRuntimeJsonLimit = 1 << 20;

const BundleSectionInfo& section_info(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("nemotron3_diarization bundle section is missing or empty: " +
                                 std::string(name));
    return *section;
}

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    (void)section_info(bundle, name);
    return bundle.read_section(name);
}

std::vector<float> float_section(const BundleReader& bundle, const char* name,
                                 std::size_t expected) {
    // The section table's length is checked before the section is read.
    check_float_section(section_info(bundle, name).length, expected, name);
    const auto raw = bundle.read_section(name);
    if (raw.size() != expected * sizeof(float))
        throw std::runtime_error("nemotron3_diarization bundle section has the wrong length: " +
                                 std::string(name));
    std::vector<float> values(expected);
    std::memcpy(values.data(), raw.data(), raw.size());
    for (const float value : values)
        if (!std::isfinite(value))
            throw std::runtime_error("nemotron3_diarization bundle section is not finite: " +
                                     std::string(name));
    return values;
}

DiarizationConfig load_config(const BundleReader& bundle) {
    check_section_limit(section_info(bundle, "runtime.json").length, kRuntimeJsonLimit,
                        "runtime.json");
    auto config = parse_runtime_json(bundle.read_section("runtime.json"));
    auto& front = config.front_end;
    front.window = float_section(bundle, "window.f32", static_cast<std::size_t>(front.win_length));
    front.mel_filters = float_section(bundle, "mel_filters.f32",
                                      static_cast<std::size_t>(front.num_mel_bins) *
                                          (static_cast<std::size_t>(front.n_fft) / 2 + 1));
    config.silence_embedding = float_section(bundle, "silence_embeds.f32",
                                             static_cast<std::size_t>(config.cache.hidden_size));
    return config;
}

EngineIo describe(const ITrtModule& engine) {
    EngineIo io;
    const auto inputs = engine.input_info();
    for (auto tensor : inputs) {
        tensor.is_input = true;
        io.tensors.push_back(std::move(tensor));
    }
    for (auto tensor : engine.output_info()) {
        tensor.is_input = false;
        io.tensors.push_back(std::move(tensor));
    }
    io.profiles = engine.optimization_profile_count();
    if (io.profiles < 1)
        return io;
    for (const auto& tensor : inputs) {
        const auto low = engine.input_profile_shape(tensor.name, 0, ProfileShapeSelector::kMin);
        const auto high = engine.input_profile_shape(tensor.name, 0, ProfileShapeSelector::kMax);
        if (!low.empty() && !high.empty())
            io.input_rows.push_back({tensor.name, {low.front(), high.front()}});
    }
    return io;
}

std::unique_ptr<ITrtModule> load_engine(const FamilyContext& context, const char* section,
                                        const std::vector<TensorSpec>& inputs,
                                        const std::vector<TensorSpec>& outputs,
                                        std::int64_t max_step_rows) {
    const auto plan = require_section(context.reader, section);
    ModuleCreateOptions options{};
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("nemotron3_diarization engine failed to load: " +
                                 std::string(section));
    check_engine_io(describe(*engine), inputs, outputs, max_step_rows, section);
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
    auto config = load_config(context.reader);
    const std::int64_t width = config.cache.hidden_size;
    const std::int64_t stacked =
        static_cast<std::int64_t>(config.cache.subsampling_factor) * config.front_end.num_mel_bins;
    const std::int64_t speakers = config.cache.num_speakers;
    auto embedder =
        load_engine(context, "embedder.plan", {{"stacked_features", {kStepRows, stacked}}},
                    {{"inputs_embeds", {kStepRows, width}}}, config.max_step_rows);
    auto encoder = load_engine(context, "encoder.plan",
                               {{"inputs_embeds", {kStepRows, width}}, {"key_bias", {kStepRows}}},
                               {{"logits", {kStepRows, speakers}}}, config.max_step_rows);
    return new DiarizationPipeline(std::move(embedder), std::move(encoder), std::move(config));
}
