/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron3_diarization/runtime/bundle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace trtmc::nemotron3_diarization {
namespace {

using Json = nlohmann::json;

[[noreturn]] void unusable(const std::string& key, const std::string& why) {
    throw std::runtime_error("nemotron3_diarization runtime.json: " + key + " " + why);
}

// The parser keeps the last of repeated keys; the callback refuses them.
Json parse_strict(const std::vector<char>& text) {
    std::vector<std::set<std::string>> keys;
    const Json::parser_callback_t callback = [&keys](int, Json::parse_event_t event, Json& parsed) {
        if (event == Json::parse_event_t::object_start) {
            keys.emplace_back();
        } else if (event == Json::parse_event_t::object_end) {
            keys.pop_back();
        } else if (event == Json::parse_event_t::key) {
            const auto key = parsed.get<std::string>();
            if (!keys.back().insert(key).second)
                unusable(key, "appears twice");
        }
        return true;
    };
    return Json::parse(text.begin(), text.end(), callback);
}

const Json& member(const Json& json, const char* key) {
    const auto found = json.find(key);
    if (found == json.end())
        unusable(key, "is missing");
    return *found;
}

std::int32_t integer(const Json& json, const char* key, std::int64_t minimum) {
    const auto& value = member(json, key);
    if (!value.is_number_integer())
        unusable(key, "must be a JSON integer");
    constexpr auto top = std::numeric_limits<std::int32_t>::max();
    if (value.is_number_unsigned() ? value.get<std::uint64_t>() > static_cast<std::uint64_t>(top)
                                   : value.get<std::int64_t>() > top)
        unusable(key, "exceeds " + std::to_string(top));
    const auto number = value.get<std::int64_t>();
    if (number < minimum)
        unusable(key, "is below " + std::to_string(minimum));
    return static_cast<std::int32_t>(number);
}

float real(const Json& json, const char* key) {
    const auto& value = member(json, key);
    if (!value.is_number())
        unusable(key, "must be a JSON number");
    const double number = value.get<double>();
    if (!std::isfinite(number) || std::fabs(number) > std::numeric_limits<float>::max())
        unusable(key, "is not a finite float32");
    return static_cast<float>(number);
}

// The step's rows are reported as dynamic (-1) or as the rows the backend's
// buffers hold now; the profile check and each output's exact shape at run time
// pin them. Every other dimension must be the expected one.
bool same_shape(const std::vector<std::int64_t>& actual,
                const std::vector<std::int64_t>& expected) {
    if (actual.size() != expected.size())
        return false;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const bool usable =
            expected[i] == kStepRows ? actual[i] == -1 || actual[i] > 0 : actual[i] == expected[i];
        if (!usable)
            return false;
    }
    return true;
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i)
        text += (i ? ", " : "") + std::to_string(shape[i]);
    return text + "]";
}

} // namespace

DiarizationConfig parse_runtime_json(const std::vector<char>& text) {
    const auto json = parse_strict(text);
    if (!json.is_object())
        unusable("document", "must be a JSON object");
    DiarizationConfig config;
    auto& front = config.front_end;
    front.sample_rate = integer(json, "sample_rate", 1);
    front.n_fft = integer(json, "n_fft", 2);
    front.win_length = integer(json, "win_length", 2);
    front.hop_length = integer(json, "hop_length", 1);
    front.num_mel_bins = integer(json, "num_mel_bins", 1);
    front.preemphasis = real(json, "preemphasis");
    front.log_guard = real(json, "log_guard");
    auto& cache = config.cache;
    cache.num_speakers = integer(json, "num_speakers", 1);
    cache.subsampling_factor = integer(json, "subsampling_factor", 1);
    cache.hidden_size = integer(json, "hidden_size", 1);
    cache.cache_length = integer(json, "speaker_cache_length", 1);
    cache.silence_frames_per_speaker = integer(json, "silence_frames_per_speaker", 0);
    cache.fifo_length = integer(json, "fifo_length", 0);
    cache.update_period = integer(json, "speaker_cache_update_period", 1);
    cache.score_threshold = real(json, "prediction_score_threshold");
    cache.latest_frames_boost = real(json, "latest_frames_score_boost");
    cache.min_positive_scores = integer(json, "min_positive_scores", 0);
    cache.strong_boosted_frames = integer(json, "strong_boosted_frames", 0);
    cache.weak_boosted_frames = integer(json, "weak_boosted_frames", 0);
    config.chunk_length = integer(json, "chunk_length", 1);
    config.chunk_left_context = integer(json, "chunk_left_context", 0);
    config.chunk_right_context = integer(json, "chunk_right_context", 0);
    config.max_step_rows = integer(json, "max_step_rows", 1);
    const auto& labels = member(json, "labels");
    if (!labels.is_array())
        unusable("labels", "must be an array of strings");
    std::set<std::string> seen;
    for (const auto& label : labels) {
        if (!label.is_string() || label.get_ref<const std::string&>().empty())
            unusable("labels", "must name every speaker with a nonempty string");
        if (!seen.insert(label.get<std::string>()).second)
            unusable("labels", "repeat a name");
        config.labels.push_back(label.get<std::string>());
    }
    return config;
}

void check_float_section(std::uint64_t length, std::size_t count, const std::string& name) {
    constexpr auto width = static_cast<std::uint64_t>(sizeof(float));
    if (static_cast<std::uint64_t>(count) > std::numeric_limits<std::uint64_t>::max() / width ||
        length != static_cast<std::uint64_t>(count) * width)
        throw std::runtime_error("nemotron3_diarization bundle section " + name + " has " +
                                 std::to_string(length) + " bytes, not " + std::to_string(count) +
                                 " float32 values");
}

void check_section_limit(std::uint64_t length, std::uint64_t limit, const std::string& name) {
    if (length > limit)
        throw std::runtime_error("nemotron3_diarization bundle section " + name + " has " +
                                 std::to_string(length) + " bytes, more than " +
                                 std::to_string(limit));
}

void check_engine_io(const EngineIo& io, const std::vector<TensorSpec>& inputs,
                     const std::vector<TensorSpec>& outputs, std::int64_t max_step_rows,
                     const std::string& engine) {
    const auto fail = [&engine](const std::string& why) {
        throw std::runtime_error("nemotron3_diarization " + engine + " " + why);
    };
    for (const bool input : {true, false}) {
        const auto& specs = input ? inputs : outputs;
        const std::string kind = input ? "input" : "output";
        const auto count =
            std::count_if(io.tensors.begin(), io.tensors.end(),
                          [input](const TensorInfo& tensor) { return tensor.is_input == input; });
        if (static_cast<std::size_t>(count) != specs.size())
            fail("has " + std::to_string(count) + " " + kind + "s, not " +
                 std::to_string(specs.size()));
        for (const auto& spec : specs) {
            const auto found =
                std::find_if(io.tensors.begin(), io.tensors.end(), [&](const TensorInfo& tensor) {
                    return tensor.is_input == input && tensor.name == spec.name;
                });
            if (found == io.tensors.end())
                fail("lacks the " + kind + " " + spec.name);
            if (found->dtype != DType::kFloat32)
                fail(kind + " " + spec.name + " is not float32");
            if (!same_shape(found->shape, spec.shape))
                fail(kind + " " + spec.name + " has the shape " + shape_text(found->shape) +
                     ", not " + shape_text(spec.shape));
        }
    }
    if (io.profiles < 1)
        fail("has no optimization profile");
    for (const auto& spec : inputs) {
        const auto found = std::find_if(io.input_rows.begin(), io.input_rows.end(),
                                        [&](const auto& rows) { return rows.first == spec.name; });
        if (found == io.input_rows.end() || found->second.first > 1 ||
            found->second.second < max_step_rows)
            fail("input " + spec.name + " does not accept 1 to " + std::to_string(max_step_rows) +
                 " rows in profile 0");
    }
}

} // namespace trtmc::nemotron3_diarization
