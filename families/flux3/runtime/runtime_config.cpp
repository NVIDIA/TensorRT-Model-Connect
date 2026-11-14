/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/flux3/runtime/runtime_config.h"

#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::flux3 {
namespace {

using Json = nlohmann::json;

constexpr const char* kFormat = "flux3-runtime-v1";
constexpr std::int64_t kSpatialScale = 16;
constexpr std::int64_t kMaxImageSide = 16384;
constexpr std::int64_t kMaxLatentChannels = 65536;
constexpr std::int64_t kMaxSequenceLength = 8192; // Diffusers' max_sequence_length
constexpr std::int64_t kMaxEmbedDim = 1 << 20;
constexpr std::int64_t kMaxProjectionDim = 65536;
constexpr std::int64_t kTrainTimesteps = 1000;
constexpr std::int64_t kInt32Max = std::numeric_limits<std::int32_t>::max();
constexpr double kMaxShift = 1000.0;

[[noreturn]] void fail(const std::string& field, const std::string& requirement) {
    throw std::runtime_error("flux3 runtime.json: " + field + " " + requirement);
}

// One JSON object of the document and its dotted path for error messages.
struct Section {
    const Json& json;
    std::string path;

    std::string name(const char* key) const { return path.empty() ? key : path + "." + key; }

    const Json& at(const char* key) const {
        const auto found = json.find(key);
        if (found == json.end())
            fail(name(key), "is missing");
        return *found;
    }

    Section object(const char* key) const {
        const auto& value = at(key);
        if (!value.is_object())
            fail(name(key), "must be an object");
        return {value, name(key)};
    }

    std::string text(const char* key) const {
        const auto& value = at(key);
        if (!value.is_string())
            fail(name(key), "must be a string");
        return value.get<std::string>();
    }

    std::int32_t integer(const char* key, std::int64_t low, std::int64_t high) const {
        const auto& value = at(key);
        if (!value.is_number_integer())
            fail(name(key), "must be an integer");
        const auto range = "must be in [" + std::to_string(low) + ", " + std::to_string(high) + "]";
        if (value.is_number_unsigned() &&
            value.get<std::uint64_t>() > static_cast<std::uint64_t>(high))
            fail(name(key), range);
        const auto number = value.get<std::int64_t>();
        if (number < low || number > high)
            fail(name(key), range);
        return static_cast<std::int32_t>(number);
    }

    double number(const char* key) const {
        const auto& value = at(key);
        if (!value.is_number())
            fail(name(key), "must be a number");
        return value.get<double>();
    }
};

Json parse_document(std::string_view text) {
    Json document;
    try {
        document = Json::parse(text.begin(), text.end());
    } catch (const Json::exception& error) {
        throw std::runtime_error(std::string("flux3 runtime.json is not valid JSON: ") +
                                 error.what());
    }
    if (!document.is_object())
        fail("document", "must be a JSON object");
    return document;
}

void parse_geometry(const Section& root, RuntimeConfig& config) {
    config.height = root.integer("height", kSpatialScale, kMaxImageSide);
    config.width = root.integer("width", kSpatialScale, kMaxImageSide);
    if (config.height % kSpatialScale != 0 || config.width % kSpatialScale != 0)
        fail("height and width", "must be multiples of 16");
    config.latent_height = root.integer("latent_height", 1, kMaxImageSide);
    config.latent_width = root.integer("latent_width", 1, kMaxImageSide);
    if (config.latent_height != config.height / kSpatialScale ||
        config.latent_width != config.width / kSpatialScale)
        fail("latent_height and latent_width", "must equal height / 16 and width / 16");
    config.latent_channels = root.integer("latent_channels", 1, kMaxLatentChannels);
}

TextConfig parse_text(const Section& text) {
    TextConfig config;
    config.prompt_prefix = text.text("prompt_prefix");
    config.prompt_suffix = text.text("prompt_suffix");
    config.pad_token_id = text.integer("pad_token_id", 0, kInt32Max);
    config.pad_multiple = text.integer("pad_multiple", 1, kMaxSequenceLength);
    config.max_tokens = text.integer("max_tokens", config.pad_multiple, kMaxSequenceLength);
    if (config.max_tokens % config.pad_multiple != 0)
        fail("text.max_tokens", "must be a multiple of text.pad_multiple");
    config.embed_dim = text.integer("embed_dim", 1, kMaxEmbedDim);
    return config;
}

ScheduleConfig parse_schedule(const Section& schedule) {
    ScheduleConfig config;
    config.default_steps = schedule.integer("default_steps", 1, kMaxSteps);
    config.shift = schedule.number("shift");
    if (!std::isfinite(config.shift) || config.shift <= 0.0 || config.shift > kMaxShift)
        fail("schedule.shift", "must be in (0, 1000]");
    // The request flow scales timesteps by exactly 1000 in both directions.
    if (schedule.integer("num_train_timesteps", 1, kInt32Max) != kTrainTimesteps)
        fail("schedule.num_train_timesteps", "must be 1000");
    return config;
}

TimestepEmbeddingConfig parse_timestep_embedding(const Section& embedding) {
    TimestepEmbeddingConfig config;
    config.dim = embedding.integer("dim", 2, kMaxProjectionDim);
    if (config.dim % 2 != 0)
        fail("timestep_embedding.dim", "must be even");
    config.max_period = embedding.integer("max_period", 2, kInt32Max);
    return config;
}

} // namespace

RuntimeConfig parse_runtime_config(std::string_view text) {
    const auto document = parse_document(text);
    const Section root{document, ""};
    if (root.text("format") != kFormat)
        fail("format", std::string("must be \"") + kFormat + "\"");
    RuntimeConfig config;
    parse_geometry(root, config);
    config.text = parse_text(root.object("text"));
    config.schedule = parse_schedule(root.object("schedule"));
    config.timestep_embedding = parse_timestep_embedding(root.object("timestep_embedding"));
    return config;
}

} // namespace trtmc::flux3
