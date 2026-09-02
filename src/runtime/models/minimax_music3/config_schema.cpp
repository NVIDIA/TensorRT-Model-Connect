/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Registration for the "music_minimax_music3" namespace schema.
// Mirrors python/tensorrt_model_connect/families/minimax_music3/
// runtime_config_schema.py -- the field names, types, defaults and bounds are
// the same on both sides, and the Python module is the source of truth.

#include "runtime/models/minimax_music3/config_schema.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>

namespace trtmc::config::schemas {

namespace {

// Mirrors runtime_config_schema.MAX_CAPTION_CHARS / MAX_AUDIO_FRAMES.
constexpr std::size_t kMaxCaptionChars = 20000;
constexpr std::int32_t kMaxAudioFrames = 9000;

bool is_bounded_caption(const std::any& value) {
    if (value.type() != typeid(std::string))
        return false;
    return std::any_cast<const std::string&>(value).size() <= kMaxCaptionChars;
}

// The frame budget drives how many windows the denoiser runs, so a zero or a
// negative would silently produce no audio rather than fail.
bool is_frame_budget(const std::any& value) {
    if (value.type() == typeid(std::int32_t)) {
        const auto frames = std::any_cast<std::int32_t>(value);
        return frames >= 1 && frames <= kMaxAudioFrames;
    }
    if (value.type() == typeid(std::int64_t)) {
        const auto frames = std::any_cast<std::int64_t>(value);
        return frames >= 1 && frames <= static_cast<std::int64_t>(kMaxAudioFrames);
    }
    return false;
}

bool is_positive_int(const std::any& value) {
    return value.type() == typeid(std::int32_t) && std::any_cast<std::int32_t>(value) >= 1;
}

bool is_positive_float(const std::any& value) {
    return value.type() == typeid(float) && std::any_cast<float>(value) > 0.0F;
}

} // namespace

Schema make_music_minimax_music3_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "music_minimax_music3",
        {
            // Empty caption means unconditioned on a description; the lyrics
            // travel in the request prompt, not here.
            ConfigField{"caption", "string", std::any{std::string{}}, session, is_bounded_caption},
            // Spelled out rather than written as kMaxAudioFrames: the
            // documentation site compares this default with the Python one as
            // text, and every other family writes the literal here too.
            ConfigField{"max_frames", "int32", std::any{std::int32_t{9000}}, session,
                        is_frame_budget},
            // The checkpoint's own draw; see runtime_config_schema.py for why
            // these are not substituted for GenerateConfig's defaults.
            ConfigField{"top_k", "int32", std::any{std::int32_t{50}}, session, is_positive_int},
            ConfigField{"temperature", "float", std::any{1.0F}, session, is_positive_float},
            ConfigField{"seed", "int64", std::any{std::int64_t{-1}}, session, nullptr},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_minimax_music3_schema,
                                             make_music_minimax_music3_schema);
} // namespace trtmc::config::schemas
