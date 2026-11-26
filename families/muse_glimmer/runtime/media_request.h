/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/language.h"

#include <string>
#include <variant>
#include <vector>

namespace trtmc::muse_glimmer {
using MediaPart = std::variant<std::string, internal::ImageView, internal::VideoView>;
struct MediaTurn {
    std::string role;
    std::vector<MediaPart> parts;
};
using MediaTurns = std::vector<MediaTurn>;

Span<const internal::ConfigField> media_fields();
TextGenerationConfig media_config(internal::ConfigView);
MediaTurns media_turns(const internal::ImagesTextToTextRequest&, const TextGenerationConfig&);
MediaTurns media_turns(const internal::VideoTextToTextRequest&, const TextGenerationConfig&);
double video_timestamp(const internal::VideoView&, std::size_t frame, double default_fps);
} // namespace trtmc::muse_glimmer
