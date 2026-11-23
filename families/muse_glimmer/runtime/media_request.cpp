/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/muse_glimmer/runtime/media_request.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace trtmc::muse_glimmer {
using namespace internal;
Span<const ConfigField> media_fields() {
    static const std::vector<ConfigField> fields{
        {"max_new_tokens", ConfigKind::I64, int64_t{128}, "Maximum generated tokens"},
        {"temperature", ConfigKind::F64, 0.0, "Sampling temperature; zero is greedy"},
        {"top_k", ConfigKind::I64, int64_t{1}, "Top-k selection"},
        {"top_p", ConfigKind::F64, 1.0, "Nucleus sampling threshold"},
        {"seed", ConfigKind::I64, int64_t{-1}, "Sampling seed; -1 is unspecified"},
        {"enable_thinking", ConfigKind::Bool, true,
         "Allow private reasoning before the public answer"},
        {"use_chat_template", ConfigKind::Bool, true, "Render the Muse conversation template"},
        {"system_prompt", ConfigKind::String, std::string_view{}, "Optional system instruction"},
    };
    return {fields.data(), fields.size()};
}
TextGenerationConfig media_config(ConfigView supplied) {
    const auto fields = media_fields();
    validate_config(fields, supplied);
    TextGenerationConfig c;
    auto integer = [&](const char* key) {
        const auto n = *config_get<int64_t>(supplied, fields, key);
        if (n < std::numeric_limits<int32_t>::min() || n > std::numeric_limits<int32_t>::max())
            throw ConfigError(std::string(key) + " exceeds the supported integer range");
        return static_cast<int32_t>(n);
    };
    c.max_new_tokens = integer("max_new_tokens");
    c.top_k = integer("top_k");
    c.seed = integer("seed");
    c.temperature = *config_get<double>(supplied, fields, "temperature");
    c.top_p = *config_get<double>(supplied, fields, "top_p");
    c.enable_thinking = *config_get<bool>(supplied, fields, "enable_thinking");
    c.use_chat_template = *config_get<bool>(supplied, fields, "use_chat_template");
    c.system_prompt = *config_get<std::string_view>(supplied, fields, "system_prompt");
    if (c.max_new_tokens < 1 || c.top_k < 0 || c.seed < -1 || !std::isfinite(c.temperature) ||
        c.temperature < 0 || !std::isfinite(c.top_p) || c.top_p <= 0 || c.top_p > 1)
        throw ConfigError("Invalid Muse generation configuration");
    return c;
}

namespace {
void validate_video(const VideoView& v) {
    if (!v.frames.data() || !v.frames.size())
        throw std::invalid_argument("Muse video requires at least one frame");
    if (v.timestamps_seconds.size() &&
        (!v.timestamps_seconds.data() || v.timestamps_seconds.size() != v.frames.size()))
        throw std::invalid_argument("Muse video requires one timestamp per frame");
    for (std::size_t i = 0; i < v.frames.size(); ++i) {
        const auto& frame = v.frames.data()[i];
        if (!frame.data || !frame.width || !frame.height ||
            frame.width != v.frames.data()[0].width || frame.height != v.frames.data()[0].height)
            throw std::invalid_argument(
                "Muse video frames must have consistent positive dimensions");
        if (v.timestamps_seconds.size()) {
            double t = v.timestamps_seconds.data()[i];
            if (!std::isfinite(t) || t < 0 || (i > 0 && t <= v.timestamps_seconds.data()[i - 1]))
                throw std::invalid_argument("Muse video timestamps must be finite and increasing");
        }
    }
}
template <typename Request>
MediaTurns parse(const Request& r, const TextGenerationConfig& cfg) {
    if (r.tools.size() || !r.messages.data() || !r.messages.size())
        throw std::invalid_argument("Muse media requires messages without tool definitions");
    MediaTurns turns;
    if (!cfg.system_prompt.empty())
        turns.push_back({"system", {cfg.system_prompt}});
    for (std::size_t i = 0; i < r.messages.size(); ++i) {
        const auto& msg = r.messages.data()[i];
        const char* role = nullptr;
        switch (msg.role) {
        case MessageRole::System:
            if (!turns.empty())
                throw std::invalid_argument("Duplicate or misplaced Muse system message");
            role = "system";
            break;
        case MessageRole::User:
            role = "user";
            break;
        case MessageRole::Assistant:
            role = "assistant";
            break;
        default:
            throw std::invalid_argument("Unsupported Muse media message role");
        }
        if (!msg.parts.data() || !msg.parts.size())
            throw std::invalid_argument("Empty Muse media turn");
        MediaTurn t{role, {}};
        for (std::size_t j = 0; j < msg.parts.size(); ++j)
            std::visit(
                [&](const auto& part) {
                    using T = std::decay_t<decltype(part)>;
                    if constexpr (std::is_same_v<T, TextPartView>) {
                        const std::string text(part.text);
                        if (text.find("<|") != std::string::npos)
                            throw std::invalid_argument(
                                "Reserved Muse control sequence in typed message text");
                        t.parts.emplace_back(text);
                    } else if constexpr (std::is_same_v<T, ImageView> ||
                                         std::is_same_v<T, VideoView>) {
                        if (msg.role != MessageRole::User)
                            throw std::invalid_argument("Muse media must belong to a user turn");
                        if constexpr (std::is_same_v<T, VideoView>)
                            validate_video(part);
                        t.parts.emplace_back(part);
                    } else
                        throw std::invalid_argument(
                            "Muse media does not support tool or reasoning content parts");
                },
                msg.parts.data()[j]);
        turns.push_back(std::move(t));
    }
    if (turns.empty() || turns.back().role != "user")
        throw std::invalid_argument("Muse generation requires a final user turn");
    if (!cfg.use_chat_template && turns.size() != 1)
        throw std::invalid_argument("Raw Muse media accepts one user turn");
    return turns;
}
} // namespace
MediaTurns media_turns(const ImagesTextToTextRequest& r, const TextGenerationConfig& c) {
    return parse(r, c);
}
MediaTurns media_turns(const VideoTextToTextRequest& r, const TextGenerationConfig& c) {
    return parse(r, c);
}
double video_timestamp(const VideoView& v, std::size_t frame, double default_fps) {
    if (frame >= v.frames.size() || !(default_fps > 0) || !std::isfinite(default_fps))
        throw std::invalid_argument("Invalid Muse video frame time");
    return v.timestamps_seconds.size() ? v.timestamps_seconds.data()[frame]
                                       : static_cast<double>(frame) / default_fps;
}
} // namespace trtmc::muse_glimmer
