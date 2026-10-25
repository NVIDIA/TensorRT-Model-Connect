/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_continuation.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace trtmc::minimax_h3 {
namespace {

int32_t narrow(int64_t value) {
    if (value < 0 || value > std::numeric_limits<int32_t>::max())
        throw std::length_error("MiniMax-H3 continuation clock exceeds its int32 index range");
    return static_cast<int32_t>(value);
}

int32_t video_time(int32_t frames) {
    if (frames < 5 || frames % 17 != 5)
        throw std::invalid_argument("MiniMax-H3 continuation requires the 17n+5 frame grid");
    return narrow(int64_t{(frames - 5) / 17} * 5 + 2);
}

int32_t audio_time(int32_t frame) {
    // frame*5/3 has no half-integer ties. This is exactly nearest-integer
    // rounding of the global boundary, without per-window rounding drift.
    return narrow((int64_t{frame} * 5 + 1) / 3);
}

ReferenceContinuationWindow make_window(int32_t start, int32_t end, int32_t overlap) {
    if (start < 0 || end <= start || overlap < 0 || overlap >= end - start ||
        (start == 0 ? overlap != 0 : overlap == 0))
        throw std::invalid_argument("MiniMax-H3 continuation window has invalid boundaries");
    const auto video = video_time(end - start);
    return {start, end, overlap, video, audio_time(end) - audio_time(start),
        overlap ? video_time(overlap) : 0,
        overlap ? audio_time(start + overlap) - audio_time(start) : 0,
        start * (40.0 / 24.0)};
}

void validate_window(const ReferenceContinuationWindow& window) {
    const auto expected = make_window(window.start_frame, window.end_frame, window.overlap_frames);
    if (window.video_latent_frames != expected.video_latent_frames ||
        window.audio_latents != expected.audio_latents ||
        window.guide_video_latent_frames != expected.guide_video_latent_frames ||
        window.guide_audio_latents != expected.guide_audio_latents ||
        window.temporal_offset != expected.temporal_offset)
        throw std::invalid_argument("MiniMax-H3 continuation window clock was modified");
}

std::size_t product(std::initializer_list<std::size_t> factors) {
    std::size_t result = 1;
    for (const auto value : factors) {
        if (!value || result > std::numeric_limits<std::size_t>::max() / value)
            throw std::length_error("MiniMax-H3 continuation latent payload size overflow");
        result *= value;
    }
    return result;
}

std::size_t video_frame_values(ReferenceTargetLatentShape shape) {
    if (shape.frames <= 0 || shape.height <= 0 || shape.width <= 0 ||
        shape.height % 2 || shape.width % 2 || shape.audio_latents <= 0)
        throw std::invalid_argument("MiniMax-H3 continuation latent shape is invalid");
    return product({static_cast<std::size_t>(shape.height / 2),
                    static_cast<std::size_t>(shape.width / 2), 96});
}

void validate_payload(const ReferenceDenoiserResult& value) {
    const auto frame_values = video_frame_values(value.shape);
    if (value.video_rows.size() != product({frame_values, static_cast<std::size_t>(value.shape.frames)}) ||
        value.audio_rows.size() != product({2, 32, static_cast<std::size_t>(value.shape.audio_latents)}) ||
        value.forwards < 0 ||
        !std::all_of(value.video_rows.begin(), value.video_rows.end(), [](float x) { return std::isfinite(x); }) ||
        !std::all_of(value.audio_rows.begin(), value.audio_rows.end(), [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("MiniMax-H3 continuation requires complete finite latent payloads");
}

void validate_history(const ReferenceDenoiserResult& history,
                      const ReferenceContinuationWindow& window) {
    validate_window(window);
    if (!window.overlap_frames ||
        history.shape.frames != video_time(window.start_frame + window.overlap_frames) ||
        history.shape.audio_latents != audio_time(window.start_frame + window.overlap_frames))
        throw std::invalid_argument("MiniMax-H3 continuation parent does not end at the guide boundary");
    validate_payload(history);
}

} // namespace

std::vector<ReferenceContinuationWindow> plan_reference_continuation(
    int32_t total_frames, int32_t window_frames, int32_t overlap_frames, int32_t maximum_windows) {
    (void)video_time(total_frames);
    (void)video_time(window_frames);
    (void)video_time(overlap_frames);
    if (maximum_windows <= 0 || overlap_frames >= window_frames)
        throw std::invalid_argument("MiniMax-H3 continuation requires a finite window cap and forward progress");
    std::vector<ReferenceContinuationWindow> result;
    int32_t end = std::min(total_frames, window_frames);
    result.push_back(make_window(0, end, 0));
    while (end < total_frames) {
        if (result.size() >= static_cast<std::size_t>(maximum_windows))
            throw std::length_error("MiniMax-H3 continuation exceeds the caller's window cap");
        const int32_t start = end - overlap_frames;
        end = narrow(std::min(int64_t{total_frames}, int64_t{start} + window_frames));
        result.push_back(make_window(start, end, overlap_frames));
    }
    return result;
}

ReferenceLatentGuide reference_latent_guide(const ReferenceDenoiserResult& history,
                                           const ReferenceContinuationWindow& window) {
    validate_history(history, window);
    const auto video_values = static_cast<std::size_t>(window.guide_video_latent_frames) *
                              video_frame_values(history.shape);
    const auto audio_values = static_cast<std::size_t>(window.guide_audio_latents) * 32;
    const auto channel_values = static_cast<std::size_t>(history.shape.audio_latents) * 32;
    ReferenceLatentGuide result{{window.guide_video_latent_frames, history.shape.height,
        history.shape.width, window.guide_audio_latents}, {}, {}};
    result.video_rows.assign(history.video_rows.end() - static_cast<std::ptrdiff_t>(video_values),
                              history.video_rows.end());
    result.audio_rows.reserve(2 * audio_values);
    for (std::size_t channel = 0; channel < 2; ++channel) {
        const auto end = history.audio_rows.begin() + static_cast<std::ptrdiff_t>((channel + 1) * channel_values);
        result.audio_rows.insert(result.audio_rows.end(), end - static_cast<std::ptrdiff_t>(audio_values), end);
    }
    return result;
}

ReferenceDenoiserResult append_reference_window(const ReferenceDenoiserResult& history,
    const ReferenceDenoiserResult& generated, const ReferenceContinuationWindow& window) {
    validate_history(history, window);
    validate_payload(generated);
    if (generated.shape.frames != window.video_latent_frames ||
        generated.shape.audio_latents != window.audio_latents ||
        generated.shape.height != history.shape.height || generated.shape.width != history.shape.width)
        throw std::invalid_argument("MiniMax-H3 continuation generated window has incompatible geometry");
    ReferenceDenoiserResult result{{video_time(window.end_frame), history.shape.height,
        history.shape.width, audio_time(window.end_frame)}, {}, {},
        narrow(int64_t{history.forwards} + generated.forwards)};
    const auto frame_values = video_frame_values(history.shape);
    result.video_rows.reserve(product({static_cast<std::size_t>(result.shape.frames), frame_values}));
    result.video_rows = history.video_rows;
    result.video_rows.insert(result.video_rows.end(), generated.video_rows.begin() +
        static_cast<std::ptrdiff_t>(window.guide_video_latent_frames * frame_values), generated.video_rows.end());
    const auto before = static_cast<std::size_t>(history.shape.audio_latents) * 32;
    const auto incoming = static_cast<std::size_t>(generated.shape.audio_latents) * 32;
    const auto skip = static_cast<std::size_t>(window.guide_audio_latents) * 32;
    result.audio_rows.reserve(product({2, 32, static_cast<std::size_t>(result.shape.audio_latents)}));
    for (std::size_t channel = 0; channel < 2; ++channel) {
        result.audio_rows.insert(result.audio_rows.end(),
            history.audio_rows.begin() + static_cast<std::ptrdiff_t>(channel * before),
            history.audio_rows.begin() + static_cast<std::ptrdiff_t>((channel + 1) * before));
        result.audio_rows.insert(result.audio_rows.end(),
            generated.audio_rows.begin() + static_cast<std::ptrdiff_t>(channel * incoming + skip),
            generated.audio_rows.begin() + static_cast<std::ptrdiff_t>((channel + 1) * incoming));
    }
    validate_payload(result);
    return result;
}

} // namespace trtmc::minimax_h3
