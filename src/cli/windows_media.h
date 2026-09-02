/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace trtmc::cli {

namespace detail {
// Overflow-safe decoded-frame allocation ceiling. Presentation timestamps and
// duration remain authoritative for the actual 15-second validity decision.
std::uint64_t reference_video_frame_ceiling(std::uint32_t fps_numerator,
                                            std::uint32_t fps_denominator);
std::pair<std::uint32_t, std::uint32_t> reference_video_decode_size(std::uint32_t source_width,
                                                                    std::uint32_t source_height);
bool reference_timeline_within_limit(std::int64_t timestamp, std::int64_t duration) noexcept;
bool reference_audio_event_timestamp_within_padding(std::int64_t timestamp,
                                                    std::uint64_t maximum_padding_ticks) noexcept;
struct ReferenceAudioDecodeState {
    std::uint64_t decoded_frames{0};
    std::uint64_t decoded_padding_frames{0};
};
// Account decoded PCM independently of container duration metadata. Compressed
// audio may decode at most maximum_padding_frames beyond the 15-second sample
// budget, and no more than that many decoded frames may fall outside [0, 15s).
bool account_reference_audio_decode(std::int64_t timestamp, std::uint64_t frame_count,
                                    std::uint32_t sample_rate, std::uint64_t maximum_padding_frames,
                                    ReferenceAudioDecodeState& state) noexcept;
// Reject a decoded Media Foundation sample/buffer that carries no payload.
// media_kind is a trusted internal label such as "audio" or "video".
void require_nonempty_decoded_buffer(std::uint32_t current_length, std::string_view media_kind);
} // namespace detail

bool is_mp4_path(std::string_view path);

// Write synchronized H.264 video and optional AAC audio using the Windows
// Media Foundation codecs shipped with the operating system. No FFmpeg or
// other runtime media dependency is involved.
void write_mp4(const VideoResult& result, const std::string& path);

// Decode a Windows-supported media container into the public THWC RGB video
// and interleaved float-audio value type used by Ref2VA. Decode fails closed
// as soon as the public 15-second reference limit is exceeded.
VideoClipInput read_video_file(const std::string& path);

// Decode the first audio stream in a Windows-supported media file (including
// MP3 and WAV) into interleaved float32 samples. The operating-system Media
// Foundation codecs are the only runtime dependency. Decode fails closed
// before appending samples beyond the public 15-second reference limit.
AudioResult read_audio_file(const std::string& path);

} // namespace trtmc::cli
