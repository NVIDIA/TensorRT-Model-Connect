/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/windows_media.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <stdexcept>
#include <string>
#include <windows.h>
#include <wrl/client.h>

namespace {

using Microsoft::WRL::ComPtr;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void require_success(HRESULT result) {
    require(SUCCEEDED(result), "Media Foundation call failed");
}

} // namespace

int main() {
    trtmc::VideoResult result;
    result.frames.width = 64;
    result.frames.height = 64;
    result.frames.channels = 3;
    result.frames.num_frames = 24;
    result.fps = 24;
    const std::size_t pixel_count = static_cast<std::size_t>(result.frames.width) *
                                    result.frames.height * result.frames.channels *
                                    result.frames.num_frames;
    result.frames.pixels.resize(pixel_count);
    for (int frame = 0; frame < result.frames.num_frames; ++frame) {
        for (int row = 0; row < result.frames.height; ++row) {
            for (int column = 0; column < result.frames.width; ++column) {
                const auto offset =
                    ((static_cast<std::size_t>(frame) * result.frames.height + row) *
                         result.frames.width +
                     column) *
                    3;
                result.frames.pixels[offset] = static_cast<float>(column) / 63.0F;
                result.frames.pixels[offset + 1] = static_cast<float>(row) / 63.0F;
                result.frames.pixels[offset + 2] = static_cast<float>(frame) / 23.0F;
            }
        }
    }

    result.audio.sample_rate = 32000;
    result.audio.channels = 2;
    result.audio.samples.resize(static_cast<std::size_t>(result.audio.sample_rate) * 2);
    for (int frame = 0; frame < result.audio.sample_rate; ++frame) {
        const float value = 0.1F * std::sin(2.0F * 3.14159265358979323846F * 440.0F *
                                            static_cast<float>(frame) / result.audio.sample_rate);
        result.audio.samples[static_cast<std::size_t>(frame) * 2] = value;
        result.audio.samples[static_cast<std::size_t>(frame) * 2 + 1] = value;
    }
    result.audio.num_samples = static_cast<int32_t>(result.audio.samples.size());

    const auto path = std::filesystem::temp_directory_path() /
                      ("trtmc_windows_media_" + std::to_string(GetCurrentProcessId()) + ".mp4");
    std::error_code cleanup_error;
    std::filesystem::remove(path, cleanup_error);
    trtmc::cli::write_mp4(result, path.string());
    require(std::filesystem::is_regular_file(path), "MP4 output file is missing");
    require(std::filesystem::file_size(path) > 1024, "MP4 output file is empty");

    const auto decoded = trtmc::cli::read_video_file(path.string());
    require(decoded.width == result.frames.width, "decoded MP4 width mismatch");
    require(decoded.height == result.frames.height, "decoded MP4 height mismatch");
    require(decoded.channels == 3, "decoded MP4 channel mismatch");
    require(decoded.num_frames == result.frames.num_frames, "decoded MP4 frame-count mismatch");
    require(decoded.fps_numerator == result.fps, "decoded MP4 frame-rate mismatch");
    require(decoded.fps_denominator == 1, "decoded MP4 frame-rate denominator mismatch");
    require(decoded.pixels.size() == result.frames.pixels.size(),
            "decoded MP4 pixel-count mismatch");
    require(decoded.soundtrack.sample_rate == result.audio.sample_rate,
            "decoded MP4 audio-rate mismatch");
    require(decoded.soundtrack.channels == result.audio.channels,
            "decoded MP4 audio-channel mismatch");
    require(!decoded.soundtrack.samples.empty(), "decoded MP4 has no soundtrack samples");

    const auto decoded_pixel = [&](int frame, int row, int column, int channel) {
        return decoded
            .pixels[((static_cast<std::size_t>(frame) * decoded.height + row) * decoded.width +
                     column) *
                        3 +
                    channel];
    };
    require(decoded_pixel(0, 32, 56, 0) > decoded_pixel(0, 32, 8, 0),
            "decoded MP4 red axis is reversed");
    require(decoded_pixel(0, 56, 32, 1) > decoded_pixel(0, 8, 32, 1),
            "decoded MP4 green axis is reversed");
    require(decoded_pixel(20, 32, 32, 2) > decoded_pixel(3, 32, 32, 2),
            "decoded MP4 frame order is reversed");

    const HRESULT com_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool owns_com = SUCCEEDED(com_result);
    require(owns_com || com_result == RPC_E_CHANGED_MODE, "CoInitializeEx failed");
    require_success(MFStartup(MF_VERSION, MFSTARTUP_FULL));
    ComPtr<IMFSourceReader> reader;
    require_success(MFCreateSourceReaderFromURL(path.wstring().c_str(), nullptr, &reader));

    ComPtr<IMFMediaType> video_type;
    require_success(
        reader->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &video_type));
    GUID video_subtype{};
    require_success(video_type->GetGUID(MF_MT_SUBTYPE, &video_subtype));
    require(video_subtype == MFVideoFormat_H264, "MP4 video track is not H.264");

    ComPtr<IMFMediaType> audio_type;
    require_success(
        reader->GetNativeMediaType(MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, &audio_type));
    GUID audio_subtype{};
    require_success(audio_type->GetGUID(MF_MT_SUBTYPE, &audio_subtype));
    require(audio_subtype == MFAudioFormat_AAC, "MP4 audio track is not AAC");

    reader.Reset();
    require_success(MFShutdown());
    if (owns_com)
        CoUninitialize();
    std::filesystem::remove(path, cleanup_error);
    require(!cleanup_error, "failed to remove temporary MP4 test artifact");
    return 0;
}
