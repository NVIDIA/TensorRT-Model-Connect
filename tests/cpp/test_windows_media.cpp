/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/windows_media.h"
#include "test_helpers.h"
#include "trtmc/trtmc_io.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
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

template <typename Fn>
void require_throws_with(Fn&& fn, const char* needle, const char* message) {
    try {
        fn();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(needle) != std::string::npos, message);
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int run_test() {
    constexpr trtmc::ReferenceMediaDecodePolicy policy{
        15,
        24,
        240,
        768,
        768ULL * 1344,
        32,
        0.25,
        4.0,
    };
    constexpr trtmc::ReferenceMediaDecodePolicy compact_policy{
        2,
        12,
        60,
        256,
        256ULL * 448,
        16,
        0.5,
        2.0,
    };
    require(trtmc::cli::detail::reference_video_frame_ceiling(compact_policy, 30'000, 1'001) ==
                60,
            "frame ceiling must follow a non-default duration policy");
    require(trtmc::cli::detail::reference_video_decode_size(compact_policy, 3840, 2160) ==
                std::pair<std::uint32_t, std::uint32_t>{448, 256},
            "decode canvas must follow non-default size and alignment policy values");
    require_throws_with(
        [&] {
            (void)trtmc::cli::detail::reference_video_decode_size(compact_policy, 2560, 1080);
        },
        "configured range", "decode aspect bounds must follow the supplied policy");
    require(trtmc::cli::detail::reference_video_frame_ceiling(policy, 30'000, 1'001) == 450,
            "configured video ceiling must round up fractional frame rates");
    require(trtmc::cli::detail::reference_video_decode_size(policy, 3840, 2160) ==
                std::pair<std::uint32_t, std::uint32_t>{1344, 768},
            "4K reference video must decode onto the policy's bounded canvas");
    require(trtmc::cli::detail::reference_timeline_within_limit(policy, 0, 150'000'000),
            "an exact configured presentation timeline must be accepted");
    require(!trtmc::cli::detail::reference_timeline_within_limit(policy, 150'000'000, 1),
            "a non-empty sample starting at the duration limit must be rejected");
    require(
        !trtmc::cli::detail::reference_timeline_within_limit(policy, 149'000'000, 2'000'000),
        "a sparse sample crossing the duration limit must be rejected");

    constexpr std::uint32_t synthetic_codec_rate = 32'000;
    constexpr std::uint64_t synthetic_mp3_access_unit_frames = 1'152;
    constexpr std::uint64_t synthetic_mp3_padding_frames = 3 * synthetic_mp3_access_unit_frames;
    constexpr std::uint64_t synthetic_mp3_padding_ticks =
        (synthetic_mp3_padding_frames * 10'000'000 + synthetic_codec_rate - 1) /
        synthetic_codec_rate;
    constexpr std::uint64_t synthetic_public_audio_frames = 15 * synthetic_codec_rate;
    require(
        trtmc::cli::detail::reference_audio_event_timestamp_within_padding(
            policy, -static_cast<std::int64_t>(synthetic_mp3_padding_ticks),
            synthetic_mp3_padding_ticks) &&
            trtmc::cli::detail::reference_audio_event_timestamp_within_padding(
                policy, 150'000'000 + static_cast<std::int64_t>(synthetic_mp3_padding_ticks),
                synthetic_mp3_padding_ticks),
        "compressed empty-event timestamps must accept the exact codec-padding window");
    require(!trtmc::cli::detail::reference_audio_event_timestamp_within_padding(
                policy, -36'000'000'000, synthetic_mp3_padding_ticks) &&
                !trtmc::cli::detail::reference_audio_event_timestamp_within_padding(
                    policy, 36'000'000'000, synthetic_mp3_padding_ticks),
            "compressed empty-event timestamps at negative or positive hours must be rejected");
    trtmc::cli::detail::ReferenceAudioDecodeState boundary_padding_state;
    require(trtmc::cli::detail::account_reference_audio_decode(
                policy, 0, synthetic_public_audio_frames + synthetic_mp3_padding_frames,
                synthetic_codec_rate, synthetic_mp3_padding_frames, boundary_padding_state),
            "an exact three-access-unit MP3 decoded tail must be accepted");
    require(boundary_padding_state.decoded_padding_frames == synthetic_mp3_padding_frames,
            "synthetic MP3 tail accounting must reach the codec-padding boundary");
    require(!trtmc::cli::detail::account_reference_audio_decode(
                policy, 150'000'001, 1, synthetic_codec_rate, synthetic_mp3_padding_frames,
                boundary_padding_state),
            "one decoded frame beyond the MP3 padding boundary must be rejected");

    trtmc::cli::detail::ReferenceAudioDecodeState falsified_duration_state;
    require(!trtmc::cli::detail::account_reference_audio_decode(
                policy, 150'000'000, synthetic_mp3_padding_frames + 1, synthetic_codec_rate,
                synthetic_mp3_padding_frames, falsified_duration_state),
            "a short or falsified presentation duration must not hide a decoded tail over three "
            "MP3 access units");

    trtmc::cli::detail::ReferenceAudioDecodeState future_timestamp_state;
    require(!trtmc::cli::detail::account_reference_audio_decode(
                policy, 36'000'000'000, 1, synthetic_codec_rate, synthetic_mp3_padding_frames,
                future_timestamp_state),
            "a tiny decoded sample with a far-future timestamp must not spend only one padding "
            "frame");
    trtmc::cli::detail::ReferenceAudioDecodeState excessive_leading_state;
    require(!trtmc::cli::detail::account_reference_audio_decode(
                policy, -130'000'000, synthetic_public_audio_frames, synthetic_codec_rate,
                synthetic_mp3_padding_frames, excessive_leading_state),
            "excessive negative-timestamp decoded PCM must not be hidden as leading padding");

    constexpr std::uint64_t synthetic_aac_padding_frames = 3 * 1'024;
    trtmc::cli::detail::ReferenceAudioDecodeState aac_padding_state;
    require(trtmc::cli::detail::account_reference_audio_decode(
                policy, 0, synthetic_public_audio_frames + synthetic_aac_padding_frames,
                synthetic_codec_rate, synthetic_aac_padding_frames, aac_padding_state),
            "an exact three-access-unit AAC decoded tail must be accepted");
    require(
        !trtmc::cli::detail::account_reference_audio_decode(
            policy, 150'000'001, 1, synthetic_codec_rate, synthetic_aac_padding_frames,
            aac_padding_state),
        "one decoded frame beyond the AAC padding boundary must be rejected");

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

    trtmc_test::TempDirGuard temporary;
    const auto temporary_root = std::filesystem::path(temporary.path());
    const auto path = temporary_root / "video.mp4";
    const auto wav_path = temporary_root / "audio.wav";
    const auto mp3_path = temporary_root / "audio.mp3";
    const auto aac_path = temporary_root / "audio.m4a";
    const auto mp3_44k_path = temporary_root / "audio-44k.mp3";
    const auto aac_44k_path = temporary_root / "audio-44k.m4a";
    const auto over_limit_mp3_path = temporary_root / "over-limit.mp3";
    const auto boundary_wav_path = temporary_root / "boundary.wav";
    const auto over_limit_wav_path = temporary_root / "over-limit.wav";
    const auto over_limit_video_path = temporary_root / "over-limit.mp4";
    trtmc::cli::write_mp4(result, path.string());
    require(std::filesystem::is_regular_file(path), "MP4 output file is missing");
    require(std::filesystem::file_size(path) > 1024, "MP4 output file is empty");

    const auto decoded = trtmc::cli::read_video_file(path.string(), policy);
    require(decoded.width == 768, "decoded MP4 width must use the configured resolver");
    require(decoded.height == 768, "decoded MP4 height must use the configured resolver");
    require(decoded.channels == 3, "decoded MP4 channel mismatch");
    require(decoded.num_frames == result.frames.num_frames, "decoded MP4 frame-count mismatch");
    require(decoded.fps_numerator == result.fps, "decoded MP4 frame-rate mismatch");
    require(decoded.fps_denominator == 1, "decoded MP4 frame-rate denominator mismatch");
    require(decoded.pixels.size() == static_cast<std::size_t>(decoded.width) * decoded.height *
                                         decoded.num_frames * decoded.channels,
            "decoded MP4 pixel-count mismatch");
    require(decoded.soundtrack.sample_rate == result.audio.sample_rate,
            "decoded MP4 audio-rate mismatch");
    require(decoded.soundtrack.channels == result.audio.channels,
            "decoded MP4 audio-channel mismatch");
    require(!decoded.soundtrack.samples.empty(), "decoded MP4 has no soundtrack samples");

    const auto compact_decoded = trtmc::cli::read_video_file(path.string(), compact_policy);
    require(compact_decoded.width == 256 && compact_decoded.height == 256,
            "decoded MP4 canvas must follow a second policy");
    require(compact_decoded.num_frames == 12 && compact_decoded.fps_numerator == 12 &&
                compact_decoded.fps_denominator == 1,
            "decoded MP4 sampling rate must follow a second policy");

    const auto extracted_audio = trtmc::cli::read_audio_file(path.string(), policy);
    require(extracted_audio.sample_rate == result.audio.sample_rate,
            "standalone media audio-rate mismatch");
    require(extracted_audio.channels == result.audio.channels,
            "standalone media audio-channel mismatch");
    require(!extracted_audio.samples.empty(), "standalone media audio decode is empty");

    trtmc::io::write_wav(result.audio, wav_path.string());
    const auto decoded_wav = trtmc::cli::read_audio_file(wav_path.string(), policy);
    require(decoded_wav.sample_rate == result.audio.sample_rate,
            "Media Foundation WAV audio-rate mismatch");
    require(decoded_wav.channels == result.audio.channels,
            "Media Foundation WAV audio-channel mismatch");
    require(decoded_wav.num_samples == static_cast<int32_t>(decoded_wav.samples.size()) &&
                !decoded_wav.samples.empty(),
            "Media Foundation WAV decode returned invalid samples");

    trtmc::AudioResult boundary_audio;
    boundary_audio.sample_rate = 8000;
    boundary_audio.channels = 1;
    boundary_audio.samples.resize(static_cast<std::size_t>(boundary_audio.sample_rate) * 15);
    boundary_audio.num_samples = static_cast<int32_t>(boundary_audio.samples.size());
    trtmc::io::write_wav(boundary_audio, boundary_wav_path.string());
    const auto decoded_boundary_audio =
        trtmc::cli::read_audio_file(boundary_wav_path.string(), policy);
    require(decoded_boundary_audio.samples.size() == boundary_audio.samples.size(),
            "configured reference audio boundary must decode successfully");

    boundary_audio.samples.push_back(0.0F);
    boundary_audio.num_samples = static_cast<int32_t>(boundary_audio.samples.size());
    trtmc::io::write_wav(boundary_audio, over_limit_wav_path.string());
    require_throws_with(
        [&] { (void)trtmc::cli::read_audio_file(over_limit_wav_path.string(), policy); },
        "configured duration limit",
        "reference audio over the policy limit must fail during Media Foundation decode");

    trtmc::VideoResult over_limit_video;
    over_limit_video.frames.width = 64;
    over_limit_video.frames.height = 64;
    over_limit_video.frames.channels = 3;
    over_limit_video.frames.num_frames = 361;
    over_limit_video.fps = 24;
    over_limit_video.frames.pixels.resize(
        static_cast<std::size_t>(over_limit_video.frames.width) * over_limit_video.frames.height *
        over_limit_video.frames.channels * over_limit_video.frames.num_frames);
    trtmc::cli::write_mp4(over_limit_video, over_limit_video_path.string());
    require_throws_with(
        [&] { (void)trtmc::cli::read_video_file(over_limit_video_path.string(), policy); },
        "configured duration limit",
        "reference video over the policy limit must fail during Media Foundation decode");

    const auto decoded_pixel = [&](int frame, int row, int column, int channel) {
        return decoded
            .pixels[((static_cast<std::size_t>(frame) * decoded.height + row) * decoded.width +
                     column) *
                        3 +
                    channel];
    };
    require(decoded_pixel(0, 384, 672, 0) > decoded_pixel(0, 384, 96, 0),
            "decoded MP4 red axis is reversed");
    require(decoded_pixel(0, 672, 384, 1) > decoded_pixel(0, 96, 384, 1),
            "decoded MP4 green axis is reversed");
    require(decoded_pixel(20, 384, 384, 2) > decoded_pixel(3, 384, 384, 2),
            "decoded MP4 frame order is reversed");

    const HRESULT com_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool owns_com = SUCCEEDED(com_result);
    require(owns_com || com_result == RPC_E_CHANGED_MODE, "CoInitializeEx failed");
    require_success(MFStartup(MF_VERSION, MFSTARTUP_FULL));

    ComPtr<IMFMediaBuffer> empty_audio_buffer;
    require_success(MFCreateMemoryBuffer(sizeof(float), &empty_audio_buffer));
    DWORD empty_audio_length = 1;
    require_success(empty_audio_buffer->GetCurrentLength(&empty_audio_length));
    require(empty_audio_length == 0,
            "synthetic Media Foundation audio buffer must start with zero current length");
    ComPtr<IMFSample> empty_audio_sample;
    require_success(MFCreateSample(&empty_audio_sample));
    require_success(empty_audio_sample->AddBuffer(empty_audio_buffer.Get()));
    DWORD empty_audio_sample_length = 1;
    require_success(empty_audio_sample->GetTotalLength(&empty_audio_sample_length));
    require(empty_audio_sample_length == 0,
            "synthetic non-null audio sample must contain no payload");
    require_throws_with(
        [&] {
            trtmc::cli::detail::require_nonempty_decoded_buffer(empty_audio_sample_length, "audio");
        },
        "empty decoded audio buffer",
        "a non-null sample with a zero-length audio buffer must fail closed");

    ComPtr<IMFSample> empty_video_sample;
    require_success(MFCreateSample(&empty_video_sample));
    ComPtr<IMFMediaBuffer> empty_video_buffer;
    require_success(MFCreateMemoryBuffer(4, &empty_video_buffer));
    require_success(empty_video_sample->AddBuffer(empty_video_buffer.Get()));
    DWORD empty_video_length = 1;
    require_success(empty_video_sample->GetTotalLength(&empty_video_length));
    require(empty_video_length == 0,
            "synthetic Media Foundation video sample must have zero total length");
    require_throws_with(
        [&] { trtmc::cli::detail::require_nonempty_decoded_buffer(empty_video_length, "video"); },
        "empty decoded video buffer",
        "a non-null sample with a zero-length video buffer must fail closed");
    trtmc::cli::detail::require_nonempty_decoded_buffer(1, "audio");

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

    constexpr int kCompressedBoundarySeconds = 15;
    const std::size_t boundary_frames =
        static_cast<std::size_t>(result.audio.sample_rate) * kCompressedBoundarySeconds;
    std::vector<std::int16_t> pcm(boundary_frames * result.audio.channels);
    for (std::size_t frame = 0; frame < boundary_frames; ++frame) {
        const float value = 0.1F * std::sin(2.0F * 3.14159265358979323846F * 440.0F *
                                            static_cast<float>(frame) / result.audio.sample_rate);
        const auto quantized = static_cast<std::int16_t>(std::lround(value * 32767.0F));
        pcm[frame * 2] = quantized;
        pcm[frame * 2 + 1] = quantized;
    }
    constexpr std::uint32_t boundary_44k_rate = 44'100;
    const std::size_t boundary_44k_frames =
        static_cast<std::size_t>(boundary_44k_rate) * kCompressedBoundarySeconds;
    std::vector<std::int16_t> pcm_44k(boundary_44k_frames * result.audio.channels);
    for (std::size_t frame = 0; frame < boundary_44k_frames; ++frame) {
        const float value = 0.1F * std::sin(2.0F * 3.14159265358979323846F * 440.0F *
                                            static_cast<float>(frame) / boundary_44k_rate);
        const auto quantized = static_cast<std::int16_t>(std::lround(value * 32767.0F));
        pcm_44k[frame * 2] = quantized;
        pcm_44k[frame * 2 + 1] = quantized;
    }
    const auto write_compressed_boundary = [&](const std::filesystem::path& output_path,
                                               const GUID& subtype,
                                               const std::vector<std::int16_t>& input_pcm,
                                               std::uint32_t sample_rate, LONGLONG duration) {
        ComPtr<IMFSinkWriter> writer;
        require_success(
            MFCreateSinkWriterFromURL(output_path.wstring().c_str(), nullptr, nullptr, &writer));
        ComPtr<IMFMediaType> output;
        require_success(MFCreateMediaType(&output));
        require_success(output->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio));
        require_success(output->SetGUID(MF_MT_SUBTYPE, subtype));
        require_success(output->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, result.audio.channels));
        require_success(output->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate));
        require_success(output->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                                          subtype == MFAudioFormat_AAC ? 24'000 : 16'000));
        if (subtype == MFAudioFormat_AAC) {
            require_success(output->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16));
            require_success(output->SetUINT32(MF_MT_AAC_PAYLOAD_TYPE, 0));
            require_success(output->SetUINT32(MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, 0x29));
        }
        DWORD stream = 0;
        require_success(writer->AddStream(output.Get(), &stream));
        ComPtr<IMFMediaType> input;
        require_success(MFCreateMediaType(&input));
        require_success(input->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio));
        require_success(input->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM));
        require_success(input->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, result.audio.channels));
        require_success(input->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate));
        require_success(input->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16));
        const auto block_alignment = static_cast<UINT32>(result.audio.channels * 2);
        require_success(input->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, block_alignment));
        require_success(
            input->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, sample_rate * block_alignment));
        require_success(writer->SetInputMediaType(stream, input.Get(), nullptr));
        require_success(writer->BeginWriting());
        ComPtr<IMFMediaBuffer> buffer;
        require_success(MFCreateMemoryBuffer(
            static_cast<DWORD>(input_pcm.size() * sizeof(std::int16_t)), &buffer));
        BYTE* bytes = nullptr;
        DWORD capacity = 0;
        require_success(buffer->Lock(&bytes, &capacity, nullptr));
        require(capacity >= input_pcm.size() * sizeof(std::int16_t),
                "Media Foundation compressed-audio input buffer is undersized");
        std::memcpy(bytes, input_pcm.data(), input_pcm.size() * sizeof(std::int16_t));
        require_success(buffer->Unlock());
        require_success(
            buffer->SetCurrentLength(static_cast<DWORD>(input_pcm.size() * sizeof(std::int16_t))));
        ComPtr<IMFSample> sample;
        require_success(MFCreateSample(&sample));
        require_success(sample->AddBuffer(buffer.Get()));
        require_success(sample->SetSampleTime(0));
        require_success(sample->SetSampleDuration(duration));
        require_success(writer->WriteSample(stream, sample.Get()));
        require_success(writer->Finalize());
    };
    write_compressed_boundary(mp3_path, MFAudioFormat_MP3, pcm, result.audio.sample_rate,
                              150'000'000);
    write_compressed_boundary(aac_path, MFAudioFormat_AAC, pcm, result.audio.sample_rate,
                              150'000'000);
    write_compressed_boundary(mp3_44k_path, MFAudioFormat_MP3, pcm_44k, boundary_44k_rate,
                              150'000'000);
    write_compressed_boundary(aac_44k_path, MFAudioFormat_AAC, pcm_44k, boundary_44k_rate,
                              150'000'000);
    auto over_limit_pcm = pcm;
    over_limit_pcm.resize(static_cast<std::size_t>(result.audio.sample_rate) *
                          result.audio.channels * 16);
    write_compressed_boundary(over_limit_mp3_path, MFAudioFormat_MP3, over_limit_pcm,
                              result.audio.sample_rate, 160'000'000);

    reader.Reset();
    require_success(MFShutdown());
    if (owns_com)
        CoUninitialize();

    require(std::filesystem::is_regular_file(mp3_path), "MP3 output file is missing");
    require(std::filesystem::file_size(mp3_path) > 1024, "MP3 output file is empty");
    const auto decoded_mp3 = trtmc::cli::read_audio_file(mp3_path.string(), policy);
    require(decoded_mp3.sample_rate == result.audio.sample_rate,
            "Media Foundation MP3 audio-rate mismatch");
    require(decoded_mp3.channels == result.audio.channels,
            "Media Foundation MP3 audio-channel mismatch");
    require(decoded_mp3.num_samples == static_cast<int32_t>(decoded_mp3.samples.size()) &&
                !decoded_mp3.samples.empty(),
            "Media Foundation MP3 decode returned invalid samples");
    require(decoded_mp3.samples.size() <= pcm.size() &&
                decoded_mp3.samples.size() >= pcm.size() - 4096,
            "configured MP3 boundary must trim only codec padding");

    require(std::filesystem::is_regular_file(aac_path), "AAC output file is missing");
    require(std::filesystem::file_size(aac_path) > 1024, "AAC output file is empty");
    const auto decoded_aac = trtmc::cli::read_audio_file(aac_path.string(), policy);
    require(decoded_aac.sample_rate == result.audio.sample_rate,
            "Media Foundation AAC audio-rate mismatch");
    require(decoded_aac.channels == result.audio.channels,
            "Media Foundation AAC audio-channel mismatch");
    require(decoded_aac.samples.size() <= pcm.size() &&
                decoded_aac.samples.size() >= pcm.size() - 4096,
            "configured AAC boundary must trim only codec padding");

    const auto decoded_mp3_44k = trtmc::cli::read_audio_file(mp3_44k_path.string(), policy);
    require(decoded_mp3_44k.sample_rate == static_cast<int32_t>(boundary_44k_rate),
            "44.1 kHz MP3 boundary audio-rate mismatch");
    require(decoded_mp3_44k.samples.size() <= pcm_44k.size() &&
                decoded_mp3_44k.samples.size() >= pcm_44k.size() - 4096,
            "configured 44.1 kHz MP3 boundary must trim only codec padding");
    const auto decoded_aac_44k = trtmc::cli::read_audio_file(aac_44k_path.string(), policy);
    require(decoded_aac_44k.sample_rate == static_cast<int32_t>(boundary_44k_rate),
            "44.1 kHz AAC boundary audio-rate mismatch");
    require(decoded_aac_44k.samples.size() <= pcm_44k.size() &&
                decoded_aac_44k.samples.size() >= pcm_44k.size() - 4096,
            "configured 44.1 kHz AAC boundary must trim only codec padding");
    require_throws_with(
        [&] { (void)trtmc::cli::read_audio_file(over_limit_mp3_path.string(), policy); },
                        "configured duration",
                        "a true 16-second MP3 must not be accepted as codec padding");

    return 0;
}

int main() {
    try {
        return run_test();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
