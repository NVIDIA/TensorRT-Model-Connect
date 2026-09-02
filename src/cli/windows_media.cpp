/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/windows_media.h"

#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <codecapi.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <windows.h>
#include <wrl/client.h>
#endif

namespace trtmc::cli {
namespace {

std::string lowercase_extension(std::string_view path) {
    const auto extension = std::filesystem::path(std::string(path)).extension().string();
    std::string lowered(extension);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return lowered;
}

constexpr std::uint64_t kAudioMediaTicksPerSecond = 10'000'000;
constexpr std::uint64_t kAudioReferenceLimitTicks = 15ULL * kAudioMediaTicksPerSecond;

struct ReferenceAudioFrameWindow {
    std::uint64_t first_frame{0};
    std::uint64_t end_frame{0};
};

std::uint64_t ticks_to_frames(std::uint64_t ticks, std::uint32_t sample_rate,
                              bool round_up) noexcept {
    const std::uint64_t whole_seconds = ticks / kAudioMediaTicksPerSecond;
    const std::uint64_t partial_ticks = ticks % kAudioMediaTicksPerSecond;
    if (whole_seconds > std::numeric_limits<std::uint64_t>::max() / sample_rate)
        return std::numeric_limits<std::uint64_t>::max();
    std::uint64_t frames = whole_seconds * sample_rate;
    const std::uint64_t partial_product = partial_ticks * sample_rate;
    const std::uint64_t partial_frames =
        round_up ? (partial_product + kAudioMediaTicksPerSecond - 1) / kAudioMediaTicksPerSecond
                 : partial_product / kAudioMediaTicksPerSecond;
    if (partial_frames > std::numeric_limits<std::uint64_t>::max() - frames)
        return std::numeric_limits<std::uint64_t>::max();
    return frames + partial_frames;
}

bool frames_to_ticks_rounded_up(std::uint64_t frames, std::uint32_t sample_rate,
                                std::uint64_t& ticks) noexcept {
    const std::uint64_t whole_seconds = frames / sample_rate;
    const std::uint64_t partial_frames = frames % sample_rate;
    if (whole_seconds > std::numeric_limits<std::uint64_t>::max() / kAudioMediaTicksPerSecond) {
        return false;
    }
    ticks = whole_seconds * kAudioMediaTicksPerSecond;
    const std::uint64_t partial_product = partial_frames * kAudioMediaTicksPerSecond;
    const std::uint64_t partial_ticks = (partial_product + sample_rate - 1) / sample_rate;
    if (partial_ticks > std::numeric_limits<std::uint64_t>::max() - ticks)
        return false;
    ticks += partial_ticks;
    return true;
}

bool account_reference_audio_decode_impl(std::int64_t timestamp, std::uint64_t frame_count,
                                         std::uint32_t sample_rate,
                                         std::uint64_t maximum_padding_frames,
                                         detail::ReferenceAudioDecodeState& state,
                                         ReferenceAudioFrameWindow* retained_window) noexcept {
    if (sample_rate == 0 || frame_count == 0)
        return false;

    const std::uint64_t public_decoded_frames = 15ULL * sample_rate;
    if (maximum_padding_frames >
        std::numeric_limits<std::uint64_t>::max() - public_decoded_frames) {
        return false;
    }
    const std::uint64_t maximum_decoded_frames = public_decoded_frames + maximum_padding_frames;
    if (state.decoded_frames > maximum_decoded_frames ||
        frame_count > maximum_decoded_frames - state.decoded_frames) {
        return false;
    }

    std::uint64_t padding_ticks = 0;
    std::uint64_t sample_duration_ticks = 0;
    if (!frames_to_ticks_rounded_up(maximum_padding_frames, sample_rate, padding_ticks) ||
        !frames_to_ticks_rounded_up(frame_count, sample_rate, sample_duration_ticks) ||
        padding_ticks > std::numeric_limits<std::uint64_t>::max() - kAudioReferenceLimitTicks) {
        return false;
    }
    const std::uint64_t latest_end_tick = kAudioReferenceLimitTicks + padding_ticks;

    std::uint64_t leading_ticks = 0;
    if (timestamp < 0) {
        leading_ticks = static_cast<std::uint64_t>(-(timestamp + 1)) + 1;
        if (leading_ticks > padding_ticks)
            return false;
        if (sample_duration_ticks > leading_ticks &&
            sample_duration_ticks - leading_ticks > latest_end_tick) {
            return false;
        }
    } else {
        const std::uint64_t start_tick = static_cast<std::uint64_t>(timestamp);
        if (start_tick > latest_end_tick || sample_duration_ticks > latest_end_tick - start_tick) {
            return false;
        }
    }

    const std::uint64_t first_frame =
        std::min(frame_count, ticks_to_frames(leading_ticks, sample_rate, true));
    std::uint64_t end_frame = first_frame;
    if (timestamp < static_cast<std::int64_t>(kAudioReferenceLimitTicks)) {
        const std::uint64_t remaining_ticks =
            timestamp < 0 ? kAudioReferenceLimitTicks + leading_ticks
                          : kAudioReferenceLimitTicks - static_cast<std::uint64_t>(timestamp);
        end_frame = std::min(frame_count, ticks_to_frames(remaining_ticks, sample_rate, false));
        end_frame = std::max(end_frame, first_frame);
    }
    const std::uint64_t padding_frames = frame_count - (end_frame - first_frame);
    if (state.decoded_padding_frames > maximum_padding_frames ||
        padding_frames > maximum_padding_frames - state.decoded_padding_frames) {
        return false;
    }

    state.decoded_frames += frame_count;
    state.decoded_padding_frames += padding_frames;
    if (retained_window != nullptr)
        *retained_window = {first_frame, end_frame};
    return true;
}

#if defined(_WIN32)

using Microsoft::WRL::ComPtr;

constexpr std::uint64_t kReferenceLimitSeconds = 15;
constexpr LONGLONG kMediaTicksPerSecond = 10'000'000;
constexpr std::size_t kMaxConsecutiveEmptyReads = 256;
constexpr std::size_t kMaxTotalEmptyReads = 16'384;
constexpr std::uint32_t kReferenceFps = 24;
constexpr std::uint32_t kMaximumReferenceSourceFps = 240;
constexpr std::uint32_t kReferenceCanvasMultiple = 32;
constexpr std::uint32_t kReferenceCanvasShortEdge = 768;
constexpr std::uint64_t kReferenceCanvasMaxPixels = 768ULL * 1344;
constexpr std::uint64_t kReferenceDecodedMaxPixels = 576ULL * 1856;

[[noreturn]] void throw_hresult(const char* operation, HRESULT result) {
    std::ostringstream message;
    message << operation << " failed with HRESULT 0x" << std::hex << std::setw(8)
            << std::setfill('0') << static_cast<std::uint32_t>(result);
    throw std::runtime_error(message.str());
}

void check_hresult(HRESULT result, const char* operation) {
    if (FAILED(result))
        throw_hresult(operation, result);
}

class MediaFoundationSession {
  public:
    MediaFoundationSession() {
        const HRESULT com_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (SUCCEEDED(com_result))
            owns_com_ = true;
        else if (com_result != RPC_E_CHANGED_MODE)
            throw_hresult("CoInitializeEx", com_result);
        const HRESULT media_result = MFStartup(MF_VERSION, MFSTARTUP_FULL);
        if (FAILED(media_result)) {
            if (owns_com_)
                CoUninitialize();
            owns_com_ = false;
            throw_hresult("MFStartup", media_result);
        }
        media_started_ = true;
    }

    ~MediaFoundationSession() {
        if (media_started_)
            (void)MFShutdown();
        if (owns_com_)
            CoUninitialize();
    }

    MediaFoundationSession(const MediaFoundationSession&) = delete;
    MediaFoundationSession& operator=(const MediaFoundationSession&) = delete;

  private:
    bool owns_com_{false};
    bool media_started_{false};
};

std::uint32_t checked_u32(std::uint64_t value, const char* label) {
    if (value > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error(std::string(label) + " exceeds the Media Foundation limit");
    return static_cast<std::uint32_t>(value);
}

LONGLONG media_time(std::uint64_t numerator, std::uint32_t denominator) {
    constexpr std::uint64_t kTicksPerSecond = 10'000'000;
    if (denominator == 0 || numerator > std::numeric_limits<std::uint64_t>::max() / kTicksPerSecond)
        throw std::runtime_error("media timestamp overflow");
    return static_cast<LONGLONG>((numerator * kTicksPerSecond) / denominator);
}

BYTE quantize(float value) {
    return static_cast<BYTE>(std::clamp(static_cast<int>(std::lround(value)), 0, 255));
}

void rgb_to_nv12(const float* rgb, std::uint32_t width, std::uint32_t height,
                 std::vector<BYTE>& nv12) {
    const std::size_t y_size = static_cast<std::size_t>(width) * height;
    nv12.resize(y_size + y_size / 2);
    auto* y_plane = nv12.data();
    auto* uv_plane = nv12.data() + y_size;

    const auto component = [](const float* pixel, int index) {
        return std::clamp(pixel[index], 0.0F, 1.0F);
    };
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            const float* pixel = rgb + (static_cast<std::size_t>(row) * width + column) * 3;
            const float red = component(pixel, 0);
            const float green = component(pixel, 1);
            const float blue = component(pixel, 2);
            y_plane[static_cast<std::size_t>(row) * width + column] =
                quantize(16.0F + 219.0F * (0.2126F * red + 0.7152F * green + 0.0722F * blue));
        }
    }

    for (std::uint32_t row = 0; row < height; row += 2) {
        for (std::uint32_t column = 0; column < width; column += 2) {
            float cb = 0.0F;
            float cr = 0.0F;
            for (std::uint32_t dy = 0; dy < 2; ++dy) {
                for (std::uint32_t dx = 0; dx < 2; ++dx) {
                    const float* pixel =
                        rgb + (static_cast<std::size_t>(row + dy) * width + column + dx) * 3;
                    const float red = component(pixel, 0);
                    const float green = component(pixel, 1);
                    const float blue = component(pixel, 2);
                    cb += -0.114572F * red - 0.385428F * green + 0.5F * blue;
                    cr += 0.5F * red - 0.454153F * green - 0.045847F * blue;
                }
            }
            const auto uv_offset = static_cast<std::size_t>(row / 2) * width + column;
            uv_plane[uv_offset] = quantize(128.0F + 224.0F * cb / 4.0F);
            uv_plane[uv_offset + 1] = quantize(128.0F + 224.0F * cr / 4.0F);
        }
    }
}

ComPtr<IMFSample> make_sample(const void* bytes, std::size_t byte_count, LONGLONG timestamp,
                              LONGLONG duration) {
    const auto media_bytes = checked_u32(byte_count, "media sample size");
    ComPtr<IMFMediaBuffer> buffer;
    check_hresult(MFCreateMemoryBuffer(media_bytes, &buffer), "MFCreateMemoryBuffer");
    BYTE* destination = nullptr;
    DWORD maximum_length = 0;
    check_hresult(buffer->Lock(&destination, &maximum_length, nullptr), "IMFMediaBuffer::Lock");
    if (maximum_length < media_bytes) {
        (void)buffer->Unlock();
        throw std::runtime_error("Media Foundation returned an undersized sample buffer");
    }
    std::memcpy(destination, bytes, byte_count);
    check_hresult(buffer->Unlock(), "IMFMediaBuffer::Unlock");
    check_hresult(buffer->SetCurrentLength(media_bytes), "IMFMediaBuffer::SetCurrentLength");

    ComPtr<IMFSample> sample;
    check_hresult(MFCreateSample(&sample), "MFCreateSample");
    check_hresult(sample->AddBuffer(buffer.Get()), "IMFSample::AddBuffer");
    check_hresult(sample->SetSampleTime(timestamp), "IMFSample::SetSampleTime");
    check_hresult(sample->SetSampleDuration(duration), "IMFSample::SetSampleDuration");
    return sample;
}

ComPtr<IMFMediaType> make_video_output_type(std::uint32_t width, std::uint32_t height,
                                            std::uint32_t fps) {
    ComPtr<IMFMediaType> type;
    check_hresult(MFCreateMediaType(&type), "MFCreateMediaType(video output)");
    check_hresult(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video), "video output major type");
    check_hresult(type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264), "video output subtype");
    const auto bitrate = checked_u32(
        std::max<std::uint64_t>(8'000'000, static_cast<std::uint64_t>(width) * height * fps / 3),
        "H.264 bitrate");
    check_hresult(type->SetUINT32(MF_MT_AVG_BITRATE, bitrate), "video output bitrate");
    check_hresult(type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive),
                  "video output interlace mode");
    check_hresult(type->SetUINT32(MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_Main),
                  "video output H.264 profile");
    check_hresult(MFSetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, width, height),
                  "video output frame size");
    check_hresult(MFSetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, fps, 1),
                  "video output frame rate");
    check_hresult(MFSetAttributeRatio(type.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1),
                  "video output pixel aspect ratio");
    return type;
}

ComPtr<IMFMediaType> make_video_input_type(std::uint32_t width, std::uint32_t height,
                                           std::uint32_t fps) {
    ComPtr<IMFMediaType> type;
    check_hresult(MFCreateMediaType(&type), "MFCreateMediaType(video input)");
    check_hresult(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video), "video input major type");
    check_hresult(type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12), "video input subtype");
    check_hresult(type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive),
                  "video input interlace mode");
    check_hresult(type->SetUINT32(MF_MT_FIXED_SIZE_SAMPLES, TRUE), "video input fixed samples");
    check_hresult(type->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE),
                  "video input independent samples");
    check_hresult(type->SetUINT32(MF_MT_SAMPLE_SIZE,
                                  checked_u32(static_cast<std::uint64_t>(width) * height * 3 / 2,
                                              "NV12 sample size")),
                  "video input sample size");
    check_hresult(type->SetUINT32(MF_MT_DEFAULT_STRIDE, width), "video input stride");
    check_hresult(MFSetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, width, height),
                  "video input frame size");
    check_hresult(MFSetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, fps, 1),
                  "video input frame rate");
    check_hresult(MFSetAttributeRatio(type.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1),
                  "video input pixel aspect ratio");
    return type;
}

ComPtr<IMFMediaType> make_audio_output_type(std::uint32_t sample_rate, std::uint32_t channels) {
    ComPtr<IMFMediaType> type;
    check_hresult(MFCreateMediaType(&type), "MFCreateMediaType(audio output)");
    check_hresult(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio), "audio output major type");
    check_hresult(type->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_AAC), "audio output subtype");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels), "audio output channels");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate),
                  "audio output sample rate");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16), "audio output bit depth");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 24'000),
                  "audio output bitrate");
    check_hresult(type->SetUINT32(MF_MT_AAC_PAYLOAD_TYPE, 0), "AAC payload type");
    check_hresult(type->SetUINT32(MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, 0x29),
                  "AAC profile level");
    return type;
}

ComPtr<IMFMediaType> make_audio_input_type(std::uint32_t sample_rate, std::uint32_t channels) {
    ComPtr<IMFMediaType> type;
    check_hresult(MFCreateMediaType(&type), "MFCreateMediaType(audio input)");
    check_hresult(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio), "audio input major type");
    check_hresult(type->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM), "audio input subtype");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels), "audio input channels");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate),
                  "audio input sample rate");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16), "audio input bit depth");
    const auto block_alignment =
        checked_u32(static_cast<std::uint64_t>(channels) * 2, "audio block alignment");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, block_alignment),
                  "audio input block alignment");
    check_hresult(
        type->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                        checked_u32(static_cast<std::uint64_t>(sample_rate) * block_alignment,
                                    "PCM byte rate")),
        "audio input byte rate");
    check_hresult(type->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE),
                  "audio input independent samples");
    return type;
}

void validate_result(const VideoResult& result) {
    const auto& frames = result.frames;
    if (frames.width <= 0 || frames.height <= 0 || frames.num_frames <= 0 || frames.channels != 3 ||
        result.fps <= 0)
        throw std::runtime_error(
            "write_mp4 requires valid THWC RGB video and a positive frame rate");
    if ((frames.width & 1) != 0 || (frames.height & 1) != 0)
        throw std::runtime_error("write_mp4 requires even frame dimensions for NV12/H.264");
    const auto pixels_per_frame = static_cast<std::uint64_t>(frames.width) * frames.height * 3;
    const auto required_pixels = pixels_per_frame * static_cast<std::uint64_t>(frames.num_frames);
    if (required_pixels > frames.pixels.size())
        throw std::runtime_error("write_mp4 frame storage is smaller than its THWC metadata");
    const auto& audio = result.audio;
    if (!audio.samples.empty()) {
        if (audio.sample_rate <= 0 || (audio.channels != 1 && audio.channels != 2) ||
            audio.samples.size() % static_cast<std::size_t>(audio.channels) != 0)
            throw std::runtime_error("write_mp4 requires valid mono or stereo interleaved audio");
    }
}

struct ReferenceVideoSize {
    std::uint32_t width{0};
    std::uint32_t height{0};
};

std::uint32_t round_canvas_axis(double value) {
    const double scaled = value / kReferenceCanvasMultiple;
    const double lower = std::floor(scaled);
    const double fraction = scaled - lower;
    double rounded = lower;
    if (fraction > 0.5 || (fraction == 0.5 && static_cast<std::uint64_t>(lower) % 2 != 0))
        rounded += 1.0;
    if (rounded > std::numeric_limits<std::uint32_t>::max() / kReferenceCanvasMultiple)
        throw std::runtime_error("reference video canvas rounding overflow");
    return std::max(kReferenceCanvasMultiple,
                    static_cast<std::uint32_t>(rounded) * kReferenceCanvasMultiple);
}

ReferenceVideoSize resolve_reference_video_size(std::uint32_t source_width,
                                                std::uint32_t source_height) {
    if (source_width == 0 || source_height == 0)
        throw std::runtime_error("reference video has invalid source dimensions");
    const double ratio = static_cast<double>(source_width) / source_height;
    if (!std::isfinite(ratio) || ratio < 0.25 || ratio > 4.0)
        throw std::runtime_error("reference video aspect must be within 1:4 through 4:1");
    double width = ratio >= 1.0 ? kReferenceCanvasShortEdge * ratio
                                : static_cast<double>(kReferenceCanvasShortEdge);
    double height = ratio >= 1.0 ? static_cast<double>(kReferenceCanvasShortEdge)
                                 : kReferenceCanvasShortEdge / ratio;
    const double area = width * height;
    if (area > static_cast<double>(kReferenceCanvasMaxPixels)) {
        const double scale = std::sqrt(static_cast<double>(kReferenceCanvasMaxPixels) / area);
        width *= scale;
        height *= scale;
    }
    ReferenceVideoSize result{round_canvas_axis(width), round_canvas_axis(height)};
    if (static_cast<std::uint64_t>(result.width) * result.height > kReferenceDecodedMaxPixels) {
        throw std::runtime_error("reference video resolver exceeded its decoded-pixel limit");
    }
    return result;
}

std::uint64_t rounded_reference_frame_slot(std::uint64_t frame, std::uint32_t fps_numerator,
                                           std::uint32_t fps_denominator) {
    if (fps_numerator == 0 || fps_denominator == 0 ||
        frame > std::numeric_limits<std::uint64_t>::max() / kReferenceFps / fps_denominator) {
        throw std::runtime_error("reference video frame-slot arithmetic overflow");
    }
    const std::uint64_t numerator = frame * kReferenceFps * fps_denominator;
    if (numerator > (std::numeric_limits<std::uint64_t>::max() - fps_numerator) / 2)
        throw std::runtime_error("reference video frame-slot rounding overflow");
    return (2 * numerator + fps_numerator) / (2ULL * fps_numerator);
}

ComPtr<IMFMediaType> source_reader_video_type(std::uint32_t width, std::uint32_t height) {
    ComPtr<IMFMediaType> type;
    check_hresult(MFCreateMediaType(&type), "MFCreateMediaType(source video)");
    check_hresult(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video), "source video major type");
    check_hresult(type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32), "source video subtype");
    check_hresult(MFSetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, width, height),
                  "source video decoded frame size");
    return type;
}

ComPtr<IMFMediaType> source_reader_audio_type(std::uint32_t sample_rate, std::uint32_t channels) {
    ComPtr<IMFMediaType> type;
    check_hresult(MFCreateMediaType(&type), "MFCreateMediaType(source audio)");
    check_hresult(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio), "source audio major type");
    check_hresult(type->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float), "source audio subtype");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels), "source audio channels");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate),
                  "source audio sample rate");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32), "source audio bit depth");
    const auto block_alignment =
        checked_u32(static_cast<std::uint64_t>(channels) * 4, "source audio block alignment");
    check_hresult(type->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, block_alignment),
                  "source audio block alignment");
    check_hresult(
        type->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                        checked_u32(static_cast<std::uint64_t>(sample_rate) * block_alignment,
                                    "source audio byte rate")),
        "source audio byte rate");
    return type;
}

void append_rgb32_frame(IMFSample* sample, std::uint32_t width, std::uint32_t height, LONG stride,
                        std::vector<float>& pixels, std::size_t maximum_scalars) {
    if (sample == nullptr)
        throw std::runtime_error("Media Foundation returned a null video sample");
    ComPtr<IMFMediaBuffer> buffer;
    check_hresult(sample->ConvertToContiguousBuffer(&buffer),
                  "IMFSample::ConvertToContiguousBuffer(video)");
    BYTE* data = nullptr;
    DWORD maximum_length = 0;
    DWORD current_length = 0;
    check_hresult(buffer->Lock(&data, &maximum_length, &current_length),
                  "IMFMediaBuffer::Lock(video)");
    const auto unlock = [&]() { (void)buffer->Unlock(); };
    if (current_length == 0) {
        unlock();
        detail::require_nonempty_decoded_buffer(current_length, "video");
    }
    const auto absolute_stride =
        static_cast<std::uint64_t>(stride < 0 ? -static_cast<std::int64_t>(stride) : stride);
    const auto required_bytes = absolute_stride * height;
    if (absolute_stride < static_cast<std::uint64_t>(width) * 4 ||
        required_bytes > current_length) {
        unlock();
        throw std::runtime_error("Media Foundation returned an invalid RGB32 stride or buffer");
    }
    const auto old_size = pixels.size();
    const auto frame_scalars = static_cast<std::uint64_t>(width) * height * 3;
    if (frame_scalars > maximum_scalars || old_size > maximum_scalars - frame_scalars) {
        unlock();
        throw std::runtime_error("decoded reference video exceeds its bounded RGB allocation");
    }
    if (frame_scalars > std::numeric_limits<std::size_t>::max() - old_size) {
        unlock();
        throw std::runtime_error("decoded video exceeds host address space");
    }
    pixels.resize(old_size + static_cast<std::size_t>(frame_scalars));
    float* destination = pixels.data() + old_size;
    for (std::uint32_t row = 0; row < height; ++row) {
        const auto source_row = stride >= 0 ? row : height - 1 - row;
        const BYTE* source = data + static_cast<std::uint64_t>(source_row) * absolute_stride;
        for (std::uint32_t column = 0; column < width; ++column) {
            const BYTE* bgra = source + static_cast<std::size_t>(column) * 4;
            float* rgb = destination + (static_cast<std::size_t>(row) * width + column) * 3;
            rgb[0] = static_cast<float>(bgra[2]) / 255.0F;
            rgb[1] = static_cast<float>(bgra[1]) / 255.0F;
            rgb[2] = static_cast<float>(bgra[0]) / 255.0F;
        }
    }
    unlock();
}

void append_float_audio(IMFSample* sample, std::uint32_t sample_rate, std::uint32_t channels,
                        LONGLONG timestamp, std::uint64_t maximum_codec_padding_frames,
                        detail::ReferenceAudioDecodeState& decode_state,
                        std::vector<float>& samples, std::size_t maximum_scalars) {
    if (sample == nullptr)
        throw std::runtime_error("Media Foundation returned a null audio sample");
    ComPtr<IMFMediaBuffer> buffer;
    check_hresult(sample->ConvertToContiguousBuffer(&buffer),
                  "IMFSample::ConvertToContiguousBuffer(audio)");
    BYTE* data = nullptr;
    DWORD maximum_length = 0;
    DWORD current_length = 0;
    check_hresult(buffer->Lock(&data, &maximum_length, &current_length),
                  "IMFMediaBuffer::Lock(audio)");
    if (current_length == 0) {
        (void)buffer->Unlock();
        detail::require_nonempty_decoded_buffer(current_length, "audio");
    }
    if (current_length % (sizeof(float) * channels) != 0) {
        (void)buffer->Unlock();
        throw std::runtime_error("Media Foundation returned misaligned float audio");
    }
    const auto scalar_count = current_length / sizeof(float);
    const std::size_t frame_count = scalar_count / channels;
    const bool trim_codec_padding = maximum_codec_padding_frames > 0;
    ReferenceAudioFrameWindow retained_window{0, frame_count};
    if (trim_codec_padding) {
        if (!account_reference_audio_decode_impl(timestamp, frame_count, sample_rate,
                                                 maximum_codec_padding_frames, decode_state,
                                                 &retained_window)) {
            (void)buffer->Unlock();
            throw std::runtime_error(
                "decoded reference audio exceeds the 15-second public limit plus codec padding");
        }
    }
    if (!trim_codec_padding) {
        LONGLONG declared_sample_duration = 0;
        if (FAILED(sample->GetSampleDuration(&declared_sample_duration)) ||
            declared_sample_duration < 0) {
            declared_sample_duration = 0;
        }
        const auto inferred_duration = static_cast<LONGLONG>(
            (static_cast<std::uint64_t>(frame_count) * kMediaTicksPerSecond + sample_rate - 1) /
            sample_rate);
        const LONGLONG timeline_duration = std::max(declared_sample_duration, inferred_duration);
        if (!detail::reference_timeline_within_limit(timestamp, timeline_duration)) {
            (void)buffer->Unlock();
            throw std::runtime_error(
                "decoded reference audio sample crosses the 15-second public timeline");
        }
    }
    const std::size_t first_frame = static_cast<std::size_t>(retained_window.first_frame);
    const std::size_t end_frame = static_cast<std::size_t>(retained_window.end_frame);
    const std::size_t retained_scalars = (end_frame - first_frame) * channels;
    const auto old_size = samples.size();
    if (retained_scalars > maximum_scalars || old_size > maximum_scalars - retained_scalars) {
        (void)buffer->Unlock();
        throw std::runtime_error("decoded reference audio exceeds the 15-second public limit");
    }
    if (retained_scalars > std::numeric_limits<std::size_t>::max() - old_size) {
        (void)buffer->Unlock();
        throw std::runtime_error("decoded audio exceeds host address space");
    }
    samples.resize(old_size + retained_scalars);
    if (retained_scalars != 0) {
        std::memcpy(samples.data() + old_size, data + first_frame * channels * sizeof(float),
                    retained_scalars * sizeof(float));
    }
    check_hresult(buffer->Unlock(), "IMFMediaBuffer::Unlock(audio)");
}

std::optional<LONGLONG> presentation_duration(IMFSourceReader& reader) {
    PROPVARIANT value;
    PropVariantInit(&value);
    const HRESULT status =
        reader.GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &value);
    std::optional<LONGLONG> result;
    if (SUCCEEDED(status)) {
        if (value.vt == VT_UI8 &&
            value.uhVal.QuadPart <= static_cast<ULONGLONG>(std::numeric_limits<LONGLONG>::max())) {
            result = static_cast<LONGLONG>(value.uhVal.QuadPart);
        } else if (value.vt == VT_I8 && value.hVal.QuadPart >= 0) {
            result = value.hVal.QuadPart;
        }
    }
    (void)PropVariantClear(&value);
    return result;
}

struct CodecPaddingAllowance {
    std::uint64_t frames{0};
    LONGLONG duration{0};
};

CodecPaddingAllowance codec_padding_allowance(IMFMediaType* native_audio) {
    if (native_audio == nullptr)
        return {};
    GUID subtype{};
    UINT32 sample_rate = 0;
    if (FAILED(native_audio->GetGUID(MF_MT_SUBTYPE, &subtype)) ||
        FAILED(native_audio->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &sample_rate)) ||
        sample_rate == 0) {
        return {};
    }
    std::uint64_t padding_frames = 0;
    if (subtype == MFAudioFormat_MP3)
        padding_frames = 3ULL * 1152;
    else if (subtype == MFAudioFormat_AAC)
        padding_frames = 3ULL * 1024;
    if (padding_frames == 0)
        return {};
    return {padding_frames,
            static_cast<LONGLONG>((padding_frames * kMediaTicksPerSecond + sample_rate - 1) /
                                  sample_rate)};
}

std::size_t maximum_reference_audio_scalars(std::uint32_t sample_rate, std::uint32_t channels) {
    const std::uint64_t public_limit =
        static_cast<std::uint64_t>(sample_rate) * channels * kReferenceLimitSeconds;
    return static_cast<std::size_t>(std::min<std::uint64_t>(
        public_limit, static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max())));
}

void reject_source_reader_error(DWORD flags, const char* label) {
    if ((flags & MF_SOURCE_READERF_ERROR) != 0)
        throw std::runtime_error(std::string(label) + " reported MF_SOURCE_READERF_ERROR");
}

void reject_empty_decoded_sample(IMFSample* sample, std::string_view media_kind) {
    if (sample == nullptr)
        throw std::runtime_error("Media Foundation returned a null decoded sample");
    DWORD total_length = 0;
    check_hresult(sample->GetTotalLength(&total_length), "IMFSample::GetTotalLength(decoded)");
    detail::require_nonempty_decoded_buffer(total_length, media_kind);
}

void reject_reference_timestamp(LONGLONG timestamp, const char* label) {
    if (timestamp >= kReferenceLimitSeconds * kMediaTicksPerSecond)
        throw std::runtime_error(std::string(label) +
                                 " timestamp exceeds the 15-second public limit");
}

void reject_video_sample_timeline(IMFSample* sample, LONGLONG timestamp,
                                  std::uint32_t fps_numerator, std::uint32_t fps_denominator) {
    LONGLONG duration = 0;
    if (sample == nullptr || FAILED(sample->GetSampleDuration(&duration)) || duration <= 0) {
        duration = static_cast<LONGLONG>(
            (static_cast<std::uint64_t>(kMediaTicksPerSecond) * fps_denominator + fps_numerator -
             1) /
            fps_numerator);
    }
    if (!detail::reference_timeline_within_limit(timestamp, duration))
        throw std::runtime_error(
            "decoded reference video sample crosses the 15-second public timeline");
}

void note_empty_source_reader_event(DWORD flags, LONGLONG timestamp, LONGLONG& last_tick_timestamp,
                                    std::size_t& consecutive_empty_reads,
                                    std::size_t& total_empty_reads,
                                    std::uint64_t maximum_codec_padding_ticks, const char* label) {
    // A stream tick deliberately carries no sample. Treat an advancing tick as
    // progress, but bound its timestamp to the public reference duration. All
    // other empty reads (including repeated/non-advancing ticks) are bounded so
    // a malformed source cannot spin forever.
    if (++total_empty_reads > kMaxTotalEmptyReads)
        throw std::runtime_error(std::string(label) + " returned too many empty stream events");
    if (maximum_codec_padding_ticks > 0 && !detail::reference_audio_event_timestamp_within_padding(
                                               timestamp, maximum_codec_padding_ticks)) {
        throw std::runtime_error(std::string(label) +
                                 " event timestamp exceeds the 15-second public limit plus codec "
                                 "padding");
    }
    if ((flags & MF_SOURCE_READERF_STREAMTICK) != 0) {
        if (maximum_codec_padding_ticks == 0)
            reject_reference_timestamp(timestamp, label);
        if (timestamp > last_tick_timestamp) {
            last_tick_timestamp = timestamp;
            consecutive_empty_reads = 0;
            return;
        }
    }
    if (++consecutive_empty_reads > kMaxConsecutiveEmptyReads)
        throw std::runtime_error(std::string(label) + " made no progress while decoding");
}

#endif

} // namespace

std::uint64_t detail::reference_video_frame_ceiling(std::uint32_t fps_numerator,
                                                    std::uint32_t fps_denominator) {
    if (fps_numerator == 0 || fps_denominator == 0)
        throw std::invalid_argument("reference video frame rate must be positive");
    constexpr std::uint64_t seconds = 15;
    const std::uint64_t numerator = seconds * fps_numerator;
    return (numerator + fps_denominator - 1) / fps_denominator;
}

std::pair<std::uint32_t, std::uint32_t>
detail::reference_video_decode_size(std::uint32_t source_width, std::uint32_t source_height) {
#if defined(_WIN32)
    const auto size = resolve_reference_video_size(source_width, source_height);
    return {size.width, size.height};
#else
    if (source_width == 0 || source_height == 0)
        throw std::invalid_argument("reference video has invalid source dimensions");
    constexpr double short_edge = 768.0;
    constexpr double max_pixels = 768.0 * 1344.0;
    const double ratio = static_cast<double>(source_width) / source_height;
    if (!std::isfinite(ratio) || ratio < 0.25 || ratio > 4.0)
        throw std::invalid_argument("reference video aspect must be within 1:4 through 4:1");
    double width = ratio >= 1.0 ? short_edge * ratio : short_edge;
    double height = ratio >= 1.0 ? short_edge : short_edge / ratio;
    if (width * height > max_pixels) {
        const double scale = std::sqrt(max_pixels / (width * height));
        width *= scale;
        height *= scale;
    }
    const auto round_axis = [](double value) {
        return static_cast<std::uint32_t>(std::nearbyint(value / 32.0)) * 32U;
    };
    return {round_axis(width), round_axis(height)};
#endif
}

bool detail::reference_timeline_within_limit(std::int64_t timestamp,
                                             std::int64_t duration) noexcept {
    constexpr std::int64_t limit = 15LL * 10'000'000;
    return timestamp >= 0 && duration >= 0 && timestamp <= limit && duration <= limit - timestamp;
}

bool detail::reference_audio_event_timestamp_within_padding(
    std::int64_t timestamp, std::uint64_t maximum_padding_ticks) noexcept {
    if (maximum_padding_ticks >
        std::numeric_limits<std::uint64_t>::max() - kAudioReferenceLimitTicks) {
        return false;
    }
    if (timestamp < 0) {
        const std::uint64_t leading_ticks = static_cast<std::uint64_t>(-(timestamp + 1)) + 1;
        return leading_ticks <= maximum_padding_ticks;
    }
    return static_cast<std::uint64_t>(timestamp) <=
           kAudioReferenceLimitTicks + maximum_padding_ticks;
}

bool detail::account_reference_audio_decode(std::int64_t timestamp, std::uint64_t frame_count,
                                            std::uint32_t sample_rate,
                                            std::uint64_t maximum_padding_frames,
                                            ReferenceAudioDecodeState& state) noexcept {
    return account_reference_audio_decode_impl(timestamp, frame_count, sample_rate,
                                               maximum_padding_frames, state, nullptr);
}

void detail::require_nonempty_decoded_buffer(std::uint32_t current_length,
                                             std::string_view media_kind) {
    if (current_length == 0) {
        throw std::runtime_error("Media Foundation returned an empty decoded " +
                                 std::string(media_kind) + " buffer");
    }
}

bool is_mp4_path(std::string_view path) {
    return lowercase_extension(path) == ".mp4";
}

void write_mp4(const VideoResult& result, const std::string& path) {
#if defined(_WIN32)
    validate_result(result);
    if (!is_mp4_path(path))
        throw std::runtime_error("write_mp4 output path must end in .mp4");
    const auto output_path = std::filesystem::path(path);
    if (!output_path.parent_path().empty())
        std::filesystem::create_directories(output_path.parent_path());

    MediaFoundationSession session;
    ComPtr<IMFAttributes> attributes;
    check_hresult(MFCreateAttributes(&attributes, 2), "MFCreateAttributes(sink writer)");
    check_hresult(attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE),
                  "enable Media Foundation hardware transforms");
    check_hresult(attributes->SetUINT32(MF_LOW_LATENCY, FALSE),
                  "configure Media Foundation latency");

    ComPtr<IMFSinkWriter> writer;
    check_hresult(MFCreateSinkWriterFromURL(output_path.wstring().c_str(), nullptr,
                                            attributes.Get(), &writer),
                  "MFCreateSinkWriterFromURL");

    const auto width = static_cast<std::uint32_t>(result.frames.width);
    const auto height = static_cast<std::uint32_t>(result.frames.height);
    const auto fps = static_cast<std::uint32_t>(result.fps);
    DWORD video_stream = 0;
    const auto video_output = make_video_output_type(width, height, fps);
    check_hresult(writer->AddStream(video_output.Get(), &video_stream),
                  "IMFSinkWriter::AddStream(video)");
    const auto video_input = make_video_input_type(width, height, fps);
    check_hresult(writer->SetInputMediaType(video_stream, video_input.Get(), nullptr),
                  "IMFSinkWriter::SetInputMediaType(video)");

    const bool has_audio = !result.audio.samples.empty();
    DWORD audio_stream = 0;
    std::uint32_t audio_rate = 0;
    std::uint32_t audio_channels = 0;
    if (has_audio) {
        audio_rate = static_cast<std::uint32_t>(result.audio.sample_rate);
        audio_channels = static_cast<std::uint32_t>(result.audio.channels);
        const auto audio_output = make_audio_output_type(audio_rate, audio_channels);
        check_hresult(writer->AddStream(audio_output.Get(), &audio_stream),
                      "IMFSinkWriter::AddStream(audio)");
        const auto audio_input = make_audio_input_type(audio_rate, audio_channels);
        check_hresult(writer->SetInputMediaType(audio_stream, audio_input.Get(), nullptr),
                      "IMFSinkWriter::SetInputMediaType(audio)");
    }

    check_hresult(writer->BeginWriting(), "IMFSinkWriter::BeginWriting");

    const std::size_t pixels_per_frame = static_cast<std::size_t>(width) * height * 3;
    const std::uint64_t audio_frames = has_audio ? result.audio.samples.size() / audio_channels : 0;
    constexpr std::uint64_t kAudioChunkFrames = 1024;
    std::uint64_t video_index = 0;
    std::uint64_t audio_offset = 0;
    std::vector<BYTE> nv12;
    std::vector<std::int16_t> pcm;

    while (video_index < static_cast<std::uint64_t>(result.frames.num_frames) ||
           audio_offset < audio_frames) {
        const auto next_video_time =
            video_index < static_cast<std::uint64_t>(result.frames.num_frames)
                ? media_time(video_index, fps)
                : std::numeric_limits<LONGLONG>::max();
        const auto next_audio_time = audio_offset < audio_frames
                                         ? media_time(audio_offset, audio_rate)
                                         : std::numeric_limits<LONGLONG>::max();
        if (next_video_time <= next_audio_time) {
            const float* source = result.frames.pixels.data() +
                                  static_cast<std::size_t>(video_index) * pixels_per_frame;
            rgb_to_nv12(source, width, height, nv12);
            const auto end_time = media_time(video_index + 1, fps);
            auto sample =
                make_sample(nv12.data(), nv12.size(), next_video_time, end_time - next_video_time);
            check_hresult(writer->WriteSample(video_stream, sample.Get()),
                          "IMFSinkWriter::WriteSample(video)");
            ++video_index;
        } else {
            const auto chunk_frames = std::min(kAudioChunkFrames, audio_frames - audio_offset);
            pcm.resize(static_cast<std::size_t>(chunk_frames) * audio_channels);
            const auto scalar_offset = static_cast<std::size_t>(audio_offset) * audio_channels;
            for (std::size_t index = 0; index < pcm.size(); ++index) {
                const float value =
                    std::clamp(result.audio.samples[scalar_offset + index], -1.0F, 1.0F);
                pcm[index] = static_cast<std::int16_t>(
                    std::lround(value * (value < 0.0F ? 32768.0F : 32767.0F)));
            }
            const auto end_time = media_time(audio_offset + chunk_frames, audio_rate);
            auto sample = make_sample(pcm.data(), pcm.size() * sizeof(std::int16_t),
                                      next_audio_time, end_time - next_audio_time);
            check_hresult(writer->WriteSample(audio_stream, sample.Get()),
                          "IMFSinkWriter::WriteSample(audio)");
            audio_offset += chunk_frames;
        }
    }

    check_hresult(writer->Finalize(), "IMFSinkWriter::Finalize");
#else
    (void)result;
    (void)path;
    throw std::runtime_error("native MP4 output is available on Windows through Media Foundation");
#endif
}

VideoClipInput read_video_file(const std::string& path) {
#if defined(_WIN32)
    if (path.empty() || std::filesystem::is_directory(path))
        throw std::runtime_error("read_video_file requires a media file path");
    MediaFoundationSession session;
    ComPtr<IMFAttributes> attributes;
    check_hresult(MFCreateAttributes(&attributes, 2), "MFCreateAttributes(source reader)");
    check_hresult(attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE),
                  "enable source-reader hardware transforms");
    check_hresult(attributes->SetUINT32(MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, TRUE),
                  "enable source-reader advanced video processing");

    ComPtr<IMFSourceReader> reader;
    check_hresult(MFCreateSourceReaderFromURL(std::filesystem::path(path).wstring().c_str(),
                                              attributes.Get(), &reader),
                  "MFCreateSourceReaderFromURL");
    const auto declared_duration = presentation_duration(*reader.Get());
    check_hresult(reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE),
                  "disable source-reader streams");

    DWORD video_stream_index = MAXDWORD;
    DWORD audio_stream_index = MAXDWORD;
    ComPtr<IMFMediaType> native_video;
    ComPtr<IMFMediaType> native_audio;
    for (DWORD stream = 0; stream < 64; ++stream) {
        ComPtr<IMFMediaType> native_type;
        const HRESULT type_result = reader->GetNativeMediaType(stream, 0, &native_type);
        if (type_result == MF_E_INVALIDSTREAMNUMBER)
            break;
        if (FAILED(type_result))
            continue;
        GUID major_type{};
        if (FAILED(native_type->GetGUID(MF_MT_MAJOR_TYPE, &major_type)))
            continue;
        if (major_type == MFMediaType_Video && video_stream_index == MAXDWORD) {
            video_stream_index = stream;
            native_video = native_type;
        } else if (major_type == MFMediaType_Audio && audio_stream_index == MAXDWORD) {
            audio_stream_index = stream;
            native_audio = native_type;
        }
    }
    if (video_stream_index == MAXDWORD || !native_video)
        throw std::runtime_error("reference media contains no video stream");
    const CodecPaddingAllowance padding = codec_padding_allowance(native_audio.Get());
    if (declared_duration &&
        *declared_duration > kReferenceLimitSeconds * kMediaTicksPerSecond + padding.duration) {
        throw std::runtime_error("reference media presentation exceeds the 15-second public limit");
    }
    const bool trim_codec_padding = padding.frames > 0;
    UINT32 fps_numerator = 0;
    UINT32 fps_denominator = 0;
    check_hresult(
        MFGetAttributeRatio(native_video.Get(), MF_MT_FRAME_RATE, &fps_numerator, &fps_denominator),
        "read source frame rate");
    if (fps_numerator == 0 || fps_denominator == 0 ||
        static_cast<std::uint64_t>(fps_numerator) >
            static_cast<std::uint64_t>(kMaximumReferenceSourceFps) * fps_denominator) {
        throw std::runtime_error(
            "reference video source frame rate must be positive and no greater than 240 fps");
    }
    UINT32 source_width = 0;
    UINT32 source_height = 0;
    check_hresult(
        MFGetAttributeSize(native_video.Get(), MF_MT_FRAME_SIZE, &source_width, &source_height),
        "read source frame size");
    const auto [requested_width, requested_height] =
        detail::reference_video_decode_size(source_width, source_height);
    const ReferenceVideoSize requested_size{requested_width, requested_height};
    check_hresult(reader->SetStreamSelection(video_stream_index, TRUE), "select source video");
    const auto requested_video =
        source_reader_video_type(requested_size.width, requested_size.height);
    check_hresult(reader->SetCurrentMediaType(video_stream_index, nullptr, requested_video.Get()),
                  "SetCurrentMediaType(video RGB32)");
    ComPtr<IMFMediaType> decoded_video;
    check_hresult(reader->GetCurrentMediaType(video_stream_index, &decoded_video),
                  "GetCurrentMediaType(video)");
    UINT32 width = 0;
    UINT32 height = 0;
    check_hresult(MFGetAttributeSize(decoded_video.Get(), MF_MT_FRAME_SIZE, &width, &height),
                  "read decoded frame size");
    if (width != requested_size.width || height != requested_size.height ||
        static_cast<std::uint64_t>(width) * height > kReferenceDecodedMaxPixels) {
        throw std::runtime_error(
            "Media Foundation did not honor the bounded reference-video decode canvas");
    }
    UINT32 raw_stride = 0;
    LONG stride = static_cast<LONG>(width * 4);
    if (SUCCEEDED(decoded_video->GetUINT32(MF_MT_DEFAULT_STRIDE, &raw_stride)))
        stride = static_cast<LONG>(raw_stride);

    bool has_audio = false;
    std::uint32_t audio_rate = 0;
    std::uint32_t audio_channels = 0;
    if (audio_stream_index != MAXDWORD && native_audio) {
        UINT32 rate = 0;
        UINT32 channels = 0;
        if (SUCCEEDED(native_audio->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &rate)) &&
            SUCCEEDED(native_audio->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &channels)) && rate > 0 &&
            (channels == 1 || channels == 2)) {
            audio_rate = rate;
            audio_channels = channels;
            check_hresult(reader->SetStreamSelection(audio_stream_index, TRUE),
                          "select source audio");
            const auto requested_audio = source_reader_audio_type(audio_rate, audio_channels);
            check_hresult(
                reader->SetCurrentMediaType(audio_stream_index, nullptr, requested_audio.Get()),
                "SetCurrentMediaType(audio float)");
            ComPtr<IMFMediaType> decoded_audio;
            check_hresult(reader->GetCurrentMediaType(audio_stream_index, &decoded_audio),
                          "GetCurrentMediaType(audio)");
            UINT32 decoded_rate = 0;
            UINT32 decoded_channels = 0;
            check_hresult(decoded_audio->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &decoded_rate),
                          "read decoded soundtrack sample rate");
            check_hresult(decoded_audio->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &decoded_channels),
                          "read decoded soundtrack channel count");
            if (decoded_rate != audio_rate || decoded_channels != audio_channels) {
                throw std::runtime_error(
                    "Media Foundation did not honor the requested soundtrack PCM format");
            }
            has_audio = true;
        }
    }

    if (width == 0 || height == 0 || fps_numerator == 0 || fps_denominator == 0 ||
        width > static_cast<UINT32>(std::numeric_limits<int32_t>::max()) ||
        height > static_cast<UINT32>(std::numeric_limits<int32_t>::max()) ||
        fps_numerator > static_cast<UINT32>(std::numeric_limits<int32_t>::max()) ||
        fps_denominator > static_cast<UINT32>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("decoded video metadata is outside the public C++ value range");

    VideoClipInput result;
    result.width = static_cast<int32_t>(width);
    result.height = static_cast<int32_t>(height);
    result.channels = 3;
    const bool downsample_video = static_cast<std::uint64_t>(fps_numerator) >
                                  static_cast<std::uint64_t>(kReferenceFps) * fps_denominator;
    result.fps_numerator = static_cast<int32_t>(downsample_video ? kReferenceFps : fps_numerator);
    result.fps_denominator = static_cast<int32_t>(downsample_video ? 1 : fps_denominator);
    result.soundtrack.sample_rate = static_cast<int32_t>(audio_rate);
    result.soundtrack.channels = static_cast<int32_t>(audio_channels);

    const std::uint64_t maximum_source_video_frames = std::min<std::uint64_t>(
        detail::reference_video_frame_ceiling(fps_numerator, fps_denominator),
        static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max()));
    const std::uint64_t maximum_video_frames =
        downsample_video ? kReferenceLimitSeconds * kReferenceFps : maximum_source_video_frames;
    const std::uint64_t frame_scalars = static_cast<std::uint64_t>(width) * height * 3;
    if (maximum_video_frames >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / frame_scalars)
        throw std::runtime_error("bounded reference-video RGB allocation exceeds host range");
    const std::size_t maximum_rgb_scalars =
        static_cast<std::size_t>(maximum_video_frames * frame_scalars);
    const std::size_t maximum_soundtrack_scalars =
        has_audio ? maximum_reference_audio_scalars(audio_rate, audio_channels) : 0;

    bool video_done = false;
    bool audio_done = !has_audio;
    LONGLONG last_video_tick = std::numeric_limits<LONGLONG>::min();
    LONGLONG last_audio_tick = std::numeric_limits<LONGLONG>::min();
    std::size_t consecutive_empty_reads = 0;
    std::size_t total_empty_reads = 0;
    std::uint64_t source_video_frames = 0;
    detail::ReferenceAudioDecodeState audio_decode_state;
    while (!video_done || !audio_done) {
        DWORD stream_index = 0;
        DWORD flags = 0;
        LONGLONG timestamp = 0;
        ComPtr<IMFSample> sample;
        check_hresult(reader->ReadSample(MF_SOURCE_READER_ANY_STREAM, 0, &stream_index, &flags,
                                         &timestamp, &sample),
                      "IMFSourceReader::ReadSample");
        reject_source_reader_error(flags, "reference media source reader");
        if ((flags & (MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED |
                      MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)) != 0)
            throw std::runtime_error("reference video changes media type mid-stream");
        if ((flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0) {
            if (stream_index == video_stream_index)
                video_done = true;
            else if (stream_index == audio_stream_index)
                audio_done = true;
            else
                throw std::runtime_error(
                    "reference media ended an unexpected source-reader stream");
            continue;
        }
        if (stream_index != video_stream_index && stream_index != audio_stream_index)
            throw std::runtime_error("reference media returned an unexpected source-reader stream");
        const bool audio_padding_event = trim_codec_padding && stream_index == audio_stream_index;
        const std::uint64_t event_padding_ticks =
            audio_padding_event ? static_cast<std::uint64_t>(padding.duration) : 0;
        if (!audio_padding_event)
            reject_reference_timestamp(timestamp, "reference media source reader");
        if ((flags & MF_SOURCE_READERF_STREAMTICK) != 0) {
            if (sample)
                throw std::runtime_error(
                    "reference media returned a sample for an empty stream tick");
            auto& last_tick =
                stream_index == video_stream_index ? last_video_tick : last_audio_tick;
            note_empty_source_reader_event(flags, timestamp, last_tick, consecutive_empty_reads,
                                           total_empty_reads, event_padding_ticks,
                                           "reference media source reader");
            continue;
        }
        if (!sample) {
            auto& last_tick =
                stream_index == video_stream_index ? last_video_tick : last_audio_tick;
            note_empty_source_reader_event(flags, timestamp, last_tick, consecutive_empty_reads,
                                           total_empty_reads, event_padding_ticks,
                                           "reference media source reader");
            continue;
        }
        reject_empty_decoded_sample(sample.Get(),
                                    stream_index == video_stream_index ? "video" : "audio");
        if (stream_index == video_stream_index) {
            if (source_video_frames >= maximum_source_video_frames)
                throw std::runtime_error(
                    "decoded reference video exceeds the 15-second public limit");
            reject_video_sample_timeline(sample.Get(), timestamp, fps_numerator, fps_denominator);
            bool retain = true;
            if (downsample_video) {
                const auto begin = rounded_reference_frame_slot(source_video_frames, fps_numerator,
                                                                fps_denominator);
                const auto end = rounded_reference_frame_slot(source_video_frames + 1,
                                                              fps_numerator, fps_denominator);
                retain = end > begin;
            }
            ++source_video_frames;
            if (retain) {
                if (static_cast<std::uint64_t>(result.num_frames) >= maximum_video_frames)
                    throw std::runtime_error(
                        "decoded reference video exceeds the bounded 24-fps frame count");
                append_rgb32_frame(sample.Get(), width, height, stride, result.pixels,
                                   maximum_rgb_scalars);
                ++result.num_frames;
            }
        } else if (stream_index == audio_stream_index) {
            append_float_audio(sample.Get(), audio_rate, audio_channels, timestamp, padding.frames,
                               audio_decode_state, result.soundtrack.samples,
                               maximum_soundtrack_scalars);
        }
        consecutive_empty_reads = 0;
    }

    if (result.num_frames <= 0)
        throw std::runtime_error("reference video contains no decoded frames");
    if (!result.soundtrack.samples.empty()) {
        if (result.soundtrack.samples.size() >
            static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("decoded soundtrack exceeds the public C++ value range");
        result.soundtrack.num_samples = static_cast<int32_t>(result.soundtrack.samples.size());
    }
    return result;
#else
    (void)path;
    throw std::runtime_error(
        "native media-file input is available on Windows through Media Foundation");
#endif
}

AudioResult read_audio_file(const std::string& path) {
#if defined(_WIN32)
    if (path.empty() || std::filesystem::is_directory(path))
        throw std::runtime_error("read_audio_file requires a media file path");
    MediaFoundationSession session;
    ComPtr<IMFAttributes> attributes;
    check_hresult(MFCreateAttributes(&attributes, 1), "MFCreateAttributes(audio source reader)");
    check_hresult(attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE),
                  "enable audio source-reader hardware transforms");

    ComPtr<IMFSourceReader> reader;
    check_hresult(MFCreateSourceReaderFromURL(std::filesystem::path(path).wstring().c_str(),
                                              attributes.Get(), &reader),
                  "MFCreateSourceReaderFromURL(audio)");
    const auto declared_duration = presentation_duration(*reader.Get());
    check_hresult(reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE),
                  "disable audio source-reader streams");

    DWORD audio_stream_index = MAXDWORD;
    ComPtr<IMFMediaType> native_audio;
    for (DWORD stream = 0; stream < 64; ++stream) {
        ComPtr<IMFMediaType> native_type;
        const HRESULT type_result = reader->GetNativeMediaType(stream, 0, &native_type);
        if (type_result == MF_E_INVALIDSTREAMNUMBER)
            break;
        if (FAILED(type_result))
            continue;
        GUID major_type{};
        if (SUCCEEDED(native_type->GetGUID(MF_MT_MAJOR_TYPE, &major_type)) &&
            major_type == MFMediaType_Audio) {
            audio_stream_index = stream;
            native_audio = native_type;
            break;
        }
    }
    if (audio_stream_index == MAXDWORD || !native_audio)
        throw std::runtime_error("reference audio file contains no audio stream");

    UINT32 native_rate = 0;
    UINT32 native_channels = 0;
    check_hresult(native_audio->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &native_rate),
                  "read source audio sample rate");
    check_hresult(native_audio->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &native_channels),
                  "read source audio channel count");
    if (native_rate == 0 ||
        native_rate > static_cast<UINT32>(std::numeric_limits<int32_t>::max()) ||
        (native_channels != 1 && native_channels != 2))
        throw std::runtime_error(
            "reference audio metadata is outside the supported mono/stereo C++ value range");
    const CodecPaddingAllowance padding = codec_padding_allowance(native_audio.Get());
    if (declared_duration &&
        *declared_duration > kReferenceLimitSeconds * kMediaTicksPerSecond + padding.duration) {
        throw std::runtime_error("reference audio presentation exceeds the 15-second public limit");
    }
    const bool trim_codec_padding = padding.frames > 0;

    check_hresult(reader->SetStreamSelection(audio_stream_index, TRUE), "select source audio");
    const auto requested_audio = source_reader_audio_type(native_rate, native_channels);
    check_hresult(reader->SetCurrentMediaType(audio_stream_index, nullptr, requested_audio.Get()),
                  "SetCurrentMediaType(audio float)");
    ComPtr<IMFMediaType> decoded_audio;
    check_hresult(reader->GetCurrentMediaType(audio_stream_index, &decoded_audio),
                  "GetCurrentMediaType(audio)");
    UINT32 decoded_rate = 0;
    UINT32 decoded_channels = 0;
    check_hresult(decoded_audio->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &decoded_rate),
                  "read decoded audio sample rate");
    check_hresult(decoded_audio->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &decoded_channels),
                  "read decoded audio channel count");
    if (decoded_rate == 0 ||
        decoded_rate > static_cast<UINT32>(std::numeric_limits<int32_t>::max()) ||
        (decoded_channels != 1 && decoded_channels != 2))
        throw std::runtime_error(
            "decoded audio metadata is outside the supported mono/stereo C++ value range");
    if (decoded_rate != native_rate || decoded_channels != native_channels) {
        throw std::runtime_error("Media Foundation did not honor the requested decoded PCM format");
    }

    AudioResult result;
    result.sample_rate = static_cast<int32_t>(decoded_rate);
    result.channels = static_cast<int32_t>(decoded_channels);
    const std::size_t maximum_scalars =
        maximum_reference_audio_scalars(decoded_rate, decoded_channels);
    LONGLONG last_tick_timestamp = std::numeric_limits<LONGLONG>::min();
    std::size_t consecutive_empty_reads = 0;
    std::size_t total_empty_reads = 0;
    detail::ReferenceAudioDecodeState decode_state;
    while (true) {
        DWORD stream_index = 0;
        DWORD flags = 0;
        LONGLONG timestamp = 0;
        ComPtr<IMFSample> sample;
        check_hresult(
            reader->ReadSample(audio_stream_index, 0, &stream_index, &flags, &timestamp, &sample),
            "IMFSourceReader::ReadSample(audio)");
        reject_source_reader_error(flags, "reference audio source reader");
        if ((flags & (MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED |
                      MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)) != 0)
            throw std::runtime_error("reference audio changes media type mid-stream");
        if ((flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0)
            break;
        if (!trim_codec_padding)
            reject_reference_timestamp(timestamp, "reference audio source reader");
        if (stream_index != audio_stream_index)
            throw std::runtime_error("reference audio returned an unexpected source-reader stream");
        if ((flags & MF_SOURCE_READERF_STREAMTICK) != 0) {
            if (sample)
                throw std::runtime_error(
                    "reference audio returned a sample for an empty stream tick");
            note_empty_source_reader_event(
                flags, timestamp, last_tick_timestamp, consecutive_empty_reads, total_empty_reads,
                static_cast<std::uint64_t>(padding.duration), "reference audio source reader");
            continue;
        }
        if (!sample) {
            note_empty_source_reader_event(
                flags, timestamp, last_tick_timestamp, consecutive_empty_reads, total_empty_reads,
                static_cast<std::uint64_t>(padding.duration), "reference audio source reader");
            continue;
        }
        reject_empty_decoded_sample(sample.Get(), "audio");
        append_float_audio(sample.Get(), decoded_rate, decoded_channels, timestamp, padding.frames,
                           decode_state, result.samples, maximum_scalars);
        consecutive_empty_reads = 0;
    }
    if (result.samples.empty())
        throw std::runtime_error("reference audio file contains no decoded samples");
    if (result.samples.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("decoded audio exceeds the public C++ value range");
    result.num_samples = static_cast<int32_t>(result.samples.size());
    return result;
#else
    if (lowercase_extension(path) == ".wav")
        return trtmc::io::read_wav_interleaved(path);
    throw std::runtime_error(
        "compressed native media-file input is available on Windows through Media Foundation");
#endif
}

} // namespace trtmc::cli
