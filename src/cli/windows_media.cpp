/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/windows_media.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
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

#if defined(_WIN32)

using Microsoft::WRL::ComPtr;

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

ComPtr<IMFMediaType> source_reader_video_type() {
    ComPtr<IMFMediaType> type;
    check_hresult(MFCreateMediaType(&type), "MFCreateMediaType(source video)");
    check_hresult(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video), "source video major type");
    check_hresult(type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32), "source video subtype");
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
                        std::vector<float>& pixels) {
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

void append_float_audio(IMFSample* sample, std::uint32_t channels, std::vector<float>& samples) {
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
    if (current_length % (sizeof(float) * channels) != 0) {
        (void)buffer->Unlock();
        throw std::runtime_error("Media Foundation returned misaligned float audio");
    }
    const auto scalar_count = current_length / sizeof(float);
    const auto old_size = samples.size();
    if (scalar_count > std::numeric_limits<std::size_t>::max() - old_size) {
        (void)buffer->Unlock();
        throw std::runtime_error("decoded audio exceeds host address space");
    }
    samples.resize(old_size + scalar_count);
    std::memcpy(samples.data() + old_size, data, current_length);
    check_hresult(buffer->Unlock(), "IMFMediaBuffer::Unlock(audio)");
}

#endif

} // namespace

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
    check_hresult(attributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE),
                  "enable source-reader video processing");

    ComPtr<IMFSourceReader> reader;
    check_hresult(MFCreateSourceReaderFromURL(std::filesystem::path(path).wstring().c_str(),
                                              attributes.Get(), &reader),
                  "MFCreateSourceReaderFromURL");
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
    UINT32 fps_numerator = 0;
    UINT32 fps_denominator = 0;
    check_hresult(
        MFGetAttributeRatio(native_video.Get(), MF_MT_FRAME_RATE, &fps_numerator, &fps_denominator),
        "read source frame rate");
    check_hresult(reader->SetStreamSelection(video_stream_index, TRUE), "select source video");
    const auto requested_video = source_reader_video_type();
    check_hresult(reader->SetCurrentMediaType(video_stream_index, nullptr, requested_video.Get()),
                  "SetCurrentMediaType(video RGB32)");
    ComPtr<IMFMediaType> decoded_video;
    check_hresult(reader->GetCurrentMediaType(video_stream_index, &decoded_video),
                  "GetCurrentMediaType(video)");
    UINT32 width = 0;
    UINT32 height = 0;
    check_hresult(MFGetAttributeSize(decoded_video.Get(), MF_MT_FRAME_SIZE, &width, &height),
                  "read decoded frame size");
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
    result.fps_numerator = static_cast<int32_t>(fps_numerator);
    result.fps_denominator = static_cast<int32_t>(fps_denominator);
    result.soundtrack.sample_rate = static_cast<int32_t>(audio_rate);
    result.soundtrack.channels = static_cast<int32_t>(audio_channels);

    bool video_done = false;
    bool audio_done = !has_audio;
    while (!video_done || !audio_done) {
        DWORD stream_index = 0;
        DWORD flags = 0;
        LONGLONG timestamp = 0;
        ComPtr<IMFSample> sample;
        check_hresult(reader->ReadSample(MF_SOURCE_READER_ANY_STREAM, 0, &stream_index, &flags,
                                         &timestamp, &sample),
                      "IMFSourceReader::ReadSample");
        (void)timestamp;
        if ((flags & (MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED |
                      MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)) != 0)
            throw std::runtime_error("reference video changes media type mid-stream");
        if ((flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0) {
            if (stream_index == video_stream_index)
                video_done = true;
            else if (stream_index == audio_stream_index)
                audio_done = true;
            continue;
        }
        if (!sample)
            continue;
        if (stream_index == video_stream_index) {
            append_rgb32_frame(sample.Get(), width, height, stride, result.pixels);
            ++result.num_frames;
        } else if (stream_index == audio_stream_index) {
            append_float_audio(sample.Get(), audio_channels, result.soundtrack.samples);
        }
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

} // namespace trtmc::cli
