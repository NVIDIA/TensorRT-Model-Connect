/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// trtmc_io.hpp — header-only file I/O utilities for trtmc pipeline results.
//
// Usage:
//   #include <trtmc/trtmc_io.hpp>
//   auto pipe = trtmc::load("model.bundle");
//   auto img = pipe->generate_image("a cat");
//   trtmc::io::save_png(img, "output.png");

#include "trtmc/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::io {

// Write a WAV file from AudioResult (IEEE float32 mono).
inline void write_wav(const AudioResult& audio, const std::string& path) {
    if (audio.samples.empty())
        throw std::runtime_error("write_wav: empty audio");

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("write_wav: cannot open " + path);

    const int32_t num_samples = static_cast<int32_t>(audio.samples.size());
    const int32_t sample_rate = audio.sample_rate;
    const int16_t num_channels = 1;
    const int16_t bits_per_sample = 32;
    const int32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
    const int16_t block_align = static_cast<int16_t>(num_channels * (bits_per_sample / 8));
    const int32_t data_size = num_samples * block_align;
    const int32_t chunk_size = 36 + data_size;

    // RIFF header
    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&chunk_size), 4);
    f.write("WAVE", 4);

    // fmt chunk
    f.write("fmt ", 4);
    int32_t fmt_size = 16;
    int16_t audio_format = 3; // IEEE float
    f.write(reinterpret_cast<const char*>(&fmt_size), 4);
    f.write(reinterpret_cast<const char*>(&audio_format), 2);
    f.write(reinterpret_cast<const char*>(&num_channels), 2);
    f.write(reinterpret_cast<const char*>(&sample_rate), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&block_align), 2);
    f.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    // data chunk
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_size), 4);
    f.write(reinterpret_cast<const char*>(audio.samples.data()),
            static_cast<std::streamsize>(data_size));
}

// Write a WAV file from channel-major multichannel audio. WAV stores sample
// frames interleaved, so [channel, sample] input is reordered on write.
inline void write_wav(const MultiChannelAudioResult& audio, const std::string& path) {
    if (audio.samples.empty())
        throw std::runtime_error("write_wav: empty audio");
    if (audio.num_samples <= 0)
        throw std::runtime_error("write_wav: num_samples must be positive");
    if (audio.sample_rate <= 0)
        throw std::runtime_error("write_wav: sample_rate must be positive");
    if (audio.num_channels <= 0 || audio.num_channels > std::numeric_limits<uint16_t>::max())
        throw std::runtime_error("write_wav: num_channels is out of range");

    const auto num_samples = static_cast<std::size_t>(audio.num_samples);
    const auto num_channels = static_cast<std::size_t>(audio.num_channels);
    if (num_samples > std::numeric_limits<std::size_t>::max() / num_channels ||
        audio.samples.size() != num_samples * num_channels)
        throw std::runtime_error(
            "write_wav: sample buffer size must equal num_samples * num_channels");

    const auto sample_count = num_samples * num_channels;
    if (sample_count > (std::numeric_limits<uint32_t>::max() - 36U) / sizeof(float))
        throw std::runtime_error("write_wav: audio is too large for a WAV file");

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("write_wav: cannot open " + path);

    const auto channels = static_cast<uint16_t>(audio.num_channels);
    constexpr uint16_t bits_per_sample = 32;
    const auto block_align_wide = static_cast<uint32_t>(channels) * (bits_per_sample / 8U);
    if (block_align_wide > std::numeric_limits<uint16_t>::max())
        throw std::runtime_error("write_wav: channel count exceeds WAV block alignment");
    const auto block_align = static_cast<uint16_t>(block_align_wide);
    const auto byte_rate_wide = static_cast<uint64_t>(audio.sample_rate) * block_align;
    if (byte_rate_wide > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("write_wav: sample rate and channel count exceed WAV limits");
    const auto sample_rate = static_cast<uint32_t>(audio.sample_rate);
    const auto byte_rate = static_cast<uint32_t>(byte_rate_wide);
    const auto data_size = static_cast<uint32_t>(sample_count * sizeof(float));
    const uint32_t chunk_size = 36U + data_size;

    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&chunk_size), 4);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    constexpr uint32_t fmt_size = 16;
    constexpr int16_t audio_format = 3; // IEEE float
    f.write(reinterpret_cast<const char*>(&fmt_size), 4);
    f.write(reinterpret_cast<const char*>(&audio_format), 2);
    f.write(reinterpret_cast<const char*>(&channels), 2);
    f.write(reinterpret_cast<const char*>(&sample_rate), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&block_align), 2);
    f.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_size), 4);
    std::vector<float> interleaved(sample_count);
    for (std::size_t sample = 0; sample < num_samples; ++sample) {
        for (std::size_t channel = 0; channel < num_channels; ++channel) {
            interleaved[sample * num_channels + channel] =
                audio.samples[channel * num_samples + sample];
        }
    }
    f.write(reinterpret_cast<const char*>(interleaved.data()),
            static_cast<std::streamsize>(data_size));
    if (!f)
        throw std::runtime_error("write_wav: failed to write " + path);
}

namespace detail {

struct ParsedWav {
    uint16_t audio_format{0};
    uint16_t num_channels{0};
    uint32_t sample_rate{0};
    uint16_t block_align{0};
    uint16_t bits_per_sample{0};
    std::vector<char> data;
};

inline uint16_t read_wav_u16(const char* bytes) {
    return static_cast<uint16_t>(static_cast<uint8_t>(bytes[0])) |
           static_cast<uint16_t>(static_cast<uint8_t>(bytes[1]) << 8U);
}

inline uint32_t read_wav_u32(const char* bytes) {
    return static_cast<uint32_t>(static_cast<uint8_t>(bytes[0])) |
           (static_cast<uint32_t>(static_cast<uint8_t>(bytes[1])) << 8U) |
           (static_cast<uint32_t>(static_cast<uint8_t>(bytes[2])) << 16U) |
           (static_cast<uint32_t>(static_cast<uint8_t>(bytes[3])) << 24U);
}

inline ParsedWav parse_wav(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("read_wav: cannot open " + path);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(input),
                                  std::istreambuf_iterator<char>()};
    if (bytes.size() < 12U || std::memcmp(bytes.data(), "RIFF", 4) != 0)
        throw std::runtime_error("read_wav: not a RIFF file");
    if (std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
        throw std::runtime_error("read_wav: not a WAVE file");
    const auto riff_size = static_cast<std::size_t>(read_wav_u32(bytes.data() + 4));
    if (riff_size < 4U)
        throw std::runtime_error("read_wav: invalid RIFF size in " + path);
    if (riff_size > bytes.size() - 8U)
        throw std::runtime_error("read_wav: truncated RIFF container in " + path);
    const std::size_t riff_end = 8U + riff_size;

    ParsedWav parsed;
    bool found_fmt = false;
    bool found_data = false;
    std::size_t offset = 12U;
    while (riff_end - offset >= 8U) {
        const char* header = bytes.data() + offset;
        const auto chunk_size = static_cast<std::size_t>(read_wav_u32(header + 4));
        const std::size_t data_offset = offset + 8U;
        if (chunk_size > riff_end - data_offset)
            throw std::runtime_error("read_wav: truncated RIFF chunk in " + path);

        if (std::memcmp(header, "fmt ", 4) == 0) {
            if (chunk_size < 16U)
                throw std::runtime_error("read_wav: fmt chunk is too small in " + path);
            const char* format = bytes.data() + data_offset;
            parsed.audio_format = read_wav_u16(format);
            parsed.num_channels = read_wav_u16(format + 2);
            parsed.sample_rate = read_wav_u32(format + 4);
            parsed.block_align = read_wav_u16(format + 12);
            parsed.bits_per_sample = read_wav_u16(format + 14);
            found_fmt = true;
        } else if (std::memcmp(header, "data", 4) == 0 && !found_data) {
            parsed.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_offset),
                               bytes.begin() +
                                   static_cast<std::ptrdiff_t>(data_offset + chunk_size));
            found_data = true;
        }

        offset = data_offset + chunk_size;
        if ((chunk_size & 1U) != 0U) {
            if (offset == riff_end)
                throw std::runtime_error("read_wav: missing RIFF chunk padding in " + path);
            ++offset;
        }
    }
    if (offset != riff_end)
        throw std::runtime_error("read_wav: truncated RIFF chunk header in " + path);

    if (!found_fmt || !found_data)
        throw std::runtime_error("read_wav: missing fmt or data chunk in " + path);
    if (parsed.num_channels == 0U)
        throw std::runtime_error("read_wav: channel count must be positive in " + path);
    if (parsed.sample_rate == 0U ||
        parsed.sample_rate > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("read_wav: sample rate is out of range in " + path);

    const bool is_pcm16 = parsed.audio_format == 1U && parsed.bits_per_sample == 16U;
    const bool is_float32 = parsed.audio_format == 3U && parsed.bits_per_sample == 32U;
    if (!is_pcm16 && !is_float32)
        throw std::runtime_error("read_wav: expected PCM16 or IEEE-float32 WAV in " + path);
    const auto bytes_per_sample = static_cast<uint16_t>(parsed.bits_per_sample / 8U);
    const auto expected_align = static_cast<uint32_t>(parsed.num_channels) * bytes_per_sample;
    if (expected_align > std::numeric_limits<uint16_t>::max() ||
        parsed.block_align != expected_align)
        throw std::runtime_error("read_wav: invalid block alignment in " + path);
    if (parsed.data.size() % parsed.block_align != 0U)
        throw std::runtime_error("read_wav: partial sample frame in " + path);
    const auto num_frames = parsed.data.size() / parsed.block_align;
    if (num_frames > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("read_wav: too many sample frames in " + path);
    return parsed;
}

inline std::vector<float> decode_wav_interleaved(const ParsedWav& parsed) {
    const std::size_t bytes_per_sample = parsed.bits_per_sample / 8U;
    const std::size_t sample_count = parsed.data.size() / bytes_per_sample;
    std::vector<float> samples(sample_count);
    for (std::size_t index = 0; index < sample_count; ++index) {
        const char* source = parsed.data.data() + index * bytes_per_sample;
        if (parsed.audio_format == 3U) {
            const uint32_t bits = read_wav_u32(source);
            std::memcpy(&samples[index], &bits, sizeof(float));
        } else {
            const uint16_t bits = read_wav_u16(source);
            int16_t pcm = 0;
            std::memcpy(&pcm, &bits, sizeof(pcm));
            samples[index] = static_cast<float>(pcm) / 32768.0F;
        }
    }
    return samples;
}

} // namespace detail

// Read a one- or two-channel PCM16/IEEE-float32 WAV without downmixing. Samples
// are returned channel-major as [channel, sample].
inline MultiChannelAudioResult read_wav_multichannel(const std::string& path) {
    const auto parsed = detail::parse_wav(path);
    if (parsed.num_channels > 2U)
        throw std::runtime_error("read_wav_multichannel: expected one or two channels in " + path);
    const auto interleaved = detail::decode_wav_interleaved(parsed);
    const auto num_channels = static_cast<std::size_t>(parsed.num_channels);
    const auto num_samples = interleaved.size() / num_channels;

    MultiChannelAudioResult result;
    result.samples.resize(interleaved.size());
    result.num_samples = static_cast<int32_t>(num_samples);
    result.sample_rate = static_cast<int32_t>(parsed.sample_rate);
    result.num_channels = static_cast<int32_t>(parsed.num_channels);
    for (std::size_t sample = 0; sample < num_samples; ++sample) {
        for (std::size_t channel = 0; channel < num_channels; ++channel) {
            result.samples[channel * num_samples + sample] =
                interleaved[sample * num_channels + channel];
        }
    }
    return result;
}

// Read a PCM16/IEEE-float32 WAV into mono float32. Multichannel files retain
// the historical behavior of averaging every channel for each sample frame.
inline AudioResult read_wav(const std::string& path) {
    const auto parsed = detail::parse_wav(path);
    const auto interleaved = detail::decode_wav_interleaved(parsed);
    const auto num_channels = static_cast<std::size_t>(parsed.num_channels);
    const auto num_samples = interleaved.size() / num_channels;

    AudioResult result;
    result.samples.resize(num_samples);
    result.num_samples = static_cast<int32_t>(num_samples);
    result.sample_rate = static_cast<int32_t>(parsed.sample_rate);
    for (std::size_t sample = 0; sample < num_samples; ++sample) {
        float sum = 0.0F;
        for (std::size_t channel = 0; channel < num_channels; ++channel)
            sum += interleaved[sample * num_channels + channel];
        result.samples[sample] = sum / static_cast<float>(num_channels);
    }
    return result;
}

// Loaded image: float RGB pixels in HWC layout [height * width * 3], values in [0, 1].
struct LoadedImage {
    std::vector<float> pixels; // [H * W * 3] float32 in [0, 1]
    int32_t height{0};
    int32_t width{0};

    bool empty() const { return pixels.empty(); }
};

// Load an image file (JPEG, PNG, BMP, etc.) and return float RGB pixels.
// Uses stb_image internally (linked via trtmc_core).
// Throws on file-not-found; returns empty LoadedImage on decode failure.
LoadedImage read_image(const std::string& path);

// Save float RGB pixels (HWC layout, values in [0, 1]) to a PNG file.
// Out-of-range values are clamped to [0, 1] before being quantised to uint8.
// Uses stb_image_write internally (linked via trtmc_core).
// Throws std::runtime_error on size mismatch or write failure.
void save_png(const std::string& path, const std::vector<float>& hwc_pixels, int width, int height);

// Convenience overload: save an ImageResult (first frame only) directly.
// For single-frame results (num_frames <= 1) writes result.pixels;
// for multi-frame results (video) writes only frame 0 — use the
// generate-video CLI command instead to dump every frame.
inline void save_png(const ImageResult& image, const std::string& path) {
    const auto frame_pixels =
        static_cast<std::size_t>(image.height) * static_cast<std::size_t>(image.width) * 3U;
    if (image.pixels.size() < frame_pixels)
        throw std::runtime_error("save_png: ImageResult pixel buffer is smaller than H*W*3");
    if (image.pixels.size() == frame_pixels) {
        save_png(path, image.pixels, image.width, image.height);
        return;
    }
    // Multi-frame: only frame 0 is written.
    std::vector<float> frame0(image.pixels.begin(),
                              image.pixels.begin() + static_cast<std::ptrdiff_t>(frame_pixels));
    save_png(path, frame0, image.width, image.height);
}

// Legacy placeholder (prefer read_image).
inline std::vector<float> decode_image(const std::string& path, int& h, int& w) {
    auto img = read_image(path);
    h = img.height;
    w = img.width;
    return std::move(img.pixels);
}

} // namespace trtmc::io
