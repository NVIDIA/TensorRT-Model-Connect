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
#include <cstddef>
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

class WavFormatError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

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

namespace wav_detail {

struct ParsedWav {
    uint16_t format{0};
    uint16_t channels{0};
    uint32_t sample_rate{0};
    uint16_t bits_per_sample{0};
    std::size_t data_offset{0};
    std::size_t data_size{0};
    bool have_format{false};
    bool have_data{false};
};

inline uint16_t read_u16_le(const char* data) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(data);
    return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8U);
}

inline uint32_t read_u32_le(const char* data) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(data);
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U) |
           (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
}

inline std::vector<char> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("read_wav: cannot open input file");
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    if (input.bad())
        throw std::runtime_error("read_wav: failed to read input file");
    return bytes;
}

inline std::size_t validate_container(const std::vector<char>& bytes) {
    if (bytes.size() < 44)
        throw WavFormatError("read_wav: WAV file is too small");
    if (std::memcmp(bytes.data(), "RIFF", 4) != 0)
        throw WavFormatError("read_wav: not a RIFF file");
    if (std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
        throw WavFormatError("read_wav: not a WAVE file");

    const uint32_t declared_size = read_u32_le(bytes.data() + 4);
    if (declared_size < 4)
        throw WavFormatError("read_wav: WAV RIFF chunk is too small");
    if (static_cast<std::size_t>(declared_size) > bytes.size() - 8U)
        throw WavFormatError("read_wav: WAV contains a truncated RIFF chunk");
    return 8U + static_cast<std::size_t>(declared_size);
}

inline void require_chunk_fits(std::size_t container_end, std::size_t data_offset,
                               uint32_t chunk_size) {
    const std::size_t available = container_end - data_offset;
    const std::size_t payload_size = static_cast<std::size_t>(chunk_size);
    const std::size_t padding_size = chunk_size & 1U;
    if (payload_size > available || padding_size > available - payload_size)
        throw WavFormatError("read_wav: WAV contains a truncated chunk");
}

inline void parse_format_chunk(const std::vector<char>& bytes, std::size_t data_offset,
                               uint32_t chunk_size, ParsedWav& parsed) {
    if (chunk_size < 16)
        throw WavFormatError("read_wav: WAV fmt chunk is too small");
    parsed.format = read_u16_le(bytes.data() + data_offset);
    parsed.channels = read_u16_le(bytes.data() + data_offset + 2);
    parsed.sample_rate = read_u32_le(bytes.data() + data_offset + 4);
    parsed.bits_per_sample = read_u16_le(bytes.data() + data_offset + 14);
    parsed.have_format = true;
}

inline void parse_chunk(const std::vector<char>& bytes, const char* chunk, std::size_t data_offset,
                        uint32_t chunk_size, ParsedWav& parsed) {
    if (std::memcmp(chunk, "fmt ", 4) == 0) {
        parse_format_chunk(bytes, data_offset, chunk_size, parsed);
        return;
    }
    if (std::memcmp(chunk, "data", 4) != 0 || chunk_size == 0)
        return;
    parsed.data_offset = data_offset;
    parsed.data_size = chunk_size;
    parsed.have_data = true;
}

inline bool supported_format(const ParsedWav& parsed) {
    return (parsed.format == 1 && parsed.bits_per_sample == 16) ||
           (parsed.format == 3 && parsed.bits_per_sample == 32);
}

inline void validate_required_fields(const ParsedWav& parsed) {
    if (!parsed.have_format || !parsed.have_data)
        throw WavFormatError("read_wav: WAV must contain non-empty fmt and data chunks");
    if (parsed.channels == 0 || parsed.sample_rate == 0)
        throw WavFormatError("read_wav: WAV channels and sample rate must be positive");
    if (parsed.sample_rate > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
        throw WavFormatError("read_wav: WAV sample rate exceeds the supported range");
    if (!supported_format(parsed))
        throw WavFormatError("read_wav: WAV samples must be PCM16 or IEEE float32");
}

inline ParsedWav parse(const std::vector<char>& bytes) {
    const std::size_t container_end = validate_container(bytes);

    ParsedWav parsed;
    std::size_t position = 12;
    while (position + 8 <= container_end) {
        const char* chunk = bytes.data() + position;
        const uint32_t chunk_size = read_u32_le(chunk + 4);
        const std::size_t data_offset = position + 8;
        require_chunk_fits(container_end, data_offset, chunk_size);
        parse_chunk(bytes, chunk, data_offset, chunk_size, parsed);
        position = data_offset + static_cast<std::size_t>(chunk_size) + (chunk_size & 1U);
    }

    validate_required_fields(parsed);
    return parsed;
}

inline float decode_sample(const char* data, uint16_t format) {
    if (format == 3) {
        const uint32_t bits = read_u32_le(data);
        float value = 0.0F;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    const uint16_t raw = read_u16_le(data);
    const int32_t value =
        raw >= 0x8000U ? static_cast<int32_t>(raw) - 0x10000 : static_cast<int32_t>(raw);
    return static_cast<float>(value) / 32768.0F;
}

inline AudioResult decode(const std::vector<char>& bytes, const ParsedWav& parsed) {
    const std::size_t sample_width = parsed.bits_per_sample / 8U;
    const std::size_t frame_width = sample_width * parsed.channels;
    if (parsed.data_size % frame_width != 0)
        throw WavFormatError("read_wav: WAV data does not contain complete audio frames");

    const std::size_t frame_count = parsed.data_size / frame_width;
    if (frame_count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw WavFormatError("read_wav: WAV contains too many audio frames");

    AudioResult result;
    result.samples.resize(frame_count);
    result.num_samples = static_cast<int32_t>(frame_count);
    result.sample_rate = static_cast<int32_t>(parsed.sample_rate);
    const char* audio = bytes.data() + parsed.data_offset;
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        float sum = 0.0F;
        for (uint16_t channel = 0; channel < parsed.channels; ++channel) {
            const std::size_t sample = frame * parsed.channels + channel;
            sum += decode_sample(audio + sample * sample_width, parsed.format);
        }
        result.samples[frame] = sum / static_cast<float>(parsed.channels);
    }
    return result;
}

} // namespace wav_detail

// Read a PCM16 or IEEE float32 WAV into mono float32 samples. Interleaved channels are
// averaged per frame. The source sample rate is preserved; model pipelines own resampling.
inline AudioResult read_wav(const std::string& path) {
    const auto bytes = wav_detail::read_file(path);
    return wav_detail::decode(bytes, wav_detail::parse(bytes));
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
