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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::io {

// Write an IEEE float32 WAV from interleaved AudioResult samples. Mono remains
// the default for AudioResult values produced by legacy pipelines.
inline void write_wav(const AudioResult& audio, const std::string& path) {
    if (audio.samples.empty())
        throw std::runtime_error("write_wav: empty audio");
    if (audio.sample_rate <= 0)
        throw std::runtime_error("write_wav: sample_rate must be positive");
    if (audio.channels <= 0 ||
        audio.channels > std::numeric_limits<int16_t>::max() / static_cast<int32_t>(sizeof(float)))
        throw std::runtime_error("write_wav: invalid channel count");
    const auto channels = static_cast<std::size_t>(audio.channels);
    if (audio.samples.size() % channels != 0)
        throw std::runtime_error(
            "write_wav: interleaved sample count is not divisible by channels");
    if (audio.samples.size() >
        static_cast<std::size_t>((std::numeric_limits<uint32_t>::max() - 36U) / sizeof(float)))
        throw std::runtime_error("write_wav: audio is too large for RIFF/WAV");

    const auto byte_rate_u64 = static_cast<uint64_t>(audio.sample_rate) * channels * sizeof(float);
    if (byte_rate_u64 > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("write_wav: byte rate exceeds the WAV field limit");

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("write_wav: cannot open " + path);

    const int32_t sample_rate = audio.sample_rate;
    const int16_t num_channels = static_cast<int16_t>(audio.channels);
    const int16_t bits_per_sample = 32;
    const uint32_t byte_rate = static_cast<uint32_t>(byte_rate_u64);
    const int16_t block_align = static_cast<int16_t>(num_channels * (bits_per_sample / 8));
    const uint32_t data_size = static_cast<uint32_t>(audio.samples.size() * sizeof(float));
    const uint32_t chunk_size = 36U + data_size;

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
    if (!f)
        throw std::runtime_error("write_wav: failed while writing " + path);
}

// Read PCM16 or IEEE-float32 WAV without changing channel layout. The returned
// samples are interleaved and num_samples is the total scalar sample count.
// This is the decoded-media path used by native video/reference generation.
inline AudioResult read_wav_interleaved(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("read_wav_interleaved: cannot open " + path);

    // Read RIFF header
    char riff[4]{};
    if (!f.read(riff, 4) || std::string(riff, 4) != "RIFF")
        throw std::runtime_error("read_wav_interleaved: not a RIFF file");

    uint32_t chunk_size = 0;
    if (!f.read(reinterpret_cast<char*>(&chunk_size), 4))
        throw std::runtime_error("read_wav_interleaved: truncated RIFF header");
    (void)chunk_size;

    char wave[4]{};
    if (!f.read(wave, 4) || std::string(wave, 4) != "WAVE")
        throw std::runtime_error("read_wav_interleaved: not a WAVE file");

    int32_t sample_rate = 0;
    int16_t num_channels = 0;
    int16_t audio_format = 0;
    int16_t bits_per_sample = 0;
    bool found_fmt = false;
    bool found_data = false;

    // Find fmt and data chunks
    std::vector<char> data_bytes;
    while (f) {
        char id[4]{};
        if (!f.read(id, 4))
            break;
        uint32_t size = 0;
        if (!f.read(reinterpret_cast<char*>(&size), 4))
            throw std::runtime_error("read_wav_interleaved: truncated chunk header");

        if (std::string(id, 4) == "fmt ") {
            if (size < 16U)
                throw std::runtime_error("read_wav_interleaved: invalid fmt chunk");
            if (!f.read(reinterpret_cast<char*>(&audio_format), 2) ||
                !f.read(reinterpret_cast<char*>(&num_channels), 2) ||
                !f.read(reinterpret_cast<char*>(&sample_rate), 4))
                throw std::runtime_error("read_wav_interleaved: truncated fmt chunk");
            f.seekg(4, std::ios::cur); // byte_rate
            f.seekg(2, std::ios::cur); // block_align
            if (!f.read(reinterpret_cast<char*>(&bits_per_sample), 2))
                throw std::runtime_error("read_wav_interleaved: truncated fmt chunk");
            if (size > 16U)
                f.seekg(static_cast<std::streamoff>(size - 16U), std::ios::cur);
            found_fmt = true;
        } else if (std::string(id, 4) == "data") {
            data_bytes.resize(static_cast<std::size_t>(size));
            if (size != 0U &&
                !f.read(data_bytes.data(), static_cast<std::streamsize>(data_bytes.size())))
                throw std::runtime_error("read_wav_interleaved: truncated data chunk");
            found_data = true;
        } else {
            f.seekg(static_cast<std::streamoff>(size), std::ios::cur);
        }
        if ((size & 1U) != 0U)
            f.seekg(1, std::ios::cur); // RIFF chunks are word aligned
        if (!f && !f.eof())
            throw std::runtime_error("read_wav_interleaved: invalid chunk size");
    }

    if (!found_fmt || !found_data)
        throw std::runtime_error("read_wav_interleaved: missing fmt or data chunk");
    if (sample_rate <= 0 || num_channels <= 0)
        throw std::runtime_error("read_wav_interleaved: invalid sample rate or channel count");
    const bool float32 = audio_format == 3 && bits_per_sample == 32;
    const bool pcm16 = audio_format == 1 && bits_per_sample == 16;
    if (!float32 && !pcm16)
        throw std::runtime_error("read_wav_interleaved: only PCM16 and IEEE float32 are supported");

    const std::size_t bytes_per_sample = float32 ? sizeof(float) : sizeof(int16_t);
    const std::size_t frame_bytes = bytes_per_sample * static_cast<std::size_t>(num_channels);
    if (data_bytes.size() % frame_bytes != 0)
        throw std::runtime_error("read_wav_interleaved: data is not frame aligned");
    const std::size_t scalar_count = data_bytes.size() / bytes_per_sample;
    if (scalar_count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("read_wav_interleaved: sample count exceeds the API limit");

    AudioResult result;
    result.sample_rate = sample_rate;
    result.channels = num_channels;
    result.samples.resize(scalar_count);
    if (float32) {
        if (!data_bytes.empty())
            std::memcpy(result.samples.data(), data_bytes.data(), data_bytes.size());
    } else {
        for (std::size_t i = 0; i < scalar_count; ++i) {
            int16_t sample = 0;
            std::memcpy(&sample, data_bytes.data() + i * sizeof(sample), sizeof(sample));
            result.samples[i] = static_cast<float>(sample) / 32768.0F;
        }
    }
    result.num_samples = static_cast<int32_t>(result.samples.size());
    return result;
}

// Read a WAV file into an AudioResult (mono float32). This legacy convenience
// API intentionally downmixes multichannel inputs. Use read_wav_interleaved()
// when channel identity is part of the model input contract.
inline AudioResult read_wav(const std::string& path) {
    AudioResult decoded = read_wav_interleaved(path);
    if (decoded.channels == 1)
        return decoded;

    const auto channels = static_cast<std::size_t>(decoded.channels);
    const auto frames = decoded.samples.size() / channels;
    AudioResult mono;
    mono.samples.resize(frames);
    mono.sample_rate = decoded.sample_rate;
    mono.channels = 1;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        float sum = 0.0F;
        for (std::size_t channel = 0; channel < channels; ++channel)
            sum += decoded.samples[frame * channels + channel];
        mono.samples[frame] = sum / static_cast<float>(channels);
    }
    mono.num_samples = static_cast<int32_t>(mono.samples.size());
    return mono;
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
void save_png(const std::string& path,
              const std::vector<float>& hwc_pixels,
              int width,
              int height);

// Convenience overload: save an ImageResult (first frame only) directly.
// For single-frame results (num_frames <= 1) writes result.pixels;
// for multi-frame results (video) writes only frame 0 — use the
// generate-video CLI command instead to dump every frame.
inline void save_png(const ImageResult& image, const std::string& path)
{
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
inline std::vector<float> decode_image(const std::string& path, int& h, int& w)
{
    auto img = read_image(path);
    h = img.height;
    w = img.width;
    return std::move(img.pixels);
}

} // namespace trtmc::io
