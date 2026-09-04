/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/io.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::cli::io {

void write_wav(const AudioResult& audio, const std::string& path) {
    if (audio.samples.empty())
        throw std::runtime_error("write_wav: empty audio");

    std::ofstream output(path, std::ios::binary);
    if (!output)
        throw std::runtime_error("write_wav: cannot open " + path);

    const auto num_samples = static_cast<std::int32_t>(audio.samples.size());
    const std::int32_t sample_rate = audio.sample_rate;
    const std::int16_t num_channels = 1;
    const std::int16_t bits_per_sample = 32;
    const std::int32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
    const auto block_align = static_cast<std::int16_t>(num_channels * (bits_per_sample / 8));
    const std::int32_t data_size = num_samples * block_align;
    const std::int32_t chunk_size = 36 + data_size;
    const std::int32_t format_size = 16;
    const std::int16_t audio_format = 3;

    output.write("RIFF", 4);
    output.write(reinterpret_cast<const char*>(&chunk_size), 4);
    output.write("WAVEfmt ", 8);
    output.write(reinterpret_cast<const char*>(&format_size), 4);
    output.write(reinterpret_cast<const char*>(&audio_format), 2);
    output.write(reinterpret_cast<const char*>(&num_channels), 2);
    output.write(reinterpret_cast<const char*>(&sample_rate), 4);
    output.write(reinterpret_cast<const char*>(&byte_rate), 4);
    output.write(reinterpret_cast<const char*>(&block_align), 2);
    output.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    output.write("data", 4);
    output.write(reinterpret_cast<const char*>(&data_size), 4);
    output.write(reinterpret_cast<const char*>(audio.samples.data()), data_size);
}

AudioResult read_wav(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("read_wav: cannot open " + path);

    char riff[4];
    input.read(riff, 4);
    if (std::string(riff, 4) != "RIFF")
        throw std::runtime_error("read_wav: not a RIFF file");
    input.seekg(4, std::ios::cur);
    char wave[4];
    input.read(wave, 4);
    if (std::string(wave, 4) != "WAVE")
        throw std::runtime_error("read_wav: not a WAVE file");

    std::int32_t sample_rate = 0;
    std::int16_t num_channels = 0;
    std::int16_t audio_format = 0;
    std::int16_t bits_per_sample = 0;
    std::vector<char> data;
    while (input) {
        char id[4];
        if (!input.read(id, 4))
            break;
        std::int32_t size = 0;
        input.read(reinterpret_cast<char*>(&size), 4);
        if (size < 0)
            throw std::runtime_error("read_wav: invalid chunk size");
        if (std::string(id, 4) == "fmt ") {
            if (size < 16)
                throw std::runtime_error("read_wav: invalid fmt chunk");
            input.read(reinterpret_cast<char*>(&audio_format), 2);
            input.read(reinterpret_cast<char*>(&num_channels), 2);
            input.read(reinterpret_cast<char*>(&sample_rate), 4);
            input.seekg(6, std::ios::cur);
            input.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (size > 16)
                input.seekg(size - 16, std::ios::cur);
        } else if (std::string(id, 4) == "data") {
            data.resize(static_cast<std::size_t>(size));
            input.read(data.data(), size);
        } else {
            input.seekg(size, std::ios::cur);
        }
    }

    const auto channels = std::max<std::int16_t>(num_channels, 1);
    AudioResult result;
    result.sample_rate = sample_rate;
    if (audio_format == 3 && bits_per_sample == 32) {
        const std::size_t count = data.size() / (sizeof(float) * channels);
        result.samples.resize(count);
        const auto* samples = reinterpret_cast<const float*>(data.data());
        for (std::size_t index = 0; index < count; ++index) {
            float sum = 0.0F;
            for (std::int16_t channel = 0; channel < channels; ++channel)
                sum += samples[index * channels + channel];
            result.samples[index] = sum / static_cast<float>(channels);
        }
    } else if (audio_format == 1 && bits_per_sample == 16) {
        const std::size_t count = data.size() / (sizeof(std::int16_t) * channels);
        result.samples.resize(count);
        const auto* samples = reinterpret_cast<const std::int16_t*>(data.data());
        for (std::size_t index = 0; index < count; ++index) {
            float sum = 0.0F;
            for (std::int16_t channel = 0; channel < channels; ++channel)
                sum += static_cast<float>(samples[index * channels + channel]);
            result.samples[index] = sum / (32768.0F * static_cast<float>(channels));
        }
    } else {
        throw std::runtime_error("read_wav: only float32 or PCM16 WAV is supported");
    }
    result.num_samples = static_cast<std::int32_t>(result.samples.size());
    return result;
}

LoadedImage read_image(const std::string& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr)
        return {};

    LoadedImage result;
    result.width = width;
    result.height = height;
    const auto count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
    result.pixels.resize(count);
    for (std::size_t index = 0; index < count; ++index)
        result.pixels[index] = static_cast<float>(raw[index]) / 255.0F;
    stbi_image_free(raw);
    return result;
}

void save_png(const std::string& path, const std::vector<float>& pixels, int width, int height) {
    if (width <= 0 || height <= 0)
        throw std::runtime_error("save_png: width and height must be positive");
    const auto expected = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
    if (pixels.size() != expected)
        throw std::runtime_error("save_png: pixel buffer size does not match dimensions");

    std::vector<std::uint8_t> output(expected);
    for (std::size_t index = 0; index < expected; ++index) {
        const float value = std::clamp(pixels[index], 0.0F, 1.0F);
        output[index] = static_cast<std::uint8_t>(value * 255.0F + 0.5F);
    }
    if (!stbi_write_png(path.c_str(), width, height, 3, output.data(), width * 3))
        throw std::runtime_error("save_png: unable to write " + path);
}

void save_png(const ImageResult& image, const std::string& path) {
    const auto count =
        static_cast<std::size_t>(image.height) * static_cast<std::size_t>(image.width) * 3U;
    if (image.pixels.size() < count)
        throw std::runtime_error("save_png: image pixel buffer is smaller than one frame");
    if (image.pixels.size() == count) {
        save_png(path, image.pixels, image.width, image.height);
        return;
    }
    save_png(path, std::vector<float>(image.pixels.begin(), image.pixels.begin() + count),
             image.width, image.height);
}

} // namespace trtmc::cli::io
