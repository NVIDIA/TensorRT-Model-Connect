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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::cli::io {

void write_wav(const AudioResult& audio, const std::string& path) {
    write_wav_interleaved({audio.samples.data(), audio.samples.size()}, audio.sample_rate, 1, path);
}

void write_wav_interleaved(Span<const float> samples, std::int32_t sample_rate,
                           std::int32_t channels, const std::string& path) {
    if (samples.empty())
        throw std::runtime_error("write_wav: empty audio");
    if (samples.data() == nullptr || sample_rate <= 0 || channels <= 0 ||
        channels > std::numeric_limits<std::uint16_t>::max() / 4 ||
        samples.size() % static_cast<std::size_t>(channels) != 0)
        throw std::runtime_error("write_wav: invalid audio format or incomplete sample frame");
    if (samples.size() > (std::numeric_limits<std::uint32_t>::max() - 36ULL) / sizeof(float) ||
        static_cast<std::uint64_t>(sample_rate) * channels * sizeof(float) >
            std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("write_wav: audio exceeds RIFF size limits");

    std::ofstream output(path, std::ios::binary);
    if (!output)
        throw std::runtime_error("write_wav: cannot open " + path);

    const auto num_channels = static_cast<std::uint16_t>(channels);
    const std::uint16_t bits_per_sample = 32;
    const auto byte_rate = static_cast<std::uint32_t>(static_cast<std::uint64_t>(sample_rate) *
                                                      channels * sizeof(float));
    const auto block_align = static_cast<std::uint16_t>(channels * sizeof(float));
    const auto data_size = static_cast<std::uint32_t>(samples.size() * sizeof(float));
    const std::uint32_t chunk_size = 36 + data_size;
    const std::uint32_t format_size = 16;
    const std::uint16_t audio_format = 3;

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
    output.write(reinterpret_cast<const char*>(samples.data()), data_size);
    output.close();
    if (!output)
        throw std::runtime_error("write_wav: failed to write " + path);
}

LoadedAudio read_wav_interleaved(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("read_wav: cannot open " + path);
    const auto file_size = input.tellg();
    if (file_size < 12)
        throw std::runtime_error("read_wav: truncated RIFF header");
    input.seekg(0);
    auto read = [&](void* destination, std::size_t count) {
        if (!input.read(static_cast<char*>(destination), static_cast<std::streamsize>(count)))
            throw std::runtime_error("read_wav: truncated chunk");
    };

    char riff[4]{};
    read(riff, 4);
    if (std::string(riff, 4) != "RIFF")
        throw std::runtime_error("read_wav: not a RIFF file");
    std::uint32_t riff_size = 0;
    read(&riff_size, 4);
    const auto riff_end = static_cast<std::uint64_t>(riff_size) + 8;
    if (riff_size < 4 || riff_end > static_cast<std::uint64_t>(file_size))
        throw std::runtime_error("read_wav: truncated RIFF payload");
    char wave[4]{};
    read(wave, 4);
    if (std::string(wave, 4) != "WAVE")
        throw std::runtime_error("read_wav: not a WAVE file");

    std::int32_t sample_rate = 0;
    std::uint16_t num_channels = 0;
    std::uint16_t audio_format = 0;
    std::uint16_t bits_per_sample = 0;
    std::vector<char> data;
    while (static_cast<std::uint64_t>(input.tellg()) < riff_end) {
        if (riff_end - static_cast<std::uint64_t>(input.tellg()) < 8)
            throw std::runtime_error("read_wav: truncated chunk header");
        char id[4]{};
        read(id, 4);
        std::uint32_t size = 0;
        read(&size, 4);
        const auto padded_size = static_cast<std::uint64_t>(size) + (size & 1U);
        if (padded_size > riff_end - static_cast<std::uint64_t>(input.tellg()))
            throw std::runtime_error("read_wav: chunk exceeds RIFF payload");
        if (std::string(id, 4) == "fmt ") {
            if (size < 16)
                throw std::runtime_error("read_wav: invalid fmt chunk");
            read(&audio_format, 2);
            read(&num_channels, 2);
            read(&sample_rate, 4);
            input.seekg(6, std::ios::cur);
            read(&bits_per_sample, 2);
            if (size > 16)
                input.seekg(size - 16, std::ios::cur);
        } else if (std::string(id, 4) == "data") {
            data.resize(static_cast<std::size_t>(size));
            read(data.data(), size);
        } else {
            input.seekg(size, std::ios::cur);
        }
        if (size & 1U)
            input.seekg(1, std::ios::cur);
        if (!input)
            throw std::runtime_error("read_wav: truncated chunk");
    }

    if (num_channels == 0 || sample_rate <= 0)
        throw std::runtime_error("read_wav: invalid sample rate or channel count");
    LoadedAudio result;
    result.sample_rate = sample_rate;
    result.channels = num_channels;
    const std::size_t bytes_per_sample = bits_per_sample / 8;
    if (bytes_per_sample == 0 || data.size() % (bytes_per_sample * num_channels) != 0)
        throw std::runtime_error("read_wav: incomplete sample frame");
    if (audio_format == 3 && bits_per_sample == 32) {
        result.samples.resize(data.size() / sizeof(float));
        if (!data.empty())
            std::memcpy(result.samples.data(), data.data(), data.size());
    } else if (audio_format == 1 && bits_per_sample == 16) {
        result.samples.resize(data.size() / sizeof(std::int16_t));
        for (std::size_t index = 0; index < result.samples.size(); ++index) {
            std::int16_t sample = 0;
            std::memcpy(&sample, data.data() + index * sizeof(sample), sizeof(sample));
            result.samples[index] = static_cast<float>(sample) / 32768.0F;
        }
    } else {
        throw std::runtime_error("read_wav: only float32 or PCM16 WAV is supported");
    }
    return result;
}

AudioResult read_wav(const std::string& path) {
    auto audio = read_wav_interleaved(path);
    const auto channels = static_cast<std::size_t>(audio.channels);
    const auto frames = audio.samples.size() / channels;
    if (frames > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error(
            "read_wav: too many sample frames for the existing mono interface");
    AudioResult result;
    result.sample_rate = audio.sample_rate;
    result.num_samples = static_cast<std::int32_t>(frames);
    if (channels == 1) {
        result.samples = std::move(audio.samples);
    } else {
        result.samples.resize(frames);
        for (std::size_t index = 0; index < frames; ++index) {
            float sum = 0.0F;
            for (std::size_t channel = 0; channel < channels; ++channel)
                sum += audio.samples[index * channels + channel];
            result.samples[index] = sum / static_cast<float>(channels);
        }
    }
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

void save_png(const std::string& path, const std::vector<float>& pixels, int width, int height,
              int channels) {
    save_png(path, Span<const float>{pixels.data(), pixels.size()}, width, height, channels);
}

void save_png(const std::string& path, Span<const float> pixels, int width, int height,
              int channels) {
    if (width <= 0 || height <= 0 || channels < 1 || channels > 4 ||
        width > std::numeric_limits<int>::max() / channels)
        throw std::runtime_error("save_png: invalid image dimensions or channel count");
    const auto expected = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                          static_cast<std::size_t>(channels);
    if (pixels.size() != expected)
        throw std::runtime_error("save_png: pixel buffer size does not match dimensions");

    std::vector<std::uint8_t> output(expected);
    for (std::size_t index = 0; index < expected; ++index) {
        if (!std::isfinite(pixels[index]))
            throw std::runtime_error("save_png: pixel is not finite");
        const float value = std::clamp(pixels[index], 0.0F, 1.0F);
        output[index] = static_cast<std::uint8_t>(value * 255.0F + 0.5F);
    }
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("save_png: unable to open " + path);
    const auto write = [](void* context, void* data, int size) {
        auto& stream = *static_cast<std::ofstream*>(context);
        stream.write(static_cast<const char*>(data), size);
    };
    const bool encoded = stbi_write_png_to_func(write, &file, width, height, channels,
                                                output.data(), width * channels) != 0;
    file.close();
    if (!encoded || !file)
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
    save_png(path, Span<const float>{image.pixels.data(), count}, image.width, image.height);
}

} // namespace trtmc::cli::io
