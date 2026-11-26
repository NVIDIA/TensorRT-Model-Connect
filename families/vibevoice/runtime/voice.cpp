/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vibevoice/runtime/voice.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace trtmc::vibevoice {
namespace {

std::uint32_t read_u32(const unsigned char* data) {
    return static_cast<std::uint32_t>(data[0]) | (static_cast<std::uint32_t>(data[1]) << 8U) |
           (static_cast<std::uint32_t>(data[2]) << 16U) |
           (static_cast<std::uint32_t>(data[3]) << 24U);
}

std::uint16_t read_u16(const unsigned char* data) {
    return static_cast<std::uint16_t>(data[0] | (data[1] << 8U));
}

constexpr std::uint16_t kPcm = 1;
constexpr std::uint16_t kFloat = 3;
constexpr std::uint16_t kExtensible = 0xFFFE;

float decode_sample(const unsigned char* data, std::uint16_t format, std::uint16_t bits) {
    if (format == kFloat) {
        float value = 0.0F;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    switch (bits) {
    case 16: {
        const auto value = static_cast<std::int16_t>(read_u16(data));
        return static_cast<float>(value) / 32768.0F;
    }
    case 24: {
        std::int32_t value =
            static_cast<std::int32_t>(data[0] | (data[1] << 8U) | (data[2] << 16U));
        if ((value & 0x800000) != 0)
            value -= 0x1000000;
        return static_cast<float>(value) / 8388608.0F;
    }
    case 32:
        return static_cast<float>(static_cast<std::int32_t>(read_u32(data))) / 2147483648.0F;
    default:
        throw std::runtime_error("unsupported WAV sample width");
    }
}

void read_at(std::ifstream& file, std::uint64_t offset, unsigned char* data, std::size_t size,
             const std::string& path) {
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!file || static_cast<std::size_t>(file.gcount()) != size)
        throw std::invalid_argument("vibevoice cannot read voice prompt: " + path);
}

} // namespace

WavAudio read_wav(const std::string& path, std::int64_t max_samples) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::invalid_argument("vibevoice cannot open voice prompt: " + path);
    const auto end = file.tellg();
    if (end < 0)
        throw std::invalid_argument("vibevoice cannot read voice prompt: " + path);
    const auto size = static_cast<std::uint64_t>(end);
    unsigned char riff[12];
    if (size < sizeof(riff))
        throw std::invalid_argument("vibevoice voice prompt is not a RIFF/WAVE file: " + path);
    read_at(file, 0, riff, sizeof(riff), path);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(riff + 8, "WAVE", 4) != 0)
        throw std::invalid_argument("vibevoice voice prompt is not a RIFF/WAVE file: " + path);
    // Chunks are read inside the RIFF container the header declares, which must fit the file.
    const std::uint64_t riff_end = 8 + static_cast<std::uint64_t>(read_u32(riff + 4));
    if (riff_end < sizeof(riff) || riff_end > size)
        throw std::invalid_argument("vibevoice voice prompt declares a RIFF size the file does not "
                                    "hold: " +
                                    path);
    std::uint16_t format = 0;
    std::uint16_t bits = 0;
    WavAudio audio;
    bool has_data = false;
    std::uint64_t payload_offset = 0;
    std::uint64_t payload_size = 0;
    std::uint64_t cursor = sizeof(riff);
    while (cursor + 8 <= riff_end) {
        unsigned char header[8];
        read_at(file, cursor, header, sizeof(header), path);
        const std::uint64_t chunk_size = read_u32(header + 4);
        const std::uint64_t padded = chunk_size + (chunk_size & 1U);
        const std::uint64_t body = cursor + 8;
        // Every chunk, with the pad byte that follows an odd size, lies inside the container.
        if (padded > riff_end - body)
            throw std::invalid_argument("vibevoice voice prompt has a truncated WAV chunk: " +
                                        path);
        if (std::memcmp(header, "fmt ", 4) == 0 && chunk_size >= 16) {
            unsigned char chunk[26] = {};
            read_at(file, body, chunk,
                    static_cast<std::size_t>(std::min<std::uint64_t>(chunk_size, sizeof(chunk))),
                    path);
            format = read_u16(chunk);
            audio.channels = read_u16(chunk + 2);
            audio.sample_rate = read_u32(chunk + 4);
            bits = read_u16(chunk + 14);
            if (format == kExtensible && chunk_size >= 26)
                format = read_u16(chunk + 24);
        } else if (std::memcmp(header, "data", 4) == 0) {
            has_data = true;
            payload_offset = body;
            payload_size = chunk_size;
        }
        cursor = body + padded;
    }
    if (!has_data || audio.channels == 0 || audio.sample_rate == 0)
        throw std::invalid_argument("vibevoice voice prompt lacks fmt or data chunks: " + path);
    if (!(format == kPcm && (bits == 16 || bits == 24 || bits == 32)) &&
        !(format == kFloat && bits == 32))
        throw std::invalid_argument(
            "vibevoice voice prompts must be 16/24/32-bit PCM or 32-bit float WAV");
    const std::size_t width = bits / 8U;
    const std::uint64_t count = payload_size / width;
    if (max_samples < 0 || count > static_cast<std::uint64_t>(max_samples))
        throw std::invalid_argument("vibevoice voice prompt " + path + " holds more than " +
                                    std::to_string(max_samples) + " samples");
    std::vector<unsigned char> payload(static_cast<std::size_t>(count) * width);
    read_at(file, payload_offset, payload.data(), payload.size(), path);
    audio.samples.resize(static_cast<std::size_t>(count));
    for (std::size_t index = 0; index < audio.samples.size(); ++index) {
        const float value = decode_sample(payload.data() + index * width, format, bits);
        if (!std::isfinite(value))
            throw std::invalid_argument("vibevoice voice prompt has a NaN or infinite sample: " +
                                        path);
        audio.samples[index] = value;
    }
    return audio;
}

bool all_finite(const float* samples, std::size_t count) {
    return std::all_of(samples, samples + count,
                       [](float sample) { return std::isfinite(sample); });
}

std::vector<float> prepare_voice(std::vector<float> samples, float target_dbfs, float eps,
                                 std::int64_t hop_length) {
    if (samples.empty())
        throw std::invalid_argument("vibevoice voice prompt contains no samples");
    if (!all_finite(samples.data(), samples.size()))
        throw std::invalid_argument("vibevoice voice prompt has a NaN or infinite sample");
    double energy = 0.0;
    for (const float sample : samples)
        energy += static_cast<double>(sample) * static_cast<double>(sample);
    const auto rms = static_cast<float>(std::sqrt(energy / static_cast<double>(samples.size())));
    const auto gain =
        static_cast<float>(std::pow(10.0, static_cast<double>(target_dbfs) / 20.0)) / (rms + eps);
    float peak = 0.0F;
    for (auto& sample : samples) {
        sample *= gain;
        peak = std::max(peak, std::fabs(sample));
    }
    if (peak > 1.0F) {
        for (auto& sample : samples)
            sample /= peak + eps;
    }
    if (!all_finite(samples.data(), samples.size()))
        throw std::runtime_error("vibevoice voice prompt normalization is not finite");
    const auto frames = (static_cast<std::int64_t>(samples.size()) + hop_length - 1) / hop_length;
    samples.resize(static_cast<std::size_t>(frames * hop_length), 0.0F);
    return samples;
}

} // namespace trtmc::vibevoice
