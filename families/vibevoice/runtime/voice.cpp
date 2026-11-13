/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vibevoice/runtime/voice.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

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

} // namespace

WavAudio read_wav(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::invalid_argument("vibevoice cannot open voice prompt: " + path);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),
                                           std::istreambuf_iterator<char>());
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
        throw std::invalid_argument("vibevoice voice prompt is not a RIFF/WAVE file: " + path);
    std::uint16_t format = 0;
    std::uint16_t bits = 0;
    WavAudio audio;
    const unsigned char* payload = nullptr;
    std::size_t payload_size = 0;
    std::size_t cursor = 12;
    while (cursor + 8 <= bytes.size()) {
        const auto chunk_size = static_cast<std::size_t>(read_u32(bytes.data() + cursor + 4));
        const unsigned char* chunk = bytes.data() + cursor + 8;
        if (chunk_size > bytes.size() - cursor - 8)
            throw std::invalid_argument("vibevoice voice prompt has a truncated WAV chunk: " +
                                        path);
        if (std::memcmp(bytes.data() + cursor, "fmt ", 4) == 0 && chunk_size >= 16) {
            format = read_u16(chunk);
            audio.channels = read_u16(chunk + 2);
            audio.sample_rate = read_u32(chunk + 4);
            bits = read_u16(chunk + 14);
            if (format == kExtensible && chunk_size >= 26)
                format = read_u16(chunk + 24);
        } else if (std::memcmp(bytes.data() + cursor, "data", 4) == 0) {
            payload = chunk;
            payload_size = chunk_size;
        }
        cursor += 8 + chunk_size + (chunk_size & 1U);
    }
    if (payload == nullptr || audio.channels == 0 || audio.sample_rate == 0)
        throw std::invalid_argument("vibevoice voice prompt lacks fmt or data chunks: " + path);
    if (!(format == kPcm && (bits == 16 || bits == 24 || bits == 32)) &&
        !(format == kFloat && bits == 32))
        throw std::invalid_argument(
            "vibevoice voice prompts must be 16/24/32-bit PCM or 32-bit float WAV");
    const std::size_t width = bits / 8U;
    const std::size_t count = payload_size / width;
    audio.samples.resize(count);
    for (std::size_t index = 0; index < count; ++index)
        audio.samples[index] = decode_sample(payload + index * width, format, bits);
    return audio;
}

std::vector<float> prepare_voice(std::vector<float> samples, float target_dbfs, float eps,
                                 std::int64_t hop_length) {
    if (samples.empty())
        throw std::invalid_argument("vibevoice voice prompt contains no samples");
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
    const auto frames = (static_cast<std::int64_t>(samples.size()) + hop_length - 1) / hop_length;
    samples.resize(static_cast<std::size_t>(frames * hop_length), 0.0F);
    return samples;
}

} // namespace trtmc::vibevoice
