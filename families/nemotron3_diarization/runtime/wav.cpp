/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron3_diarization/runtime/wav.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace trtmc::nemotron3_diarization {
namespace {

static_assert(sizeof(float) == sizeof(std::uint32_t), "float32 samples are copied bit for bit");

constexpr std::uint32_t kPcm = 1;
constexpr std::uint32_t kFloat = 3;
[[noreturn]] void refuse(const std::string& reason) {
    throw std::invalid_argument("invalid WAV input: " + reason);
}

// RIFF stores integers little-endian, whatever the host's byte order.
std::uint32_t little_endian(const char* data, unsigned width) {
    std::uint32_t value = 0;
    for (unsigned i = width; i-- > 0;)
        value = (value << 8U) | static_cast<unsigned char>(data[i]);
    return value;
}

} // namespace

WavAudio parse_wav(const std::vector<char>& bytes) {
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
        refuse("not a RIFF/WAVE file");
    const std::uint64_t end = 8ULL + little_endian(bytes.data() + 4, 4);
    if (end < 12)
        refuse("the RIFF size does not cover its WAVE tag");
    if (end > bytes.size())
        refuse("the RIFF payload is truncated");

    const char* format = nullptr;
    const char* data = nullptr;
    std::uint64_t data_size = 0;
    for (std::uint64_t at = 12; at < end;) {
        if (end - at < 8)
            refuse("a chunk header is truncated");
        const char* chunk = bytes.data() + at;
        const std::uint64_t size = little_endian(chunk + 4, 4);
        // An odd-sized chunk is followed by one pad byte, inside the payload too.
        const std::uint64_t padded = size + (size & 1U);
        if (padded > end - at - 8)
            refuse("a chunk exceeds the RIFF payload");
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            if (format != nullptr)
                refuse("more than one fmt chunk");
            if (size < 16)
                refuse("the fmt chunk is shorter than 16 bytes");
            format = chunk + 8;
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            if (data != nullptr)
                refuse("more than one data chunk");
            data = chunk + 8;
            data_size = size;
        }
        at += 8 + padded;
    }
    if (format == nullptr)
        refuse("no fmt chunk");
    if (data == nullptr)
        refuse("no data chunk");

    const auto encoding = little_endian(format, 2);
    const auto channels = little_endian(format + 2, 2);
    const auto sample_rate = little_endian(format + 4, 4);
    const auto block_align = little_endian(format + 12, 2);
    const auto bits = little_endian(format + 14, 2);
    const bool pcm16 = encoding == kPcm && bits == 16;
    if (!pcm16 && !(encoding == kFloat && bits == 32))
        refuse("only PCM16 and float32 WAV are supported");
    if (channels == 0 || sample_rate == 0)
        refuse("the sample rate and the channel count must be positive");
    const std::uint32_t width = bits / 8;
    if (block_align != channels * width)
        refuse("the block alignment does not match the channels and the sample width");
    if (data_size == 0)
        refuse("no sample frames");
    if (data_size % block_align != 0)
        refuse("the data chunk ends inside a sample frame");

    WavAudio audio;
    audio.sample_rate = sample_rate;
    audio.channels = channels;
    audio.samples.resize(static_cast<std::size_t>(data_size / width));
    for (std::size_t i = 0; i < audio.samples.size(); ++i) {
        const char* sample = data + i * width;
        if (pcm16) {
            const auto code = static_cast<std::int32_t>(little_endian(sample, 2));
            audio.samples[i] = static_cast<float>(code >= 32768 ? code - 65536 : code) / 32768.0F;
        } else {
            const auto word = little_endian(sample, 4);
            std::memcpy(&audio.samples[i], &word, sizeof(float));
        }
    }
    return audio;
}

WavAudio read_wav(std::istream& input, std::uint64_t size) {
    std::vector<char> bytes(static_cast<std::size_t>(std::min<std::uint64_t>(size, 12)));
    if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("cannot read the WAV header");
    if (bytes.size() == 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0) {
        const std::uint64_t end = 8ULL + little_endian(bytes.data() + 4, 4);
        bytes.resize(static_cast<std::size_t>(std::max<std::uint64_t>(12, std::min(end, size))));
        if (!input.read(bytes.data() + 12, static_cast<std::streamsize>(bytes.size() - 12)))
            throw std::runtime_error("cannot read the WAV payload");
    }
    return parse_wav(bytes);
}

WavAudio read_wav(const std::string& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error))
        throw std::invalid_argument("the WAV input is not a readable file: " + path);
    const std::uint64_t size = std::filesystem::file_size(path, error);
    std::ifstream file(path, std::ios::binary);
    if (error || !file)
        throw std::invalid_argument("cannot open the WAV input: " + path);
    return read_wav(file, size);
}

} // namespace trtmc::nemotron3_diarization
