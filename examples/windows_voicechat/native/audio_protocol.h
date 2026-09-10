/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::examples::windows_voicechat {

constexpr int kInputSampleRate = 16000;
constexpr int kOutputSampleRate = 48000;
constexpr std::size_t kMaxInputSamples = 16000;
constexpr std::size_t kMaxCommandBytes = 1024 * 1024;

class CommandError : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
};

inline int base64_digit(char character) {
    if (character >= 'A' && character <= 'Z')
        return character - 'A';
    if (character >= 'a' && character <= 'z')
        return character - 'a' + 26;
    if (character >= '0' && character <= '9')
        return character - '0' + 52;
    if (character == '+')
        return 62;
    if (character == '/')
        return 63;
    throw CommandError("audio must contain standard padded base64");
}

inline std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
    if (encoded.empty() || encoded.size() % 4 != 0 ||
        encoded.size() > ((kMaxInputSamples * 4 + 2) / 3) * 4)
        throw CommandError("audio base64 has an invalid or excessive length");
    std::vector<std::uint8_t> bytes;
    bytes.reserve(encoded.size() / 4 * 3);
    for (std::size_t offset = 0; offset < encoded.size(); offset += 4) {
        const auto a = base64_digit(encoded[offset]);
        const auto b = base64_digit(encoded[offset + 1]);
        const bool pad_c = encoded[offset + 2] == '=';
        const bool pad_d = encoded[offset + 3] == '=';
        if ((pad_c || pad_d) && offset + 4 != encoded.size())
            throw CommandError("audio base64 padding must be at the end");
        if (pad_c && !pad_d)
            throw CommandError("audio base64 padding is invalid");
        const auto c = pad_c ? 0 : base64_digit(encoded[offset + 2]);
        const auto d = pad_d ? 0 : base64_digit(encoded[offset + 3]);
        if ((pad_c && (b & 15) != 0) || (pad_d && !pad_c && (c & 3) != 0))
            throw CommandError("audio base64 has noncanonical padding bits");
        bytes.push_back(static_cast<std::uint8_t>((a << 2) | (b >> 4)));
        if (!pad_c)
            bytes.push_back(static_cast<std::uint8_t>((b << 4) | (c >> 2)));
        if (!pad_d)
            bytes.push_back(static_cast<std::uint8_t>((c << 6) | d));
    }
    return bytes;
}

inline std::string encode_base64(const std::vector<std::uint8_t>& bytes) {
    constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve((bytes.size() + 2) / 3 * 4);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3) {
        const auto a = bytes[offset];
        const auto b = offset + 1 < bytes.size() ? bytes[offset + 1] : 0;
        const auto c = offset + 2 < bytes.size() ? bytes[offset + 2] : 0;
        result.push_back(alphabet[a >> 2]);
        result.push_back(alphabet[((a & 3) << 4) | (b >> 4)]);
        result.push_back(offset + 1 < bytes.size() ? alphabet[((b & 15) << 2) | (c >> 6)] : '=');
        result.push_back(offset + 2 < bytes.size() ? alphabet[c & 63] : '=');
    }
    return result;
}

inline std::vector<float> decode_audio(std::string_view encoded) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "VoiceChat wire format requires IEEE 754 float32");
    const auto bytes = decode_base64(encoded);
    if (bytes.size() % 4 != 0 || bytes.size() / 4 > kMaxInputSamples)
        throw CommandError("audio must contain 1 to 16000 little-endian float32 samples");
    std::vector<float> samples(bytes.size() / 4);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto offset = index * 4;
        const auto bits = static_cast<std::uint32_t>(bytes[offset]) |
                          (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
                          (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
                          (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
        std::memcpy(&samples[index], &bits, 4);
        if (!std::isfinite(samples[index]) || std::abs(samples[index]) > 1.0F)
            throw CommandError("microphone samples must be finite and between -1 and 1");
    }
    return samples;
}

inline std::string encode_audio(const std::vector<float>& samples) {
    std::vector<std::uint8_t> bytes(samples.size() * 4);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (!std::isfinite(samples[index]))
            throw std::runtime_error("speech session produced a nonfinite audio sample");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &samples[index], 4);
        for (std::size_t byte = 0; byte < 4; ++byte)
            bytes[index * 4 + byte] = static_cast<std::uint8_t>(bits >> (byte * 8));
    }
    return encode_base64(bytes);
}

} // namespace trtmc::examples::windows_voicechat
