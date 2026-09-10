/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_protocol.h"

#include <cstdlib>
#include <iostream>
#include <limits>

using namespace trtmc::examples::windows_voicechat;

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

template <typename Function>
void rejects(Function&& operation) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "malformed wire audio was accepted");
}

int main() {
    try {
        // Independent wire fixture: little-endian bytes for 0, +1 and -1.
        const std::string reference = "AAAAAAAAgD8AAIC/";
        const auto decoded = decode_audio(reference);
        require(decoded == std::vector<float>({0.0F, 1.0F, -1.0F}), "float32 wire fixture failed");
        require(encode_audio(decoded) == reference, "float32 wire encoder disagrees with fixture");
        require(decode_audio("AACAPw==")[0] == 1.0F, "one-sample padding failed");

        // Exercise every base64 remainder and a realistic microphone block.
        for (std::size_t count = 1; count <= 16000; count = count < 10 ? count + 1 : count * 2) {
            std::vector<float> samples(count);
            for (std::size_t index = 0; index < count; ++index)
                samples[index] = static_cast<float>(index % 201) / 100.0F - 1.0F;
            require(decode_audio(encode_audio(samples)) == samples,
                    "audio block round trip failed");
        }
        require(decode_audio(encode_audio(std::vector<float>(kMaxInputSamples))).size() ==
                    kMaxInputSamples,
                "maximum microphone block was rejected");
        rejects([] { decode_audio(encode_audio(std::vector<float>(kMaxInputSamples + 1))); });
        rejects([] { decode_audio(""); });
        rejects([] { decode_audio("AAAA"); }); // Three bytes, not a float32.
        rejects([] { decode_audio("AACAPw="); });
        rejects([] { decode_audio("AACAPw==AAAA"); });
        rejects([] { decode_audio("AACAPx=="); }); // Noncanonical padding bits.
        rejects([] { decode_audio("AACAPw$="); });
        rejects([] { decode_audio("AACAPw= ="); });
        rejects([] { decode_audio("AACAfw=="); }); // Positive infinity.
        rejects([] { decode_audio("AADAfw=="); }); // Quiet NaN.
        rejects([] { decode_audio("AAAAQA=="); }); // Out-of-range microphone amplitude.
        rejects([] { encode_audio({std::numeric_limits<float>::infinity()}); });
        std::cout << "VoiceChat PCM transport: wire fixtures, limits and malformed payload checks "
                     "passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
