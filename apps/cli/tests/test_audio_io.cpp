/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/io.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
int failures = 0;

void check(bool value, const char* name) {
    if (!value) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

std::uint32_t little_endian(const std::vector<unsigned char>& bytes, std::size_t offset,
                            std::size_t length) {
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < length; ++i)
        value |= static_cast<std::uint32_t>(bytes.at(offset + i)) << (8U * i);
    return value;
}

std::vector<unsigned char> read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "trtmc-multichannel-audio.wav";
    try {
        // Existing three-field construction must continue to mean mono.
        const trtmc::AudioResult mono{{-1.0F, 0.25F, 1.0F}, 3, 16000};
        check(mono.num_channels == 1, "legacy aggregate defaults to mono");
        trtmc::cli::io::write_wav(mono, path.string());
        const auto mono_bytes = read_bytes(path);
        check(little_endian(mono_bytes, 22, 2) == 1 && little_endian(mono_bytes, 28, 4) == 64000 &&
                  little_endian(mono_bytes, 32, 2) == 4,
              "mono WAV header remains unchanged");
        const auto restored = trtmc::cli::io::read_wav(path.string());
        check(restored.samples == mono.samples && restored.num_samples == 3 &&
                  restored.sample_rate == 16000 && restored.num_channels == 1,
              "mono round trip");

        // Three distinct left/right frames; comparing bytes avoids a reader
        // accidentally hiding a writer error by downmixing the output.
        trtmc::AudioResult stereo{{1.0F, -1.0F, 0.5F, 0.0F, -0.5F, 1.0F}, 6, 48000, 2};
        trtmc::cli::io::write_wav(stereo, path.string());
        const auto bytes = read_bytes(path);
        check(bytes.size() == 44 + 6 * sizeof(float), "stereo payload is not counted twice");
        check(little_endian(bytes, 4, 4) == bytes.size() - 8 && little_endian(bytes, 20, 2) == 3 &&
                  little_endian(bytes, 22, 2) == 2 && little_endian(bytes, 24, 4) == 48000 &&
                  little_endian(bytes, 28, 4) == 384000 && little_endian(bytes, 32, 2) == 8 &&
                  little_endian(bytes, 34, 2) == 32 && little_endian(bytes, 40, 4) == 24,
              "stereo float32 WAV header");
        check(static_cast<double>(little_endian(bytes, 40, 4)) / little_endian(bytes, 28, 4) ==
                  3.0 / 48000.0,
              "duration counts frames, not scalar samples");
        for (std::size_t i = 0; i < stereo.samples.size(); ++i) {
            const auto bits = little_endian(bytes, 44 + 4 * i, 4);
            float sample = 0.0F;
            std::memcpy(&sample, &bits, sizeof(sample));
            check(sample == stereo.samples[i], "interleaved channel samples round trip");
        }
        const auto downmixed = trtmc::cli::io::read_wav(path.string());
        check(downmixed.num_channels == 1 && downmixed.num_samples == 3 &&
                  downmixed.samples == std::vector<float>({0.0F, 0.25F, 0.25F}),
              "existing input downmix is preserved");

        stereo.num_samples = 0;
        trtmc::cli::io::write_wav(stereo, path.string());
        check(read_bytes(path) == bytes, "unspecified sample count uses buffer size");

        const auto rejects = [&](trtmc::AudioResult invalid) {
            bool rejected = false;
            try {
                trtmc::cli::io::write_wav(invalid, path.string());
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            check(rejected, "invalid audio is rejected");
            check(read_bytes(path) == bytes, "validation does not truncate an existing file");
        };
        rejects({});
        rejects({{1.0F, 2.0F}, 2, 48000, 0});
        rejects({{1.0F, 2.0F}, 2, 48000, -1});
        rejects({{1.0F, 2.0F}, 2, 0, 2});
        rejects({{1.0F, 2.0F}, 2, -1, 2});
        rejects({{1.0F, 2.0F, 3.0F}, 3, 48000, 2});
        rejects({{1.0F, 2.0F}, 1, 48000, 2});
        rejects({{1.0F, 2.0F}, -1, 48000, 2});
        rejects({{1.0F, 2.0F}, 2, std::numeric_limits<std::int32_t>::max(), 2});
        rejects({std::vector<float>(8192), 8192, 48000, 8192});
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove(path);
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures == 0 ? 0 : 1;
}
