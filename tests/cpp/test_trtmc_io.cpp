/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-IO-CPP-02
// Architecture:   ARCH-AUD-001
// Unit Design:    UD-IO-02
// Intent:         trtmc::io::write_wav(AudioResult) and read_wav: round-trip,
//                 sample rate preservation, empty-audio error, bad path error
// Preconditions:  Writable temp directory
// Postconditions: Written WAV is readable and matches input; exceptions thrown
//                 on empty audio or unwritable path
// =============================================================================

// test_trtmc_io.cpp — Unit tests for include/trtmc/trtmc_io.hpp
//
// Purpose:
//   Validates the inline write_wav(AudioResult, path) and read_wav(path)
//   functions in trtmc::io. These operate on AudioResult structs and throw on
//   errors.
//
// Dependencies:
//   - trtmc/trtmc_io.hpp  : trtmc::io::write_wav, trtmc::io::read_wav
//   - test_helpers.h    : TempDirGuard
//   No TRT, GPU, or CUDA required.

#include "test_helpers.h"
#include "trtmc/trtmc_io.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

template <typename T>
static T read_binary_value(const std::vector<char>& bytes, std::size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

static void write_pcm16_wav(const std::string& path, uint16_t channels, uint32_t sample_rate,
                            const std::vector<int16_t>& interleaved) {
    const uint16_t block_align = static_cast<uint16_t>(channels * sizeof(int16_t));
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t data_size = static_cast<uint32_t>(interleaved.size() * sizeof(int16_t));
    const uint32_t chunk_size = 36U + data_size;
    constexpr uint32_t fmt_size = 16U;
    constexpr uint16_t audio_format = 1U;
    constexpr uint16_t bits_per_sample = 16U;

    std::ofstream file(path, std::ios::binary);
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&chunk_size), sizeof(chunk_size));
    file.write("WAVEfmt ", 8);
    file.write(reinterpret_cast<const char*>(&fmt_size), sizeof(fmt_size));
    file.write(reinterpret_cast<const char*>(&audio_format), sizeof(audio_format));
    file.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    file.write(reinterpret_cast<const char*>(&sample_rate), sizeof(sample_rate));
    file.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
    file.write(reinterpret_cast<const char*>(&block_align), sizeof(block_align));
    file.write(reinterpret_cast<const char*>(&bits_per_sample), sizeof(bits_per_sample));
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    file.write(reinterpret_cast<const char*>(interleaved.data()),
               static_cast<std::streamsize>(data_size));
}

static void rewrite_wav_u32(const std::string& path, std::streamoff offset, uint32_t value) {
    const char bytes[] = {
        static_cast<char>(value & 0xFFU),
        static_cast<char>((value >> 8U) & 0xFFU),
        static_cast<char>((value >> 16U) & 0xFFU),
        static_cast<char>((value >> 24U) & 0xFFU),
    };
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(offset);
    file.write(bytes, sizeof(bytes));
}

// ---------------------------------------------------------------------------
// write_wav + read_wav round-trip
// ---------------------------------------------------------------------------

// Intention: write_wav(AudioResult) writes a valid WAV that read_wav can
//            decode, recovering the original samples and sample rate exactly.
// Preconditions:  temp dir writable
// Postconditions: decoded samples match input within float32 precision
static bool test_io_write_read_roundtrip() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "rt.wav").string();

    trtmc::AudioResult ar;
    ar.samples = {0.0F, 0.5F, 1.0F, -1.0F, -0.5F, 0.25F};
    ar.num_samples = static_cast<int32_t>(ar.samples.size());
    ar.sample_rate = 22050;

    try {
        trtmc::io::write_wav(ar, path);
    } catch (const std::exception& e) {
        std::cerr << "io_write_read_roundtrip: write threw: " << e.what() << '\n';
        return false;
    }

    trtmc::AudioResult result;
    try {
        result = trtmc::io::read_wav(path);
    } catch (const std::exception& e) {
        std::cerr << "io_write_read_roundtrip: read threw: " << e.what() << '\n';
        return false;
    }

    if (result.sample_rate != ar.sample_rate) {
        std::cerr << "io_write_read_roundtrip: sample_rate mismatch " << result.sample_rate
                  << " vs " << ar.sample_rate << '\n';
        return false;
    }
    if (result.samples.size() != ar.samples.size()) {
        std::cerr << "io_write_read_roundtrip: sample count mismatch " << result.samples.size()
                  << " vs " << ar.samples.size() << '\n';
        return false;
    }
    for (std::size_t i = 0; i < ar.samples.size(); ++i) {
        if (std::abs(result.samples[i] - ar.samples[i]) > 1e-6F) {
            std::cerr << "io_write_read_roundtrip: sample[" << i << "] mismatch "
                      << result.samples[i] << " vs " << ar.samples[i] << '\n';
            return false;
        }
    }
    return true;
}

// Intention: channel-major multichannel audio is written as interleaved IEEE
//            float32 WAV sample frames with correct stereo metadata.
// Preconditions:  two channels with three samples per channel
// Postconditions: WAV header describes stereo float32 and payload is LRLRLR
static bool test_io_write_stereo_channel_major_layout() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "stereo.wav").string();

    trtmc::MultiChannelAudioResult audio;
    audio.samples = {0.1F, 0.2F, 0.3F, -0.1F, -0.2F, -0.3F};
    audio.num_samples = 3;
    audio.sample_rate = 48000;
    audio.num_channels = 2;

    try {
        trtmc::io::write_wav(audio, path);
    } catch (const std::exception& e) {
        std::cerr << "io_write_stereo_channel_major_layout: write threw: " << e.what() << '\n';
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(file),
                                  std::istreambuf_iterator<char>()};
    if (bytes.size() != 44U + 6U * sizeof(float))
        return false;
    if (std::string(bytes.data(), 4) != "RIFF" || std::string(bytes.data() + 8, 4) != "WAVE" ||
        std::string(bytes.data() + 12, 4) != "fmt " || std::string(bytes.data() + 36, 4) != "data")
        return false;
    if (read_binary_value<uint32_t>(bytes, 4) != 60U ||
        read_binary_value<uint32_t>(bytes, 16) != 16U ||
        read_binary_value<int16_t>(bytes, 20) != 3 || read_binary_value<int16_t>(bytes, 22) != 2 ||
        read_binary_value<int32_t>(bytes, 24) != 48000 ||
        read_binary_value<int32_t>(bytes, 28) != 384000 ||
        read_binary_value<int16_t>(bytes, 32) != 8 || read_binary_value<int16_t>(bytes, 34) != 32 ||
        read_binary_value<uint32_t>(bytes, 40) != 24U)
        return false;

    const std::vector<float> expected{0.1F, -0.1F, 0.2F, -0.2F, 0.3F, -0.3F};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(read_binary_value<float>(bytes, 44U + i * sizeof(float)) - expected[i]) >
            1e-6F)
            return false;
    }
    return true;
}

// Intention: read_wav_multichannel reverses WAV interleaving and preserves a
//            float32 stereo signal in channel-major layout.
// Preconditions:  write_wav receives two channels with distinct samples
// Postconditions: channel count, frame count, sample rate, and samples survive
static bool test_io_read_stereo_float_channel_major() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "stereo-float.wav").string();

    trtmc::MultiChannelAudioResult input;
    input.samples = {0.1F, 0.2F, 0.3F, -0.4F, -0.5F, -0.6F};
    input.num_samples = 3;
    input.sample_rate = 44100;
    input.num_channels = 2;
    try {
        trtmc::io::write_wav(input, path);
        const auto result = trtmc::io::read_wav_multichannel(path);
        if (result.num_channels != 2 || result.num_samples != 3 || result.sample_rate != 44100 ||
            result.samples.size() != input.samples.size())
            return false;
        for (std::size_t i = 0; i < input.samples.size(); ++i) {
            if (std::abs(result.samples[i] - input.samples[i]) > 1e-6F)
                return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "io_read_stereo_float_channel_major: " << e.what() << '\n';
        return false;
    }
    return true;
}

// Intention: one-channel inputs use the same channel-major API without adding
//            or dropping samples.
// Preconditions:  valid mono IEEE-float32 WAV
// Postconditions: reader reports one channel and preserves all samples
static bool test_io_read_mono_multichannel_contract() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "mono-float.wav").string();
    trtmc::AudioResult input;
    input.samples = {-0.75F, 0.0F, 0.5F};
    input.num_samples = 3;
    input.sample_rate = 22050;
    try {
        trtmc::io::write_wav(input, path);
        const auto result = trtmc::io::read_wav_multichannel(path);
        return result.num_channels == 1 && result.num_samples == 3 && result.sample_rate == 22050 &&
               result.samples == input.samples;
    } catch (const std::exception& e) {
        std::cerr << "io_read_mono_multichannel_contract: " << e.what() << '\n';
        return false;
    }
}

// Intention: PCM16 stereo is decoded channel-major while read_wav keeps its
//            historical average-to-mono behavior.
// Preconditions:  hand-authored interleaved PCM16 stereo WAV
// Postconditions: multichannel samples preserve L/R and mono samples average
static bool test_io_read_stereo_pcm16_and_mono_compatibility() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "stereo-pcm16.wav").string();
    write_pcm16_wav(path, 2, 16000, {32767, -32768, 16384, -16384, 0, 8192});

    try {
        const auto multichannel = trtmc::io::read_wav_multichannel(path);
        const std::vector<float> expected_channels{
            32767.0F / 32768.0F, 0.5F, 0.0F, -1.0F, -0.5F, 0.25F};
        if (multichannel.num_channels != 2 || multichannel.num_samples != 3 ||
            multichannel.sample_rate != 16000 ||
            multichannel.samples.size() != expected_channels.size())
            return false;
        for (std::size_t i = 0; i < expected_channels.size(); ++i) {
            if (std::abs(multichannel.samples[i] - expected_channels[i]) > 1e-6F)
                return false;
        }

        const auto mono = trtmc::io::read_wav(path);
        const std::vector<float> expected_mono{(-1.0F / 32768.0F) * 0.5F, 0.0F, 0.125F};
        if (mono.num_samples != 3 || mono.sample_rate != 16000 ||
            mono.samples.size() != expected_mono.size())
            return false;
        for (std::size_t i = 0; i < expected_mono.size(); ++i) {
            if (std::abs(mono.samples[i] - expected_mono[i]) > 1e-6F)
                return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "io_read_stereo_pcm16_and_mono_compatibility: " << e.what() << '\n';
        return false;
    }
    return true;
}

// Intention: the new reader has an explicit one/two-channel contract without
//            narrowing the legacy mono reader's multichannel behavior.
// Preconditions:  valid three-channel IEEE-float32 WAV
// Postconditions: multichannel reader rejects it; mono reader averages all 3
static bool test_io_multichannel_channel_limit_preserves_mono_behavior() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "three-channel.wav").string();
    trtmc::MultiChannelAudioResult input;
    input.samples = {0.0F, 0.3F, 0.3F, 0.6F, 0.6F, 0.9F};
    input.num_samples = 2;
    input.sample_rate = 24000;
    input.num_channels = 3;
    try {
        trtmc::io::write_wav(input, path);
        bool rejected = false;
        try {
            (void)trtmc::io::read_wav_multichannel(path);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        if (!rejected)
            return false;
        const auto mono = trtmc::io::read_wav(path);
        return mono.samples.size() == 2U && std::abs(mono.samples[0] - 0.3F) < 1e-6F &&
               std::abs(mono.samples[1] - 0.6F) < 1e-6F;
    } catch (const std::exception& e) {
        std::cerr << "io_multichannel_channel_limit_preserves_mono_behavior: " << e.what() << '\n';
        return false;
    }
}

// Intention: chunk traversal is bounded by the RIFF container's declared size.
// Preconditions:  valid WAV whose RIFF-size word is rewritten too small/large
// Postconditions: chunks outside the extent and truncated extents are rejected
static bool test_io_riff_declared_size_bounds_chunks() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "riff-size.wav").string();
    const auto rejected_with = [&](const char* expected) {
        try {
            (void)trtmc::io::read_wav_multichannel(path);
        } catch (const std::runtime_error& e) {
            return std::string(e.what()).find(expected) != std::string::npos;
        }
        return false;
    };

    write_pcm16_wav(path, 1, 16000, {0, 1});
    rewrite_wav_u32(path, 4, 4U);
    if (!rejected_with("missing fmt or data chunk"))
        return false;

    write_pcm16_wav(path, 1, 16000, {0, 1});
    rewrite_wav_u32(path, 4, 4096U);
    return rejected_with("truncated RIFF container");
}

// Intention: write_wav preserves the sample rate into the WAV header.
// Preconditions:  AudioResult with sample_rate=44100
// Postconditions: read_wav returns sample_rate==44100
static bool test_io_sample_rate_preserved() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "rate.wav").string();

    trtmc::AudioResult ar;
    ar.samples = {0.1F, 0.2F, 0.3F};
    ar.num_samples = 3;
    ar.sample_rate = 44100;

    try {
        trtmc::io::write_wav(ar, path);
    } catch (...) {
        return false;
    }

    trtmc::AudioResult result;
    try {
        result = trtmc::io::read_wav(path);
    } catch (...) {
        return false;
    }

    return result.sample_rate == 44100;
}

// Intention: write_wav with a single sample produces a valid WAV.
// Preconditions:  AudioResult with one sample
// Postconditions: read_wav recovers exactly that one sample
static bool test_io_single_sample() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "one.wav").string();

    trtmc::AudioResult ar;
    ar.samples = {0.75F};
    ar.num_samples = 1;
    ar.sample_rate = 16000;

    try {
        trtmc::io::write_wav(ar, path);
    } catch (...) {
        return false;
    }

    trtmc::AudioResult result;
    try {
        result = trtmc::io::read_wav(path);
    } catch (...) {
        return false;
    }

    return result.samples.size() == 1 && std::abs(result.samples[0] - 0.75F) < 1e-6F;
}

// Intention: write_wav throws std::runtime_error when AudioResult is empty.
// Preconditions:  AudioResult.samples is empty
// Postconditions: std::runtime_error is thrown; no file written
static bool test_io_write_empty_throws() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "empty.wav").string();

    trtmc::AudioResult ar;
    // ar.samples is default-empty

    bool threw = false;
    try {
        trtmc::io::write_wav(ar, path);
    } catch (const std::runtime_error&) {
        threw = true;
    } catch (...) {
    }

    return threw;
}

// Intention: write_wav throws when the output path is not writable.
// Preconditions:  "/nonexistent/dir/" does not exist
// Postconditions: std::runtime_error is thrown
static bool test_io_write_bad_path_throws() {
    trtmc::AudioResult ar;
    ar.samples = {0.0F};
    ar.num_samples = 1;
    ar.sample_rate = 16000;

    bool threw = false;
    try {
        trtmc::io::write_wav(ar, "/nonexistent/dir/out.wav");
    } catch (const std::runtime_error&) {
        threw = true;
    } catch (...) {
    }

    return threw;
}

// Intention: read_wav throws when the file does not exist.
// Preconditions:  path does not exist on filesystem
// Postconditions: std::runtime_error is thrown
static bool test_io_read_missing_throws() {
    bool threw = false;
    try {
        trtmc::io::read_wav("/nonexistent/path/to/audio.wav");
    } catch (const std::runtime_error&) {
        threw = true;
    } catch (...) {
    }

    return threw;
}

// Intention: write_wav then read_wav preserves num_samples field.
// Preconditions:  AudioResult with known num_samples
// Postconditions: result.num_samples == original sample count
static bool test_io_num_samples_field() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "ns.wav").string();

    trtmc::AudioResult ar;
    ar.samples = {0.1F, 0.2F, 0.3F, 0.4F};
    ar.num_samples = 4;
    ar.sample_rate = 8000;

    try {
        trtmc::io::write_wav(ar, path);
    } catch (...) {
        return false;
    }

    trtmc::AudioResult result;
    try {
        result = trtmc::io::read_wav(path);
    } catch (...) {
        return false;
    }

    return result.num_samples == 4 && result.samples.size() == 4;
}

int main() {
    bool all_passed = true;
    std::cout << "test_trtmc_io:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << '\n';
        all_passed &= ok;
    };

    run("io_write_read_roundtrip", test_io_write_read_roundtrip);
    run("io_write_stereo_channel_major_layout", test_io_write_stereo_channel_major_layout);
    run("io_read_stereo_float_channel_major", test_io_read_stereo_float_channel_major);
    run("io_read_mono_multichannel_contract", test_io_read_mono_multichannel_contract);
    run("io_read_stereo_pcm16_and_mono_compatibility",
        test_io_read_stereo_pcm16_and_mono_compatibility);
    run("io_multichannel_channel_limit_preserves_mono_behavior",
        test_io_multichannel_channel_limit_preserves_mono_behavior);
    run("io_riff_declared_size_bounds_chunks", test_io_riff_declared_size_bounds_chunks);
    run("io_sample_rate_preserved", test_io_sample_rate_preserved);
    run("io_single_sample", test_io_single_sample);
    run("io_write_empty_throws", test_io_write_empty_throws);
    run("io_write_bad_path_throws", test_io_write_bad_path_throws);
    run("io_read_missing_throws", test_io_read_missing_throws);
    run("io_num_samples_field", test_io_num_samples_field);

    if (all_passed) {
        std::cout << "test_trtmc_io passed" << std::endl;
        return 0;
    }
    std::cerr << "test_trtmc_io FAILED" << std::endl;
    return 1;
}
