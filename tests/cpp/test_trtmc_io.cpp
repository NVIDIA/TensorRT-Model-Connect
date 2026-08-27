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
#include <filesystem>
#include <fstream>
#include <iostream>
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

static void write_wav_fixture(const std::string& path, uint16_t format, uint16_t channels,
                              uint32_t sample_rate, uint16_t bits_per_sample, const void* samples,
                              uint32_t data_size, bool add_odd_junk = false) {
    std::ofstream output(path, std::ios::binary);
    const uint32_t file_size = 36 + data_size + (add_odd_junk ? 10 : 0);
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t fmt_size = 16;

    output.write("RIFF", 4);
    output.write(reinterpret_cast<const char*>(&file_size), 4);
    output.write("WAVE", 4);
    if (add_odd_junk) {
        const uint32_t junk_size = 1;
        const char junk[2] = {'x', '\0'};
        output.write("JUNK", 4);
        output.write(reinterpret_cast<const char*>(&junk_size), 4);
        output.write(junk, 2); // RIFF chunks are word-aligned.
    }
    output.write("fmt ", 4);
    output.write(reinterpret_cast<const char*>(&fmt_size), 4);
    output.write(reinterpret_cast<const char*>(&format), 2);
    output.write(reinterpret_cast<const char*>(&channels), 2);
    output.write(reinterpret_cast<const char*>(&sample_rate), 4);
    output.write(reinterpret_cast<const char*>(&byte_rate), 4);
    output.write(reinterpret_cast<const char*>(&block_align), 2);
    output.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    output.write("data", 4);
    output.write(reinterpret_cast<const char*>(&data_size), 4);
    output.write(static_cast<const char*>(samples), data_size);
}

static void write_truncated_chunk_fixture(const std::string& path) {
    std::ofstream output(path, std::ios::binary);
    const uint32_t file_size = 36;
    const uint32_t declared_chunk_size = 64;
    const char short_payload[24] = {};
    output.write("RIFF", 4);
    output.write(reinterpret_cast<const char*>(&file_size), 4);
    output.write("WAVE", 4);
    output.write("JUNK", 4);
    output.write(reinterpret_cast<const char*>(&declared_chunk_size), 4);
    output.write(short_payload, sizeof(short_payload));
}

static void write_riff_bound_violation_fixture(const std::string& path) {
    const int16_t sample = 1;
    write_wav_fixture(path, 1, 1, 16000, 16, &sample, static_cast<uint32_t>(sizeof(sample)));

    // The physical file contains the sample, but the RIFF container ends after
    // the data header. A parser must not consume bytes outside that boundary.
    const uint32_t declared_riff_size = 36;
    std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(4);
    output.write(reinterpret_cast<const char*>(&declared_riff_size), 4);
}

static void write_partial_chunk_header_fixture(const std::string& path) {
    const int16_t sample = 1;
    write_wav_fixture(path, 1, 1, 16000, 16, &sample, static_cast<uint32_t>(sizeof(sample)));

    {
        std::ofstream output(path, std::ios::binary | std::ios::app);
        output.write("JUNK", 4);
    }
    const uint32_t declared_riff_size =
        static_cast<uint32_t>(std::filesystem::file_size(path) - 8U);
    std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(4);
    output.write(reinterpret_cast<const char*>(&declared_riff_size), 4);
}

static void append_trailing_truncated_chunk(const std::string& path) {
    const uint32_t declared_chunk_size = 64;
    const char short_payload = 'x';
    std::ofstream output(path, std::ios::binary | std::ios::app);
    output.write("JUNK", 4);
    output.write(reinterpret_cast<const char*>(&declared_chunk_size), 4);
    output.write(&short_payload, 1);
}

static bool read_throws_with(const std::string& path, const std::string& expected) {
    try {
        (void)trtmc::io::read_wav(path);
    } catch (const trtmc::io::WavFormatError& error) {
        return std::string(error.what()).find(expected) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
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
    try {
        trtmc::io::read_wav("/nonexistent/path/to/audio.wav");
    } catch (const trtmc::io::WavFormatError&) {
        return false;
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
    }
    return false;
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

static bool test_io_multichannel_pcm16_downmix() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "pcm16-4ch.wav").string();
    const std::vector<int16_t> interleaved = {
        16384, 8192, 0,     -8192, // 0.125 after four-channel averaging
        0,     8192, 16384, 24576, // 0.375 after four-channel averaging
    };
    write_wav_fixture(path, 1, 4, 48000, 16, interleaved.data(),
                      static_cast<uint32_t>(interleaved.size() * sizeof(int16_t)), true);

    const auto result = trtmc::io::read_wav(path);
    return result.sample_rate == 48000 && result.num_samples == 2 && result.samples.size() == 2 &&
           std::abs(result.samples[0] - 0.125F) < 1e-6F &&
           std::abs(result.samples[1] - 0.375F) < 1e-6F;
}

static bool test_io_multichannel_float32_downmix() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "float32-3ch.wav").string();
    const std::vector<float> interleaved = {
        1.0F,  0.5F,  -0.75F, // 0.25 after three-channel averaging
        -1.0F, 0.25F, 0.75F,  // 0.0 after three-channel averaging
    };
    write_wav_fixture(path, 3, 3, 44100, 32, interleaved.data(),
                      static_cast<uint32_t>(interleaved.size() * sizeof(float)));

    const auto result = trtmc::io::read_wav(path);
    return result.sample_rate == 44100 && result.num_samples == 2 && result.samples.size() == 2 &&
           std::abs(result.samples[0] - 0.25F) < 1e-6F && std::abs(result.samples[1]) < 1e-6F;
}

static bool test_io_ignores_bytes_after_riff_container() {
    trtmc_test::TempDirGuard dir;
    const auto path = (std::filesystem::path(dir.path()) / "trailing-bytes.wav").string();
    const std::vector<int16_t> samples = {16384, -16384};
    write_wav_fixture(path, 1, 1, 16000, 16, samples.data(),
                      static_cast<uint32_t>(samples.size() * sizeof(int16_t)));
    append_trailing_truncated_chunk(path);

    const auto result = trtmc::io::read_wav(path);
    return result.sample_rate == 16000 && result.num_samples == 2 && result.samples.size() == 2 &&
           std::abs(result.samples[0] - 0.5F) < 1e-6F && std::abs(result.samples[1] + 0.5F) < 1e-6F;
}

static bool test_io_rejects_invalid_wav_contract() {
    trtmc_test::TempDirGuard dir;
    const auto root = std::filesystem::path(dir.path());
    const std::vector<uint8_t> pcm8 = {0, 255};
    const auto unsupported = (root / "pcm8.wav").string();
    write_wav_fixture(unsupported, 1, 1, 16000, 8, pcm8.data(), static_cast<uint32_t>(pcm8.size()));

    const std::vector<int16_t> incomplete_frame = {1};
    const auto incomplete = (root / "incomplete-frame.wav").string();
    write_wav_fixture(incomplete, 1, 2, 16000, 16, incomplete_frame.data(),
                      static_cast<uint32_t>(incomplete_frame.size() * sizeof(int16_t)));

    const std::vector<int16_t> sample = {1};
    const auto zero_channels = (root / "zero-channels.wav").string();
    write_wav_fixture(zero_channels, 1, 0, 16000, 16, sample.data(),
                      static_cast<uint32_t>(sample.size() * sizeof(int16_t)));

    const auto truncated = (root / "truncated-chunk.wav").string();
    write_truncated_chunk_fixture(truncated);

    const auto outside_riff = (root / "outside-riff-bound.wav").string();
    write_riff_bound_violation_fixture(outside_riff);

    const auto partial_header = (root / "partial-chunk-header.wav").string();
    write_partial_chunk_header_fixture(partial_header);

    return read_throws_with(unsupported, "PCM16 or IEEE float32") &&
           read_throws_with(incomplete, "complete audio frames") &&
           read_throws_with(zero_channels, "channels and sample rate must be positive") &&
           read_throws_with(truncated, "truncated chunk") &&
           read_throws_with(outside_riff, "truncated chunk") &&
           read_throws_with(partial_header, "truncated chunk header");
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
    run("io_sample_rate_preserved", test_io_sample_rate_preserved);
    run("io_single_sample", test_io_single_sample);
    run("io_write_empty_throws", test_io_write_empty_throws);
    run("io_write_bad_path_throws", test_io_write_bad_path_throws);
    run("io_read_missing_throws", test_io_read_missing_throws);
    run("io_num_samples_field", test_io_num_samples_field);
    run("io_multichannel_pcm16_downmix", test_io_multichannel_pcm16_downmix);
    run("io_multichannel_float32_downmix", test_io_multichannel_float32_downmix);
    run("io_ignores_bytes_after_riff_container", test_io_ignores_bytes_after_riff_container);
    run("io_rejects_invalid_wav_contract", test_io_rejects_invalid_wav_contract);

    if (all_passed) {
        std::cout << "test_trtmc_io passed" << std::endl;
        return 0;
    }
    std::cerr << "test_trtmc_io FAILED" << std::endl;
    return 1;
}
