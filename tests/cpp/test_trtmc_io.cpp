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
#include <filesystem>
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
