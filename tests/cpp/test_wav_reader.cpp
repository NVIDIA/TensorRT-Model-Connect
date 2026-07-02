/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-17
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         WAV reader: read/write round-trip for PCM int16 and IEEE float32, linear
// resampling Preconditions:  Synthetic WAV files written to temp directory Postconditions: Read
// samples match written values, sample rate preserved, resample produces correct length
// =============================================================================

// Test suite: WAV reader utility.
//
// Purpose:
//   Validates read_wav() and resample_linear() from utils/wav_reader.h.
//   Creates synthetic WAV files (PCM int16 and IEEE float32), reads them
//   back, and verifies sample values and sample rate.
//
// Dependencies:
//   - utils/wav_reader.h: read_wav, resample_linear
//   - No TRT, GPU, or CUDA required.

#include "test_helpers.h"
#include "utils/wav_reader.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static std::filesystem::path make_temp_dir() {
    char pattern[] = "/tmp/trtmc_wav_test_XXXXXX";
    char* dir = mkdtemp(pattern);
    if (dir == nullptr)
        throw std::runtime_error("mkdtemp failed");
    return std::filesystem::path(dir);
}

// Write a minimal WAV file with PCM int16 mono data.
static void write_pcm16_wav(const std::string& path, const std::vector<int16_t>& samples,
                            uint32_t sample_rate) {
    std::ofstream out(path, std::ios::binary);
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t file_size = 36 + data_size;
    const uint16_t channels = 1;
    const uint16_t bits = 16;
    const uint32_t byte_rate = sample_rate * channels * bits / 8;
    const uint16_t block_align = channels * bits / 8;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&file_size), 4);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    uint32_t fmt_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    uint16_t fmt_tag = 1; // PCM
    out.write(reinterpret_cast<const char*>(&fmt_tag), 2);
    out.write(reinterpret_cast<const char*>(&channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits), 2);
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);
    out.write(reinterpret_cast<const char*>(samples.data()),
              static_cast<std::streamsize>(data_size));
}

// Write a minimal WAV file with IEEE float32 mono data.
static void write_float32_wav(const std::string& path, const std::vector<float>& samples,
                              uint32_t sample_rate) {
    std::ofstream out(path, std::ios::binary);
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(float));
    const uint32_t file_size = 36 + data_size;
    const uint16_t channels = 1;
    const uint16_t bits = 32;
    const uint32_t byte_rate = sample_rate * channels * bits / 8;
    const uint16_t block_align = channels * bits / 8;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&file_size), 4);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    uint32_t fmt_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    uint16_t fmt_tag = 3; // IEEE float
    out.write(reinterpret_cast<const char*>(&fmt_tag), 2);
    out.write(reinterpret_cast<const char*>(&channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits), 2);
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);
    out.write(reinterpret_cast<const char*>(samples.data()),
              static_cast<std::streamsize>(data_size));
}

// Write a stereo PCM16 WAV file.
static void write_stereo_pcm16_wav(const std::string& path, const std::vector<int16_t>& samples,
                                   uint32_t sample_rate) {
    std::ofstream out(path, std::ios::binary);
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t file_size = 36 + data_size;
    const uint16_t channels = 2;
    const uint16_t bits = 16;
    const uint32_t byte_rate = sample_rate * channels * bits / 8;
    const uint16_t block_align = channels * bits / 8;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&file_size), 4);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    uint32_t fmt_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    uint16_t fmt_tag = 1;
    out.write(reinterpret_cast<const char*>(&fmt_tag), 2);
    out.write(reinterpret_cast<const char*>(&channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits), 2);
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);
    out.write(reinterpret_cast<const char*>(samples.data()),
              static_cast<std::streamsize>(data_size));
}

int main() {
    auto tmp = make_temp_dir();

    // Test 1: Read PCM int16 mono WAV
    {
        std::vector<int16_t> pcm = {0, 16384, 32767, -32768, -16384, 0};
        auto path = (tmp / "mono16.wav").string();
        write_pcm16_wav(path, pcm, 16000);
        auto wav = trtmc::read_wav(path);
        check(wav.sample_rate == 16000, "pcm16_mono: sample_rate");
        check(wav.samples.size() == 6, "pcm16_mono: sample count");
        check(std::abs(wav.samples[0]) < 1e-5F, "pcm16_mono: sample[0] == 0");
        check(std::abs(wav.samples[1] - 0.5F) < 0.01F, "pcm16_mono: sample[1] ~ 0.5");
        check(wav.samples[3] < -0.99F, "pcm16_mono: sample[3] ~ -1.0");
    }

    // Test 2: Read IEEE float32 mono WAV
    {
        std::vector<float> flt = {0.0F, 0.5F, 1.0F, -1.0F, -0.5F};
        auto path = (tmp / "monof32.wav").string();
        write_float32_wav(path, flt, 44100);
        auto wav = trtmc::read_wav(path);
        check(wav.sample_rate == 44100, "float32_mono: sample_rate");
        check(wav.samples.size() == 5, "float32_mono: sample count");
        check(std::abs(wav.samples[1] - 0.5F) < 1e-6F, "float32_mono: sample[1]");
        check(std::abs(wav.samples[3] + 1.0F) < 1e-6F, "float32_mono: sample[3]");
    }

    // Test 3: Read stereo WAV -> mono (averaged)
    {
        // stereo: [L0, R0, L1, R1] = [1000, 3000, -2000, -4000]
        std::vector<int16_t> stereo = {1000, 3000, -2000, -4000};
        auto path = (tmp / "stereo16.wav").string();
        write_stereo_pcm16_wav(path, stereo, 16000);
        auto wav = trtmc::read_wav(path);
        check(wav.samples.size() == 2, "stereo_to_mono: 2 mono samples");
        // Expected: (1000+3000)/2/32768 ~ 0.061, (-2000-4000)/2/32768 ~ -0.0916
        float expected0 = (1000.0F / 32768.0F + 3000.0F / 32768.0F) * 0.5F;
        check(std::abs(wav.samples[0] - expected0) < 0.001F, "stereo_to_mono: sample[0]");
    }

    // Test 4: resample_linear (16kHz -> 8kHz: half the samples)
    {
        std::vector<float> src = {0.0F, 1.0F, 0.0F, -1.0F, 0.0F, 1.0F, 0.0F, -1.0F};
        auto resampled = trtmc::resample_linear(src.data(), 8, 16000, 8000);
        check(resampled.size() == 4, "resample: half count");
    }

    // Test 5: resample_linear identity (same rate)
    {
        std::vector<float> src = {0.1F, 0.2F, 0.3F};
        auto resampled = trtmc::resample_linear(src.data(), 3, 16000, 16000);
        check(resampled.size() == 3, "resample_identity: same count");
        check(std::abs(resampled[1] - 0.2F) < 1e-6F, "resample_identity: values match");
    }

    // Test 6: Invalid file throws
    {
        bool caught = false;
        try {
            trtmc::read_wav("/nonexistent/file.wav");
        } catch (...) {
            caught = true;
        }
        check(caught, "invalid_file: throws");
    }

    trtmc_test::remove_all_safe(tmp);
    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
    }
    return failures;
}
