/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-04-SAM2-HOI
// Architecture:   ARCH-MODPLUG-001
// Unit Design:    UD-SEG-01
// Intent:         SAM2 HOI JPEG decoding matches the Pillow/libjpeg RGB boundary
// Preconditions:  A fixed, valid baseline JPEG stream is available
// Postconditions: Dimensions and every decoded RGB byte match the Pillow golden
// =============================================================================

#include "runtime/models/sam2_hoi/jpeg_decoder.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_jpeg_decode_matches_pillow_rgb_golden() {
    // Encoded with Pillow 12.2.0 from a 4x3 RGB fixture using JPEG quality 87
    // and 4:2:0 subsampling. The expected bytes below are Pillow's RGB decode.
    static constexpr char jpeg_bytes[] = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46\x00\x01\x01\x00"
                                         "\x00\x01\x00\x01\x00\x00\xFF\xDB\x00\x43"
                                         "\x00\x04\x03\x03\x04\x03\x03\x04\x04\x03\x04\x05\x04\x04"
                                         "\x05\x06\x0A\x07\x06\x06\x06\x06\x0D\x09"
                                         "\x0A\x08\x0A\x0F\x0D\x10\x10\x0F\x0D\x0F\x0E\x11\x13\x18"
                                         "\x14\x11\x12\x17\x12\x0E\x0F\x15\x1C\x15"
                                         "\x17\x19\x19\x1B\x1B\x1B\x10\x14\x1D\x1F\x1D\x1A\x1F\x18"
                                         "\x1A\x1B\x1A\xFF\xDB\x00\x43\x01\x04\x05"
                                         "\x05\x06\x05\x06\x0C\x07\x07\x0C\x1A\x11\x0F\x11\x1A\x1A"
                                         "\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A"
                                         "\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A"
                                         "\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A"
                                         "\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A\x1A"
                                         "\xFF\xC0\x00\x11\x08\x00\x03\x00\x04\x03"
                                         "\x01\x22\x00\x02\x11\x01\x03\x11\x01\xFF\xC4\x00\x1F\x00"
                                         "\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
                                         "\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07"
                                         "\x08\x09\x0A\x0B\xFF\xC4\x00\xB5\x10\x00"
                                         "\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01"
                                         "\x7D\x01\x02\x03\x00\x04\x11\x05\x12\x21"
                                         "\x31\x41\x06\x13\x51\x61\x07\x22\x71\x14\x32\x81\x91\xA1"
                                         "\x08\x23\x42\xB1\xC1\x15\x52\xD1\xF0\x24"
                                         "\x33\x62\x72\x82\x09\x0A\x16\x17\x18\x19\x1A\x25\x26\x27"
                                         "\x28\x29\x2A\x34\x35\x36\x37\x38\x39\x3A"
                                         "\x43\x44\x45\x46\x47\x48\x49\x4A\x53\x54\x55\x56\x57\x58"
                                         "\x59\x5A\x63\x64\x65\x66\x67\x68\x69\x6A"
                                         "\x73\x74\x75\x76\x77\x78\x79\x7A\x83\x84\x85\x86\x87\x88"
                                         "\x89\x8A\x92\x93\x94\x95\x96\x97\x98\x99"
                                         "\x9A\xA2\xA3\xA4\xA5\xA6\xA7\xA8\xA9\xAA\xB2\xB3\xB4\xB5"
                                         "\xB6\xB7\xB8\xB9\xBA\xC2\xC3\xC4\xC5\xC6"
                                         "\xC7\xC8\xC9\xCA\xD2\xD3\xD4\xD5\xD6\xD7\xD8\xD9\xDA\xE1"
                                         "\xE2\xE3\xE4\xE5\xE6\xE7\xE8\xE9\xEA\xF1"
                                         "\xF2\xF3\xF4\xF5\xF6\xF7\xF8\xF9\xFA\xFF\xC4\x00\x1F\x01"
                                         "\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01"
                                         "\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07"
                                         "\x08\x09\x0A\x0B\xFF\xC4\x00\xB5\x11\x00"
                                         "\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02"
                                         "\x77\x00\x01\x02\x03\x11\x04\x05\x21\x31"
                                         "\x06\x12\x41\x51\x07\x61\x71\x13\x22\x32\x81\x08\x14\x42"
                                         "\x91\xA1\xB1\xC1\x09\x23\x33\x52\xF0\x15"
                                         "\x62\x72\xD1\x0A\x16\x24\x34\xE1\x25\xF1\x17\x18\x19\x1A"
                                         "\x26\x27\x28\x29\x2A\x35\x36\x37\x38\x39"
                                         "\x3A\x43\x44\x45\x46\x47\x48\x49\x4A\x53\x54\x55\x56\x57"
                                         "\x58\x59\x5A\x63\x64\x65\x66\x67\x68\x69"
                                         "\x6A\x73\x74\x75\x76\x77\x78\x79\x7A\x82\x83\x84\x85\x86"
                                         "\x87\x88\x89\x8A\x92\x93\x94\x95\x96\x97"
                                         "\x98\x99\x9A\xA2\xA3\xA4\xA5\xA6\xA7\xA8\xA9\xAA\xB2\xB3"
                                         "\xB4\xB5\xB6\xB7\xB8\xB9\xBA\xC2\xC3\xC4"
                                         "\xC5\xC6\xC7\xC8\xC9\xCA\xD2\xD3\xD4\xD5\xD6\xD7\xD8\xD9"
                                         "\xDA\xE2\xE3\xE4\xE5\xE6\xE7\xE8\xE9\xEA"
                                         "\xF2\xF3\xF4\xF5\xF6\xF7\xF8\xF9\xFA\xFF\xDA\x00\x0C\x03"
                                         "\x01\x00\x02\x11\x03\x11\x00\x3F\x00\xDB"
                                         "\xF0\x4E\xB5\xA9\x78\x72\x0D\x62\xCB\x42\xD4\x6F\x34\xFB"
                                         "\x54\xD4\xA4\x41\x1C\x37\x0C\xA0\x88\xD1"
                                         "\x22\x42\x79\xE4\x88\xE2\x45\xDC\x79\x3B\x72\x49\x3C\xD1"
                                         "\x45\x15\xFA\x6E\x57\x80\xC2\x55\xC0\x50"
                                         "\x94\xE9\x45\xBE\x58\xEF\x14\xFA\x25\xD8\xFE\x64\xE2\x6C"
                                         "\x56\x22\x39\xB5\x55\x1A\x8D\x69\x1E\xAF"
                                         "\xAC\x22\xDF\xE2\x7F\xFF\xD9";
    const std::vector<std::uint8_t> expected{
        73, 83, 0,  157, 167, 81,  14,  33,  73,  237, 255, 255, 34, 44,  0,   183, 193, 107,
        34, 53, 93, 135, 154, 194, 128, 195, 144, 29,  96,  45,  57, 108, 200, 112, 163, 255,
    };

    const std::string path = "/tmp/trtmc_sam2_hoi_pillow_golden.jpg";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(jpeg_bytes, static_cast<std::streamsize>(sizeof(jpeg_bytes) - 1U));
        check(output.good(), "SAM2 HOI JPEG fixture write succeeds");
    }

    const auto frame = trtmc::sam2_hoi::decode_jpeg_pillow_rgb(path);
    check(frame.height == 3 && frame.width == 4, "SAM2 HOI JPEG dimensions match Pillow");
    check(frame.pixels.size() == expected.size(), "SAM2 HOI JPEG RGB shape matches Pillow");
    if (frame.pixels.size() == expected.size()) {
        for (std::size_t index = 0; index < expected.size(); ++index) {
            if (frame.pixels[index] != static_cast<float>(expected[index]) / 255.0F) {
                std::cerr << "FAIL: SAM2 HOI JPEG pixel " << index
                          << " actual=" << frame.pixels[index]
                          << " expected_byte=" << static_cast<int>(expected[index]) << '\n';
                ++g_failures;
                break;
            }
        }
    }

    const std::vector<std::string> batch_paths(11U, path);
    const auto batch = trtmc::sam2_hoi::decode_jpeg_pillow_rgb_batch(batch_paths);
    check(batch.size() == batch_paths.size(), "SAM2 HOI JPEG batch returns every frame");
    for (const auto& batch_frame : batch) {
        check(batch_frame.height == frame.height && batch_frame.width == frame.width &&
                  batch_frame.pixels == frame.pixels,
              "parallel JPEG decode is bitwise identical to serial decode");
    }
    (void)std::remove(path.c_str());
}

void test_jpeg_decode_failure_is_empty() {
    const auto frame = trtmc::sam2_hoi::decode_jpeg_pillow_rgb("/tmp/trtmc_sam2_hoi_missing.jpg");
    check(frame.empty(), "SAM2 HOI missing JPEG returns an empty frame");
}

void test_bounded_decode_joins_all_work_and_reports_lowest_failure() {
    constexpr std::size_t path_count = 12U;
    std::vector<std::string> paths;
    paths.reserve(path_count);
    for (std::size_t index = 0; index < path_count; ++index)
        paths.push_back("frame-" + std::to_string(index));

    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
    std::atomic<std::size_t> completed{0U};
    std::mutex gate_mutex;
    std::condition_variable gate_condition;
    std::size_t gate_arrivals = 0U;
    bool gate_open = false;

    const auto decoder = [&](const std::string& path) {
        const int active_now = active.fetch_add(1, std::memory_order_relaxed) + 1;
        int observed_max = max_active.load(std::memory_order_relaxed);
        while (observed_max < active_now &&
               !max_active.compare_exchange_weak(observed_max, active_now,
                                                 std::memory_order_relaxed)) {
        }

        {
            std::unique_lock<std::mutex> lock(gate_mutex);
            if (!gate_open) {
                ++gate_arrivals;
                if (gate_arrivals == trtmc::sam2_hoi::kMaxConcurrentJpegDecodes) {
                    gate_open = true;
                    gate_condition.notify_all();
                } else if (!gate_condition.wait_for(lock, std::chrono::seconds(2),
                                                    [&]() { return gate_open; })) {
                    gate_open = true;
                    gate_condition.notify_all();
                }
            }
        }

        const std::size_t index = static_cast<std::size_t>(std::stoul(path.substr(6U)));
        active.fetch_sub(1, std::memory_order_relaxed);
        completed.fetch_add(1U, std::memory_order_relaxed);
        if (index == 1U || index == 7U)
            throw std::runtime_error("decode failure " + std::to_string(index));
        return trtmc::VideoFrame{{static_cast<float>(index)}, 1, 1};
    };

    bool reported_lowest_failure = false;
    try {
        (void)trtmc::sam2_hoi::decode_jpeg_paths_bounded(
            paths, trtmc::sam2_hoi::kMaxConcurrentJpegDecodes, decoder);
    } catch (const std::runtime_error& error) {
        reported_lowest_failure = std::string(error.what()) == "decode failure 1";
    }

    check(max_active.load(std::memory_order_relaxed) ==
              static_cast<int>(trtmc::sam2_hoi::kMaxConcurrentJpegDecodes),
          "SAM2 HOI JPEG batch executes with exactly the five-worker cap");
    check(completed.load(std::memory_order_relaxed) == path_count,
          "SAM2 HOI JPEG batch joins and completes all paths before throwing");
    check(reported_lowest_failure, "SAM2 HOI JPEG batch reports the lowest-index decode failure");
}

} // namespace

int main() {
    test_jpeg_decode_matches_pillow_rgb_golden();
    test_jpeg_decode_failure_is_empty();
    test_bounded_decode_joins_all_work_and_reports_lowest_failure();

    if (g_failures != 0) {
        std::cerr << g_failures << " SAM2 HOI JPEG test(s) failed\n";
        return 1;
    }
    std::cout << "SAM2 HOI JPEG tests passed\n";
    return 0;
}
