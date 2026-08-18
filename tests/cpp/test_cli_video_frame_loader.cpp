/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CLI-CPP-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-CABI-01
// Intent:         Track-HOI frame loading selects and validates batch capability results
// Preconditions:  Fake tracking capabilities provide deterministic owned frames
// Postconditions: Ordered JPEG batches are used once; other inputs remain serial
// =============================================================================

#include "cli/video_frame_loader.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

trtmc::VideoFrame marker_frame(float marker) {
    return {{marker}, 1, 1};
}

class SerialTracker : public trtmc::IVideoTrackingPipeline {
  public:
    trtmc::VideoFrame load_video_frame(const std::string& path) override {
        serial_paths.push_back(path);
        return marker_frame(static_cast<float>(serial_paths.size()));
    }

    int32_t track_video(const std::vector<trtmc::VideoFrameView>&, const std::string&,
                        const std::string&) override {
        return 0;
    }

    std::vector<std::string> serial_paths;
};

class BatchTracker final : public SerialTracker, public trtmc::IVideoFrameBatchLoader {
  public:
    std::vector<trtmc::VideoFrame>
    load_video_frames(const std::vector<std::string>& paths) override {
        ++batch_calls;
        batch_paths = paths;
        return batch_result;
    }

    std::size_t max_video_frame_load_concurrency() const noexcept override {
        return max_concurrency;
    }

    std::size_t max_concurrency{5U};
    int batch_calls{0};
    std::vector<std::string> batch_paths;
    std::vector<trtmc::VideoFrame> batch_result;
};

void test_batch_capability_receives_one_ordered_request() {
    BatchTracker tracker;
    tracker.batch_result = {marker_frame(2.0F), marker_frame(10.0F)};
    const std::vector<std::filesystem::path> paths{"frame2.JPG", "frame10.jpeg"};

    const auto clip = trtmc::cli::decode_video_clip(tracker, paths);

    check(tracker.batch_calls == 1, "JPEG clip invokes model batch capability exactly once");
    check(tracker.batch_paths == std::vector<std::string>{"frame2.JPG", "frame10.jpeg"},
          "batch capability receives exact ordered string paths");
    check(tracker.serial_paths.empty(), "batch-capable JPEG clip skips serial decoder");
    check(clip.paths == paths, "decoded clip retains ordered source paths");
    check(clip.owned_frames.size() == 2U && clip.views.size() == 2U &&
              clip.views[0].pixels == clip.owned_frames[0].pixels.data() &&
              clip.views[1].pixels == clip.owned_frames[1].pixels.data() &&
              clip.views[0].pixels[0] == 2.0F && clip.views[1].pixels[0] == 10.0F,
          "decoded clip retains batch result order and stable owned views");
    check(clip.frame_decode_mode == "model_batch" && clip.frame_decode_max_concurrency == 5U,
          "decoded clip records model batch loading provenance");
}

void test_absent_or_ineligible_capability_uses_serial_fallback() {
    SerialTracker serial_tracker;
    const std::vector<std::filesystem::path> jpeg_paths{"frame1.jpg", "frame2.jpeg"};
    const auto serial_clip = trtmc::cli::decode_video_clip(serial_tracker, jpeg_paths);
    check(serial_tracker.serial_paths == std::vector<std::string>{"frame1.jpg", "frame2.jpeg"},
          "missing batch capability loads frames serially in order");
    check(serial_clip.frame_decode_mode == "serial" &&
              serial_clip.frame_decode_max_concurrency == 1U,
          "serial fallback records single-decoder provenance");

    BatchTracker mixed_tracker;
    const std::vector<std::filesystem::path> mixed_paths{"frame1.jpg", "frame2.png"};
    const auto mixed_clip = trtmc::cli::decode_video_clip(mixed_tracker, mixed_paths);
    check(mixed_tracker.batch_calls == 0 &&
              mixed_tracker.serial_paths == std::vector<std::string>{"frame1.jpg", "frame2.png"},
          "non-JPEG clip remains on the serial model decoder path");
    check(mixed_clip.frame_decode_mode == "serial" && mixed_clip.frame_decode_max_concurrency == 1U,
          "mixed clip reports its actual serial loading mode");
}

void test_batch_results_are_validated_before_views_escape() {
    const std::vector<std::filesystem::path> paths{"frame1.jpg", "frame2.jpg", "frame3.jpg"};

    BatchTracker short_tracker;
    short_tracker.batch_result = {marker_frame(1.0F), marker_frame(2.0F)};
    bool rejected_count = false;
    try {
        (void)trtmc::cli::decode_video_clip(short_tracker, paths);
    } catch (const std::runtime_error& error) {
        rejected_count = std::string(error.what()).find("invalid frame count") != std::string::npos;
    }
    check(rejected_count && short_tracker.batch_calls == 1,
          "batch loader rejects a non-corresponding result count");

    BatchTracker empty_tracker;
    empty_tracker.batch_result = {marker_frame(1.0F), {}, marker_frame(3.0F)};
    bool rejected_empty = false;
    try {
        (void)trtmc::cli::decode_video_clip(empty_tracker, paths);
    } catch (const std::runtime_error& error) {
        rejected_empty = std::string(error.what()).find("frame2.jpg") != std::string::npos;
    }
    check(rejected_empty && empty_tracker.batch_calls == 1,
          "batch loader rejects the lowest-index empty result with its source path");

    BatchTracker zero_cap_tracker;
    zero_cap_tracker.max_concurrency = 0U;
    zero_cap_tracker.batch_result = {marker_frame(1.0F), marker_frame(2.0F), marker_frame(3.0F)};
    bool rejected_zero_cap = false;
    try {
        (void)trtmc::cli::decode_video_clip(zero_cap_tracker, paths);
    } catch (const std::runtime_error& error) {
        rejected_zero_cap =
            std::string(error.what()).find("zero maximum concurrency") != std::string::npos;
    }
    check(rejected_zero_cap && zero_cap_tracker.batch_calls == 0,
          "batch loader rejects invalid capability metadata before decoding");
}

} // namespace

int main() {
    test_batch_capability_receives_one_ordered_request();
    test_absent_or_ineligible_capability_uses_serial_fallback();
    test_batch_results_are_validated_before_views_escape();

    if (g_failures != 0) {
        std::cerr << g_failures << " CLI video frame loader test(s) failed\n";
        return 1;
    }
    std::cout << "CLI video frame loader tests passed\n";
    return 0;
}
