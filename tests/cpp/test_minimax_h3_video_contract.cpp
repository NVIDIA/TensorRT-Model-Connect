/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/minimax_h3_video_contract.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

trtmc::VideoImageInput image(int32_t width, int32_t height) {
    trtmc::VideoImageInput result;
    result.width = width;
    result.height = height;
    return result;
}

void test_text_default_and_frame_alignment() {
    trtmc::VideoGenerationRequest request;
    const auto contract = trtmc::cli::resolve_minimax_h3_video_contract(request);
    check(contract.num_frames == 124, "H3 CLI resolves the true default frame count");
    check(contract.height == 768 && contract.width == 1344,
          "H3 T2VA defaults to the public 768x1344 canvas");
    check(request.config.video_num_frames == contract.num_frames &&
              request.config.height == contract.height && request.config.width == contract.width,
          "H3 CLI makes resolved defaults explicit at the plugin boundary");

    trtmc::VideoGenerationRequest aligned;
    aligned.config.video_num_frames = 120;
    const auto aligned_contract = trtmc::cli::resolve_minimax_h3_video_contract(aligned);
    check(aligned_contract.num_frames == 124 && aligned.config.video_num_frames == 124,
          "H3 CLI aligns the five-second request exactly like plugin");

    trtmc::VideoGenerationRequest maximum;
    maximum.config.video_num_frames = 345;
    check(trtmc::cli::resolve_minimax_h3_video_contract(maximum).num_frames == 345,
          "H3 CLI preserves the largest aligned public frame count");
}

void test_fl2va_keyframe_derived_canvases() {
    trtmc::VideoGenerationRequest square;
    square.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    square.first_frame = image(1024, 1024);
    const auto square_contract = trtmc::cli::resolve_minimax_h3_video_contract(square);
    check(square_contract.height == 768 && square_contract.width == 768,
          "H3 square first-only FL2VA resolves before generation");

    trtmc::VideoGenerationRequest portrait;
    portrait.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    portrait.last_frame = image(720, 1280);
    const auto portrait_contract = trtmc::cli::resolve_minimax_h3_video_contract(portrait);
    check(portrait_contract.height == 1344 && portrait_contract.width == 768,
          "H3 portrait last-only FL2VA resolves before generation");

    trtmc::VideoGenerationRequest both;
    both.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    both.first_frame = image(640, 480);
    both.last_frame = image(720, 1280);
    const auto both_contract = trtmc::cli::resolve_minimax_h3_video_contract(both);
    check(both_contract.height == 768 && both_contract.width == 1024,
          "H3 two-endpoint FL2VA uses the first-frame aspect without rejecting the last");

    trtmc::VideoGenerationRequest explicit_override;
    explicit_override.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    explicit_override.first_frame = image(1024, 1024);
    explicit_override.config.height = 544;
    explicit_override.config.width = 960;
    const auto explicit_contract = trtmc::cli::resolve_minimax_h3_video_contract(explicit_override);
    check(explicit_contract.height == 544 && explicit_contract.width == 960,
          "H3 explicit FL2VA canvas overrides the keyframe aspect");

    trtmc::VideoGenerationRequest invalid_first;
    invalid_first.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    invalid_first.first_frame = image(5000, 1000);
    invalid_first.last_frame = image(1024, 1024);
    bool invalid_first_rejected = false;
    try {
        (void)trtmc::cli::resolve_minimax_h3_video_contract(invalid_first);
    } catch (const std::invalid_argument&) {
        invalid_first_rejected = true;
    }
    check(invalid_first_rejected,
          "H3 two-endpoint FL2VA never falls back from an invalid first-frame aspect");
}

void test_explicit_profile_and_invalid_requests() {
    trtmc::VideoGenerationRequest explicit_profile;
    explicit_profile.config.height = 544;
    explicit_profile.config.width = 960;
    const auto contract = trtmc::cli::resolve_minimax_h3_video_contract(explicit_profile);
    check(contract.height == 544 && contract.width == 960,
          "H3 CLI preserves the explicit public performance canvas");

    trtmc::VideoGenerationRequest reference_default;
    reference_default.mode = trtmc::VideoGenerationMode::kReferenceToVideoAudio;
    const auto reference_contract =
        trtmc::cli::resolve_minimax_h3_video_contract(reference_default);
    check(reference_contract.height == 768 && reference_contract.width == 1344,
          "H3 Ref2VA target geometry remains independent of reference aspect");

    trtmc::VideoGenerationRequest missing_endpoint;
    missing_endpoint.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    bool missing_endpoint_rejected = false;
    try {
        (void)trtmc::cli::resolve_minimax_h3_video_contract(missing_endpoint);
    } catch (const std::invalid_argument&) {
        missing_endpoint_rejected = true;
    }
    check(missing_endpoint_rejected, "H3 CLI rejects FL2VA without an endpoint before generation");

    trtmc::VideoGenerationRequest partial_dimensions;
    partial_dimensions.config.height = 768;
    bool partial_dimensions_rejected = false;
    try {
        (void)trtmc::cli::resolve_minimax_h3_video_contract(partial_dimensions);
    } catch (const std::invalid_argument&) {
        partial_dimensions_rejected = true;
    }
    check(partial_dimensions_rejected,
          "H3 CLI rejects a partial explicit canvas before generation");

    trtmc::VideoGenerationRequest unsupported_canvas;
    unsupported_canvas.config.height = 1024;
    unsupported_canvas.config.width = 1024;
    bool unsupported_canvas_rejected = false;
    try {
        (void)trtmc::cli::resolve_minimax_h3_video_contract(unsupported_canvas);
    } catch (const std::invalid_argument&) {
        unsupported_canvas_rejected = true;
    }
    check(unsupported_canvas_rejected, "H3 CLI rejects a non-profile canvas before generation");

    trtmc::VideoGenerationRequest too_long;
    too_long.config.video_num_frames = 346;
    bool too_long_rejected = false;
    try {
        (void)trtmc::cli::resolve_minimax_h3_video_contract(too_long);
    } catch (const std::invalid_argument&) {
        too_long_rejected = true;
    }
    check(too_long_rejected, "H3 CLI rejects a duration beyond the public 15-second profile");

    trtmc::VideoGenerationRequest too_short;
    too_short.config.video_num_frames = 107;
    bool too_short_rejected = false;
    try {
        (void)trtmc::cli::resolve_minimax_h3_video_contract(too_short);
    } catch (const std::invalid_argument&) {
        too_short_rejected = true;
    }
    check(too_short_rejected, "H3 CLI rejects a duration below the public five-second profile");
}

} // namespace

int main() {
    test_text_default_and_frame_alignment();
    test_fl2va_keyframe_derived_canvases();
    test_explicit_profile_and_invalid_requests();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All MiniMax-H3 CLI video contract tests passed\n";
    return 0;
}
