/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/options.h"
#include "trtmc/pipeline.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Callable>
void check_throws(Callable&& callable, const std::string& expected, const char* label,
                  bool exact = false) {
    try {
        callable();
        std::cerr << "FAIL: " << label << " did not throw\n";
        ++failures;
    } catch (const std::exception& error) {
        const std::string actual = error.what();
        check(exact ? actual == expected : actual.find(expected) != std::string::npos, label);
    }
}

trtmc::Wan22TI2VOptions parse_official_options() {
    // Python's default json.dumps output uses these \u escapes.  The C++
    // runtime must decode them before prompt cleaning and tokenization.
    constexpr auto config =
        R"({"negative_prompt":"\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd\uff0c\u9759\u6001\uff0c\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05\uff0c\u5b57\u5e55\uff0c\u98ce\u683c\uff0c\u4f5c\u54c1\uff0c\u753b\u4f5c\uff0c\u753b\u9762\uff0c\u9759\u6b62\uff0c\u6574\u4f53\u53d1\u7070\uff0c\u6700\u5dee\u8d28\u91cf\uff0c\u4f4e\u8d28\u91cf\uff0cJPEG\u538b\u7f29\u6b8b\u7559\uff0c\u4e11\u964b\u7684\uff0c\u6b8b\u7f3a\u7684\uff0c\u591a\u4f59\u7684\u624b\u6307\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8\uff0c\u7578\u5f62\u7684\uff0c\u6bc1\u5bb9\u7684\uff0c\u5f62\u6001\u7578\u5f62\u7684\u80a2\u4f53\uff0c\u624b\u6307\u878d\u5408\uff0c\u9759\u6b62\u4e0d\u52a8\u7684\u753b\u9762\uff0c\u6742\u4e71\u7684\u80cc\u666f\uff0c\u4e09\u6761\u817f\uff0c\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012\u7740\u8d70","num_inference_steps":50,"guidance_scale":5.0,"flow_shift":5.0,"seed":42,"video_height":704,"video_width":1280,"video_num_frames":121,"frame_rate":24})";
    return trtmc::parse_wan22_options(config);
}

trtmc::Wan22TI2VOptions parse_l0_options() {
    constexpr auto config =
        R"({"negative_prompt":"x","num_inference_steps":15,"guidance_scale":5.0,"flow_shift":5.0,"seed":42,"video_height":384,"video_width":672,"video_num_frames":5,"frame_rate":24,"text_seq_len":512})";
    return trtmc::parse_wan22_options(config);
}

void test_official_bundle_contract() {
    constexpr auto official =
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

    const auto options = parse_official_options();
    check(options.negative_prompt == official,
          "Wan2.2 escaped negative prompt decodes to official UTF-8");
    check(options.num_inference_steps == 50 && options.guidance_scale == 5.0F &&
              options.flow_shift == 5.0F && options.seed == 42,
          "Wan2.2 numeric bundle options are parsed");
    check(options.video_height == 704 && options.video_width == 1280 &&
              options.video_num_frames == 121 && options.frame_rate == 24 &&
              options.text_seq_len == 512,
          "Wan2.2 official video profile is parsed");
}

void test_l0_bundle_contract() {
    const auto options = parse_l0_options();
    check(options.num_inference_steps == 15 && options.guidance_scale == 5.0F &&
              options.flow_shift == 5.0F && options.video_height == 384 &&
              options.video_width == 672 && options.video_num_frames == 5 &&
              options.frame_rate == 24 && options.text_seq_len == 512,
          "Wan2.2 exact L0 profile is parsed");
}

void test_bundle_rejects_accuracy_changing_profile() {
    struct InvalidBundleCase {
        const char* config_json;
        const char* expected;
        const char* label;
        bool exact;
    };
    const InvalidBundleCase tests[] = {
        {R"({"negative_prompt":"x","flow_shift":3,"frame_rate":24})", "flow_shift=5",
         "Wan2.2 rejects non-official flow shift", false},
        {R"({"negative_prompt":"x","flow_shift":5,"frame_rate":16})", "frame_rate=24",
         "Wan2.2 rejects non-official frame rate", false},
        {R"({"negative_prompt":"x","video_height":720,"video_width":1280,"video_num_frames":121})",
         "one complete qualified profile", "Wan2.2 rejects mismatched bundle geometry", false},
        {R"({"negative_prompt":"x","num_inference_steps":15,"video_height":704,"video_width":1280,"video_num_frames":121,"frame_rate":24})",
         "one complete qualified profile", "Wan2.2 rejects L0 steps with official geometry", false},
        {R"({"negative_prompt":"x","num_inference_steps":50,"video_height":384,"video_width":672,"video_num_frames":5,"frame_rate":24})",
         "one complete qualified profile", "Wan2.2 rejects official steps with L0 geometry", false},
        {R"({"negative_prompt":"x","text_seq_len":500})", "text_seq_len=512",
         "Wan2.2 rejects a non-512 text contract", false},
        {R"({"negative_prompt":"x","seed":-2})", "Wan2.2-TI2V-5B bundle seed must be non-negative",
         "Wan2.2 rejects a negative bundle seed before unsigned RNG conversion", true},
    };

    for (const auto& test : tests) {
        check_throws([&] { (void)trtmc::parse_wan22_options(test.config_json); }, test.expected,
                     test.label, test.exact);
    }
}

void test_request_resolution_and_validation() {
    const auto options = parse_official_options();
    trtmc::GenerateConfig config;
    config.negative_prompt = "customer negative prompt";
    config.num_steps = 50;
    config.guidance_scale = 5.0F;
    config.seed = 7;
    config.height = 704;
    config.width = 1280;
    const auto request = trtmc::resolve_wan22_request(options, config);
    check(request.negative_prompt == config.negative_prompt,
          "Wan2.2 request forwards negative-prompt override");
    check(request.seed == 7 && request.num_inference_steps == 50 &&
              request.guidance_scale == 5.0F && request.flow_shift == 5.0F,
          "Wan2.2 request forwards fixed generation profile");
    check(request.video_height == 704 && request.video_width == 1280 &&
              request.video_num_frames == 121 && request.frame_rate == 24,
          "Wan2.2 request preserves official output geometry and fps");

    config.height = 720;
    check_throws([&] { (void)trtmc::resolve_wan22_request(options, config); },
                 "--height must match bundle profile height 704",
                 "Wan2.2 rejects unsupported request height");
    config.height = 704;
    config.width = 720;
    check_throws([&] { (void)trtmc::resolve_wan22_request(options, config); },
                 "--width must match bundle profile width 1280",
                 "Wan2.2 rejects unsupported request width");
}

void test_l0_request_resolution_and_validation() {
    const auto options = parse_l0_options();
    trtmc::GenerateConfig config;
    config.num_steps = 15;
    config.guidance_scale = 5.0F;
    config.height = 384;
    config.width = 672;
    const auto request = trtmc::resolve_wan22_request(options, config);
    check(request.video_height == 384 && request.video_width == 672 &&
              request.video_num_frames == 5 && request.num_inference_steps == 15 &&
              request.frame_rate == 24 && request.text_seq_len == 512,
          "Wan2.2 request preserves the complete L0 profile");
    config.num_steps = 50;
    check_throws([&] { (void)trtmc::resolve_wan22_request(options, config); },
                 "--num-steps must match the bundle's complete profile",
                 "Wan2.2 rejects an official step override on an L0 bundle");
}

void test_request_rejects_invalid_sentinel_values() {
    const auto options = parse_official_options();
    trtmc::GenerateConfig config;

    config.num_steps = 0;
    check_throws([&] { (void)trtmc::resolve_wan22_request(options, config); }, "num_steps must be",
                 "Wan2.2 rejects zero steps instead of defaulting");
    config.num_steps = -1;

    config.guidance_scale = std::numeric_limits<float>::quiet_NaN();
    check_throws([&] { (void)trtmc::resolve_wan22_request(options, config); },
                 "guidance_scale must be", "Wan2.2 rejects NaN guidance instead of defaulting");
    config.guidance_scale = -1.0F;

    config.seed = -2;
    check_throws([&] { (void)trtmc::resolve_wan22_request(options, config); }, "seed must be",
                 "Wan2.2 rejects invalid negative seed");
}

} // namespace

int main() {
    test_official_bundle_contract();
    test_l0_bundle_contract();
    test_bundle_rejects_accuracy_changing_profile();
    test_request_resolution_and_validation();
    test_l0_request_resolution_and_validation();
    test_request_rejects_invalid_sentinel_values();
    return failures == 0 ? 0 : 1;
}
