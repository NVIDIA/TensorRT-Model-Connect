/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/openpi_data_plane.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

template <typename Function>
void check_throws(Function&& function, const char* name) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, name);
}

void test_quantile_normalization_matches_openpi_broadcast_and_tail_rules() {
    const trtmc::openpi::QuantileStats input_stats{{0.0F, -10.0F}, {10.0F, 10.0F}};
    const std::vector<float> values{0.0F, -10.0F, 5.0F, 0.0F, 10.0F, 10.0F};
    const auto normalized = trtmc::openpi::quantile_normalize(values, 2, input_stats);
    check(normalized.size() == values.size(), "quantile normalization preserves shape");
    check_close(normalized[0], -1.0F, 1e-7F, "quantile normalization q01 maps to -1");
    check_close(normalized[2], -1.1920929e-7F, 2e-7F,
                "quantile normalization midpoint includes upstream epsilon");
    check_close(normalized[5], 0.99999988F, 2e-7F,
                "quantile normalization q99 includes upstream epsilon");

    const trtmc::openpi::QuantileStats output_stats{{0.0F, -10.0F}, {10.0F, 10.0F}};
    const std::vector<float> model_actions{-1.0F, 0.0F, 123.0F, 1.0F, 1.0F, -7.0F};
    const auto actions = trtmc::openpi::quantile_unnormalize(model_actions, 3, output_stats);
    check_close(actions[0], 0.0F, 1e-7F, "quantile unnormalize -1 maps to q01");
    check_close(actions[1], 9.536743e-7F, 1e-8F,
                "quantile unnormalize midpoint includes float32 epsilon");
    check_close(actions[2], 123.0F, 0.0F, "quantile unnormalize preserves padded tail");
    check_close(actions[5], -7.0F, 0.0F, "quantile unnormalize preserves every padded row tail");

    check_throws([&] { (void)trtmc::openpi::quantile_normalize(values, 3, input_stats); },
                 "quantile normalize rejects undersized stats");
    check_throws([&] { (void)trtmc::openpi::quantile_unnormalize(values, 1, output_stats); },
                 "quantile unnormalize rejects oversized stats");
}

void test_state_binning_and_prompt_format_match_pi05() {
    check(trtmc::openpi::discretize_state_value(-2.0F) == -1,
          "state values below range remain bin -1");
    check(trtmc::openpi::discretize_state_value(-1.0F) == 0, "state -1 maps to bin zero");
    check(trtmc::openpi::discretize_state_value(-0.9921875F) == 1,
          "state exact second boundary maps to bin one");
    check(trtmc::openpi::discretize_state_value(0.0F) == 128, "state zero maps to bin 128");
    check(trtmc::openpi::discretize_state_value(0.9921875F) == 255,
          "state last boundary maps to bin 255");
    check(trtmc::openpi::discretize_state_value(std::numeric_limits<float>::infinity()) == 255,
          "state positive infinity follows numpy digitize");
    check(trtmc::openpi::discretize_state_value(std::numeric_limits<float>::quiet_NaN()) == 255,
          "state NaN follows numpy digitize");

    const std::string padded_prompt =
        std::string("\xC2\xA0") + "pick_up\nblock" + std::string("\xE3\x80\x80");
    check(trtmc::openpi::clean_prompt(padded_prompt) == "pick up block",
          "prompt cleaning matches unicode strip and replacements");
    check(trtmc::openpi::clean_prompt("  keep\rreturn  ") == "keep\rreturn",
          "prompt cleaning only replaces newline, not carriage return");
    check(trtmc::openpi::format_pi05_prompt(padded_prompt, {-1.0F, 0.0F, 1.0F}) ==
              "Task: pick up block, State: 0 128 255;\nAction: ",
          "pi05 prompt formatting is exact");
    check_throws(
        [] {
            const std::string invalid("\xC0\xAF", 2);
            (void)trtmc::openpi::clean_prompt(invalid);
        },
        "prompt cleaning rejects non-canonical UTF-8");
}

void test_camera_slot_validation_and_order() {
    const std::vector<uint8_t> base{1, 2, 3};
    const std::vector<uint8_t> wrist{4, 5, 6};
    const std::vector<uint8_t> black{0, 0, 0};
    auto camera = [](std::string_view name, const std::vector<uint8_t>& data, bool valid) {
        trtmc::openpi::CameraFrameView view;
        view.name = name;
        view.height = 1;
        view.width = 1;
        view.channels = 3;
        view.valid = valid;
        view.pixel_type = trtmc::openpi::CameraPixelType::kUint8;
        view.uint8_data = data.data();
        view.element_count = data.size();
        return view;
    };
    const auto ordered = trtmc::openpi::validate_and_order_camera_slots(
        {camera("right_wrist_0_rgb", black, false), camera("base_0_rgb", base, true),
         camera("left_wrist_0_rgb", wrist, true)});
    check(ordered[0].name == "base_0_rgb" && ordered[1].name == "left_wrist_0_rgb" &&
              ordered[2].name == "right_wrist_0_rgb",
          "camera slots are returned in canonical order");
    trtmc::openpi::validate_pi05_two_camera_masks(ordered);

    check_throws(
        [&] {
            (void)trtmc::openpi::validate_and_order_camera_slots(
                {camera("base_0_rgb", base, true), camera("base_0_rgb", base, true),
                 camera("right_wrist_0_rgb", black, false)});
        },
        "camera validation rejects duplicates");
    const std::vector<uint8_t> non_black{0, 1, 0};
    check_throws(
        [&] {
            (void)trtmc::openpi::validate_and_order_camera_slots(
                {camera("base_0_rgb", base, true), camera("left_wrist_0_rgb", wrist, true),
                 camera("right_wrist_0_rgb", non_black, false)});
        },
        "camera validation rejects non-black masked slot");
}

void test_resize_with_pad_matches_jax_linear_geometry_and_values() {
    const auto geometry = trtmc::openpi::compute_resize_with_pad_geometry(256, 320, 60, 80);
    check(geometry.resized_height == 60 && geometry.resized_width == 75,
          "resize geometry preserves aspect ratio with truncation");
    check(geometry.pad_left == 2 && geometry.pad_right == 3 && geometry.pad_top == 0 &&
              geometry.pad_bottom == 0,
          "resize geometry puts odd extra padding on right/bottom");

    // JAX LINEAR uses a widened triangular kernel when downsampling. For each
    // identical row [0, 0, 255, 255], a 4 -> 2 resize yields [36, 219] after
    // weight normalization and ties-to-even uint8 rounding, not [0, 255].
    const std::vector<uint8_t> downsample_source{0, 0, 255, 255, 0, 0, 255, 255};
    const auto downsampled =
        trtmc::openpi::resize_with_pad_uint8(downsample_source.data(), 2, 4, 1, 1, 2);
    check(downsampled == std::vector<uint8_t>({36, 219}),
          "uint8 resize matches JAX antialiased triangle kernel");

    const std::vector<float> upsample_source{0.0F, 1.0F, 1.0F, 0.0F};
    const auto upsampled =
        trtmc::openpi::resize_with_pad_float(upsample_source.data(), 2, 2, 1, 3, 3);
    check(upsampled.size() == 9, "float resize produces target geometry");
    check_close(upsampled[4], 0.5F, 1e-6F, "float resize uses half-pixel bilinear interpolation");

    const std::vector<float> wide_source{0.0F, 0.25F, 0.5F, 1.0F, 1.0F, 0.5F, 0.25F, 0.0F};
    const auto padded = trtmc::openpi::resize_with_pad_float(wide_source.data(), 2, 4, 1, 4, 4);
    check(padded.size() == 16, "float resize-with-pad target size");
    check_close(padded[0], -1.0F, 0.0F, "float resize top padding is normalized black");
    check_close(padded[4], 0.0F, 0.0F, "float resize preserves identity row");
    check_close(padded[15], -1.0F, 0.0F, "float resize bottom padding is normalized black");

    const auto normalized = trtmc::openpi::uint8_to_openpi_float({0, 64, 127, 128, 249, 255});
    const std::vector<float> expected_normalized{
        -0x1.0p+0F, -0x1.fdfe00p-2F, -0x1.0102p-8F, 0x1.01p-8F, 0x1.e7e7e4p-1F, 0x1.fffffcp-1F,
    };
    check(normalized == expected_normalized,
          "uint8 normalization is bit-exact with pinned JAX/XLA lowering");
}

void test_resize_matches_pinned_jax_0_5_3_goldens() {
    std::vector<uint8_t> uint8_source(45);
    for (std::size_t i = 0; i < uint8_source.size(); ++i) {
        uint8_source[i] = static_cast<uint8_t>(i * 5U);
    }
    // Generated directly with jax==0.5.3, ResizeMethod.LINEAR (default
    // antialias=True), followed by jnp.round/clip/uint8 and jnp.pad.
    const std::vector<uint8_t> expected_uint8{
        0,   0,   0,   0,   0,  0,  0,  0,  0,   0,   0,   0,   32,  37,  42,  49,
        54,  59,  67,  72,  77, 84, 89, 94, 126, 131, 136, 143, 148, 153, 161, 166,
        171, 178, 183, 188, 0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,
    };
    const auto actual_uint8 =
        trtmc::openpi::resize_with_pad_uint8(uint8_source.data(), 3, 5, 3, 4, 4);
    check(actual_uint8 == expected_uint8, "uint8 resize is exact against pinned JAX golden");

    const std::vector<float> float_source{
        -1.0F,         -0.931034505F, -0.862068951F,  -0.793103456F, -0.724137902F, -0.655172408F,
        -0.586206913F, -0.517241359F, -0.448275864F,  -0.37931034F,  -0.310344815F, -0.241379306F,
        -0.172413796F, -0.103448279F, -0.0344827585F, 0.0344827585F, 0.103448279F,  0.172413796F,
        0.241379306F,  0.310344815F,  0.37931034F,    0.448275864F,  0.517241359F,  0.586206913F,
        0.655172408F,  0.724137902F,  0.793103456F,   0.862068951F,  0.931034505F,  1.0F,
    };
    const std::vector<float> expected_float{
        -1.0F,         -1.0F,         -1.0F,         -1.0F,         -1.0F,         -1.0F,
        -1.0F,         -1.0F,         -0.706896544F, -0.637930989F, -0.545976996F, -0.477011472F,
        -0.385057449F, -0.316091925F, -0.224137917F, -0.155172408F, 0.155172408F,  0.224137932F,
        0.316091925F,  0.385057449F,  0.477011502F,  0.545976996F,  0.637931049F,  0.706896544F,
        -1.0F,         -1.0F,         -1.0F,         -1.0F,         -1.0F,         -1.0F,
        -1.0F,         -1.0F,
    };
    const auto actual_float =
        trtmc::openpi::resize_with_pad_float(float_source.data(), 3, 5, 2, 4, 4);
    check(actual_float.size() == expected_float.size(),
          "float JAX golden has matching element count");
    for (std::size_t i = 0; i < actual_float.size() && i < expected_float.size(); ++i) {
        check_close(actual_float[i], expected_float[i], 2e-7F,
                    "float resize matches pinned JAX golden");
    }
}

void test_fixed_noise_and_euler_helpers_are_reproducible() {
    const auto schedule = trtmc::openpi::make_euler_schedule(10);
    check(schedule.timesteps.size() == 10, "Euler schedule has ten denoising steps");
    check_close(schedule.dt, -0.1F, 0.0F, "Euler schedule dt matches OpenPI");
    check_close(schedule.timesteps.front(), 1.0F, 0.0F, "Euler schedule starts at noise time");
    check_close(schedule.timesteps.back(), 0.099999927F, 1e-7F,
                "Euler schedule retains iterative float32 drift");

    const std::vector<float> source_noise{0.25F, -0.5F, 1.0F, -1.0F};
    auto sample = trtmc::openpi::make_fixed_noise(source_noise, 1, 2, 2);
    check(sample == source_noise, "fixed noise helper preserves every supplied bit");
    trtmc::openpi::euler_step_in_place(sample, {0.5F, -1.0F, 2.0F, 4.0F}, schedule.dt);
    check_close(sample[0], 0.2F, 1e-7F, "Euler update first component");
    check_close(sample[1], -0.4F, 1e-7F, "Euler update second component");
    check_close(sample[2], 0.8F, 1e-7F, "Euler update third component");
    check_close(sample[3], -1.4F, 1e-7F, "Euler update fourth component");
    check_throws([&] { (void)trtmc::openpi::make_fixed_noise(source_noise, 1, 3, 2); },
                 "fixed noise rejects incorrect shape");
    check_throws([&] { trtmc::openpi::euler_step_in_place(sample, {1.0F}, -0.1F); },
                 "Euler update rejects mismatched velocity");
}

} // namespace

int main() {
    test_quantile_normalization_matches_openpi_broadcast_and_tail_rules();
    test_state_binning_and_prompt_format_match_pi05();
    test_camera_slot_validation_and_order();
    test_resize_with_pad_matches_jax_linear_geometry_and_values();
    test_resize_matches_pinned_jax_0_5_3_goldens();
    test_fixed_noise_and_euler_helpers_are_reproducible();

    if (g_failures != 0) {
        std::cerr << g_failures << " OpenPI data-plane test(s) failed\n";
        return 1;
    }
    return 0;
}
