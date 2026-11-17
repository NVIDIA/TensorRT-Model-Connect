/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host sampling math of the flux3 bundle contract (runtime/sampling.h).
//
// The expected values were computed independently with NumPy 2.4 float32 and
// bfloat16 emulation and are embedded as float32 bit patterns:
//   bf16(x)     u = x.view(uint32); ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16,
//               any NaN -> 0x7FC00000 (non-NaN results agree with torch and ml_dtypes)
//   schedule    u = np.linspace(1.0, 1 / N, N).astype(np.float32)
//               sigma = 6.93 * u / (1 + (6.93 - 1) * u)        (float32, NEP 50 scalars)
//               t = sigma * np.float32(1000), each with a terminal 0 appended;
//               bit-equal to Diffusers' FlowMatchEulerDiscreteScheduler(shift=6.93)
//               .set_timesteps(N, sigmas=np.linspace(1.0, 1 / N, N))
//   feed        bf16(bf16(t / np.float32(1000)) * np.float32(1000))
//   projection  e = np.float32(-9.210340371976184) * k / np.float32(128)
//               arg = np.float32(t) * np.exp(e.astype(np.float64)).astype(np.float32)
//               np.cos(arg.astype(np.float64)).astype(np.float32), same for np.sin
//   packing     bf16(x.reshape(C, H * W).T.reshape(-1)) with
//               x[c, h, w] = f32(100c + 10h + w) + f32(0.3)
//   Euler       bf16(x + bf16(bf16(d) * v)) with d = sigma[14] - sigma[13] of the N = 16
//               schedule; x and v are bf16(np.random.RandomState(20261111).standard_normal)
//               samples where the product rounding changes the result.

#include "families/flux3/runtime/sampling.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace flux3 = trtmc::flux3;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float from_bits(std::uint32_t value) {
    float result = 0.0F;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::string hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08X", static_cast<unsigned>(value));
    return text;
}

void check_bits(float actual, std::uint32_t expected, const std::string& label) {
    check(bits(actual) == expected,
          label + ": got " + hex(bits(actual)) + ", expected " + hex(expected));
}

template <std::size_t Size>
void check_all_bits(const std::vector<float>& actual,
                    const std::array<std::uint32_t, Size>& expected, const std::string& label) {
    check(actual.size() == Size, label + ": size " + std::to_string(actual.size()));
    for (std::size_t index = 0; index < Size && index < actual.size(); ++index)
        check_bits(actual[index], expected[index], label + "[" + std::to_string(index) + "]");
}

// Distance in representable float32 values, valid across zero.
std::int64_t ulp_distance(float left, float right) {
    const auto ordered = [](float value) {
        const auto raw = bits(value);
        const auto magnitude = static_cast<std::int64_t>(raw & 0x7FFFFFFFU);
        return (raw & 0x80000000U) != 0 ? -magnitude : magnitude;
    };
    const auto distance = ordered(left) - ordered(right);
    return distance < 0 ? -distance : distance;
}

template <typename Exception, typename Function>
void check_throws(Function&& function, const std::string& label) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    check(false, label + ": no exception");
}

void test_round_bf16() {
    struct Case {
        std::uint32_t input;
        std::uint32_t expected;
    };
    const std::array<Case, 20> cases = {{
        {0x00000000U, 0x00000000U}, // +0
        {0x80000000U, 0x80000000U}, // -0
        {0x3F800000U, 0x3F800000U}, // 1, exact
        {0x3F807FFFU, 0x3F800000U}, // just below a tie
        {0x3F808000U, 0x3F800000U}, // tie, even neighbour below
        {0x3F808001U, 0x3F810000U}, // just above a tie
        {0x3F818000U, 0x3F820000U}, // tie, odd neighbour below: up to even
        {0xBF818000U, 0xBF820000U}, // negative tie
        {0x3EAAAAABU, 0x3EAB0000U}, // 1/3
        {0x00008000U, 0x00000000U}, // subnormal tie down to zero
        {0x00018000U, 0x00020000U}, // subnormal tie up to even
        {0x7F7F7FFFU, 0x7F7F0000U}, // largest input that stays finite
        {0x7F7F8000U, 0x7F800000U}, // tie at the top overflows
        {0x7F7FFFFFU, 0x7F800000U}, // FLT_MAX overflows to +inf
        {0xFF7FFFFFU, 0xFF800000U}, // -FLT_MAX overflows to -inf
        {0x7F800000U, 0x7F800000U}, // +inf
        {0xFF800000U, 0xFF800000U}, // -inf
        {0x7FC00000U, 0x7FC00000U}, // quiet NaN
        {0x7F800001U, 0x7FC00000U}, // signalling NaN must not become inf
        {0xFFFFFFFFU, 0x7FC00000U}, // negative NaN with a full payload
    }};
    for (const auto& item : cases)
        check_bits(flux3::round_bf16(from_bits(item.input)), item.expected,
                   "bf16(" + hex(item.input) + ")");
    check(std::isnan(flux3::round_bf16(from_bits(0x7F800001U))), "bf16 keeps NaN");
}

template <std::size_t Size>
void check_schedule(std::int32_t steps, const std::array<std::uint32_t, Size>& sigmas,
                    const std::array<std::uint32_t, Size>& timesteps) {
    const auto schedule = flux3::make_schedule(steps, 6.93);
    const auto label = "N=" + std::to_string(steps);
    check_all_bits(schedule.sigmas, sigmas, label + " sigma");
    check_all_bits(schedule.timesteps, timesteps, label + " t");
}

void test_schedules() {
    check_schedule<17>(
        16,
        {0x3F800000U, 0x3F7D8F8DU, 0x3F7AD44FU, 0x3F77BFFBU, 0x3F744062U, 0x3F703E06U, 0x3F6B9A00U,
         0x3F662AE3U, 0x3F5FB7B0U, 0x3F57EFFAU, 0x3F4E5E4DU, 0x3F425030U, 0x3F32A89AU, 0x3F1D8262U,
         0x3EFEB6ACU, 0x3EA1CB78U, 0x00000000U},
        {0x447A0000U, 0x44779E30U, 0x4474F355U, 0x4471F17BU, 0x446E86E0U, 0x446A9C92U, 0x44661464U,
         0x4460C5E2U, 0x445A7962U, 0x4452E05AU, 0x44498817U, 0x443DC24FU, 0x442E78A6U, 0x4419D154U,
         0x43F8BE64U, 0x439E00B3U, 0x00000000U});
    check_schedule<5>(4, {0x3F800000U, 0x3F744062U, 0x3F5FB7B0U, 0x3F32A89AU, 0x00000000U},
                      {0x447A0000U, 0x446E86E0U, 0x445A7962U, 0x442E78A6U, 0x00000000U});
    check_schedule<2>(1, {0x3F800000U, 0x00000000U}, {0x447A0000U, 0x00000000U});
    check_throws<std::invalid_argument>([] { flux3::make_schedule(0, 6.93); }, "zero steps");
}

void check_feeds(std::int32_t steps, const std::vector<float>& expected) {
    const auto schedule = flux3::make_schedule(steps, 6.93);
    check(schedule.timesteps.size() == expected.size(), "feed count N=" + std::to_string(steps));
    for (std::size_t index = 0; index < expected.size() && index < schedule.timesteps.size();
         ++index)
        check(flux3::timestep_feed(schedule.timesteps[index]) == expected[index],
              "feed N=" + std::to_string(steps) + " i=" + std::to_string(index));
}

// The feed of step i is the timestep embedding argument and that of i + 1 its target.
void test_timestep_feeds() {
    check_feeds(
        16, {1000, 992, 980, 968, 952, 936, 920, 900, 876, 844, 804, 756, 700, 616, 498, 316, 0});
    check_feeds(4, {1000, 952, 876, 700, 0});
    check_feeds(1, {1000, 0});
    check(flux3::timestep_feed(990.47168F) == 992.0F, "feed rounds t / 1000 to bf16 first");
}

void check_projection(float feed, const std::array<std::uint32_t, 7>& cosines,
                      const std::array<std::uint32_t, 7>& sines) {
    constexpr std::array<std::size_t, 7> picks = {0, 1, 5, 31, 64, 100, 127};
    const auto projection = flux3::timestep_projection(feed, 256, 10000);
    check(projection.size() == 256, "projection size");
    if (projection.size() != 256)
        return;
    for (std::size_t index = 0; index < picks.size(); ++index) {
        const auto k = picks[index];
        const auto label = "t=" + std::to_string(feed) + " k=" + std::to_string(k);
        check(ulp_distance(projection[k], from_bits(cosines[index])) <= 2, label + " cos");
        check(ulp_distance(projection[128 + k], from_bits(sines[index])) <= 2, label + " sin");
    }
}

void test_timestep_projection() {
    check_projection(
        1000.0F,
        {0x3F0FF813U, 0x3F4A24F9U, 0x3F6C1703U, 0x3F4C51EEU, 0xBF56CD64U, 0x3F3B54B1U, 0x3F7E85F7U},
        {0x3F53AE61U, 0x3F1D1414U, 0x3EC5F71CU, 0x3F1A3CE6U, 0xBF0B44F8U, 0x3F2E7ACCU,
         0x3DDBA804U});
    check_projection(
        992.0F,
        {0x3F3C7B23U, 0x3F609023U, 0x3EE95CD0U, 0x3F7A3603U, 0xBF613EB7U, 0x3F3C5FCBU, 0x3F7E8BFCU},
        {0xBF2D3C90U, 0xBEF5D31FU, 0x3F63DCE4U, 0xBE5888F0U, 0xBEF350F7U, 0x3F2D5A4CU,
         0x3DD9E7DFU});
    check_projection(
        316.0F,
        {0xBE88893BU, 0x3EA1F4A5U, 0x3F52DB44U, 0xBF534C50U, 0xBF7FF4E6U, 0x3F78D891U, 0x3F7FDA38U},
        {0x3F76BAC5U, 0xBF72DAE7U, 0x3F112C96U, 0x3F1087DAU, 0xBC96C881U, 0x3E70639EU,
         0x3D0B1048U});
    const auto zero = flux3::timestep_projection(0.0F, 256, 10000);
    for (std::size_t k = 0; k < 128; ++k)
        check(zero[k] == 1.0F && zero[128 + k] == 0.0F, "t=0 k=" + std::to_string(k));
    check_throws<std::invalid_argument>([] { flux3::timestep_projection(1.0F, 255, 10000); },
                                        "odd dim");
}

void test_text_window() {
    const std::array<std::array<std::size_t, 2>, 5> cases = {
        {{1, 80}, {79, 80}, {80, 80}, {81, 160}, {1280, 1280}}};
    for (const auto& item : cases)
        check(flux3::text_window(item[0], 80, 1280) == static_cast<std::int32_t>(item[1]),
              "window for " + std::to_string(item[0]) + " tokens");
    check_throws<std::invalid_argument>([] { flux3::text_window(1281, 80, 1280); }, "over budget");
    check_throws<std::invalid_argument>([] { flux3::text_window(0, 80, 1280); }, "no tokens");
    check(flux3::pad_tokens({151644, 872, 198}, 5, 151643) ==
              std::vector<std::int32_t>{151644, 872, 198, 151643, 151643},
          "right padding");
    check_throws<std::invalid_argument>([] { flux3::pad_tokens({1, 2, 3}, 2, 0); },
                                        "ids above window");
    check(flux3::frame_prompt("<|im_start|>user\n", "a red fox",
                              "<|im_end|>\n<|im_start|>assistant\n") ==
              "<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n",
          "chat framing");
}

void test_pack_latents() {
    const flux3::LatentShape shape{3, 2, 3};
    std::vector<float> latents;
    for (int channel = 0; channel < shape.channels; ++channel) {
        for (int row = 0; row < shape.height; ++row) {
            for (int column = 0; column < shape.width; ++column)
                latents.push_back(static_cast<float>(100 * channel + 10 * row + column) + 0.3F);
        }
    }
    const std::array<std::uint32_t, 18> expected = {
        0x3E9A0000U, 0x42C90000U, 0x43480000U, 0x3FA60000U, 0x42CB0000U, 0x43490000U,
        0x40130000U, 0x42CD0000U, 0x434A0000U, 0x41250000U, 0x42DD0000U, 0x43520000U,
        0x41350000U, 0x42DF0000U, 0x43530000U, 0x41450000U, 0x42E10000U, 0x43540000U};
    check_all_bits(flux3::pack_latents({latents.data(), latents.size()}, shape), expected,
                   "tokens");
    check_throws<std::invalid_argument>(
        [&] { flux3::pack_latents({latents.data(), latents.size() - 1}, shape); }, "short latents");
}

void test_euler_update() {
    const float sigma = from_bits(0x3F1D8262U);      // N = 16, i = 13
    const float next_sigma = from_bits(0x3EFEB6ACU); // i = 14
    check_bits(next_sigma - sigma, 0xBDF13860U, "float32 sigma step");
    const std::array<std::uint32_t, 8> x = {0x3D190000U, 0x3F050000U, 0x3DD10000U, 0x3F070000U,
                                            0xBE3F0000U, 0xBF860000U, 0xBEF40000U, 0xBE620000U};
    const std::array<std::uint32_t, 8> v = {0xBE690000U, 0x3EFA0000U, 0xBF9E0000U, 0x3F570000U,
                                            0x3F930000U, 0xBFA40000U, 0xBE470000U, 0xBE120000U};
    // The reference rounds d to bfloat16 before the product (verified against
    // FlowMatchEulerDiscreteScheduler.step on CUDA and CPU); with an unrounded d
    // the first value would be 0x3D840000.
    const std::array<std::uint32_t, 8> expected = {0x3D830000U, 0x3EED0000U, 0x3E7E0000U,
                                                   0x3EDC0000U, 0xBEA40000U, 0xBF660000U,
                                                   0xBEE80000U, 0xBE510000U};
    std::vector<float> latents;
    std::vector<float> velocity;
    for (std::size_t index = 0; index < x.size(); ++index) {
        latents.push_back(from_bits(x[index]));
        velocity.push_back(from_bits(v[index]));
    }
    flux3::euler_update(latents, {velocity.data(), velocity.size()}, sigma, next_sigma);
    check_all_bits(latents, expected, "euler");
    check_throws<std::invalid_argument>(
        [&] { flux3::euler_update(latents, {velocity.data(), 1}, sigma, next_sigma); },
        "short velocity");
}

void test_image_to_pixels() {
    const std::vector<float> planar = {-1.0F, 0.5F, 2.0F, -0.25F, 0.0F, 1.0F}; // [3, 1, 2]
    const auto pixels = flux3::image_to_pixels({planar.data(), planar.size()}, 1, 2);
    check(pixels == std::vector<float>{0.0F, 1.0F, 0.5F, 0.75F, 0.375F, 1.0F},
          "HWC clamp(x / 2 + 0.5)");
    const std::vector<float> invalid = {0.0F, NAN, 0.0F, 0.0F, 0.0F, 0.0F};
    check_throws<std::runtime_error>(
        [&] { flux3::image_to_pixels({invalid.data(), invalid.size()}, 1, 2); },
        "non-finite decoder output");
    check_throws<std::invalid_argument>(
        [&] { flux3::image_to_pixels({planar.data(), planar.size()}, 2, 2); }, "image shape");
}

void test_gaussian_latents() {
    const auto first = flux3::gaussian_latents(7, 1024);
    check(first.size() == 1024, "noise count");
    check(first == flux3::gaussian_latents(7, 1024), "noise is deterministic per seed");
    check(first != flux3::gaussian_latents(8, 1024), "noise depends on the seed");
}

} // namespace

int main() {
    test_round_bf16();
    test_schedules();
    test_timestep_feeds();
    test_timestep_projection();
    test_text_window();
    test_pack_latents();
    test_euler_update();
    test_image_to_pixels();
    test_gaussian_latents();
    if (failures != 0) {
        std::cerr << failures << " flux3 sampling check(s) failed\n";
        return 1;
    }
    std::cout << "flux3 sampling: all checks passed\n";
    return 0;
}
