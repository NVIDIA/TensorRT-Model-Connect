/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU checks of the Qwen-Image 2.1 sampling helpers against values computed with the
// reference scheduler formulas (diffusers FlowMatchEulerDiscreteScheduler, NumPy float32).
#include "families/qwen_image2_1/runtime/runtime_config.h"
#include "families/qwen_image2_1/runtime/sampling.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace trtmc::qwen_image2_1;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

void expect_near(float actual, float expected, float tolerance, const std::string& message) {
    expect(std::fabs(actual - expected) <= tolerance,
           message + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}

nlohmann::json runtime_document(int height, int width) {
    return {{"tensor_parallel_size", 1},
            {"image_height", height},
            {"image_width", width},
            {"image_channels", 4},
            {"latent_channels", 64},
            {"vae_scale_factor", 16},
            {"text_tokens", 512},
            {"text_hidden_size", 4096},
            {"prefix_tokens", 12},
            {"pad_token_id", 151643},
            {"prompt_prefix", "<|im_start|>user\n"},
            {"prompt_suffix", "<|im_end|>\n"},
            {"num_train_timesteps", 1000},
            {"base_shift", 0.5},
            {"max_shift", 0.9},
            {"base_image_seq_len", 256},
            {"max_image_seq_len", 8192},
            {"shift_terminal", 0.02},
            {"default_num_steps", 40}};
}

void test_runtime_config() {
    const auto config = parse_runtime_config(runtime_document(1024, 768).dump());
    expect(config.latent_height() == 64 && config.latent_width() == 48, "latent grid");
    expect(config.image_tokens() == 3072 && config.latent_count() == 3072U * 64U, "token count");

    auto parallel = runtime_document(1024, 1024);
    parallel["tensor_parallel_size"] = 2;
    auto misaligned = runtime_document(1000, 1024);
    auto missing = runtime_document(1024, 1024);
    missing.erase("prefix_tokens");
    auto mistyped = runtime_document(1024, 1024);
    mistyped["text_tokens"] = "many";
    auto oversized = runtime_document(1024, 1024);
    oversized["image_height"] = 1 << 20;
    auto rescaled = runtime_document(1024, 1024);
    rescaled["num_train_timesteps"] = 500;
    std::vector<std::string> documents{"{ not json"};
    for (const auto& document : {parallel, misaligned, missing, mistyped, oversized, rescaled})
        documents.push_back(document.dump());
    for (const auto& document : documents) {
        bool rejected = false;
        try {
            parse_runtime_config(document);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).rfind("Qwen-Image 2.1 runtime.json: ", 0) == 0;
        }
        expect(rejected, "malformed runtime.json must be rejected as std::runtime_error");
    }
}

void test_sigmas() {
    // 1024 x 1024: 4096 image tokens, mu = 0.6935483870967742.
    const auto large = parse_runtime_config(runtime_document(1024, 1024).dump());
    const auto sigmas = flow_match_sigmas(large, 40);
    expect(sigmas.size() == 41, "40 steps give 41 sigmas");
    expect_near(sigmas[0], 1.0F, 1e-6F, "first sigma");
    expect_near(sigmas[1], 0.9869637489318848F, 2e-6F, "second sigma");
    expect_near(sigmas[20], 0.6566662788391113F, 2e-6F, "middle sigma");
    expect_near(sigmas[39], 0.019999980926513672F, 2e-6F, "terminal stretch");
    expect(sigmas[40] == 0.0F, "appended terminal sigma");
    for (std::size_t index = 0; index + 1 < sigmas.size(); ++index)
        expect(sigmas[index] > sigmas[index + 1], "sigmas must decrease");

    // 512 x 512: 1024 image tokens, mu = 0.5387096774193548.
    const auto small = parse_runtime_config(runtime_document(512, 512).dump());
    const auto short_schedule = flow_match_sigmas(small, 20);
    expect(short_schedule.size() == 21, "20 steps give 21 sigmas");
    expect_near(short_schedule[1], 0.9681665897369385F, 2e-6F, "shift follows the token count");
    expect_near(short_schedule[10], 0.606309175491333F, 2e-6F, "middle sigma at 512");

    // One step leaves the terminal stretch undefined (0 / 0 in the reference).
    for (const int steps : {0, 1}) {
        bool rejected = false;
        try {
            flow_match_sigmas(large, steps);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "fewer than two steps must be rejected");
    }
}

void test_bfloat16_timestep() {
    expect(round_to_bfloat16(0.1F) == 0.10009765625F, "round 0.1");
    expect(round_to_bfloat16(987.3F) == 988.0F, "round 987.3");
    expect(round_to_bfloat16(-3.0000002F) == -3.0F, "round a negative value");
    expect(round_to_bfloat16(1.00390625F) == 1.0F, "tie rounds to the even significand");
    expect(round_to_bfloat16(1.01171875F) == 1.015625F, "tie rounds up to the even significand");
    expect(round_to_bfloat16(0.0F) == 0.0F && round_to_bfloat16(1.0F) == 1.0F, "exact values");

    expect(reference_timestep(1.0F, 1000) == 1.0F, "sigma 1");
    expect(reference_timestep(0.9869637489318848F, 1000) == 0.98828125F, "second step");
    expect(reference_timestep(0.6566662788391113F, 1000) == 0.65625F, "middle step");
    expect(reference_timestep(0.019999980926513672F, 1000) == 0.02001953125F, "last step");
}

void test_latent_layout() {
    // [channels = 2, height = 2, width = 3] -> tokens [6, 2] and back.
    const std::vector<float> chw = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15};
    const auto tokens = pack_latents(chw.data(), 2, 2, 3);
    const std::vector<float> expected = {0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15};
    expect(tokens == expected, "tokens are row-major with channels last");
    expect(unpack_latents(tokens, 2, 2, 3) == chw, "unpack inverts pack");

    std::vector<float> latents = {1.0F, -2.0F};
    euler_step(latents, {0.5F, 4.0F}, 1.0F, 0.75F);
    expect(latents[0] == 0.875F && latents[1] == -3.0F, "Euler step moves along the velocity");

    const auto first = gaussian_latents(4096, 7);
    expect(first == gaussian_latents(4096, 7), "a seed reproduces its latents");
    expect(first != gaussian_latents(4096, 8), "seeds differ");
    double sum = 0.0;
    double squares = 0.0;
    for (const float value : first) {
        sum += value;
        squares += static_cast<double>(value) * value;
    }
    expect(std::fabs(sum / 4096.0) < 0.1 && std::fabs(squares / 4096.0 - 1.0) < 0.1,
           "initial latents are standard normal");
}

void test_image_layout() {
    // [channels = 2, height = 1, width = 2] in [-1, 1] -> HWC in [0, 1], clamped.
    const std::vector<float> chw = {-1.0F, 0.0F, 3.0F, 1.0F};
    const auto hwc = image_to_hwc_unit(chw.data(), 2, 1, 2);
    const std::vector<float> expected = {0.0F, 1.0F, 0.5F, 1.0F};
    expect(hwc == expected, "image is interleaved and mapped to the unit range");
}

} // namespace

int main() {
    test_runtime_config();
    test_sigmas();
    test_bfloat16_timestep();
    test_latent_layout();
    test_image_layout();
    std::cout << "qwen_image2_1 sampling tests passed\n";
    return 0;
}
