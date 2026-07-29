/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/cosmos3/conditioning.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void test_chat_template() {
    const auto positive = trtmc::cosmos3::format_chat_prompt("A robot.", false);
    if (positive != "<|im_start|>system\n"
                    "You are a helpful assistant who will generate videos from a give prompt."
                    "<|im_end|>\n<|im_start|>user\n"
                    "A robot. The video is 7.9 seconds long and is of 24 FPS. "
                    "This video is of 720x1280 resolution."
                    "<|im_end|>\n<|im_start|>assistant\n") {
        throw std::runtime_error("Cosmos3 positive chat template differs from Diffusers");
    }
    const auto negative = trtmc::cosmos3::format_chat_prompt("artifacts", true);
    if (negative.find("The video is not 7.9 seconds long") == std::string::npos ||
        negative.find("This video is not of 720x1280 resolution.") == std::string::npos) {
        throw std::runtime_error("Cosmos3 negative metadata template is missing");
    }
}

void test_patch_round_trip() {
    constexpr std::size_t count = static_cast<std::size_t>(trtmc::cosmos3::kLatentChannels) *
                                  trtmc::cosmos3::kLatentFrames * trtmc::cosmos3::kLatentHeight *
                                  trtmc::cosmos3::kLatentWidth;
    std::vector<float> latents(count);
    for (std::size_t index = 0; index < latents.size(); ++index)
        latents[index] = static_cast<float>(index % 8191U) / 8191.0F;
    const auto restored =
        trtmc::cosmos3::unpatchify_latents(trtmc::cosmos3::patchify_latents(latents));
    if (restored != latents)
        throw std::runtime_error("Cosmos3 patchify/unpatchify is not lossless");
}

} // namespace

int main() {
    try {
        test_chat_template();
        test_patch_round_trip();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
