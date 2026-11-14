/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "audio_observation.h"

#include <iostream>

int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char* name) {
        if (!ok) {
            std::cerr << "FAIL: " << name << '\n';
            ++failures;
        }
    };
    for (const int channels : {1, 2}) {
        trtmc::AudioResult audio{std::vector<float>(48000 * channels), 0, 48000, channels};
        for (const int count : {0, 48000 * channels}) {
            audio.num_samples = count;
            const auto summary = audio_observation(audio);
            check(summary.at("num_samples") == audio.samples.size(), "resolved sample count");
            check(summary.at("output_samples") == audio.samples.size(), "output sample count");
            check(summary.at("output_audio_seconds") == 1.0, "channel-aware duration");
            check(summary.at("num_channels") == channels, "channel metadata");
        }
        for (const int count : {-1, 1, 48000 * channels + 1}) {
            audio.num_samples = count;
            bool rejected = false;
            try {
                audio_observation(audio);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            check(rejected, "reject inconsistent sample count");
        }
    }
    for (const auto& audio :
         {trtmc::AudioResult{{0.0F}, 0, 48000, 2}, trtmc::AudioResult{{0.0F}, 0, 0, 1},
          trtmc::AudioResult{{0.0F}, 0, 48000, 0}}) {
        bool rejected = false;
        try {
            audio_observation(audio);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "reject invalid frame or metadata");
    }
    return failures ? 1 : 0;
}
