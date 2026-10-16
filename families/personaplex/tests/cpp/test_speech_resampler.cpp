/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The Native reference loads audio with librosa.resample, whose default is
// soxr's "HQ" recipe. The expected samples below are librosa 0.11 soxr-HQ
// outputs for the deterministic 16 kHz signal this test regenerates.

#include "families/personaplex/runtime/resampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

std::vector<float> test_signal() {
    constexpr double pi = 3.14159265358979323846;
    std::vector<float> signal(1600);
    std::int64_t state = 12345;
    for (std::size_t i = 0; i < signal.size(); ++i) {
        state = (1103515245LL * state + 12345LL) % (1LL << 31);
        const double noise = static_cast<double>(state) / static_cast<double>(1LL << 31) - 0.5;
        const double t = static_cast<double>(i) / 16000.0;
        signal[i] = static_cast<float>(0.3 * std::sin(2.0 * pi * 440.0 * t) +
                                       0.2 * std::sin(2.0 * pi * 3100.0 * t) +
                                       0.1 * std::sin(2.0 * pi * 6500.0 * t) + 0.01 * noise);
    }
    return signal;
}

void test_matches_librosa_soxr_hq() {
    const std::vector<std::pair<std::size_t, float>> expected = {
        {0, 0.011917337775230408F},       {60, 0.07347443699836731F},
        {120, 0.287638783454895F},        {180, 0.38372278213500977F},
        {240, 0.1780434101819992F},       {300, -0.10074397921562195F},
        {360, -0.1764850914478302F},      {420, -0.1912091076374054F},
        {480, -0.28382164239883423F},     {540, -0.2777569890022278F},
        {600, 0.004780381917953491F},     {660, 0.279096394777298F},
        {720, 0.2877911329269409F},       {780, 0.1882505565881729F},
        {840, 0.17866060137748718F},      {900, 0.10334031283855438F},
        {960, -0.1773723065853119F},      {1020, -0.3880367875099182F},
        {1080, -0.2856844663619995F},     {1140, -0.07874905318021774F},
        {1200, -0.00021627545356750488F}, {1260, 0.07542504370212555F},
        {1320, 0.2867894172668457F},      {1380, 0.38249242305755615F},
        {1440, 0.1745864748954773F},      {1500, -0.1018492728471756F},
        {1560, -0.1743355095386505F},     {1620, -0.18630856275558472F},
        {1680, -0.28137555718421936F},    {1740, -0.276859313249588F},
        {1800, 0.0029449015855789185F},   {1860, 0.27915632724761963F},
        {1920, 0.28788119554519653F},     {1980, 0.18228313326835632F},
        {2040, 0.17793279886245728F},     {2100, 0.09714777767658234F},
        {2160, -0.17875248193740845F},    {2220, -0.38646167516708374F},
        {2280, -0.2867954671382904F},     {2340, -0.07849051058292389F},
        {2399, -0.22822284698486328F},
    };
    const auto signal = test_signal();
    const auto output = trtmc::resample_audio(
        signal.data(), static_cast<std::int32_t>(signal.size()), 16000, 24000);
    check(output.size() == 2400U, "output length is ceil(samples * 24000 / 16000)");
    double worst = 0.0;
    for (const auto& [index, value] : expected) {
        if (index < output.size())
            worst = std::max(worst, static_cast<double>(std::abs(output[index] - value)));
    }
    check(worst <= 2.0e-4,
          "16 -> 24 kHz matches soxr HQ within 2e-4 (worst " + std::to_string(worst) + ")");
}

void test_identity_and_validation() {
    const std::vector<float> signal = {0.25F, -0.5F, 0.75F};
    check(trtmc::resample_audio(signal.data(), 3, 24000, 24000) == signal,
          "equal rates copy the samples");
    bool threw = false;
    try {
        trtmc::resample_audio(signal.data(), 3, 0, 24000);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "non-positive rates are rejected");
}

} // namespace

int main() {
    test_matches_librosa_soxr_hq();
    test_identity_and_validation();
    if (g_failures != 0) {
        std::cerr << g_failures << " personaplex resampler test(s) failed\n";
        return 1;
    }
    return 0;
}
