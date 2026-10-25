/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Clock and latent assembly only; no model, decoder or media-quality claim.
#include "families/minimax_h3/runtime/reference_continuation.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
using namespace trtmc::minimax_h3;

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

template <class F> void rejects(F&& call) {
    try { call(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error("invalid continuation was accepted");
}

ReferenceDenoiserResult fixture(int32_t video_frames, int32_t audio_frames, float base) {
    ReferenceDenoiserResult result{{video_frames, 2, 2, audio_frames}, {}, {}, 49};
    result.video_rows.resize(static_cast<std::size_t>(video_frames) * 96);
    result.audio_rows.resize(static_cast<std::size_t>(audio_frames) * 64);
    for (std::size_t i = 0; i < result.video_rows.size(); ++i)
        result.video_rows[i] = base + i;
    for (int32_t channel = 0; channel < 2; ++channel)
        for (int32_t time = 0; time < audio_frames; ++time)
            for (int32_t feature = 0; feature < 32; ++feature)
                result.audio_rows[(channel * audio_frames + time) * 32 + feature] =
                    (channel ? -1.0F : 1.0F) * (base + time * 32 + feature);
    return result;
}

void test_assembly() {
    const auto windows = plan_reference_continuation(328, 124, 22, 3);
    require(windows.size() == 3 && windows[1].start_frame == 102 &&
                windows[1].end_frame == 226 && windows[1].temporal_offset == 170.0,
            "full native windows retain the source-defined frame and global AV clock");
    auto first = fixture(37, 207, 10000.0F);
    auto second = fixture(37, 207, 30000.0F);
    const auto original = first;
    const auto guide = reference_latent_guide(first, windows[1]);
    require(guide.shape.frames == 7 && guide.shape.audio_latents == 37 &&
                guide.video_rows.size() == 7 * 96 && guide.audio_rows.size() == 74 * 32,
            "guide uses exactly the hidden tail, not the full prior window");
    require(guide.video_rows[0] == first.video_rows[30 * 96] &&
                guide.audio_rows[0] == first.audio_rows[170 * 32] &&
                guide.audio_rows[37 * 32] == first.audio_rows[(207 + 170) * 32],
            "extract the tail of each audio channel separately");
    const auto combined = append_reference_window(first, second, windows[1]);
    require(combined.shape.frames == 67 && combined.shape.audio_latents == 377 &&
                combined.forwards == 98, "two full windows retain the cumulative latent lengths");
    require(std::equal(first.video_rows.begin(), first.video_rows.end(), combined.video_rows.begin()),
            "the parent video latent prefix remains bit-identical");
    require(combined.video_rows[37 * 96] == second.video_rows[7 * 96],
            "discard the new window's overlapping target video, not the accepted parent");
    for (int32_t channel = 0; channel < 2; ++channel) {
        const auto begin = first.audio_rows.begin() + channel * 207 * 32;
        const auto joined = combined.audio_rows.begin() + channel * 377 * 32;
        require(std::equal(begin, begin + 207 * 32, joined),
                "each channel retains its complete old temporal prefix");
        require(*(joined + 207 * 32) == second.audio_rows[(channel * 207 + 37) * 32],
                "append each generated channel after its own hidden overlap");
    }
    const auto third = fixture(37, 207, 50000.0F);
    const auto final = append_reference_window(combined, third, windows[2]);
    require(windows[2].temporal_offset == 340.0 && final.shape.frames == 97 &&
                final.shape.audio_latents == 547 && final.forwards == 147,
            "cumulative continuation retains the global time origin");
    require(first.video_rows == original.video_rows && first.audio_rows == original.audio_rows,
            "guide extraction and append never mutate a parent");
    auto bad = second;
    bad.audio_rows[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { append_reference_window(first, bad, windows[1]); });
    bad = second;
    bad.video_rows.pop_back();
    rejects([&] { append_reference_window(first, bad, windows[1]); });
    auto clock = windows[1];
    clock.guide_audio_latents = 36;
    rejects([&] { reference_latent_guide(first, clock); });
    rejects([&] { reference_latent_guide(first, windows[0]); });
    rejects([&] { reference_latent_guide(first, windows[2]); });
}

void test_rounding_and_rejections() {
    const auto windows = plan_reference_continuation(209, 107, 22, 3);
    require(windows.size() == 3 && windows[1].audio_latents == 178 &&
                windows[1].guide_audio_latents == 36 && windows[2].guide_audio_latents == 37 &&
                windows[2].audio_latents == 65,
            "audio overlap lengths follow cumulative boundary rounding");
    require(windows[1].temporal_offset == 85 * (40.0 / 24.0),
            "global rotary positions retain fractional audio-clock units");
    const auto first = fixture(32, 178, 1.0F);
    const auto second = fixture(32, 178, 2.0F);
    const auto third = fixture(12, 65, 3.0F);
    const auto joined = append_reference_window(first, second, windows[1]);
    const auto final = append_reference_window(joined, third, windows[2]);
    require(final.shape.frames == 62 && final.shape.audio_latents == 348,
            "varying rounded overlaps cannot accumulate AV drift");
    rejects([] { plan_reference_continuation(226, 124, 22, 1); });
    rejects([] { plan_reference_continuation(226, 124, 22, 0); });
    rejects([] { plan_reference_continuation(225, 124, 22, 2); });
    rejects([] { plan_reference_continuation(226, 124, 23, 2); });
    rejects([] { plan_reference_continuation(226, 124, 124, 2); });
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "--dump") {
            nlohmann::json result = nlohmann::json::array();
            for (const auto& w : plan_reference_continuation(std::stoi(argv[2]),
                    std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5])))
                result.push_back({{"start", w.start_frame}, {"end", w.end_frame},
                    {"overlap", w.overlap_frames}, {"video_latent_frames", w.video_latent_frames},
                    {"audio_latents", w.audio_latents}, {"guide_video_latent_frames", w.guide_video_latent_frames},
                    {"guide_audio_latents", w.guide_audio_latents}, {"temporal_offset", w.temporal_offset}});
            std::cout << result.dump() << '\n';
            return 0;
        }
        if (argc != 1)
            throw std::invalid_argument("expected --dump total_frames window_frames overlap_frames maximum_windows");
        test_assembly();
        test_rounding_and_rejections();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
