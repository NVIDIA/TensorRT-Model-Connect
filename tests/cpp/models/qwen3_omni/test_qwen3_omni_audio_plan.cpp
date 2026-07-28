/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-09
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Omni audio plan: encode padding/trimming and official codec decode
// Preconditions:  OmniConfig with valid audio parameters
// Postconditions: Frames padded/trimmed correctly and codec tensor/sample contracts hold
// =============================================================================

#include "runtime/models/qwen3_omni/omni_audio_plan.h"
#include "runtime/models/qwen3_omni/omni_thinker_plan.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

trtmc::OmniConfig make_config() {
    trtmc::OmniConfig cfg;
    cfg.audio_embed_dim = 1280;
    cfg.audio_num_frames = 8;
    cfg.talker_n_codebooks = 4;
    return cfg;
}

void test_audio_encode_plan_pads_and_trims_frames() {
    const trtmc::OmniConfig cfg = make_config();
    const auto plan = trtmc::make_omni_audio_encode_plan(cfg, 3, 10);

    check(plan.actual_frames == 8, "omni audio plan clamps to max frames");
    check(plan.output_frames == 4, "omni audio plan derives output frames");
    check(plan.input_size == 24, "omni audio plan computes padded input size");
    check(plan.copy_size == 24, "omni audio plan computes copy size");
    check(plan.output_elements == 5120, "omni audio plan computes output element count");
}

void test_audio_encode_input_builder_zero_pads_tail() {
    const trtmc::OmniConfig cfg = make_config();
    const auto plan = trtmc::make_omni_audio_encode_plan(cfg, 2, 3);
    const std::vector<float> mel = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};

    const auto padded = trtmc::build_omni_audio_encoder_input(mel.data(), plan);
    check(padded.size() == plan.input_size, "omni input builder uses padded size");
    check(padded[0] == 1.0F && padded[5] == 6.0F, "omni input builder copies active mel frames");
    check(padded[6] == 0.0F && padded.back() == 0.0F, "omni input builder zero-pads unused tail");
}

void test_codec_plan_derives_official_frame_shape() {
    const trtmc::OmniConfig cfg = make_config();
    const auto codec_plan = trtmc::make_omni_codec_plan(cfg, 8);
    check(codec_plan.should_run_codec, "omni codec plan enables codec with token payload");
    check(codec_plan.n_codebooks == 4, "omni codec plan forwards codebook count");
    check(codec_plan.n_frames == 2, "omni codec plan derives frame count");
}

void test_code2wav_input_builder_transposes_frame_major_tokens() {
    const std::vector<int32_t> codec_tokens = {
        10, 20, 30, 11, 21, 31,
    };
    const auto input_codes = trtmc::build_omni_code2wav_input_codes(codec_tokens, 3, 4, 2);

    check(input_codes.size() == 12, "omni code2wav input builder allocates padded tensor");
    check(input_codes[0] == 10 && input_codes[1] == 11,
          "omni code2wav input builder copies first codebook frames");
    check(input_codes[4] == 20 && input_codes[5] == 21,
          "omni code2wav input builder transposes second codebook frames");
    check(input_codes[8] == 30 && input_codes[9] == 31,
          "omni code2wav input builder transposes third codebook frames");
    check(input_codes[10] == 0 && input_codes[11] == 0,
          "omni code2wav input builder zero-pads trailing frames");
}

void test_code2wav_output_uses_official_stride_and_causal_delay() {
    trtmc::OmniConfig cfg;
    cfg.code2wav_upsample_factor = 1920;
    cfg.code2wav_output_delay = 555;

    check(trtmc::code2wav_output_samples(cfg, 16, 60885) == 30165,
          "omni code2wav output trims the official causal decoder delay");
    check(trtmc::code2wav_output_samples(cfg, 32, 60885) == 60885,
          "omni code2wav output preserves the full static engine result");
    check(trtmc::code2wav_output_samples(cfg, 0, 60885) == 0,
          "omni code2wav output rejects zero frames");
}

void test_thinker_stops_only_on_configured_eos() {
    check(!trtmc::omni_thinker_should_stop(0, 151645),
          "omni thinker preserves token zero as ordinary text");
    check(trtmc::omni_thinker_should_stop(151645, 151645),
          "omni thinker stops on configured im_end token");
}

} // namespace

int main() {
    test_audio_encode_plan_pads_and_trims_frames();
    test_audio_encode_input_builder_zero_pads_tail();
    test_codec_plan_derives_official_frame_shape();
    test_code2wav_input_builder_transposes_frame_major_tokens();
    test_code2wav_output_uses_official_stride_and_causal_delay();
    test_thinker_stops_only_on_configured_eos();

    if (g_failures != 0) {
        std::cerr << g_failures << " omni audio plan test(s) failed\n";
        return 1;
    }
    return 0;
}
