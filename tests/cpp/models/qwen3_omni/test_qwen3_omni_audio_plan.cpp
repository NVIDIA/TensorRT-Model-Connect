// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-09
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Omni audio plan: encode padding/trimming, generation gating, codec decode
// Preconditions:  OmniConfig with valid audio parameters
// Postconditions: Frames padded/trimmed correctly, talker/codec gates match config, codebook argmax
// correct
// =============================================================================

#include "runtime/models/qwen3_omni/omni_audio_plan.h"

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

void test_generation_plans_gate_talker_and_codec() {
    const trtmc::OmniConfig cfg = make_config();
    const auto talker_plan = trtmc::make_omni_talker_plan(5, 0, true);
    check(!talker_plan.should_run_talker, "omni talker plan requires hidden states");

    const auto active_talker_plan = trtmc::make_omni_talker_plan(5, 10, true);
    check(active_talker_plan.should_run_talker, "omni talker plan enables active talker stage");
    check(active_talker_plan.num_tokens == 5, "omni talker plan forwards token count");

    const auto codec_plan = trtmc::make_omni_codec_plan(cfg, 8);
    check(codec_plan.should_run_codec, "omni codec plan enables codec with token payload");
    check(codec_plan.n_codebooks == 4, "omni codec plan forwards codebook count");
    check(codec_plan.n_frames == 2, "omni codec plan derives frame count");
}

void test_talker_decode_helpers_extract_codebook_argmax() {
    const auto decode_plan = trtmc::make_omni_talker_decode_plan(2, 4, 3);
    std::vector<int32_t> all_codes;
    const std::vector<float> logits = {
        0.1F, 0.4F, 0.3F, 0.2F, 0.0F, 1.0F, 0.5F, 0.6F,
    };

    trtmc::append_omni_talker_codes_from_logits(logits, decode_plan, all_codes);

    check(decode_plan.num_tokens == 3, "omni talker decode plan forwards token count");
    check(all_codes.size() == 2, "omni talker decode helper appends one code per codebook");
    check(all_codes[0] == 1 && all_codes[1] == 1,
          "omni talker decode helper selects argmax in each codebook slice");
    check(trtmc::select_omni_codebook_argmax(logits, 20, 4) == 0,
          "omni talker argmax helper returns zero for out-of-range slices");
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

} // namespace

int main() {
    test_audio_encode_plan_pads_and_trims_frames();
    test_audio_encode_input_builder_zero_pads_tail();
    test_generation_plans_gate_talker_and_codec();
    test_talker_decode_helpers_extract_codebook_argmax();
    test_code2wav_input_builder_transposes_frame_major_tokens();

    if (g_failures != 0) {
        std::cerr << g_failures << " omni audio plan test(s) failed\n";
        return 1;
    }
    return 0;
}
