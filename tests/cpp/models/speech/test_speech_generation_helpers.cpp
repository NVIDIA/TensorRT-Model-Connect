// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-12
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Speech generation helpers: delay cache, waveform trim/normalize, postprocess
// safety Preconditions:  Delay cache with known codebook delays, waveform samples Postconditions:
// Delays generalize to num_codebooks, cache reads correct, waveform trimmed/normalized
// =============================================================================

#include "runtime/models/speech/speech_delay_cache.h"
#include "runtime/models/speech/speech_waveform_postprocess.h"

#include <cmath>
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

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

void test_default_speech_delays_generalize_to_num_codebooks() {
    const auto delays = trtmc::make_default_speech_delays(6);
    check(delays.size() == 7, "default delays size");
    check(delays[0] == 0, "default delays text stream");
    check(delays[1] == 0, "default delays first moshi stream");
    check(delays[4] == 0, "default delays first user stream");
    check(delays[2] == 1 && delays[3] == 1 && delays[5] == 1 && delays[6] == 1,
          "default delays remaining streams");
}

void test_delay_cache_reads_and_collects_outputs() {
    auto state = trtmc::make_delay_cache_state({0, 0, 0, 1, 1}, 4);
    check(state.total_k == 5, "delay cache total streams");
    check(state.max_delay == 1, "delay cache max delay");

    const std::vector<int32_t> codec_tokens = {
        101,
        102,
        201,
        202,
    };

    trtmc::seed_delay_offset_zero(state, 9000, 8000);
    trtmc::write_user_tokens_to_delay_cache(state, codec_tokens, 0, 2, 2, 2, 8000);
    trtmc::fill_initial_delay_tokens(state, 0, 9000, 8000);

    int32_t text_input = -1;
    std::vector<int32_t> moshi_input(2, -1);
    std::vector<int32_t> user_input(2, -1);
    trtmc::read_model_inputs_from_delay_cache(state, 0, 2, text_input, moshi_input, user_input);
    check(text_input == 9000, "delay cache reads text input");
    check(moshi_input == std::vector<int32_t>({8000, 8000}), "delay cache reads moshi input");
    check(user_input == std::vector<int32_t>({8000, 8000}), "delay cache reads delayed user input");

    std::vector<int32_t> target_audio_tokens(4, -1);
    std::vector<uint8_t> target_audio_provided(4, 0);
    trtmc::build_target_audio_arrays(state, 1, 4, 8000, target_audio_tokens, target_audio_provided);
    check(target_audio_tokens == std::vector<int32_t>({-2, -2, 101, 102}),
          "target audio exposes delayed user tokens");
    check(target_audio_provided == std::vector<uint8_t>({0, 0, 1, 1}),
          "target audio marks provided delayed streams");

    const std::vector<int32_t> frame_codes = {501, 502, 503, 504};
    trtmc::write_generated_tokens_to_delay_cache(state, 1, 9100, false, frame_codes, 4);
    std::vector<int32_t> output_codes;
    const bool collected =
        trtmc::collect_output_codes_from_delay_cache(state, 2, state.max_delay, 2, output_codes);
    check(collected, "delay cache collects output after max delay");
    check(output_codes == std::vector<int32_t>({501, 502}),
          "delay cache collects mimi codebooks only");
}

void test_waveform_trim_and_peak_normalize() {
    std::vector<float> waveform(20, 0.0F);
    const auto trim_result = trtmc::trim_speech_waveform_to_generated_frames(10, 2.0F, 3, waveform);
    check(trim_result.trimmed, "waveform trim applied");
    check(trim_result.expected_samples == 15, "waveform trim expected samples");
    check(waveform.size() == 15, "waveform trim resized output");

    waveform = {2.0F, -1.0F, 0.5F};
    const auto normalize_result = trtmc::peak_normalize_speech_waveform(waveform);
    check(normalize_result.normalized, "waveform normalize applied");
    check_close(normalize_result.peak, 2.0F, 1e-6F, "waveform normalize peak");
    check_close(normalize_result.scale, 0.475F, 1e-6F, "waveform normalize scale");
    check_close(waveform[0], 0.95F, 1e-6F, "waveform normalize sample 0");
    check_close(waveform[1], -0.475F, 1e-6F, "waveform normalize sample 1");
}

void test_waveform_postprocess_skips_invalid_or_safe_inputs() {
    std::vector<float> waveform = {0.1F, 0.2F, 0.3F};
    std::vector<float> empty_waveform;
    const auto no_trim_empty =
        trtmc::trim_speech_waveform_to_generated_frames(10, 2.0F, 2, empty_waveform);
    check(!no_trim_empty.trimmed && no_trim_empty.expected_samples == 0,
          "waveform trim skips empty waveform");

    const auto no_trim_bad_frames =
        trtmc::trim_speech_waveform_to_generated_frames(10, 0.0F, 2, waveform);
    check(!no_trim_bad_frames.trimmed, "waveform trim skips invalid frame rate");
    check(waveform.size() == 3, "waveform trim preserves waveform when skipped");

    const auto no_trim_expected_ge_size =
        trtmc::trim_speech_waveform_to_generated_frames(10, 2.0F, 1, waveform);
    check(!no_trim_expected_ge_size.trimmed,
          "waveform trim skips when expected samples do not shrink output");

    const auto no_norm_empty = trtmc::peak_normalize_speech_waveform(waveform, 0.8F);
    check(!no_norm_empty.normalized, "waveform normalize skips already safe waveform");
    check_close(no_norm_empty.peak, 0.3F, 1e-6F, "waveform normalize reports measured safe peak");

    const auto no_norm_zero = trtmc::peak_normalize_speech_waveform(empty_waveform);
    check(!no_norm_zero.normalized && no_norm_zero.peak == 0.0F,
          "waveform normalize skips empty waveform");
}

} // namespace

int main() {
    test_default_speech_delays_generalize_to_num_codebooks();
    test_delay_cache_reads_and_collects_outputs();
    test_waveform_trim_and_peak_normalize();
    test_waveform_postprocess_skips_invalid_or_safe_inputs();

    if (g_failures != 0) {
        std::cerr << g_failures << " speech generation helper test(s) failed\n";
        return 1;
    }
    return 0;
}
