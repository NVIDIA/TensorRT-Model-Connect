// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-13
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Speech Mimi decode plan: layout byte computation, input transposition, waveform
// stats Preconditions:  Mimi config with valid frame and codebook parameters Postconditions: Layout
// bytes and output elems correct, transposition frame-major, RMS/peak reported
// =============================================================================

#include "runtime/models/personaplex/speech_mimi_decode_plan.h"

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

void test_layout_computes_bytes_and_output_elems() {
    const auto layout = trtmc::build_mimi_decode_layout(8, 5, {1, 1, 40});
    check(layout.dec_codebooks == 8, "layout codebooks");
    check(layout.dec_frames == 5, "layout frames");
    check(layout.total_output_elems == 40, "layout output elems");
    check(layout.input_elems == 40, "layout input elems");
    check(layout.input_bytes == 40 * sizeof(float), "layout input bytes");
    check(layout.output_bytes == 40 * sizeof(float), "layout output bytes");
}

void test_decoder_input_transposes_frame_major_tokens() {
    const std::vector<int32_t> codec_tokens = {
        1, 2, 3, 4, 5, 6,
    };
    const auto input = trtmc::build_mimi_decoder_input(codec_tokens, 3, 2, 4, 3);
    const std::vector<float> expected = {
        1.0F, 3.0F, 5.0F, 0.0F, 2.0F, 4.0F, 6.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    check(input == expected, "decoder input transposes and pads");
}

void test_waveform_stats_report_rms_and_peak() {
    float rms = 0.0F;
    float peak = 0.0F;
    trtmc::waveform_stats({1.0F, -2.0F, 2.0F}, 3, rms, peak);
    check_close(peak, 2.0F, 1e-6F, "waveform peak");
    check_close(rms, std::sqrt(3.0F), 1e-6F, "waveform rms");
}

} // namespace

int main() {
    test_layout_computes_bytes_and_output_elems();
    test_decoder_input_transposes_frame_major_tokens();
    test_waveform_stats_report_rms_and_peak();

    if (g_failures != 0) {
        std::cerr << g_failures << " speech mimi decode plan test(s) failed\n";
        return 1;
    }
    return 0;
}
