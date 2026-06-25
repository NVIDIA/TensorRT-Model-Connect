// Unit tests for RNNT-owned audio feature extraction helpers.

#include "runtime/models/nemotron_speech_streaming/audio_helpers.h"

#include <cmath>
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

void test_rnnt_mel_matches_configured_owned_options() {
    const std::vector<float> samples{0.0F, 0.1F, -0.2F, 0.4F, -0.1F, 0.0F, 0.2F, -0.3F};
    const int32_t n_fft = 4;
    const int32_t n_freq_bins = 3;
    const int32_t n_mel_bins = 2;
    const std::vector<float> mel_filters{
        0.5F, 0.1F, 0.2F, 0.7F, 0.3F, 0.4F,
    };

    const auto rnnt = trtmc::rnnt::extract_rnnt_mel_spectrogram(
        samples.data(), static_cast<int32_t>(samples.size()), mel_filters.data(), n_freq_bins,
        n_mel_bins, n_fft, /*win_length=*/3, /*hop_length=*/2, /*chunk_length_s=*/1,
        /*sample_rate=*/8, /*preemphasis=*/0.97F);

    trtmc::rnnt::MelSpectrogramOptions options;
    options.n_fft = n_fft;
    options.win_length = 3;
    options.hop_length = 2;
    options.chunk_length_s = 1;
    options.sample_rate = 8;
    options.symmetric_window = true;
    options.center_window_in_fft = true;
    options.preemphasis = 0.97F;
    options.log_scale = trtmc::rnnt::MelLogScale::kNaturalLog;
    const auto configured = trtmc::rnnt::extract_configured_mel_spectrogram(
        samples.data(), static_cast<int32_t>(samples.size()), mel_filters.data(), n_freq_bins,
        n_mel_bins, options);

    check(rnnt.n_mels == configured.n_mels, "rnnt mel keeps configured mel count");
    check(rnnt.n_frames == configured.n_frames, "rnnt mel keeps configured frame count");
    check(rnnt.data.size() == configured.data.size(), "rnnt mel keeps configured data size");
    for (std::size_t i = 0; i < rnnt.data.size(); ++i) {
        check_close(rnnt.data[i], configured.data[i], 1e-6F, "rnnt mel value matches configured");
    }
}

} // namespace

int main() {
    test_rnnt_mel_matches_configured_owned_options();

    if (g_failures != 0) {
        std::cerr << g_failures << " rnnt audio helper test(s) failed\n";
        return 1;
    }
    return 0;
}
