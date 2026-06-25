#include "runtime/models/canary/canary_mel_spectrogram.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

std::vector<float> make_identity_filterbank(int32_t n_freq_bins, int32_t n_mel_bins) {
    std::vector<float> fb(static_cast<std::size_t>(n_freq_bins) * n_mel_bins, 0.0F);
    const int32_t mapped = std::min(n_freq_bins, n_mel_bins);
    for (int32_t i = 0; i < mapped; ++i) {
        fb[static_cast<std::size_t>(i) * n_mel_bins + i] = 1.0F;
    }
    return fb;
}

void test_canary_shape_and_empty_audio() {
    const int32_t sample_rate = 16000;
    const int32_t n_fft = 400;
    const int32_t hop_length = 160;
    const int32_t chunk_length_s = 30;
    const int32_t n_freq_bins = n_fft / 2 + 1;
    const int32_t n_mel_bins = 80;
    auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);

    const auto mel =
        trtmc::canary::extract_mel_spectrogram(nullptr, 0, fb.data(), n_freq_bins, n_mel_bins,
                                               n_fft, hop_length, chunk_length_s, sample_rate);

    check(mel.n_mels == n_mel_bins, "canary mel keeps mel bin count");
    check(mel.n_frames == 3000, "canary mel frame count matches 30s HF window");
    check(static_cast<int32_t>(mel.data.size()) == n_mel_bins * mel.n_frames,
          "canary mel data size matches shape");
}

} // namespace

int main() {
    test_canary_shape_and_empty_audio();
    return failures;
}
