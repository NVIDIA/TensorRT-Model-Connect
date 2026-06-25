#include "runtime/models/whisper/whisper_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
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

void test_whisper_shape_and_energy() {
    const int32_t sample_rate = 16000;
    const int32_t n_fft = 400;
    const int32_t hop_length = 160;
    const int32_t chunk_length_s = 30;
    const int32_t n_freq_bins = n_fft / 2 + 1;
    const int32_t n_mel_bins = 80;
    auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);

    std::vector<float> sine(sample_rate);
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t i = 0; i < sample_rate; ++i) {
        sine[static_cast<std::size_t>(i)] =
            static_cast<float>(std::sin(pi2 * 440.0 * static_cast<double>(i) / sample_rate));
    }

    const auto mel = trtmc::whisper::extract_mel_spectrogram(
        sine.data(), static_cast<int32_t>(sine.size()), fb.data(), n_freq_bins, n_mel_bins, n_fft,
        hop_length, chunk_length_s, sample_rate);

    check(mel.n_mels == n_mel_bins, "whisper mel keeps mel bin count");
    check(mel.n_frames == 3000, "whisper mel frame count matches 30s HF window");
    check(static_cast<int32_t>(mel.data.size()) == n_mel_bins * mel.n_frames,
          "whisper mel data size matches shape");

    float energy_target = 0.0F;
    float energy_quiet = 0.0F;
    const int32_t check_frames = std::min(mel.n_frames, 50);
    for (int32_t t = 0; t < check_frames; ++t) {
        energy_target += mel.data[static_cast<std::size_t>(11) * mel.n_frames + t];
        energy_quiet += mel.data[static_cast<std::size_t>(50) * mel.n_frames + t];
    }
    check(energy_target > energy_quiet, "whisper mel keeps 440Hz energy concentrated");
}

} // namespace

int main() {
    test_whisper_shape_and_energy();
    return failures;
}
