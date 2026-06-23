#pragma once

#include "runtime/domains/audio/mel_spectrogram.h"

#include <cstdint>

namespace trtmc {
namespace rnnt {

MelResult extract_rnnt_mel_spectrogram(const float* samples, int32_t n_samples,
                                       const float* mel_filters, int32_t n_freq_bins,
                                       int32_t n_mel_bins, int32_t n_fft, int32_t win_length,
                                       int32_t hop_length, int32_t chunk_length_s,
                                       int32_t sample_rate, float preemphasis);

} // namespace rnnt
} // namespace trtmc
