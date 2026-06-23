#include "audio_helpers.h"

namespace trtmc {
namespace rnnt {

MelResult extract_rnnt_mel_spectrogram(const float* samples, int32_t n_samples,
                                       const float* mel_filters, int32_t n_freq_bins,
                                       int32_t n_mel_bins, int32_t n_fft, int32_t win_length,
                                       int32_t hop_length, int32_t chunk_length_s,
                                       int32_t sample_rate, float preemphasis) {
    MelSpectrogramOptions options;
    options.n_fft = n_fft;
    options.win_length = win_length;
    options.hop_length = hop_length;
    options.chunk_length_s = chunk_length_s;
    options.sample_rate = sample_rate;
    options.symmetric_window = true;
    options.center_window_in_fft = true;
    options.preemphasis = preemphasis;
    options.log_scale = MelLogScale::kNaturalLog;
    return extract_configured_mel_spectrogram(samples, n_samples, mel_filters, n_freq_bins,
                                              n_mel_bins, options);
}

} // namespace rnnt
} // namespace trtmc
