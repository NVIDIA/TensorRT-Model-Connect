/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/personaplex/runtime/resampler.h"

#include <soxr.h>
#include <stdexcept>
#include <string>

namespace trtmc {

// The Native reference loads audio with librosa.resample, whose default is
// soxr's "HQ" recipe. Mimi tokenizes the resampled waveform and the language
// model is conditioned on those codes, so the runtime resamples with libsoxr's
// HQ recipe itself: an approximation of its filter changes some codes.
std::vector<float> resample_audio(const float* samples, std::int32_t num_samples,
                                  std::int32_t source_rate, std::int32_t target_rate) {
    if (num_samples < 0 || source_rate <= 0 || target_rate <= 0)
        throw std::invalid_argument("resampler requires a sample count and positive rates");
    if (num_samples == 0)
        return {};
    if (samples == nullptr)
        throw std::invalid_argument("resampler requires samples");
    if (source_rate == target_rate)
        return {samples, samples + num_samples};

    // librosa trims or pads the result to ceil(num_samples * target / source).
    const auto output_count = static_cast<std::size_t>(
        (static_cast<std::int64_t>(num_samples) * target_rate + source_rate - 1) / source_rate);
    std::vector<float> output(output_count, 0.0F);
    const soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
    const soxr_quality_spec_t quality = soxr_quality_spec(SOXR_HQ, 0);
    std::size_t consumed = 0;
    std::size_t produced = 0;
    const soxr_error_t error = soxr_oneshot(
        source_rate, target_rate, 1, samples, static_cast<std::size_t>(num_samples), &consumed,
        output.data(), output_count, &produced, &io_spec, &quality, nullptr);
    if (error != nullptr)
        throw std::runtime_error(std::string("resample_audio: soxr: ") + error);
    return output;
}

} // namespace trtmc
