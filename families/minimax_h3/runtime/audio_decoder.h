/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/reference_conditioner.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace trtmc::minimax_h3 {

enum class AudioDecoderResidency { Staged, Resident };

struct DecodedH3Audio {
    std::vector<float> interleaved_samples;
    int32_t sample_rate{32000};
    int32_t channels{2};
};

// Decodes generated channel-major [stereo * frames, 32] rows. Reference
// prefixes must already have been removed by ImageReferenceDenoiser.
class AudioDecoder final {
  public:
    AudioDecoder(ReferenceModuleLoader loader, int32_t frames = 207, cudaStream_t stream = nullptr,
                 AudioDecoderResidency residency = AudioDecoderResidency::Staged);

    DecodedH3Audio run(const std::vector<float>& generated_rows);

  private:
    ReferenceModuleLoader loader_;
    int32_t frames_;
    cudaStream_t stream_;
    AudioDecoderResidency residency_;
    std::unique_ptr<ITrtModule> module_;
    std::mutex mutex_;
};

} // namespace trtmc::minimax_h3
