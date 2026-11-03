/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/reference_conditioner.h"
#include "families/minimax_h3/runtime/reference_layout.h"
#include "families/minimax_h3/runtime/reference_visual_encoder.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc::minimax_h3 {

enum class ReferenceSamplingRecipe { OriginalH3, OmniRefPdd8 };

struct ReferenceDenoiserConfig {
    ReferenceTargetLatentShape target{37, 84, 48, 207};
    int32_t layers{50};
    int32_t hidden_size{5376};
    int32_t text_width{5120};
    int32_t max_sequence_rows{0}; // Required exact loaded Ref2VA profile cap.
    ReferenceSamplingRecipe recipe{ReferenceSamplingRecipe::OriginalH3};
    int32_t max_vsa_tiles{0}; // Explicit only for the OmniRef VSA engine ABI.
};

enum class ReferenceDenoiserResidency { Staged, Resident };

struct ReferenceDenoiserResult {
    ReferenceTargetLatentShape shape;
    std::vector<float> video_rows; // Generated-only [target rows,96], FP32.
    std::vector<float> audio_rows; // Generated-only stereo channel-major [rows,32].
    int32_t forwards{0};
};

// Family-owned, already normalized latent tail. Video rows use the same
// patchified layout as generated output; stereo audio rows are channel-major.
// This is not an encoded-media reference or a public cross-request state ABI.
struct ReferenceLatentGuide {
    ReferenceTargetLatentShape shape;
    std::vector<float> video_rows;
    std::vector<float> audio_rows;
};

// Image Ref2VA denoising with an explicit checkpoint/sampling recipe, using
// encoded Qwen/VisualVAE outputs and family plans through ITrtModule. The
// original-H3 default retains 49 forwards; the OmniRef student uses its eight
// trained heads and native 128-token VSA geometry. No implicit recipe fallback.
// This private component does not register a public Task or decode a video.
class ImageReferenceDenoiser {
  public:
    ImageReferenceDenoiser(ReferenceModuleLoader loader, ReferenceDenoiserConfig config,
                           cudaStream_t stream, ReferenceDenoiserResidency residency);
    ~ImageReferenceDenoiser();

    ReferenceDenoiserResult run(const ReferenceImageConditioning& text,
                                const std::vector<EncodedVisualReference>& images, uint64_t seed);

    ReferenceDenoiserResult
    run_continuation(const ReferenceImageConditioning& text,
                     const std::vector<EncodedVisualReference>& images,
                     const ReferenceLatentGuide& guide, uint64_t seed, double temporal_offset,
                     std::optional<int32_t> media_time_origin = std::nullopt);

    // One bounded request with the same canonical images in both windows.
    // Optional child conditioning pins the media clock after the longest text
    // prefix, matching the fixed upstream per-window conditioning behavior.
    // A 226-frame request uses two complete 124-frame windows and
    // returns cumulative normalized latents, not cross-request accepted state.
    ReferenceDenoiserResult
    run_windows(const ReferenceImageConditioning& text,
                const std::vector<EncodedVisualReference>& images, uint64_t seed,
                int32_t output_frames,
                const ReferenceImageConditioning* continuation_text = nullptr);

  private:
    struct Modulations;
    void precompute(const ImageReferenceLayout& layout, const std::vector<float>& video_times,
                    const std::vector<float>& audio_times);
    ImageReferenceLayout validate_conditioning(const ReferenceImageConditioning& text,
                                               const std::vector<EncodedVisualReference>& images,
                                               std::optional<ReferenceContinuationGuideShape> guide,
                                               double temporal_offset,
                                               std::optional<int32_t> media_time_origin) const;
    // The public family entry points hold mutex_ for the whole request.
    ReferenceDenoiserResult run_unlocked(const ReferenceImageConditioning& text,
                                         const std::vector<EncodedVisualReference>& images,
                                         uint64_t seed, const ReferenceLatentGuide* guide,
                                         double temporal_offset,
                                         std::optional<int32_t> media_time_origin);

    ReferenceModuleLoader loader_;
    ReferenceDenoiserConfig config_;
    cudaStream_t stream_;
    ReferenceDenoiserResidency residency_;
    std::unique_ptr<Modulations> modulations_;
    std::unique_ptr<ITrtModule> denoiser_;
    std::mutex mutex_;
};

} // namespace trtmc::minimax_h3
