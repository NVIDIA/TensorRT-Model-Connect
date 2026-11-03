/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/reference_image.h"
#include "trtmc/runtime/trt_module.h"

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc::minimax_h3 {

struct ReferenceLatentNormalization {
    std::array<float, 24> mean{};
    std::array<float, 24> std{};
};

enum class VisualReferenceResidency { Staged, Resident };

using VisualReferenceModuleLoader =
    std::function<std::unique_ptr<ITrtModule>(const std::string&, cudaStream_t)>;

struct EncodedVisualReference {
    int32_t latent_height;
    int32_t latent_width;
    std::vector<float> latents; // Owned CPU FP32 [1,24,1,H/16,W/16].
};

// Native original-H3 image-reference path: sequential 256-pixel tile plans,
// exact spatial moment stitching, fresh seed-42 CPU noise, a TensorRT posterior
// plan that rounds to FP16, then checkpoint channel normalization on CPU.
// One image stays one temporal frame. Staged mode releases the encoder before
// loading posterior sampling; resident mode reuses both actual contexts.
// This is not public Ref2VA Task registration or trained-encoder qualification.
class ReferenceVisualEncoder {
  public:
    ReferenceVisualEncoder(VisualReferenceModuleLoader loader,
                           ReferenceLatentNormalization normalization, cudaStream_t stream,
                           VisualReferenceResidency residency);

    EncodedVisualReference encode_image(const PreparedReferenceImage& image);

  private:
    VisualReferenceModuleLoader loader_;
    ReferenceLatentNormalization normalization_;
    cudaStream_t stream_;
    VisualReferenceResidency residency_;
    std::unique_ptr<ITrtModule> encoder_;
    std::unique_ptr<ITrtModule> posterior_;
    std::mutex mutex_;
};

} // namespace trtmc::minimax_h3
