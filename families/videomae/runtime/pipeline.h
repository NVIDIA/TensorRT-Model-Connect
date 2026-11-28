/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/videomae/runtime/clip.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <vector>

namespace trtmc::videomae {

// Reconstructs the masked tubes of one clip with VideoMAEForPreTraining. The
// result keeps every conditioning tube's source pixels and replaces each
// masked tube with the model's prediction.
class MaskedVideoReconstruction final : public internal::IModel,
                                        public internal::IMaskedVideoTextToVideo {
  public:
    MaskedVideoReconstruction(std::unique_ptr<ITrtModule> engine, ClipGeometry geometry,
                              PixelNormalization normalization);

    const char* task() const noexcept override { return IMaskedVideoTextToVideo::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IMaskedVideoTextToVideo>(*this)};
    }
    internal::VideoResult run(const internal::MaskedVideoTextToVideoRequest& request,
                              internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    ClipGeometry geometry_;
    PixelNormalization normalization_;
};

} // namespace trtmc::videomae
