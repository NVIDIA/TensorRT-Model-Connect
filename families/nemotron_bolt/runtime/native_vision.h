/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "native_language.h"
namespace trtmc::nemotron_bolt {
class NativeVision {
  public:
    NativeVision(IBackend& backend, const NativeLanguage::PlanReader& read,
                 std::vector<float> positions, int position_side, int max_patches);
    // One image or a true two-frame tubelet; RGB rows are resized, in [0,1] before interpolation.
    std::vector<float> encode(const std::vector<std::vector<float>>& frames, int height, int width);

  private:
    std::vector<float> positions_;
    int side_, maximum_;
    std::unique_ptr<ITrtModule> image_, tubelet_, visual_;
};
} // namespace trtmc::nemotron_bolt
