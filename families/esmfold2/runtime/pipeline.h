/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/esmfold2/runtime/bundle_config.h"
#include "families/esmfold2/runtime/fold.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc::esmfold2 {

struct EngineSet {
    std::unique_ptr<ITrtModule> esmc;
    std::unique_ptr<ITrtModule> embed;
    std::unique_ptr<ITrtModule> msa; // ESMFold2 only
    std::unique_ptr<ITrtModule> loop;
    std::unique_ptr<ITrtModule> decode;
    std::unique_ptr<ITrtModule> conditioning;
    std::unique_ptr<ITrtModule> denoise;
    std::unique_ptr<ITrtModule> confidence;
};

class EsmFold2Pipeline final : public IStructurePrediction {
  public:
    EsmFold2Pipeline(EngineSet engines, BundleConfig config);

    StructurePredictionResult predict_structure(const StructurePredictionRequest& request) override;

  private:
    EngineSet engines_;
    BundleConfig config_;
    cudaStream_t stream_{nullptr};
};

} // namespace trtmc::esmfold2
