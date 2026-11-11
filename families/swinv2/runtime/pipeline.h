/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/swinv2/runtime/image_preprocess.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::swinv2 {

struct RuntimeConfig {
    PreprocessConfig preprocess;
    std::int32_t num_classes{0};
    std::vector<std::string> labels;
    std::string vocabulary_id;
};

class Pipeline final : public internal::IModel, public internal::IImageToClassScores {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, RuntimeConfig config, std::string primary_task);

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::LabelScoresResult run(const internal::ImageToClassScoresRequest& request,
                                    internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    RuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc::swinv2
