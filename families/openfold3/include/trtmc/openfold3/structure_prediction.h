/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/task.h"

#include <string>

namespace trtmc::openfold3 {

struct StructurePredictionResult {
    std::string structure;
    std::string metadata_json;
};

class IStructurePrediction : public virtual ITask {
  public:
    static constexpr const char* kTask = "structure_prediction";
    const char* task() const noexcept override { return kTask; }
    virtual StructurePredictionResult predict_structure(const std::string& input) = 0;
};

} // namespace trtmc::openfold3
