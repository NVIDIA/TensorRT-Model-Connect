/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"

#include <map>
#include <string>

namespace trtmc::qwen {

/// Qwen owns the section schema and per-engine names; the backend sees only views.
/// Mapping storage remains alive through both synchronous module creations.
class QwenRefitSource {
  public:
    explicit QwenRefitSource(const BundleReader& reader);
    const RefitWeightMap* weights_for(const std::string& engine) const;

  private:
    MappedBundleSection storage_;
    std::map<std::string, RefitWeightMap> engines_;
};

} // namespace trtmc::qwen
