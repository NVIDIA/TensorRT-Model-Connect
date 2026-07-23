/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bundle/bundle_format.h"
#include "runtime/models/openpi/config.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string_view>
#include <vector>

namespace trtmc::openpi {

struct VerifiedOpenPIBundle {
    OpenPIConfig config;
    OpenPINormalization normalization;
    std::string_view config_json;
    std::string_view tokenizer_bytes;
    std::string_view normalization_bytes;
    const std::vector<char>* prefill_plan{nullptr};
    const std::vector<char>* action_plan{nullptr};
};

// Validate the complete OpenPI bundle integrity contract before either plan is
// handed to TensorRT. Missing, duplicate, empty, or hash-mismatched sections
// fail closed.
VerifiedOpenPIBundle verify_openpi_bundle_integrity(const BundleFile& bundle);

// Deserialize one required OpenPI TensorRT plan.
std::unique_ptr<ITrtModule> load_openpi_module(IBackend* backend, const std::vector<char>* plan,
                                               const char* section_name, const char* timing_label,
                                               const ModuleCreateOptions& options);

} // namespace trtmc::openpi
