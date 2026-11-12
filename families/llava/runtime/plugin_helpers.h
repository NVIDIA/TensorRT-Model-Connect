/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/llava/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct LoadedModule {
    std::unique_ptr<ITrtModule> module;
};

LoadedModule load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan,
                                       const char* label, const ModuleCreateOptions& options = {});

std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleReader& bundle);

} // namespace trtmc
