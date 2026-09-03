/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>

namespace trtmc {

class ITokenizer;
class ITrtModule;

// Load the one required K2-Horizon engine_plan through the selected backend.
std::unique_ptr<ITrtModule> load_k2_horizon_engine_plan(IBackend* backend, const BundleFile& bundle,
                                                        const ModuleCreateOptions& options = {});

// Create K2-Horizon's required native BPE tokenizer from tokenizer.json.
std::shared_ptr<ITokenizer> create_k2_horizon_bpe_tokenizer(const BundleFile& bundle);

} // namespace trtmc
