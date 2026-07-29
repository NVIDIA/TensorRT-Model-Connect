/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Minimal runtime helpers used by the GLM pipeline plugin.

#include "bundle/bundle_format.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <memory>
#include <vector>

namespace trtmc {

// A loaded TRT engine, ready for inference.
// The stream is owned internally by the module — callers get it via module->stream().
struct LoadedModule {
    std::unique_ptr<ITrtModule> module;
};

// Load a TRT engine from a serialized plan via the backend. Throws on failure.
LoadedModule load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan,
                                       const char* label, const ModuleCreateOptions& options = {});

// Create a native tokenizer from bundle. Tries BPE -> WordPiece -> Unigram.
// Returns nullptr if no native tokenizer matches.
std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleFile& bundle);

} // namespace trtmc
