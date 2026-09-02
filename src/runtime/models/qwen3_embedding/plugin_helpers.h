/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Qwen3-Embedding runtime helper surface.

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstddef>
#include <memory>
#include <string>
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

// Convert an untrusted kernel name into one safe filename component.
std::string sanitize_kernel_filename_component(const std::string& global_name);

// Load all TVM-FFI kernels listed in the bundle's kernel_manifest.json.
// Must be called BEFORE deserializing any TRT engine that uses FFI plugins.
// No-op if the bundle has no kernel_manifest.json section (non-FFI bundles).
void load_ffi_kernels_from_bundle(const BundleFile& bundle);

} // namespace trtmc
