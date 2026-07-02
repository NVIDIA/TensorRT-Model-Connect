/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Shared helper functions for pipeline plugins.
// Extracted from pipeline_factory.cpp's anonymous namespace so all
// strategy plugins can reuse TRT module loading, tokenizer creation,
// KV-dim computation, and data-section conversion utilities.

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/models/qwen3_omni/inference_state.h"
#include "runtime/models/qwen3_omni/kv_cache.h"
#include "trtmc/runtime/pipeline_plugin.h"
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

// Emit a parseable runtime load/deserialization timing line.
void log_trt_load_timing(const char* label, double load_deserialize_ms, std::size_t plan_bytes);

// Like load_trt_module_from_plan but returns empty LoadedModule on failure
// instead of throwing (for optional engines).
LoadedModule try_load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan,
                                           const char* label,
                                           const ModuleCreateOptions& options = {});

// Load an optional TRT module, returning nullptr if the plan is absent.
// On deserialization failure, returns nullptr (does not throw).
std::unique_ptr<ITrtModule> extract_optional_module(IBackend* backend,
                                                    const std::vector<char>* plan,
                                                    const char* label,
                                                    const ModuleCreateOptions& options = {});

// Dual-profile TRT module group: one shared backend engine, two module
// contexts (one per optimization profile). Weights live once in GPU memory
// and both modules share the CUDA stream. Use `decode->stream()` to obtain
// the shared stream.
struct DualProfileModules {
    std::unique_ptr<ITrtModule> prefill; // batched Sq profile (null if single-profile)
    std::unique_ptr<ITrtModule> decode;  // Sq=1 profile, or the only profile if single-profile
};

// Load an engine from a serialized plan via the backend and create two
// execution contexts — one per optimization profile — sharing the engine.
// When the engine has fewer than 2 profiles, `prefill` is left null and
// `decode` holds the single-profile context (legacy bundles).
DualProfileModules load_dual_profile_modules(IBackend* backend, const std::vector<char>* plan,
                                             const char* label,
                                             const ModuleCreateOptions& options = {});

// Detect whether the bundle's config requests add_special_tokens for the tokenizer.
bool detect_add_special_tokens(const BundleFile& bundle);

// Check if the bundle's tokenizer.json describes a BPE model.
bool is_bpe_tokenizer_json(const BundleFile& bundle);

// Try to create a native C++ BPE tokenizer from the bundle's tokenizer.json.
// Returns nullptr if the section is absent or the model is non-BPE.
// If throw_on_failure is true, throws instead of returning nullptr on BPE parse errors.
std::shared_ptr<ITokenizer> try_create_native_bpe(const BundleFile& bundle, bool add_special,
                                                  bool throw_on_failure);

// Try to create a native C++ tokenizer from the bundle's tokenizer.json.
// Attempts: BPE -> WordPiece -> Unigram. Returns nullptr if none match.
std::shared_ptr<ITokenizer> try_create_native_tokenizer(const BundleFile& bundle,
                                                        bool add_special_tokens);

// Create a native tokenizer from bundle. Tries BPE -> WordPiece -> Unigram.
// Returns nullptr if no native tokenizer matches.
std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleFile& bundle);

// Compute the KV cache dimension from model config.
int32_t compute_kv_dim(const BaseConfig& cfg);

// Convert the BaseConfig precision string ("fp16", "bf16", "fp32") to a DType
// for use as KV cache element type.
DType cache_dtype_from_precision(const std::string& precision);

// Reinterpret a raw char section as a vector of floats.
std::vector<float> section_to_floats(const std::vector<char>* sec);

// Reinterpret a raw char section as a vector of int32_t.
std::vector<int32_t> section_to_int32s(const std::vector<char>* sec);

// Return true if the section pointer is non-null and non-empty.
bool has_section_data(const std::vector<char>* d);

// BundleFile-based helpers.

// Mel filterbank loaded from bundle (for Whisper native mel extraction).
struct MelFilterbank {
    std::vector<float> data; // [n_freq_bins * n_mel_bins] row-major
    int32_t n_freq_bins{0};
    int32_t n_mel_bins{0};
};

// Load mel filterbank from the "mel_filterbank" bundle section.
// Returns empty MelFilterbank if section is not present (old bundles).
MelFilterbank load_mel_filterbank(const BundleFile& bundle);

// Create a native BPE tokenizer from the CLIP tokenizer sections in the bundle.
// Used for dual-tokenizer models (e.g., FLUX: CLIP + T5).
// Returns nullptr if clip_tokenizer.json section is absent.
std::unique_ptr<ITokenizer> create_clip_tokenizer_from_bundle(const BundleFile& bundle);

// Load all TVM-FFI kernels listed in the bundle's kernel_manifest.json.
// Must be called BEFORE deserializing any TRT engine that uses FFI plugins.
// No-op if the bundle has no kernel_manifest.json section (non-FFI bundles).
void load_ffi_kernels_from_bundle(const BundleFile& bundle);

} // namespace trtmc
