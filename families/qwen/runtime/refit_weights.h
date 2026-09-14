/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Refit weights for a stripped Qwen bundle.
//
// A bundle built with --strip-weights carries no weight values inside its
// engine plans. They arrive one of two ways:
//
//   refit_weights        a safetensors section embedded in the bundle, used
//                        whenever the engine's weights are not byte-identical
//                        to the checkpoint (fp16 from a bf16 checkpoint,
//                        quantized, tensor-parallel).
//
//   refit_manifest.json  offsets into the original HF checkpoint, used when
//                        --native-layout made every weight byte-identical to a
//                        tensor already on disk. The bundle then carries no
//                        weights at all and the runtime maps the checkpoint.
//
// Either way the result is a RefitWeightMap handed to IRefitter before any
// execution context exists. The backend only iterates it.

#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

// Bundle section names the build may emit.
inline constexpr const char* kQwenRefitWeightsSection = "refit_weights";
inline constexpr const char* kQwenRefitManifestSection = "refit_manifest.json";

// Overrides the checkpoint directory recorded in the manifest, for bundles
// that have moved since they were built.
inline constexpr const char* kQwenRefitCheckpointEnv = "TRTMC_REFIT_CHECKPOINT_DIR";

// Owns whatever backs a RefitWeightMap: nothing for the embedded case (the
// bundle's own buffer is borrowed), or the checkpoint mappings for the
// manifest case. Must outlive module creation.
class QwenRefitSource {
  public:
    QwenRefitSource() = default;
    ~QwenRefitSource();
    QwenRefitSource(QwenRefitSource&&) noexcept;
    QwenRefitSource& operator=(QwenRefitSource&&) noexcept;
    QwenRefitSource(const QwenRefitSource&) = delete;
    QwenRefitSource& operator=(const QwenRefitSource&) = delete;

    const RefitWeightMap& weights() const { return weights_; }
    bool empty() const { return weights_.empty(); }
    // Bytes mapped from the checkpoint; 0 for the embedded case.
    std::size_t mapped_bytes() const;

    friend QwenRefitSource parse_qwen_refit_weights(const std::vector<char>&);
    friend QwenRefitSource load_qwen_refit_from_manifest(const std::string&);

  private:
    struct Mapping {
        void* base{nullptr};
        std::size_t size{0};
    };
    std::vector<Mapping> mappings_;
    RefitWeightMap weights_;
};

// Parse an embedded `refit_weights` safetensors section. Views point INTO
// `section`, so the caller's buffer must outlive the result.
QwenRefitSource parse_qwen_refit_weights(const std::vector<char>& section);

// Resolve a `refit_manifest.json` section: map the checkpoint shards it names
// and point the weights at them. Throws if the checkpoint is missing, a shard
// has an unexpected size, or an entry runs past the end of its file.
QwenRefitSource load_qwen_refit_from_manifest(const std::string& manifest_json);

} // namespace trtmc
