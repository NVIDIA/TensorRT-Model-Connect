/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Pipeline plugin interface for registry-based pipeline dispatch.
// Each plugin handles one or more runtime_strategy values, parsing its own
// config from raw JSON and extracting its own bundle sections.

#include "trtmc/pipeline.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

// Forward declaration — defined in src/bundle/bundle_format.h (internal).
// PipelineContext holds a const reference, so no full definition needed here.
struct BundleFile;
class IBackend;

// Tensor I/O name mapping — read from the bundle's config.json "io_map" object.
// Per-layer patterns use tokens: {i}=layer, {2i}=2*layer, {2i+1}=2*layer+1, {2i+2}=2*layer+2.
struct IoMap {
    std::string token_id{"token_id"};
    std::string position_id{"position_id"};
    std::string attention_mask{"attention_mask"};
    std::string logits{"logits"};
    std::string cache_k_pattern{"cache_k_{i}"};
    std::string cache_v_pattern{"cache_v_{i}"};
    std::string present_k_pattern{"present_k_{i}"};
    std::string present_v_pattern{"present_v_{i}"};
};

// Universal base config — the ~10 fields every pipeline needs.
// Plugins parse strategy-specific fields directly from config_json.
struct BaseConfig {
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t num_layers{1};
    int32_t num_heads{1};
    int32_t num_kv_heads{1};
    int32_t head_dim{0};
    int32_t attention_size{0};
    int32_t max_cache_length{32};
    int32_t id_bos{-1};
    int32_t id_eos{-1};
    std::string runtime_strategy;
    std::string precision{"fp32"};
    bool tokenizer_add_special_tokens{false};
    bool tokenizer_add_special_tokens_present{false};
    IoMap io_map;
    std::vector<int32_t> id_eos_ids;
};

// Parse universal base config from config.json text.
BaseConfig parse_base_config(const std::string& config_text, int32_t max_cache_length_override);

// Forward declaration; full type in trtmc/config/config_bundle.h.
namespace config {
class ConfigBundle;
}

// Context passed to each plugin's create() method. Non-owning references
// to the bundle and parsed base config. The BundleFile must outlive the
// pipeline being created (it does — PipelineFactory::from_bundle() holds it).
struct PipelineContext {
    const BundleFile& bundle;
    const BaseConfig& config;
    const std::string& config_json;        // raw JSON text from bundle
    const std::string& hf_python;          // path to HF Python interpreter
    const std::string& bundle_path;        // filesystem path to .bundle file
    IBackend* backend;                     // Backend for creating ITrtModule instances
    const std::string& runtime_cache_path; // RTX: JIT kernel cache file path
    bool cuda_graphs;                      // Enable backend-supported CUDA Graph execution
    std::uint64_t kv_cache_size_bytes{0};  // 0 = use bundle max_cache_length (TriAttention)
    // Resolved layered config (schema-driven). Nullable: plugins not yet
    // migrated to the registry ignore it; migrated plugins query their
    // namespace via ctx.runtime_config->get<T>("ns", "field"). The bundle
    // the pointer refers to is owned by the factory and outlives create().
    const ::trtmc::config::ConfigBundle* runtime_config{nullptr};
    // When false, plugins validate header metadata and section bounds but do not
    // attest payload contents against the declared digests.
    bool validate_bundle_payloads{true};
};

// Plugin interface. Each plugin registers itself with the PipelineRegistry
// via the REGISTER_PIPELINE_PLUGIN macro. The registry calls create() when
// the bundle's runtime_strategy matches one of the plugin's registered keys.
class IPipelinePlugin {
  public:
    virtual ~IPipelinePlugin() = default;

    // Create a pipeline from the given context. The plugin is responsible for:
    // 1. Parsing strategy-specific config from ctx.config_json
    // 2. Extracting needed sections from ctx.bundle via find_section()
    // 3. Loading TRT engines, creating caches, tokenizers, etc.
    // 4. Returning the fully constructed pipeline
    virtual std::unique_ptr<IPipeline> create(const PipelineContext& ctx) = 0;

    // Create independent request lanes for PipelinePool. The default is
    // correct for every model but deserializes one engine per lane. Plugins
    // can override this to share engine weights across execution contexts.
    virtual std::vector<std::unique_ptr<IPipeline>> create_pool(const PipelineContext& ctx,
                                                                std::size_t count) {
        if (count == 0)
            throw std::invalid_argument("Pipeline pool size must be positive");
        std::vector<std::unique_ptr<IPipeline>> pipelines;
        pipelines.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
            pipelines.push_back(create(ctx));
        return pipelines;
    }
};

} // namespace trtmc
