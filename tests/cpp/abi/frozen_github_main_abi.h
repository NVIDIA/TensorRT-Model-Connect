/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Frozen ABI slice from github/main commit
// 26abe4c31944fbd4219837bc8b6b54524fa5d7af.
//
// This header intentionally does not include any current TRTMC public header.
// It contains the pre-runtime-memory declarations needed by independently
// compiled compatibility fixtures. Keep it mechanically aligned with:
//   include/trtmc/pipeline.h
//   include/trtmc/runtime/pipeline_plugin.h
//   include/trtmc/runtime/pipeline_registry.h
// at the commit above.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc {

// The compatibility calls never receive a live pipeline. A complete opaque
// type is required only so an old std::unique_ptr return value can be destroyed
// on failure paths. The factory/plugin function signatures and type name are
// the frozen ABI surface under test.
class IPipeline {
  public:
    virtual ~IPipeline() = default;
};

struct LoadOptions {
    std::string hf_python;
    std::string runtime_cache_path;
    bool cuda_graphs{false};
    std::uint64_t kv_cache_size_bytes{0};
    std::string config_path;
    std::vector<std::string> set_tokens;
    std::vector<std::string> backend_search_paths;
    std::vector<std::string> model_plugin_search_paths;
};

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const std::string& hf_python = "",
                                const std::string& runtime_cache_path = "",
                                bool cuda_graphs = false);
std::unique_ptr<IPipeline> load(const std::string& bundle_path, const LoadOptions& options);

struct BundleFile;
class IBackend;

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
};

namespace config {
class ConfigBundle;
}

struct PipelineContext {
    const BundleFile& bundle;
    const BaseConfig& config;
    const std::string& config_json;
    const std::string& hf_python;
    const std::string& bundle_path;
    IBackend* backend;
    const std::string& runtime_cache_path;
    bool cuda_graphs;
    std::uint64_t kv_cache_size_bytes{0};
    const ::trtmc::config::ConfigBundle* runtime_config{nullptr};
};

class IPipelinePlugin {
  public:
    virtual ~IPipelinePlugin() = default;
    virtual std::unique_ptr<IPipeline> create(const PipelineContext& ctx) = 0;

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

class PipelineRegistry {
  public:
    static PipelineRegistry& instance();
    void register_plugin(const std::string& strategy, IPipelinePlugin* plugin);
    IPipelinePlugin* lookup(const std::string& strategy) const;
    std::vector<std::string> registered_strategies() const;

  private:
    PipelineRegistry() = default;
    std::unordered_map<std::string, IPipelinePlugin*> registry_;
};

} // namespace trtmc

extern "C" {

struct TrtmcPipelineOptions {
    int max_new_tokens;
    const char* hf_python;
    const char* image_path;
    const char* runtime_cache;
    int cuda_graphs;
};

typedef trtmc::IPipeline* trtmc_pipeline_t;

trtmc::IPipeline* trtmc_create_pipeline(const char* bundle_path, int flags);
trtmc::IPipeline* trtmc_create_pipeline_ex(const char* bundle_path,
                                           const TrtmcPipelineOptions* options);
const char* trtmc_last_error(void);
const char* trtmc_version(void);
int trtmc_has_trt(void);
}
