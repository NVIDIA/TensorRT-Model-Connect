/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// VLPlugin: handles "qwen_vl_vision_language" strategy.
// Two-engine pipeline: vision encoder + text decoder with KV cache.

#include "plugin_helpers.h"
#include "runtime/models/qwen_vl/cuda_stream.h"
#include "runtime/models/qwen_vl/image_preprocessor.h"
#include "runtime/models/qwen_vl/pipeline.h"
#include "runtime/models/qwen_vl/tensor_names.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

struct TextModuleRuntime {
    ModuleCreateOptions options;
    DistributedRuntimeGroup tp_group;
    std::string engine_section{"engine_plan"};
};

TensorParallelRuntimeConfig parse_tensor_parallel_runtime_config(const std::string& config_json) {
    TensorParallelRuntimeConfig cfg;
    cfg.tp_size = extract_json_int(config_json, "tensor_parallel_size", 1);
    const auto mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    cfg.enabled = (mode == "tensor_parallel" && cfg.tp_size > 1);
    return cfg;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine_plan_tp_rank" + std::to_string(rank);
}

int32_t dim_at(const std::vector<int64_t>& shape, std::size_t idx) {
    return shape.size() > idx ? static_cast<int32_t>(shape[idx]) : 0;
}

int32_t decoder_cache_row_width(const TrtModule& module, const std::string& tensor_name,
                                const BaseConfig& config) {
    const auto shape = module.tensor_shape(tensor_name);
    if (shape.size() == 4) {
        const int32_t heads = dim_at(shape, 1);
        const int32_t head_dim = dim_at(shape, 3);
        if (heads > 0 && head_dim > 0 && heads <= std::numeric_limits<int32_t>::max() / head_dim) {
            return heads * head_dim;
        }
    }
    const int32_t from_engine = dim_at(shape, 1);
    return from_engine > 0 ? from_engine : compute_kv_dim(config);
}

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream oss;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        oss << (static_cast<double>(bytes) / kGiB) << " GiB";
    } else if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        oss << (static_cast<double>(bytes) / kMiB) << " MiB";
    } else {
        oss.unsetf(std::ios::floatfield);
        oss << bytes << " B";
    }
    return oss.str();
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error("Qwen-VL native KV byte accounting overflow");
    return lhs * rhs;
}

void validate_native_scalar(const ITrtModule& module, const std::string& name,
                            const char* module_label) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error(std::string("Qwen-VL ") + module_label + " native KV input '" +
                                 name + "' must be int32 [1]");
    }
}

void validate_native_module(const PipelineContext& ctx, const ITrtModule& module,
                            const QwenVlKvCacheNames& names,
                            const TensorParallelRuntimeConfig& tp_config,
                            const char* module_label) {
    validate_native_scalar(module, names.cache_write_indices, module_label);
    validate_native_scalar(module, names.key_value_lengths, module_label);

    const int32_t tp_size = tp_config.enabled ? tp_config.tp_size : 1;
    if (tp_size <= 0 || ctx.config.num_kv_heads <= 0 || ctx.config.num_kv_heads % tp_size != 0) {
        throw std::runtime_error(
            "Qwen-VL native KV requires num_key_value_heads divisible by tensor parallel size");
    }
    const int32_t local_kv_heads = ctx.config.num_kv_heads / tp_size;
    const std::vector<int64_t> expected_shape{1, local_kv_heads, ctx.config.max_cache_length, 128};
    for (int32_t layer = 0; layer < ctx.config.num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        const auto validate_pair = [&](const std::string& cache_name,
                                       const std::string& present_name) {
            if (!module.has_input(cache_name) || !module.has_output(present_name) ||
                module.tensor_shape(cache_name) != expected_shape ||
                module.tensor_shape(present_name) != expected_shape ||
                module.tensor_dtype(cache_name) != DType::kBFloat16 ||
                module.tensor_dtype(present_name) != DType::kBFloat16) {
                throw std::runtime_error(std::string("Qwen-VL ") + module_label +
                                         " native KV cache/present must be BF16 "
                                         "[1,local_num_kv_heads,capacity,128]");
            }
        };
        validate_pair(names.cache_k[index], names.present_k[index]);
        validate_pair(names.cache_v[index], names.present_v[index]);
    }
}

std::uint64_t native_kv_cache_bytes(const PipelineContext& ctx, int32_t kv_dim,
                                    std::size_t lane_count) {
    if (ctx.config.num_layers <= 0 || ctx.config.max_cache_length <= 0 || kv_dim <= 0)
        throw std::runtime_error("Qwen-VL native KV geometry must be positive");
    auto bytes = checked_multiply(static_cast<std::uint64_t>(ctx.config.num_layers),
                                  static_cast<std::uint64_t>(ctx.config.max_cache_length));
    bytes = checked_multiply(bytes, static_cast<std::uint64_t>(kv_dim));
    bytes = checked_multiply(bytes, 2); // BF16 bytes per element.
    bytes = checked_multiply(bytes, 2); // K and V.
    return checked_multiply(bytes, static_cast<std::uint64_t>(lane_count));
}

void admit_native_kv_allocation(const PipelineContext& ctx, std::uint64_t cache_bytes) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Qwen-VL native KV CUDA memory query failed: ") +
                                 cudaGetErrorString(status));
    }
    constexpr std::uint64_t kTwoGiB = 2ULL << 30;
    const auto free = static_cast<std::uint64_t>(free_bytes);
    const auto total = static_cast<std::uint64_t>(total_bytes);
    const auto reserve = std::max(kTwoGiB, total / 10);
    const auto available = free > reserve ? free - reserve : 0;
    if (cache_bytes > available) {
        throw std::runtime_error(
            "Qwen-VL native KV cache admission failed before allocation: capacity=" +
            std::to_string(ctx.config.max_cache_length) +
            " tokens, required=" + format_bytes(cache_bytes) + ", free=" + format_bytes(free) +
            ", reserve=" + format_bytes(reserve));
    }
}

TextModuleRuntime initialize_text_module_runtime(const TensorParallelRuntimeConfig& tp_config) {
    TextModuleRuntime runtime;
    if (!tp_config.enabled)
        return runtime;

    runtime.tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
    runtime.engine_section = tp_engine_section_name(runtime.tp_group.rank);
    return runtime;
}

void configure_text_module_options(TextModuleRuntime& runtime,
                                   const ModuleCreateOptions& base_options,
                                   const TensorParallelRuntimeConfig& tp_config) {
    runtime.options = base_options;
    if (!tp_config.enabled)
        return;

    runtime.options.distributed_communicator = runtime.tp_group.communicator;
    runtime.options.distributed_owner = runtime.tp_group.owner;
}

QwenVlKvCacheNames build_kv_cache_names(const BaseConfig& config) {
    const auto& io = config.io_map;
    QwenVlKvCacheNames kv_names;
    for (int32_t i = 0; i < config.num_layers; ++i) {
        kv_names.cache_k.push_back(qwen_vl_expand_layer_name(io.cache_k_pattern, i));
        kv_names.cache_v.push_back(qwen_vl_expand_layer_name(io.cache_v_pattern, i));
        kv_names.present_k.push_back(qwen_vl_expand_layer_name(io.present_k_pattern, i));
        kv_names.present_v.push_back(qwen_vl_expand_layer_name(io.present_v_pattern, i));
    }
    return kv_names;
}

BackendContextModules load_context_modules(IBackend* backend, const std::vector<char>* plan,
                                           const char* label,
                                           const std::vector<ModuleCreateOptions>& options) {
    if (!plan || plan->empty())
        throw std::runtime_error(std::string("Bundle missing ") + label);
    if (!backend)
        throw std::runtime_error("No backend loaded");
    const auto started = std::chrono::steady_clock::now();
    auto loaded = backend->create_context_modules(plan->data(), plan->size(), options);
    const auto finished = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(finished - started).count();
    log_trt_load_timing(label, elapsed, plan->size());
    if (loaded.modules.size() != options.size())
        throw std::runtime_error(std::string("Failed to create all execution contexts for ") +
                                 label);
    for (auto& module : loaded.modules) {
        if (!module || !module->ok())
            throw std::runtime_error(std::string("Failed to create execution context for ") +
                                     label);
        module->set_timing_label(label);
    }
    return loaded;
}

bool is_lora_input(const std::string& name) {
    return name.rfind("lora_a_", 0) == 0 || name.rfind("lora_b_", 0) == 0;
}

std::vector<TensorInfo> lora_input_contract(const ITrtModule& module) {
    std::vector<TensorInfo> contract;
    for (const auto& info : module.input_info()) {
        if (is_lora_input(info.name))
            contract.push_back(info);
    }
    return contract;
}

DualProfileModules load_text_modules(const PipelineContext& ctx, TextModuleRuntime& runtime,
                                     const std::shared_ptr<QwenVlCudaStream>& stream) {
    auto loaded =
        load_dual_profile_modules(ctx.backend, find_section(ctx.bundle, runtime.engine_section),
                                  runtime.engine_section.c_str(), runtime.options);
    loaded.decode->keep_alive(stream);
    if (loaded.prefill)
        loaded.prefill->keep_alive(stream);
    if (runtime.tp_group.owner) {
        loaded.decode->keep_alive(runtime.tp_group.owner);
        if (loaded.prefill)
            loaded.prefill->keep_alive(runtime.tp_group.owner);
    }
    return loaded;
}

std::unique_ptr<TrtModule> load_vision_module(IBackend* backend, const BundleFile& bundle,
                                              const ModuleCreateOptions& options,
                                              const std::shared_ptr<QwenVlCudaStream>& stream,
                                              bool declared_in_config) {
    const auto* plan = find_section(bundle, "vision_engine_plan");
    const bool required = declared_in_config || plan != nullptr;
    auto loaded = required
                      ? load_trt_module_from_plan(backend, plan, "vision_engine_plan", options)
                      : try_load_trt_module_from_plan(backend, plan, "vision_engine_plan", options);
    if (loaded.module && loaded.module->ok()) {
        loaded.module->keep_alive(stream);
        std::cerr << "[trtmc] Vision encoder loaded" << std::endl;
        return std::move(loaded.module);
    }
    return nullptr;
}

std::string bundle_section_text(const BundleFile& bundle, const std::string& section_name) {
    const auto* section = find_section(bundle, section_name);
    if (section && !section->empty())
        return std::string(section->begin(), section->end());
    return {};
}

struct LaneResources {
    std::vector<std::shared_ptr<QwenVlCudaStream>> streams;
    std::vector<ModuleCreateOptions> options;
};

LaneResources create_lane_resources(std::size_t count, const PipelineContext& ctx,
                                    TextModuleRuntime& text_runtime,
                                    const TensorParallelRuntimeConfig& tp_config) {
    LaneResources resources;
    resources.streams.reserve(count);
    resources.options.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        auto stream = std::make_shared<QwenVlCudaStream>();
        if (!stream->ok())
            throw std::runtime_error("VLPlugin: failed to create CUDA stream");
        ModuleCreateOptions options;
        options.stream = stream->get();
        options.runtime_cache_path = ctx.runtime_cache_path.c_str();
        options.cuda_graphs = ctx.cuda_graphs;
        configure_text_module_options(text_runtime, options, tp_config);
        resources.options.push_back(text_runtime.options);
        resources.streams.push_back(std::move(stream));
    }
    return resources;
}

struct TextLaneModules {
    std::vector<std::unique_ptr<ITrtModule>> decode;
    std::vector<std::unique_ptr<ITrtModule>> prefill;
};

void validate_native_contract(const PipelineContext& ctx, const TextLaneModules& modules,
                              const QwenVlKvCacheNames& names,
                              const TensorParallelRuntimeConfig& tp_config) {
    if (!extract_json_bool(ctx.config_json, "native_kv_cache", false) ||
        extract_json_int(ctx.config_json, "native_kv_contract_version", 0) != 1) {
        throw std::runtime_error(
            "Qwen-VL bundle metadata does not declare native KV contract version 1");
    }
    for (const auto& decode : modules.decode)
        validate_native_module(ctx, *decode, names, tp_config, "decode engine");
    for (const auto& prefill : modules.prefill) {
        if (!prefill)
            throw std::runtime_error("Qwen-VL native KV bundle is missing its prefill engine");
        validate_native_module(ctx, *prefill, names, tp_config, "prefill engine");
    }
    if (ctx.kv_cache_size_bytes != 0) {
        throw std::invalid_argument(
            "Qwen-VL native TensorRT KV cache allocates the model's complete fixed capacity; "
            "kv_cache_size_bytes is not supported");
    }
}

TextLaneModules load_split_text_lane_modules(const PipelineContext& ctx, TextModuleRuntime& runtime,
                                             const LaneResources& lanes,
                                             const std::vector<char>& prefill_plan) {
    const std::size_t count = lanes.streams.size();
    TextLaneModules modules;
    modules.decode.resize(count);
    modules.prefill.resize(count);
    std::vector<ModuleCreateOptions> decode_options = lanes.options;
    std::vector<ModuleCreateOptions> prefill_options = lanes.options;
    for (auto& options : decode_options)
        options.optimization_profile = 0;
    for (auto& options : prefill_options)
        options.optimization_profile = 0;
    auto decode =
        load_context_modules(ctx.backend, find_section(ctx.bundle, runtime.engine_section),
                             runtime.engine_section.c_str(), decode_options);
    auto prefill =
        load_context_modules(ctx.backend, &prefill_plan, "prefill_engine_plan", prefill_options);
    for (std::size_t index = 0; index < count; ++index) {
        modules.decode[index] = std::move(decode.modules[index]);
        modules.prefill[index] = std::move(prefill.modules[index]);
        modules.decode[index]->keep_alive(lanes.streams[index]);
        modules.prefill[index]->keep_alive(lanes.streams[index]);
        if (runtime.tp_group.owner) {
            modules.decode[index]->keep_alive(runtime.tp_group.owner);
            modules.prefill[index]->keep_alive(runtime.tp_group.owner);
        }
    }
    return modules;
}

TextLaneModules load_text_lane_modules(const PipelineContext& ctx, TextModuleRuntime& runtime,
                                       const LaneResources& lanes) {
    const std::size_t count = lanes.streams.size();
    TextLaneModules modules;
    modules.decode.resize(count);
    modules.prefill.resize(count);
    const auto* separate_prefill = find_section(ctx.bundle, "prefill_engine_plan");
    if (separate_prefill != nullptr && !separate_prefill->empty()) {
        return load_split_text_lane_modules(ctx, runtime, lanes, *separate_prefill);
    }
    if (count == 1) {
        auto loaded = load_text_modules(ctx, runtime, lanes.streams.front());
        modules.decode.front() = std::move(loaded.decode);
        modules.prefill.front() = std::move(loaded.prefill);
        return modules;
    }

    std::vector<ModuleCreateOptions> profile_options;
    profile_options.reserve(count * 2);
    for (const auto& options : lanes.options) {
        auto prefill_options = options;
        prefill_options.optimization_profile = 0;
        profile_options.push_back(prefill_options);
        auto decode_options = options;
        decode_options.optimization_profile = 1;
        profile_options.push_back(decode_options);
    }
    auto loaded =
        load_context_modules(ctx.backend, find_section(ctx.bundle, runtime.engine_section),
                             runtime.engine_section.c_str(), profile_options);
    for (std::size_t index = 0; index < count; ++index) {
        modules.prefill[index] = std::move(loaded.modules[index * 2]);
        modules.decode[index] = std::move(loaded.modules[index * 2 + 1]);
        modules.prefill[index]->keep_alive(lanes.streams[index]);
        modules.decode[index]->keep_alive(lanes.streams[index]);
        if (runtime.tp_group.owner) {
            modules.prefill[index]->keep_alive(runtime.tp_group.owner);
            modules.decode[index]->keep_alive(runtime.tp_group.owner);
        }
    }
    return modules;
}

std::vector<std::unique_ptr<ITrtModule>>
load_vision_lane_modules(const PipelineContext& ctx, const LaneResources& lanes, bool declared) {
    const std::size_t count = lanes.streams.size();
    std::vector<std::unique_ptr<ITrtModule>> modules(count);
    auto vision_options = lanes.options;
    for (auto& options : vision_options) {
        options.distributed_communicator = nullptr;
        options.distributed_owner.reset();
    }
    if (count == 1) {
        modules.front() = load_vision_module(ctx.backend, ctx.bundle, vision_options.front(),
                                             lanes.streams.front(), declared);
        return modules;
    }
    const auto* plan = find_section(ctx.bundle, "vision_engine_plan");
    const bool required = declared || plan != nullptr;
    if (!plan || plan->empty()) {
        if (required)
            throw std::runtime_error("Bundle missing vision_engine_plan");
        return modules;
    }
    try {
        auto loaded = load_context_modules(ctx.backend, plan, "vision_engine_plan", vision_options);
        modules = std::move(loaded.modules);
        for (std::size_t index = 0; index < count; ++index)
            modules[index]->keep_alive(lanes.streams[index]);
        std::cerr << "[trtmc] Vision encoder loaded" << std::endl;
    } catch (...) {
        if (required)
            throw;
    }
    return modules;
}

int32_t prefill_profile_max_length(const ITrtModule& prefill, const std::string& token_name) {
    const auto static_shape = prefill.tensor_shape(token_name);
    const int32_t static_length = dim_at(static_shape, 0);
    if (static_length > 1)
        return static_length;
    const int32_t profile_idx = prefill.profile_idx();
    if (profile_idx < 0 || profile_idx >= prefill.optimization_profile_count())
        return static_length;
    return dim_at(prefill.input_profile_shape(token_name, profile_idx, ProfileShapeSelector::kMax),
                  0);
}

QwenVlConfig make_pipeline_config(const PipelineContext& ctx, const ITrtModule& decode,
                                  const ITrtModule& prefill) {
    QwenVlConfig config;
    config.vocab_size = ctx.config.vocab_size;
    config.id_bos = ctx.config.id_bos;
    config.id_eos = ctx.config.id_eos;
    config.image_token_id = extract_json_int(ctx.config_json, "image_token_id", -1);
    config.vision_output_dim = extract_json_int(ctx.config_json, "vision_output_dim", 0);
    config.has_position_input = decode.has_input("position_id");
    config.num_layers = ctx.config.num_layers;
    const std::string token_name =
        ctx.config.io_map.token_id.empty() ? std::string("token_id") : ctx.config.io_map.token_id;
    config.prefill_max_length = prefill_profile_max_length(prefill, token_name);
    if (config.prefill_max_length <= 0)
        throw std::runtime_error("Qwen-VL native KV prefill engine has no valid profile capacity");
    config.present_k_pattern = ctx.config.io_map.present_k_pattern;
    config.present_v_pattern = ctx.config.io_map.present_v_pattern;
    return config;
}

std::vector<std::unique_ptr<IPipeline>>
make_pipeline_lanes(const PipelineContext& ctx, TextLaneModules modules,
                    std::vector<std::unique_ptr<ITrtModule>> vision_modules,
                    const LaneResources& lanes, const QwenVlConfig& config,
                    const QwenVlPreprocessConfig& preprocess, int32_t kv_dim, DType cache_dtype,
                    std::shared_ptr<ITokenizer> tokenizer) {
    auto adapter_cache = std::make_shared<qwen_vl::LoraAdapterCache>(
        lora_input_contract(*modules.decode.front()), lanes.streams.front()->get());
    std::vector<std::unique_ptr<IPipeline>> pipelines;
    pipelines.reserve(lanes.streams.size());
    for (std::size_t index = 0; index < lanes.streams.size(); ++index) {
        auto state = std::make_unique<QwenVlKvCache>(
            ctx.config.num_layers, ctx.config.max_cache_length, kv_dim, lanes.streams[index]->get(),
            cache_dtype, build_kv_cache_names(ctx.config));
        pipelines.push_back(std::make_unique<QwenVlPipeline>(
            std::move(modules.decode[index]), std::move(vision_modules[index]), std::move(state),
            config, preprocess, lanes.streams[index]->get(), tokenizer, ctx.bundle.info.model_id,
            nullptr, std::move(modules.prefill[index]), adapter_cache));
    }
    return pipelines;
}

} // namespace

class VLPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto pipelines = create_lanes(ctx, 1);
        return std::move(pipelines.front());
    }

    std::vector<std::unique_ptr<IPipeline>> create_pool(const PipelineContext& ctx,
                                                        std::size_t count) override {
        return create_lanes(ctx, count);
    }

  private:
    std::vector<std::unique_ptr<IPipeline>> create_lanes(const PipelineContext& ctx,
                                                         std::size_t count) {
        if (count == 0)
            throw std::invalid_argument("Qwen-VL pipeline pool size must be positive");
        load_ffi_kernels_from_bundle(ctx.bundle);

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        if (count > 1 && tp_config.enabled)
            throw std::runtime_error(
                "Qwen-VL pipeline pools do not yet support tensor parallelism");
        auto text_runtime = initialize_text_module_runtime(tp_config);
        auto lanes = create_lane_resources(count, ctx, text_runtime, tp_config);
        auto text_modules = load_text_lane_modules(ctx, text_runtime, lanes);

        const auto kv_names = build_kv_cache_names(ctx.config);
        validate_native_contract(ctx, text_modules, kv_names, tp_config);

        const std::string cache_k_name =
            ctx.config.io_map.cache_k_pattern.empty()
                ? std::string("cache_k_0")
                : qwen_vl_expand_layer_name(ctx.config.io_map.cache_k_pattern, 0);
        int32_t kv_dim =
            decoder_cache_row_width(*text_modules.decode.front(), cache_k_name, ctx.config);
        DType cache_dtype = text_modules.decode.front()->tensor_dtype(cache_k_name);
        if (cache_dtype != DType::kBFloat16)
            throw std::runtime_error("Qwen-VL native KV runtime requires BF16 cache tensors");

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        const auto vlc =
            make_pipeline_config(ctx, *text_modules.decode.front(), *text_modules.prefill.front());
        const bool has_vision_engine =
            extract_json_bool(ctx.config_json, "has_vision_engine", false);
        auto vision_modules = load_vision_lane_modules(ctx, lanes, has_vision_engine);

        // Build VL preprocessing config from bundle's config.json +
        // preprocessor_config.json sections.
        const std::string config_text = bundle_section_text(ctx.bundle, "config.json");
        const std::string preproc_text =
            bundle_section_text(ctx.bundle, "preprocessor_config.json");
        auto vl_preprocess = qwen_vl_parse_preprocess_config(config_text, preproc_text);

        // Engine/context deserialization (including the vision engine) can consume
        // significant memory. Check the complete user-owned cache allocation only
        // after every module required by this pipeline is resident.
        const auto cache_bytes = native_kv_cache_bytes(ctx, kv_dim, count);
        admit_native_kv_allocation(ctx, cache_bytes);
        std::cerr << "[trtmc] Qwen-VL native KV cache capacity=" << ctx.config.max_cache_length
                  << " tokens, lanes=" << count << ", allocation=" << format_bytes(cache_bytes)
                  << '\n';

        return make_pipeline_lanes(ctx, std::move(text_modules), std::move(vision_modules), lanes,
                                   vlc, vl_preprocess, kv_dim, cache_dtype, std::move(tokenizer));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_qwen_vl_plugin, VLPlugin,
                                       "qwen_vl_vision_language");

} // namespace trtmc
