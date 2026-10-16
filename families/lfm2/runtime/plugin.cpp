/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2/runtime/conv_state.h"
#include "families/lfm2/runtime/hybrid_state.h"
#include "families/lfm2/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::lfm2 {

namespace {

constexpr const char* kSamplerMode = "philox_exponential_v1";
constexpr const char* kW2Mode = "cublas_bf16_all_w2_v1";
constexpr const char* kFirstTwoW2Mode = "cublas_bf16_first_two_v1";
constexpr const char* kAttentionMode = "cudnn_bf16_dense_v1";

struct RuntimeConfig {
    std::int32_t hidden_size;
    std::int32_t num_layers;
    std::int32_t num_attention_heads;
    std::int32_t num_key_value_heads;
    std::int32_t head_dim;
    std::int32_t num_attention_layers;
    std::int32_t num_conv_layers;
    std::int32_t conv_cache_length;
    std::int32_t conv_state_dim;
    std::int32_t vocab_size;
    std::int32_t bos_token_id;
    std::vector<std::int32_t> eos_token_ids;
    std::int32_t max_cache_length;
    std::string precision;
    std::string sampler_mode;
    std::string w2_mode;
    std::string attention_mode;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("lfm2 runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("lfm2 runtime.json has invalid '") + name + "'");
    }
}

std::vector<std::int32_t> require_eos_ids(const nlohmann::json& json) {
    if (!json.contains("eos_token_id"))
        throw std::runtime_error("lfm2 runtime.json missing 'eos_token_id'");
    const auto& value = json.at("eos_token_id");
    if (value.is_number_integer())
        return {value.get<std::int32_t>()};
    if (value.is_array() && !value.empty())
        return value.get<std::vector<std::int32_t>>();
    throw std::runtime_error("lfm2 runtime.json has invalid 'eos_token_id'");
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const std::string text = require_text_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("lfm2 invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("lfm2 runtime.json must be an object");
    const bool has_sampler = json.contains("sampler_mode");
    const bool has_w2 = json.contains("w2_mode");
    const bool has_attention = json.contains("attention_mode");
    if (has_w2 && !has_sampler)
        throw std::runtime_error("lfm2 w2_mode requires the Philox sampler contract");
    if (has_attention && !has_w2)
        throw std::runtime_error("lfm2 attention_mode requires the W2 and Philox contracts");
    if (json.size() != 23U + static_cast<unsigned>(has_sampler) + static_cast<unsigned>(has_w2) +
                           static_cast<unsigned>(has_attention))
        throw std::runtime_error("lfm2 runtime.json has an unexpected field set");
    const std::string sampler_mode =
        has_sampler ? require_value<std::string>(json, "sampler_mode") : std::string{};
    if (has_sampler && sampler_mode != kSamplerMode)
        throw std::runtime_error("lfm2 runtime.json has an unsupported sampler_mode");
    const std::string w2_mode =
        has_w2 ? require_value<std::string>(json, "w2_mode") : std::string{};
    if (has_w2 && w2_mode != kW2Mode && w2_mode != kFirstTwoW2Mode)
        throw std::runtime_error("lfm2 runtime.json has an unsupported w2_mode");
    const std::string attention_mode =
        has_attention ? require_value<std::string>(json, "attention_mode") : std::string{};
    if (has_attention && (attention_mode != kAttentionMode || w2_mode != kW2Mode))
        throw std::runtime_error("lfm2 runtime.json has an unsupported attention contract");
    const auto* sampler_plan = bundle.find_section("sampler.plan");
    if (has_sampler && (sampler_plan == nullptr || sampler_plan->length == 0))
        throw std::runtime_error("lfm2 sampler_mode requires a nonempty sampler.plan");
    if (!has_sampler && sampler_plan != nullptr)
        throw std::runtime_error("lfm2 legacy runtime contract cannot contain sampler.plan");
    if (!require_value<bool>(json, "native_kv_cache") ||
        require_value<std::int32_t>(json, "native_kv_contract_version") !=
            (has_attention ? 2 : 1)) {
        throw std::runtime_error(
            "lfm2 runtime.json has an incompatible native KV contract version");
    }
    const std::int32_t intermediate_size = require_value<std::int32_t>(json, "intermediate_size");
    const float norm_eps = require_value<float>(json, "norm_eps");
    const float rope_theta = require_value<float>(json, "rope_theta");
    const auto layer_types = require_value<std::vector<std::string>>(json, "layer_types");
    (void)require_value<bool>(json, "tie_word_embeddings");
    (void)require_value<std::int32_t>(json, "pad_token_id");
    const std::string layout = require_value<std::string>(json, "decoder_engine_layout");

    RuntimeConfig config{
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_attention_heads"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "num_attention_layers"),
        require_value<std::int32_t>(json, "num_conv_layers"),
        require_value<std::int32_t>(json, "conv_L_cache"),
        require_value<std::int32_t>(json, "conv_dim"),
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_eos_ids(json),
        require_value<std::int32_t>(json, "max_cache_length"),
        require_value<std::string>(json, "precision"),
        sampler_mode,
        w2_mode,
        attention_mode,
    };
    if (config.hidden_size <= 0 || config.num_layers <= 0 || config.num_attention_heads <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 ||
        config.num_attention_layers <= 0 || config.num_conv_layers <= 0 ||
        config.num_attention_layers + config.num_conv_layers != config.num_layers ||
        config.conv_cache_length <= 0 || config.conv_state_dim != config.hidden_size ||
        config.vocab_size <= 0 || config.max_cache_length <= 0 || config.eos_token_ids.empty()) {
        throw std::runtime_error("lfm2 runtime.json contains invalid geometry");
    }
    if (intermediate_size <= 0 || norm_eps <= 0.0F || rope_theta <= 0.0F ||
        layer_types.size() != static_cast<std::size_t>(config.num_layers) || layout != "single" ||
        (config.precision != "fp16" && config.precision != "bf16")) {
        throw std::runtime_error("lfm2 runtime.json contains invalid model contract");
    }
    if (has_sampler && config.vocab_size != 128000)
        throw std::runtime_error("lfm2 Philox sampler requires a 128000-token vocabulary");
    if (has_w2 &&
        (config.precision != "bf16" || config.hidden_size != 2048 || intermediate_size != 10752 ||
         config.num_layers != 30 || layer_types[0] != "conv" || layer_types[1] != "conv"))
        throw std::runtime_error("lfm2 W2 mode requires its qualified BF16 model geometry");
    if (has_attention && (config.num_attention_heads != 32 || config.num_key_value_heads != 8 ||
                          config.head_dim != 64 || config.num_attention_layers != 8 ||
                          config.max_cache_length > 131072))
        throw std::runtime_error("lfm2 cuDNN attention requires its qualified BF16 model geometry");
    return config;
}

void register_sampler() {
    // Failed initialization is retryable; a successful registration is retained
    // once per loaded family library, including concurrent pipeline creation.
    static void* const registered = [] {
        Dl_info location{};
        if (!dladdr(reinterpret_cast<void*>(&register_sampler), &location) || !location.dli_fname)
            throw std::runtime_error("lfm2 cannot locate its family runtime library");
        const auto root = std::filesystem::absolute(location.dli_fname).parent_path();
        const auto helper = root / "libtrtmc_lfm2_sampler_byok.so";
        const auto bridge = root / "libtrtmc_byok_tvm_ffi.so";
        if (!std::filesystem::is_regular_file(helper) || !std::filesystem::is_regular_file(bridge))
            throw std::runtime_error(
                "lfm2 Philox sampler requires adjacent family sampler and TVM-FFI libraries");
        void* library = dlopen(bridge.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
        if (library == nullptr)
            throw std::runtime_error("lfm2 cannot load its adjacent TVM-FFI bridge: " +
                                     std::string(dlerror()));
        using LoadKernel = const char* (*)(const char*, const char*, const char*) noexcept;
        const auto load = reinterpret_cast<LoadKernel>(dlsym(library, "trtmc_load_byok_kernel"));
        if (load == nullptr) {
            dlclose(library);
            throw std::runtime_error("lfm2 TVM-FFI bridge has no kernel registration entrypoint");
        }
        if (const char* error =
                load(helper.c_str(), "sample", "lfm2.sampler.philox_exponential_v1")) {
            const std::string message(error);
            dlclose(library);
            throw std::runtime_error("lfm2 cannot register its sampler: " + message);
        }
        // Registration stores callable pointers; retain this successful handle.
        return library;
    }();
    (void)registered;
}

void register_w2() {
    static void* const registered = [] {
        Dl_info location{};
        if (!dladdr(reinterpret_cast<void*>(&register_w2), &location) || !location.dli_fname)
            throw std::runtime_error("lfm2 cannot locate its family runtime library");
        const auto root = std::filesystem::absolute(location.dli_fname).parent_path();
        const auto helper = root / "libtrtmc_lfm2_w2_byok.so";
        const auto bridge = root / "libtrtmc_byok_tvm_ffi.so";
        if (!std::filesystem::is_regular_file(helper) || !std::filesystem::is_regular_file(bridge))
            throw std::runtime_error(
                "lfm2 W2 mode requires adjacent family W2 and TVM-FFI libraries");
        void* library = dlopen(bridge.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
        if (library == nullptr)
            throw std::runtime_error("lfm2 cannot load its adjacent TVM-FFI bridge: " +
                                     std::string(dlerror()));
        using LoadKernel = const char* (*)(const char*, const char*, const char*) noexcept;
        const auto load = reinterpret_cast<LoadKernel>(dlsym(library, "trtmc_load_byok_kernel"));
        if (load == nullptr) {
            dlclose(library);
            throw std::runtime_error("lfm2 TVM-FFI bridge has no kernel registration entrypoint");
        }
        if (const char* error = load(helper.c_str(), "run", "lfm2.w2.cublas_bf16_v1")) {
            const std::string message(error);
            dlclose(library);
            throw std::runtime_error("lfm2 cannot register its W2 operation: " + message);
        }
        return library;
    }();
    (void)registered;
}

void register_attention() {
    static void* const registered = [] {
        Dl_info location{};
        if (!dladdr(reinterpret_cast<void*>(&register_attention), &location) || !location.dli_fname)
            throw std::runtime_error("lfm2 cannot locate its family runtime library");
        const auto root = std::filesystem::absolute(location.dli_fname).parent_path();
        const auto helper = root / "libtrtmc_lfm2_sdpa_byok.so";
        const auto bridge = root / "libtrtmc_byok_tvm_ffi.so";
        if (!std::filesystem::is_regular_file(helper) || !std::filesystem::is_regular_file(bridge))
            throw std::runtime_error(
                "lfm2 cuDNN attention requires adjacent SDPA and TVM-FFI libraries");
        void* library = dlopen(bridge.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
        if (library == nullptr)
            throw std::runtime_error("lfm2 cannot load its adjacent TVM-FFI bridge: " +
                                     std::string(dlerror()));
        using LoadKernel = const char* (*)(const char*, const char*, const char*) noexcept;
        const auto load = reinterpret_cast<LoadKernel>(dlsym(library, "trtmc_load_byok_kernel"));
        if (load == nullptr) {
            dlclose(library);
            throw std::runtime_error("lfm2 TVM-FFI bridge has no kernel registration entrypoint");
        }
        if (const char* error = load(helper.c_str(), "run", "lfm2.attention.cudnn_bf16_dense_v1")) {
            const std::string message(error);
            dlclose(library);
            throw std::runtime_error("lfm2 cannot register its attention operation: " + message);
        }
        return library;
    }();
    (void)registered;
}

} // namespace

ITask* create(const FamilyContext& context) {
    const RuntimeConfig config = parse_runtime_config(context.reader);
    if (!config.sampler_mode.empty()) {
        const char* backend = context.backend.name();
        if (backend == nullptr || std::string(backend) != "trt")
            throw std::runtime_error("lfm2 Philox sampler requires the TensorRT backend");
        register_sampler();
    }
    if (!config.w2_mode.empty())
        register_w2();
    if (!config.attention_mode.empty())
        register_attention();
    auto decoder = load_module(context.backend, require_section(context.reader, "engine.plan"));
    const cudaStream_t stream = decoder->stream();
    std::unique_ptr<ITrtModule> sampler;
    if (!config.sampler_mode.empty()) {
        const auto plan = require_section(context.reader, "sampler.plan");
        ModuleCreateOptions options;
        options.stream = stream;
        sampler = context.backend.create_module(plan.data(), plan.size(), options);
        if (!sampler || !sampler->ok() || sampler->stream() != stream)
            throw std::runtime_error("lfm2 failed to load sampler.plan on the decoder stream");
        lfm2_validate_sampler_module(*sampler);
        sampler->set_timing_label("lfm2 sampler");
    }
    const DType dtype = state_dtype(config.precision);

    auto cache_names = kv_names(config.num_attention_layers);
    if (!config.attention_mode.empty()) {
        cache_names.key_value_lengths.clear();
        cache_names.active_length_extent = "active_length_extent";
    }
    auto kv = std::make_unique<Lfm2KvCache>(config.num_attention_layers, config.max_cache_length,
                                            config.num_key_value_heads, config.head_dim, stream,
                                            dtype, std::move(cache_names));
    auto conv = std::make_unique<Lfm2ConvState>(config.num_conv_layers, config.conv_state_dim,
                                                config.conv_cache_length, stream, dtype);
    auto state = std::make_unique<Lfm2HybridState>(std::move(kv), std::move(conv));
    if (!state->ok())
        throw std::runtime_error("lfm2 failed to allocate inference state");

    Lfm2TextGenConfig generation;
    generation.vocab_size = config.vocab_size;
    generation.id_bos = config.bos_token_id;
    generation.id_eos = config.eos_token_ids.front();
    generation.id_eos_ids = config.eos_token_ids;
    apply_chat_template(context.reader, generation);
    auto tokenizer = create_tokenizer(context.reader);

    return new Lfm2TextGenerationPipeline(std::move(decoder), std::move(state),
                                          std::move(generation), stream, std::move(tokenizer),
                                          std::string{}, std::move(sampler));
}

} // namespace trtmc::lfm2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("lfm2 does not support --kv-cache-size");
    return trtmc::lfm2::create(context);
}
