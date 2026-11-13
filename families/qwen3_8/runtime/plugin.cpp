/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_8/runtime/chat_templates.h"
#include "families/qwen3_8/runtime/dflash2.h"
#include "families/qwen3_8/runtime/dspark.h"
#include "families/qwen3_8/runtime/hybrid_state.h"
#include "families/qwen3_8/runtime/mtp_scheduler.h"
#include "families/qwen3_8/runtime/native_media.h"
#include "families/qwen3_8/runtime/pipeline.h"
#include "families/qwen3_8/runtime/plugin_helpers.h"
#include "families/qwen3_8/runtime/prefill_runner.h"
#include "families/qwen3_8/runtime/recurrent_state.h"
#include "families/qwen3_8/runtime/tensor_parallel.h"
#include "trtmc/runtime/family_factory.h"
#ifdef TRTMC_HAS_EDGE_LLM
#include "families/qwen3_8/runtime/edge_llm/adapter.h"
#endif

#include <algorithm>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::qwen3_8 {

namespace {

struct RuntimeConfig {
    std::int32_t hidden_size;
    std::int32_t num_layers;
    std::int32_t num_attention_heads;
    std::int32_t num_key_value_heads;
    std::int32_t head_dim;
    std::int32_t vocab_size;
    std::int32_t bos_token_id;
    std::vector<std::int32_t> eos_token_ids;
    std::int32_t num_attention_layers;
    std::int32_t num_mamba_layers;
    std::int32_t d_inner;
    std::int32_t mamba_d_state;
    std::int32_t mamba_d_conv;
    std::int32_t mamba_nheads;
    std::int32_t mamba_head_dim;
    std::int32_t conv_dim;
    std::int32_t max_cache_length;
    std::string precision;
    bool has_mtp;
    std::int32_t mtp_num_draft_tokens;
    std::int32_t prefill_chunk_len; // 0: no prefill_engine.plan, single-token prefill
    std::string kv_cache_quantization;
    int32_t tensor_parallel_size;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("qwen3_8 runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("qwen3_8 runtime.json has invalid '") + name + "'");
    }
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const std::string text = require_text_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("qwen3_8 invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("qwen3_8 runtime.json must be an object");
    const bool has_prefill_field = json.contains("prefill_chunk_len");
    if (json.size() != 22U + (has_prefill_field ? 1U : 0U) +
                           (json.contains("kv_cache_quantization") ? 1U : 0U) +
                           (json.contains("tensor_parallel_size") ? 1U : 0U))
        throw std::runtime_error("qwen3_8 runtime.json has an unexpected field set");
    if (require_value<std::string>(json, "decoder_engine_layout") != "single")
        throw std::runtime_error("qwen3_8 requires decoder_engine_layout='single'");

    RuntimeConfig config{
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_attention_heads"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_value<std::vector<std::int32_t>>(json, "eos_token_ids"),
        require_value<std::int32_t>(json, "num_attention_layers"),
        require_value<std::int32_t>(json, "num_mamba_layers"),
        require_value<std::int32_t>(json, "d_inner"),
        require_value<std::int32_t>(json, "mamba_d_state"),
        require_value<std::int32_t>(json, "mamba_d_conv"),
        require_value<std::int32_t>(json, "mamba_nheads"),
        require_value<std::int32_t>(json, "mamba_head_dim"),
        require_value<std::int32_t>(json, "conv_dim"),
        require_value<std::int32_t>(json, "max_cache_length"),
        require_value<std::string>(json, "precision"),
        require_value<bool>(json, "has_mtp"),
        require_value<std::int32_t>(json, "mtp_num_draft_tokens"),
        has_prefill_field ? require_value<std::int32_t>(json, "prefill_chunk_len") : 0,
        json.contains("kv_cache_quantization")
            ? require_value<std::string>(json, "kv_cache_quantization")
            : "none",
        json.contains("tensor_parallel_size") ? require_value<int32_t>(json, "tensor_parallel_size")
                                              : 1,
    };
    const auto layer_types = require_value<std::vector<std::string>>(json, "layer_types");
    if (config.hidden_size <= 0 || config.num_layers <= 0 || config.num_attention_heads <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 || config.vocab_size <= 0 ||
        config.num_attention_layers <= 0 || config.num_mamba_layers <= 0 ||
        config.num_attention_layers + config.num_mamba_layers != config.num_layers ||
        config.d_inner <= 0 || config.mamba_d_state <= 0 || config.mamba_d_conv <= 0 ||
        config.mamba_nheads <= 0 || config.mamba_head_dim <= 0 || config.conv_dim <= 0 ||
        config.max_cache_length <= 0 || config.eos_token_ids.empty() ||
        layer_types.size() != static_cast<std::size_t>(config.num_layers)) {
        throw std::runtime_error("qwen3_8 runtime.json contains invalid geometry");
    }
    if (config.prefill_chunk_len < 0 || config.prefill_chunk_len == 1)
        throw std::runtime_error("qwen3_8 runtime.json has invalid prefill_chunk_len");
    if (config.precision != "fp16" && config.precision != "fp32" && config.precision != "bf16")
        throw std::runtime_error("qwen3_8 runtime.json contains invalid precision");
    if (config.has_mtp && config.mtp_num_draft_tokens < 1)
        throw std::runtime_error(
            "qwen3_8 runtime.json has has_mtp set but mtp_num_draft_tokens < 1");
    if (config.kv_cache_quantization != "none" && config.kv_cache_quantization != "fp8")
        throw std::runtime_error("qwen3_8 runtime.json has invalid KV cache quantization");
    const int tp = config.tensor_parallel_size;
    if ((tp != 1 && tp != 2) ||
        (tp == 2 &&
         (config.has_mtp || config.precision != "fp16" || config.kv_cache_quantization != "none" ||
          config.num_attention_heads % tp || config.num_key_value_heads % tp ||
          config.mamba_nheads % tp || config.d_inner % tp || config.conv_dim % tp)))
        throw std::runtime_error("Invalid Qwen native TP2 metadata");
    return config;
}

DType state_dtype(const std::string& precision) {
    if (precision == "bf16")
        return DType::kBFloat16;
    return precision == "fp16" ? DType::kFloat16 : DType::kFloat32;
}

std::string chat_template(const BundleReader& bundle) {
    const auto* section = bundle.find_section("chat_template.jinja");
    if (section == nullptr || section->length == 0)
        return {};
    const auto data = bundle.read_section("chat_template.jinja");
    return {data.begin(), data.end()};
}

} // namespace

ITask* create_native(const FamilyContext& context, const RuntimeConfig& config, int rank,
                     const ModuleCreateOptions& options) {
    const bool parallel = config.tensor_parallel_size == 2;
    const auto plan = require_section(
        context.reader, parallel ? "engine.rank" + std::to_string(rank) + ".plan" : "engine.plan");
    auto decoder = context.backend.create_module(plan.data(), plan.size(), options);
    if (!decoder || !decoder->ok())
        throw std::runtime_error("Qwen native decoder failed to load");
    decoder->set_timing_label("qwen3_8 decoder");
    const cudaStream_t stream = decoder->stream();
    const std::int32_t kv_dim =
        config.num_key_value_heads / config.tensor_parallel_size * config.head_dim;

    const auto cache_dtype =
        config.kv_cache_quantization == "fp8" ? DType::kFloat8E4M3 : state_dtype(config.precision);
    // Validate the storage contract before binding one-byte buffers. A stale
    // or malformed metadata entry must not allocate FP8 for an FP16 engine.
    for (int32_t i = 0; i < config.num_attention_layers; ++i) {
        for (const char* name : {"cache_k_", "cache_v_", "present_k_", "present_v_"}) {
            if (decoder->tensor_dtype(std::string(name) + std::to_string(i)) != cache_dtype)
                throw std::runtime_error("qwen3_8 engine KV dtype disagrees with runtime.json");
        }
    }
    auto kv = std::make_unique<Qwen38KvCache>(config.num_attention_layers, config.max_cache_length,
                                              kv_dim, stream, cache_dtype);
    if (!kv->ok())
        throw std::runtime_error("qwen3_8 failed to create KV cache");

    const std::int64_t conv_elements =
        static_cast<std::int64_t>(config.conv_dim / config.tensor_parallel_size) *
        config.mamba_d_conv;
    const std::int64_t state_elements =
        static_cast<std::int64_t>(config.mamba_nheads / config.tensor_parallel_size) *
        config.mamba_head_dim * config.mamba_d_state;
    auto recurrent =
        std::make_unique<Qwen38RecurrentState>(config.num_mamba_layers,
                                               std::vector<Qwen38RecurrentState::TensorSpec>{
                                                   {"conv_state", {conv_elements}, "present_conv"},
                                                   {"ssm_state", {state_elements}, "present_ssm"},
                                               },
                                               stream);
    auto state = std::make_unique<Qwen38HybridState>(std::move(kv), std::move(recurrent));
    if (!state->ok())
        throw std::runtime_error("qwen3_8 failed to create hybrid state");

    // Built BEFORE state is moved into RecurrentPipeline: Qwen38MtpScheduler
    // stores a reference to the Qwen38HybridState object, not the
    // unique_ptr -- moving a unique_ptr does not change the pointee's
    // address, so the reference stays valid once ownership transfers below.
    std::unique_ptr<Qwen38MtpScheduler> mtp_scheduler;
    if (config.has_mtp) {
        auto mtp_module = load_engine(
            context.backend, require_section(context.reader, "mtp_engine.plan"), "qwen3_8 MTP");
        // New bundles orchestrate repeated single-step drafts in C++.
        // Preserve the in-graph execution path for previously built bundles.
        std::unique_ptr<ITrtModule> draft_chain_module;
        if (context.reader.find_section("mtp_draft_chain_engine.plan")) {
            draft_chain_module = load_engine(
                context.backend, require_section(context.reader, "mtp_draft_chain_engine.plan"),
                "qwen3_8 MTP draft chain");
        }
        auto multi_token_module =
            load_engine(context.backend, require_section(context.reader, "multi_token_engine.plan"),
                        "qwen3_8 multi-token verification");
        std::unique_ptr<ITrtModule> mtp_prefill_module;
        if (config.prefill_chunk_len > 0 &&
            context.reader.find_section("mtp_prefill_engine.plan")) {
            mtp_prefill_module = load_engine(
                context.backend, require_section(context.reader, "mtp_prefill_engine.plan"),
                "qwen3_8 MTP prefill");
        }
        mtp_scheduler = std::make_unique<Qwen38MtpScheduler>(
            std::move(mtp_module), std::move(draft_chain_module), std::move(multi_token_module),
            *state, config.hidden_size, config.vocab_size, config.mtp_num_draft_tokens, stream,
            std::move(mtp_prefill_module), config.prefill_chunk_len);
    }

    std::unique_ptr<Qwen38PrefillRunner> prefill_runner;
    if (config.prefill_chunk_len > 0) {
        auto prefill_module =
            load_engine(context.backend, require_section(context.reader, "prefill_engine.plan"),
                        "qwen3_8 prefill");
        prefill_runner = std::make_unique<Qwen38PrefillRunner>(
            std::move(prefill_module), *state, config.prefill_chunk_len, config.hidden_size,
            config.vocab_size, stream);
    }

    std::unique_ptr<Qwen38ExternalDraft> external_draft;
    if (context.reader.find_section("native_dspark.json")) {
        if (config.has_mtp || config.precision != "fp16" ||
            context.reader.find_section("native_dflash2.json"))
            throw std::runtime_error(
                "Qwen3.8 DSpark requires FP16 without another speculative variant");
        const auto j =
            nlohmann::json::parse(require_text_section(context.reader, "native_dspark.json"));
        const auto layers = require_value<std::vector<int32_t>>(j, "feature_layers");
        auto sorted = layers;
        std::sort(sorted.begin(), sorted.end());
        if (!j.is_object() || j.size() != 10 || require_value<int32_t>(j, "schema_version") != 1 ||
            require_value<int32_t>(j, "hidden_size") != config.hidden_size ||
            require_value<int32_t>(j, "vocab_size") != config.vocab_size ||
            require_value<int32_t>(j, "max_cache_length") != config.max_cache_length ||
            sorted.empty() || sorted.front() < 0 || sorted.back() >= config.num_layers ||
            std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::runtime_error("Invalid Qwen3.8 DSpark metadata/target agreement");
        Qwen38DSparkConfig draft_config{
            config.hidden_size,
            config.vocab_size,
            config.max_cache_length,
            static_cast<int32_t>(layers.size()),
            require_value<int32_t>(j, "rows"),
            require_value<int32_t>(j, "mask_token_id"),
            require_value<bool>(j, "sample_from_anchor"),
            require_value<bool>(j, "causal"),
            require_value<bool>(j, "confidence"),
        };
        auto draft =
            load_engine(context.backend, require_section(context.reader, "native_dspark.plan"),
                        "qwen3_8 native DSpark");
        auto heads = load_engine(context.backend,
                                 require_section(context.reader, "native_dspark_heads.plan"),
                                 "qwen3_8 DSpark heads");
        external_draft = std::make_unique<Qwen38DSpark>(
            draft_config, std::move(draft), std::move(heads),
            require_section(context.reader, "native_dspark_embeddings.bin"));
    }

    if (context.reader.find_section("native_dflash2.json")) {
        if (config.has_mtp || config.precision != "fp16")
            throw std::runtime_error("Qwen3.8 DFlash2 requires FP16 without embedded MTP");
        const auto j =
            nlohmann::json::parse(require_text_section(context.reader, "native_dflash2.json"));
        const auto layers = require_value<std::vector<int32_t>>(j, "feature_layers");
        auto sorted = layers;
        std::sort(sorted.begin(), sorted.end());
        if (!j.is_object() || j.size() != 10 || require_value<int32_t>(j, "schema_version") != 1 ||
            require_value<int32_t>(j, "hidden_size") != config.hidden_size ||
            require_value<int32_t>(j, "vocab_size") != config.vocab_size ||
            require_value<int32_t>(j, "max_cache_length") != config.max_cache_length ||
            require_value<int32_t>(j, "rows") != 8 || sorted.empty() || sorted.front() < 0 ||
            sorted.back() >= config.num_layers ||
            std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::runtime_error("Invalid Qwen3.8 DFlash2 metadata/target agreement");
        Qwen38DFlash2Config draft_config{
            config.hidden_size,
            config.vocab_size,
            config.max_cache_length,
            static_cast<int32_t>(layers.size()),
            require_value<int32_t>(j, "mask_token_id"),
            require_value<int32_t>(j, "window"),
            require_value<int32_t>(j, "rank"),
            require_value<int32_t>(j, "top_k"),
        };
        auto draft =
            load_engine(context.backend, require_section(context.reader, "native_dflash2.plan"),
                        "qwen3_8 native DFlash2");
        auto selector = load_engine(context.backend,
                                    require_section(context.reader, "native_dflash2_selector.plan"),
                                    "qwen3_8 DFlash2 selector");
        external_draft = std::make_unique<Qwen38DFlash2>(
            draft_config, std::move(draft), std::move(selector),
            require_section(context.reader, "native_dflash2_embeddings.bin"));
    }

    RecurrentGenConfig generation;
    generation.vocab_size = config.vocab_size;
    generation.id_bos = config.bos_token_id;
    generation.id_eos_ids = config.eos_token_ids;
    generation.has_position_input = decoder->has_input("position_id");
    generation.chat_template_format =
        qwen3_8_detect_chat_template_format(chat_template(context.reader));

    auto tokenizer = create_tokenizer(context.reader);
    auto pipeline = std::make_unique<RecurrentPipeline>(
        std::move(decoder), std::move(state), std::move(generation), stream, "Qwen3.8", tokenizer,
        std::string{}, nullptr, std::move(mtp_scheduler), std::move(prefill_runner),
        std::move(external_draft));
    if (context.reader.find_section("native_visual.json")) {
        auto visual = load_engine(
            context.backend,
            require_section(context.reader,
                            parallel ? "native_visual.rank" + std::to_string(rank) + ".plan"
                                     : "native_visual.plan"),
            "qwen3_8 native visual");
        return create_native_media_task(context.reader, std::move(pipeline), std::move(visual),
                                        std::move(tokenizer));
    }
    if (context.reader.info().task != "text_generation")
        throw std::runtime_error("Qwen native media bundle is missing its visual metadata");
    return pipeline.release();
}

ITask* create(const FamilyContext& context) {
    if (context.reader.find_section("edge_llm.json")) {
#ifdef TRTMC_HAS_EDGE_LLM
        return edge_llm::create(context.reader);
#else
        throw std::runtime_error("Qwen3.8 Edge bundle requires a runtime configured with "
                                 "-DTRTMC_ENABLE_EDGELLM=ON; rebuild and install Model Connect");
#endif
    }
    const auto config = parse_runtime_config(context.reader);
    if (config.tensor_parallel_size == 2) {
        if (context.reader.find_section("native_dspark.json") ||
            context.reader.find_section("native_dflash2.json"))
            throw std::runtime_error("Qwen native TP2 does not map speculative companions");
        // The factory can be called again after a failed request. FamilyContext
        // borrows a reader used only during loading; retain its cheap metadata/path
        // copy, not a dangling reference or another in-memory copy of the plans.
        auto bundle = std::make_shared<BundleReader>(context.reader);
        return create_native_tensor_parallel_task(
            [bundle, backend = &context.backend, config](int rank,
                                                         const ModuleCreateOptions& options) {
                const FamilyContext local{*bundle, *backend, 0};
                return create_native(local, config, rank, options);
            },
            context.reader.find_section("native_visual.json") != nullptr);
    }
    return create_native(context, config, 0, {});
}

} // namespace trtmc::qwen3_8

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen3_8 does not support --kv-cache-size");
    return trtmc::qwen3_8::create(context);
}
