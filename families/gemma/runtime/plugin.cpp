/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gemma/runtime/chat_templates.h"
#include "families/gemma/runtime/distributed_runtime.h"
#include "families/gemma/runtime/kv_cache.h"
#include "families/gemma/runtime/pipeline.h"
#include "families/gemma/runtime/plugin_helpers.h"
#include "families/gemma/runtime/tensor_names.h"
#include "trtmc/runtime/family_factory.h"
#ifdef TRTMC_HAS_EDGE_LLM
#include "families/gemma/runtime/edge_llm/adapter.h"
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::gemma {

namespace {

struct RuntimeConfig {
    std::int32_t hidden_size;
    std::int32_t num_layers;
    std::int32_t num_heads;
    std::int32_t num_key_value_heads;
    std::int32_t head_dim;
    std::int32_t vocab_size;
    std::int32_t bos_token_id;
    std::int32_t eos_token_id;
    std::int32_t pad_token_id;
    std::int32_t max_cache_length;
    std::int32_t tensor_parallel_size;
    std::string tensor_parallel_mode;
    std::string precision;
    std::string decoder_engine_layout;
    // Empty unless the bundle names more than one stop token.
    std::vector<std::int32_t> eos_token_ids;
    // Gemma4 owns only a prefix of cache layers, with local/global head widths.
    std::vector<std::int32_t> kv_cache_dimensions;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("gemma runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("gemma runtime.json has invalid '") + name + "'");
    }
}

std::int32_t require_eos_token(const nlohmann::json& json) {
    if (!json.contains("eos_token_id"))
        throw std::runtime_error("gemma runtime.json missing 'eos_token_id'");
    const auto& value = json.at("eos_token_id");
    if (value.is_number_integer())
        return value.get<std::int32_t>();
    if (value.is_array() && !value.empty() && value.front().is_number_integer())
        return value.front().get<std::int32_t>();
    throw std::runtime_error("gemma runtime.json has invalid 'eos_token_id'");
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const std::string text = require_text_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("gemma invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("gemma runtime.json must be an object");
    const bool multi_eos = json.contains("eos_token_ids");
    if (json.size() != static_cast<std::size_t>(14 + (multi_eos ? 1 : 0) +
                                                (json.contains("kv_cache_dimensions") ? 1 : 0)))
        throw std::runtime_error("gemma runtime.json has an unexpected field set");

    RuntimeConfig config{
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_attention_heads"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_eos_token(json),
        require_value<std::int32_t>(json, "pad_token_id"),
        require_value<std::int32_t>(json, "max_cache_length"),
        require_value<std::int32_t>(json, "tensor_parallel_size"),
        require_value<std::string>(json, "tensor_parallel_mode"),
        require_value<std::string>(json, "precision"),
        require_value<std::string>(json, "decoder_engine_layout"),
        {},
        {},
    };
    if (config.hidden_size <= 0 || config.num_layers <= 0 || config.num_heads <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 || config.vocab_size <= 0 ||
        config.max_cache_length <= 0 || config.tensor_parallel_size <= 0 ||
        config.num_key_value_heads * config.head_dim > config.hidden_size) {
        throw std::runtime_error("gemma runtime.json contains invalid dimensions");
    }
    if (json.contains("kv_cache_dimensions")) {
        config.kv_cache_dimensions =
            require_value<std::vector<std::int32_t>>(json, "kv_cache_dimensions");
        if (config.tensor_parallel_size != 1 || config.kv_cache_dimensions.empty() ||
            config.kv_cache_dimensions.size() > static_cast<std::size_t>(config.num_layers) ||
            std::any_of(config.kv_cache_dimensions.begin(), config.kv_cache_dimensions.end(),
                        [](std::int32_t dim) { return dim <= 0; }))
            throw std::runtime_error("gemma runtime.json has invalid KV cache dimensions");
    }
    if (config.num_key_value_heads % config.tensor_parallel_size != 0)
        throw std::runtime_error("gemma KV heads must be divisible by tensor parallel size");
    if (multi_eos) {
        config.eos_token_ids = require_value<std::vector<std::int32_t>>(json, "eos_token_ids");
        if (config.eos_token_ids.empty())
            throw std::runtime_error("gemma runtime.json has an empty 'eos_token_ids'");
        for (const std::int32_t token_id : config.eos_token_ids) {
            if (token_id < 0 || token_id >= config.vocab_size)
                throw std::runtime_error("gemma runtime.json has an out-of-range 'eos_token_ids'");
        }
    }
    if (config.precision != "fp16" && config.precision != "bf16" && config.precision != "fp32") {
        throw std::runtime_error("gemma runtime.json contains invalid precision");
    }
    if (config.decoder_engine_layout != "split" && config.decoder_engine_layout != "dual_profile" &&
        config.decoder_engine_layout != "single_profile") {
        throw std::runtime_error("gemma runtime.json contains invalid decoder_engine_layout");
    }
    if (config.decoder_engine_layout != "dual_profile" && config.tensor_parallel_size != 1)
        throw std::runtime_error("gemma single-rank engine layout requires tensor_parallel_size=1");
    const std::string expected_mode =
        config.tensor_parallel_size > 1 ? "tensor_parallel" : "single";
    if (config.tensor_parallel_mode != expected_mode)
        throw std::runtime_error("gemma runtime.json has inconsistent tensor_parallel_mode");
    return config;
}

DType cache_dtype(const std::string& precision) {
    if (precision == "fp16")
        return DType::kFloat16;
    if (precision == "bf16")
        return DType::kBFloat16;
    return DType::kFloat32;
}

GemmaKvCacheNames make_kv_names(std::int32_t num_layers) {
    GemmaKvCacheNames names;
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        names.cache_k.push_back(gemma_layer_tensor_name("cache_k", layer));
        names.cache_v.push_back(gemma_layer_tensor_name("cache_v", layer));
        names.present_k.push_back(gemma_layer_tensor_name("present_k", layer));
        names.present_v.push_back(gemma_layer_tensor_name("present_v", layer));
    }
    return names;
}

std::string chat_template(const BundleReader& bundle) {
    const auto* section = bundle.find_section("chat_template.jinja");
    if (section != nullptr && section->length > 0) {
        const auto data = bundle.read_section("chat_template.jinja");
        return {data.begin(), data.end()};
    }

    section = bundle.find_section("tokenizer_config.json");
    if (section == nullptr || section->length == 0)
        return {};
    const auto data = bundle.read_section("tokenizer_config.json");
    try {
        const auto config = nlohmann::json::parse(data.begin(), data.end());
        const auto template_value = config.find("chat_template");
        if (template_value != config.end() && template_value->is_string())
            return template_value->get<std::string>();
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("gemma invalid tokenizer_config.json: " +
                                 std::string(error.what()));
    }
    return {};
}

struct DecoderModules {
    std::unique_ptr<ITrtModule> prefill;
    std::unique_ptr<ITrtModule> decode;
    std::shared_ptr<void> distributed_owner;
};

DecoderModules load_modules(const FamilyContext& context, const RuntimeConfig& config) {
    DistributedRuntimeGroup group = initialize_tensor_parallel_group(config.tensor_parallel_size);
    DecoderModules modules;
    modules.distributed_owner = group.owner;
    if (config.decoder_engine_layout == "split") {
        modules.decode = load_engine(
            context.backend, require_section(context.reader, "engine.plan"), "gemma decoder");
        modules.prefill = load_engine(
            context.backend, require_section(context.reader, "prefill.plan"), "gemma prefill");
        return modules;
    }

    ModuleCreateOptions options;
    if (config.tensor_parallel_size > 1) {
        options.distributed_communicator = group.communicator;
        options.distributed_owner = group.owner;
    }
    const std::string section = config.tensor_parallel_size > 1
                                    ? "engine.rank" + std::to_string(group.rank) + ".plan"
                                    : std::string("engine.plan");
    const auto& plan = require_section(context.reader, section);
    if (config.decoder_engine_layout == "single_profile") {
        modules.decode = context.backend.create_module(plan.data(), plan.size(), options);
        if (!modules.decode || !modules.decode->ok() ||
            modules.decode->optimization_profile_count() != 1)
            throw std::runtime_error("gemma single-profile engine did not create a valid context");
        return modules;
    }
    auto dual = context.backend.create_dual_profile_modules(plan.data(), plan.size(), options);
    if (dual.decode == nullptr || !dual.decode->ok() || dual.prefill == nullptr ||
        !dual.prefill->ok()) {
        throw std::runtime_error("gemma dual-profile engine did not create both contexts");
    }
    modules.prefill = std::move(dual.prefill);
    modules.decode = std::move(dual.decode);
    return modules;
}

} // namespace

template <typename T>
std::vector<T> audio_constants(const BundleReader& bundle, const char* name) {
    const auto& bytes = require_section(bundle, name);
    if (bytes.size() % sizeof(T))
        throw std::runtime_error("Gemma4 malformed audio frontend constants");
    std::vector<T> result(bytes.size() / sizeof(T));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

ITask* create(const FamilyContext& context) {
    if (context.reader.find_section("edge_llm.json")) {
#ifdef TRTMC_HAS_EDGE_LLM
        return edge_llm::create(context.reader);
#else
        throw std::runtime_error("Gemma4 Edge bundle requires TRTMC_ENABLE_EDGELLM=ON");
#endif
    }
    const RuntimeConfig config = parse_runtime_config(context.reader);
    DecoderModules modules = load_modules(context, config);
    const cudaStream_t stream = modules.decode->stream();
    const std::int32_t kv_dim =
        (config.num_key_value_heads / config.tensor_parallel_size) * config.head_dim;
    const auto cache_layers = config.kv_cache_dimensions.empty()
                                  ? config.num_layers
                                  : static_cast<std::int32_t>(config.kv_cache_dimensions.size());
    auto state = std::make_unique<GemmaKvCache>(
        cache_layers, config.max_cache_length, kv_dim, stream, cache_dtype(config.precision),
        make_kv_names(cache_layers), config.kv_cache_dimensions);
    if (!state->ok())
        throw std::runtime_error("gemma failed to create KV cache");

    GemmaTextGenConfig text_config;
    text_config.primary_task = context.reader.info().task;
    text_config.vocab_size = config.vocab_size;
    text_config.id_bos = config.bos_token_id;
    text_config.id_eos = config.eos_token_id;
    text_config.id_eos_ids = config.eos_token_ids;
    text_config.chat_template_format =
        gemma_detect_chat_template_format(chat_template(context.reader));
    text_config.prefill_max_length = config.max_cache_length;
    text_config.num_layers = cache_layers;
    text_config.single_context = config.decoder_engine_layout == "single_profile";

    std::unique_ptr<GemmaVisionEncoder> vision;
    if (context.reader.find_section("vision.json")) {
        const auto media =
            nlohmann::json::parse(require_text_section(context.reader, "vision.json"));
        GemmaImageConfig image{
            require_value<int>(media, "patch_size"),  require_value<int>(media, "pooling_kernel"),
            require_value<int>(media, "soft_tokens"), require_value<int>(media, "hidden_size"),
            require_value<int>(media, "image_token"), require_value<int>(media, "begin_token"),
            require_value<int>(media, "end_token"),   media.value("merged_patches", false)};
        if (image.patch_size <= 0 || image.patch_size > 64 || image.pooling_kernel < 2 ||
            image.pooling_kernel > 16 || image.soft_tokens <= 0 || image.soft_tokens > 1120 ||
            image.hidden_size != config.hidden_size || !image.valid_token_ids(config.vocab_size))
            throw std::runtime_error("Gemma4 invalid vision configuration");
        vision = std::make_unique<GemmaVisionEncoder>(
            load_engine(context.backend, require_section(context.reader, "vision.plan"),
                        "gemma4 vision"),
            image);
    }
    std::unique_ptr<GemmaAudioEncoder> audio;
    if (context.reader.find_section("audio.json")) {
        const auto media =
            nlohmann::json::parse(require_text_section(context.reader, "audio.json"));
        GemmaAudioConfig acoustic;
        acoustic.sample_rate = require_value<int>(media, "sample_rate");
        acoustic.mel_bins = require_value<int>(media, "mel_bins");
        acoustic.max_frames = require_value<int>(media, "max_frames");
        acoustic.hidden_size = require_value<int>(media, "hidden_size");
        acoustic.audio_token = require_value<int>(media, "audio_token");
        acoustic.begin_token = require_value<int>(media, "begin_token");
        acoustic.end_token = require_value<int>(media, "end_token");
        acoustic.waveform = media.value("waveform", false);
        if (!acoustic.waveform) {
            acoustic.frame_length = require_value<int>(media, "frame_length");
            acoustic.hop_length = require_value<int>(media, "hop_length");
            acoustic.fft_length = require_value<int>(media, "fft_length");
            acoustic.input_scale = require_value<double>(media, "input_scale");
            acoustic.preemphasis = require_value<double>(media, "preemphasis");
            acoustic.preemphasis_htk = require_value<bool>(media, "preemphasis_htk");
            acoustic.mel_floor = require_value<double>(media, "mel_floor");
            acoustic.mean = require_value<std::vector<double>>(media, "mean");
            acoustic.stddev = require_value<std::vector<double>>(media, "std");
            acoustic.window = audio_constants<float>(context.reader, "audio_window.f32");
            acoustic.mel_filters = audio_constants<double>(context.reader, "audio_mel.f64");
        }
        if (acoustic.hidden_size != config.hidden_size ||
            int64_t{acoustic.max_frames} !=
                int64_t{config.max_cache_length} * (acoustic.waveform ? 1 : 4) ||
            acoustic.audio_token < 0 || acoustic.audio_token >= config.vocab_size ||
            acoustic.begin_token < 0 || acoustic.begin_token >= config.vocab_size ||
            acoustic.end_token < 0 || acoustic.end_token >= config.vocab_size)
            throw std::runtime_error("Gemma4 invalid audio configuration");
        audio = std::make_unique<GemmaAudioEncoder>(
            load_engine(context.backend, require_section(context.reader, "audio.plan"),
                        "gemma4 audio"),
            std::move(acoustic));
    }
    std::unique_ptr<ITrtModule> assistant;
    if (context.reader.find_section("assistant.json")) {
        const auto metadata =
            nlohmann::json::parse(require_text_section(context.reader, "assistant.json"));
        const auto variant = require_value<std::string>(metadata, "variant");
        if ((variant != "mtp" && variant != "dflash" && variant != "dspark" &&
             variant != "eagle3") ||
            require_value<int>(metadata, "hidden_size") != config.hidden_size ||
            require_value<int>(metadata, "vocab_size") != config.vocab_size ||
            config.tensor_parallel_size != 1 || vision || audio)
            throw std::runtime_error("Gemma4 invalid native companion configuration");
        if (variant == "dflash" || variant == "dspark" || variant == "eagle3") {
            const auto layers = require_value<std::vector<int>>(metadata, "target_layers");
            const auto block = require_value<int>(metadata, "block_size");
            const auto mask =
                variant == "eagle3" ? 0 : require_value<int>(metadata, "mask_token_id");
            const bool skip_first = metadata.value("skip_first", false);
            if (layers.empty() ||
                (skip_first ? (variant != "dflash" || block != 16) : block != 7) || mask < 0 ||
                mask >= config.vocab_size || !std::is_sorted(layers.begin(), layers.end()) ||
                std::adjacent_find(layers.begin(), layers.end()) != layers.end() ||
                layers.front() < 0 || layers.back() >= config.num_layers - 1)
                throw std::runtime_error("Gemma4 invalid native block-draft configuration");
            text_config.draft_markov = variant == "dspark";
            text_config.draft_eagle = variant == "eagle3";
            text_config.draft_skip_first = skip_first;
            text_config.draft_block_size = block;
            text_config.draft_mask_token_id = mask;
            text_config.draft_feature_width =
                static_cast<int32_t>(layers.size()) * config.hidden_size;
        } else {
            const auto layers = require_value<nlohmann::json>(metadata, "target_layers");
            if (!layers.is_object() || layers.empty())
                throw std::runtime_error("Gemma4 MTP shared target layers are missing");
            for (const auto& [kind, layer] : layers.items()) {
                if ((kind != "sliding_attention" && kind != "full_attention") ||
                    !layer.is_number_integer() || layer.get<int>() < 0 ||
                    layer.get<int>() >= cache_layers)
                    throw std::runtime_error("Gemma4 MTP shared target layer is invalid");
                text_config.mtp_target_layers.emplace_back(kind, layer.get<int>());
            }
            text_config.mtp_hidden_size = config.hidden_size;
        }
        assistant = load_engine(context.backend, require_section(context.reader, "assistant.plan"),
                                variant == "mtp" ? "gemma4 MTP assistant" : "gemma4 block draft");
    }
    return new GemmaTextGenerationPipeline(std::move(modules.decode), std::move(state),
                                           std::move(text_config), create_tokenizer(context.reader),
                                           std::move(modules.prefill),
                                           std::move(modules.distributed_owner), std::move(vision),
                                           std::move(audio), std::move(assistant));
}

} // namespace trtmc::gemma

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("gemma does not support --kv-cache-size");
    return trtmc::gemma::create(context);
}
