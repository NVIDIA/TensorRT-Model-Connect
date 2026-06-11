/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// NemotronSpeechStreamingPlugin: handles
// "nemotron_speech_streaming_speech_to_text_rnnt" strategy.

#include "plugin_helpers.h"
#include "runtime/models/nemotron_speech_streaming/pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cctype>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
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

// Best-effort JSON object parser: pulls "key": <int> pairs out of the object
// stored under `key` in `json`. Returns an empty map if the key is missing or
// the value is not an object. Sufficient for the small prompt_dictionary
// objects emitted by the Python builder.
std::unordered_map<std::string, int32_t>
extract_string_int_map(const std::string& json, const std::string& key) {
    std::unordered_map<std::string, int32_t> out;
    const std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos)
        return out;
    pos = json.find('{', pos);
    if (pos == std::string::npos)
        return out;
    int depth = 1;
    auto end = pos + 1;
    while (end < json.size() && depth > 0) {
        if (json[end] == '{')
            ++depth;
        else if (json[end] == '}')
            --depth;
        ++end;
    }
    if (depth != 0)
        return out;
    auto i = pos + 1;
    while (i < end - 1) {
        i = json.find('"', i);
        if (i == std::string::npos || i >= end - 1)
            break;
        auto ke = json.find('"', i + 1);
        if (ke == std::string::npos || ke >= end - 1)
            break;
        std::string k_str = json.substr(i + 1, ke - i - 1);
        auto colon = json.find(':', ke + 1);
        if (colon == std::string::npos || colon >= end - 1)
            break;
        auto j = colon + 1;
        while (j < end - 1 && std::isspace(static_cast<unsigned char>(json[j])))
            ++j;
        std::string num;
        while (j < end - 1 &&
               (std::isdigit(static_cast<unsigned char>(json[j])) || json[j] == '-')) {
            num.push_back(json[j]);
            ++j;
        }
        if (!num.empty()) {
            try {
                out.emplace(std::move(k_str), std::stoi(num));
            } catch (...) {
                // ignore unparseable entry
            }
        }
        i = j;
    }
    return out;
}

} // namespace

class NemotronSpeechStreamingPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        DistributedRuntimeGroup tp_group;
        ModuleCreateOptions pred_opts = opts;
        std::string pred_section = "engine_plan";
        if (tp_config.enabled) {
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
            pred_opts.distributed_communicator = tp_group.communicator;
            pred_opts.distributed_owner = tp_group.owner;
            pred_section = tp_engine_section_name(tp_group.rank);
        }

        auto enc_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "vision_engine_plan"), "rnnt encoder", opts);
        auto pred_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, pred_section), "rnnt predictor", pred_opts);
        auto joint_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "joint_engine_plan"), "rnnt joint", opts);
        std::unique_ptr<TrtModule> prompt_kernel_module;
        const bool has_prompt_kernel =
            ctx.config_json.find("\"rnnt_has_prompt_kernel\": true") != std::string::npos ||
            ctx.config_json.find("\"rnnt_has_prompt_kernel\":true") != std::string::npos;
        if (has_prompt_kernel) {
            const auto* pk_plan = find_section(ctx.bundle, "prompt_kernel_plan");
            if (!pk_plan)
                throw std::runtime_error(
                    "rnnt_has_prompt_kernel=true but bundle missing prompt_kernel_plan section");
            auto pk_loaded =
                load_trt_module_from_plan(ctx.backend, pk_plan, "rnnt prompt_kernel", opts);
            prompt_kernel_module = std::move(pk_loaded.module);
        }
        std::map<int32_t, std::string> streaming_encoder_sections;
        std::map<int32_t, std::string> streaming_first_encoder_sections;
        auto right_contexts =
            extract_json_int_array(ctx.config_json, "rnnt_streaming_right_contexts");
        if (right_contexts.empty())
            right_contexts = {13, 6, 1, 0};
        for (int32_t right_context : right_contexts) {
            const std::string section_name =
                "streaming_encoder_plan_ctx" + std::to_string(right_context);
            const auto* plan = find_section(ctx.bundle, section_name);
            if (plan)
                streaming_encoder_sections.emplace(right_context, section_name);
            const std::string first_section_name =
                "streaming_encoder_first_plan_ctx" + std::to_string(right_context);
            const auto* first_plan = find_section(ctx.bundle, first_section_name);
            if (first_plan)
                streaming_first_encoder_sections.emplace(right_context, first_section_name);
        }

        const auto& json = ctx.config_json;
        RnntConfig cfg;
        cfg.sample_rate = extract_json_int(json, "mel_sampling_rate", 16000);
        cfg.num_mel_bins = extract_json_int(json, "num_mel_bins", 128);
        cfg.mel_n_fft = extract_json_int(json, "mel_n_fft", 512);
        cfg.mel_win_length = extract_json_int(json, "mel_win_length", 400);
        cfg.mel_hop_length = extract_json_int(json, "mel_hop_length", 160);
        cfg.mel_chunk_length = extract_json_int(json, "mel_chunk_length", 30);
        cfg.mel_length = extract_json_int(json, "mel_length", 3000);
        cfg.encoder_hidden_size =
            extract_json_int(json, "rnnt_encoder_hidden_size", ctx.config.hidden_size);
        cfg.pred_hidden_size =
            extract_json_int(json, "rnnt_pred_hidden_size", ctx.config.hidden_size);
        cfg.pred_num_layers = extract_json_int(json, "rnnt_pred_num_layers", ctx.config.num_layers);
        cfg.encoder_layers = extract_json_int(json, "rnnt_encoder_layers", 0);
        cfg.vocab_size = extract_json_int(json, "rnnt_vocab_size", ctx.config.vocab_size);
        cfg.blank_id = extract_json_int(json, "rnnt_blank_id", cfg.vocab_size);
        cfg.max_symbols_per_step = extract_json_int(json, "rnnt_max_symbols_per_step", 10);
        cfg.encoder_seq_len = extract_json_int(json, "max_source_positions", cfg.mel_length / 8);
        cfg.att_context_left = extract_json_int(json, "rnnt_att_context_left", 70);
        cfg.att_context_right = extract_json_int(json, "rnnt_att_context_right", 13);
        cfg.subsampling_factor = extract_json_int(json, "subsampling_factor", 8);
        cfg.streaming_cache_left = extract_json_int(json, "rnnt_streaming_cache_left", 70);
        cfg.streaming_time_cache = extract_json_int(json, "rnnt_streaming_time_cache", 8);
        cfg.streaming_pre_encode_cache =
            extract_json_int(json, "rnnt_streaming_pre_encode_cache", 9);
        cfg.streaming_drop_pre_encoded =
            extract_json_int(json, "rnnt_streaming_drop_pre_encoded", 2);
        cfg.causal_downsampling =
            json.find("\"rnnt_causal_downsampling\": true") != std::string::npos;
        cfg.has_prompt_kernel = has_prompt_kernel;
        cfg.num_prompts = extract_json_int(json, "rnnt_num_prompts", 0);
        cfg.prompt_dictionary = extract_string_int_map(json, "rnnt_prompt_dictionary");
        cfg.supported_right_contexts = right_contexts;

        auto mel_fb = load_mel_filterbank(ctx.bundle);
        auto tok = create_tokenizer_from_bundle(ctx.bundle);
        cudaStream_t stream = pred_loaded.module->stream();

        return std::make_unique<RnntPipeline>(
            std::move(enc_loaded.module), std::move(pred_loaded.module),
            std::move(joint_loaded.module), std::move(prompt_kernel_module),
            std::move(streaming_encoder_sections), ctx.backend, opts,
            std::move(streaming_first_encoder_sections), ctx.bundle_path, std::move(cfg),
            std::move(mel_fb), stream, std::move(tok), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_nemotron_speech_streaming_plugin,
                                       NemotronSpeechStreamingPlugin,
                                       "nemotron_speech_streaming_speech_to_text_rnnt");

} // namespace trtmc
