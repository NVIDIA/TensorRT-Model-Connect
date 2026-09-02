/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// ParakeetTDTPlugin: handles
// "parakeet_tdt_speech_to_text" strategy.

#include "plugin_helpers.h"
#include "runtime/models/parakeet_tdt/pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>
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

// Parse the small `prompt_dictionary` object emitted by the Python builder.
// Defers the actual parsing to nlohmann/json after extracting the object's
// text via the shared helper, so the schema-tolerant behaviour matches the
// rest of the bundle-config readers.
std::unordered_map<std::string, int32_t> extract_string_int_map(const std::string& json,
                                                                const std::string& key) {
    const auto obj_text = extract_json_object_text(json, key);
    if (obj_text.empty())
        return {};
    try {
        const auto parsed = nlohmann::json::parse(obj_text);
        if (!parsed.is_object())
            return {};
        std::unordered_map<std::string, int32_t> out;
        for (auto it = parsed.begin(); it != parsed.end(); ++it) {
            if (it.value().is_number_integer())
                out.emplace(it.key(), it.value().get<int32_t>());
        }
        return out;
    } catch (const nlohmann::json::exception&) {
        return {};
    }
}

} // namespace

class ParakeetTDTPlugin final : public IPipelinePlugin {
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
            ctx.backend, find_section(ctx.bundle, "vision_engine_plan"), "tdt encoder", opts);
        auto pred_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, pred_section), "tdt predictor", pred_opts);
        auto joint_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "joint_engine_plan"), "tdt joint", opts);
        std::unique_ptr<TrtModule> prompt_kernel_module;
        const bool has_prompt_kernel =
            extract_json_bool(ctx.config_json, "tdt_has_prompt_kernel", false);
        if (has_prompt_kernel) {
            const auto* pk_plan = find_section(ctx.bundle, "prompt_kernel_plan");
            if (!pk_plan)
                throw std::runtime_error(
                    "tdt_has_prompt_kernel=true but bundle missing prompt_kernel_plan section");
            auto pk_loaded =
                load_trt_module_from_plan(ctx.backend, pk_plan, "tdt prompt_kernel", opts);
            prompt_kernel_module = std::move(pk_loaded.module);
        }
        std::map<int32_t, std::string> streaming_encoder_sections;
        std::map<int32_t, std::string> streaming_first_encoder_sections;
        auto right_contexts =
            extract_json_int_array(ctx.config_json, "tdt_streaming_right_contexts");
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
        TdtConfig cfg;
        cfg.sample_rate = extract_json_int(json, "mel_sampling_rate", 16000);
        cfg.num_mel_bins = extract_json_int(json, "num_mel_bins", 128);
        cfg.mel_n_fft = extract_json_int(json, "mel_n_fft", 512);
        cfg.mel_win_length = extract_json_int(json, "mel_win_length", 400);
        cfg.mel_hop_length = extract_json_int(json, "mel_hop_length", 160);
        cfg.mel_chunk_length = extract_json_int(json, "mel_chunk_length", 30);
        cfg.mel_preemph = extract_json_float(json, "mel_preemph", 0.97F);
        cfg.mel_length = extract_json_int(json, "mel_length", 3000);
        cfg.encoder_hidden_size =
            extract_json_int(json, "tdt_encoder_hidden_size", ctx.config.hidden_size);
        cfg.pred_hidden_size =
            extract_json_int(json, "tdt_pred_hidden_size", ctx.config.hidden_size);
        cfg.pred_num_layers = extract_json_int(json, "tdt_pred_num_layers", ctx.config.num_layers);
        cfg.encoder_layers = extract_json_int(json, "tdt_encoder_layers", 0);
        cfg.vocab_size = extract_json_int(json, "tdt_vocab_size", ctx.config.vocab_size);
        cfg.blank_id = extract_json_int(json, "tdt_blank_id", cfg.vocab_size);
        cfg.max_symbols_per_step = extract_json_int(json, "tdt_max_symbols_per_step", 10);
        cfg.duration_values = extract_json_int_array(json, "tdt_duration_values");
        if (cfg.duration_values.empty())
            cfg.duration_values = {0, 1, 2, 3, 4};
        cfg.encoder_seq_len = extract_json_int(json, "max_source_positions", cfg.mel_length / 8);
        cfg.att_context_left = extract_json_int(json, "tdt_att_context_left", -1);
        cfg.att_context_right = extract_json_int(json, "tdt_att_context_right", -1);
        cfg.subsampling_factor = extract_json_int(json, "subsampling_factor", 8);
        cfg.streaming_cache_left = extract_json_int(json, "tdt_streaming_cache_left", 70);
        cfg.streaming_time_cache = extract_json_int(json, "tdt_streaming_time_cache", 8);
        cfg.streaming_pre_encode_cache =
            extract_json_int(json, "tdt_streaming_pre_encode_cache", 9);
        cfg.streaming_drop_pre_encoded =
            extract_json_int(json, "tdt_streaming_drop_pre_encoded", 2);
        cfg.causal_downsampling = extract_json_bool(json, "tdt_causal_downsampling", false);
        cfg.has_prompt_kernel = has_prompt_kernel;
        cfg.num_prompts = extract_json_int(json, "tdt_num_prompts", 0);
        cfg.prompt_dictionary = extract_string_int_map(json, "tdt_prompt_dictionary");
        cfg.supported_right_contexts = right_contexts;

        auto mel_fb = load_mel_filterbank(ctx.bundle);
        auto tok = create_tokenizer_from_bundle(ctx.bundle);
        cudaStream_t stream = pred_loaded.module->stream();

        return std::make_unique<TdtPipeline>(
            std::move(enc_loaded.module), std::move(pred_loaded.module),
            std::move(joint_loaded.module), std::move(prompt_kernel_module),
            std::move(streaming_encoder_sections), ctx.backend, opts,
            std::move(streaming_first_encoder_sections), ctx.bundle_path, std::move(cfg),
            std::move(mel_fb), stream, std::move(tok), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_parakeet_tdt_plugin, ParakeetTDTPlugin,
                                       "parakeet_tdt_speech_to_text");

} // namespace trtmc
