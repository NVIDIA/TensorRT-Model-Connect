// RnntPlugin: handles "speech_to_text_rnnt" strategy.

#include "runtime/models/rnnt/pipeline.h"
#include "runtime/plugins/shared/audio_helpers.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

class RnntPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto enc_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "vision_engine_plan"), "rnnt encoder", opts);
        auto pred_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "rnnt predictor", opts);
        auto joint_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "joint_engine_plan"), "rnnt joint", opts);
        std::map<int32_t, std::string> streaming_encoder_sections;
        std::map<int32_t, std::string> streaming_first_encoder_sections;
        for (int32_t right_context : {13, 6, 1, 0}) {
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

        auto mel_fb = load_mel_filterbank(ctx.bundle);
        auto tok = create_tokenizer_from_bundle(ctx.bundle);
        cudaStream_t stream = pred_loaded.module->stream();

        return std::make_unique<RnntPipeline>(
            std::move(enc_loaded.module), std::move(pred_loaded.module),
            std::move(joint_loaded.module), std::move(streaming_encoder_sections), ctx.backend,
            opts, std::move(streaming_first_encoder_sections), ctx.bundle_path, std::move(cfg),
            std::move(mel_fb), stream, std::move(tok), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_rnnt_plugin, RnntPlugin, "speech_to_text_rnnt");

} // namespace trtmc
