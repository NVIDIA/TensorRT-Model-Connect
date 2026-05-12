// WhisperPlugin: handles "speech_to_text" strategy.
// Whisper encoder-decoder pipeline with mel spectrogram input.

#include "runtime/models/whisper/pipeline.h"
#include "runtime/plugins/shared/audio_helpers.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

namespace trtmc {

class WhisperPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto& json = ctx.config_json;

        // Load encoder (stored as vision_engine_plan in Whisper bundles)
        const auto* enc_plan = find_section(ctx.bundle, "vision_engine_plan");
        if (!enc_plan || enc_plan->empty())
            enc_plan = find_section(ctx.bundle, "coarse_engine_plan");
        auto enc_loaded = load_trt_module_from_plan(ctx.backend, enc_plan, "whisper encoder", opts);

        // Load decoder (main engine_plan)
        auto dec_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "whisper decoder", opts);

        // Build WhisperConfig
        int32_t encoder_layers = extract_json_int(json, "encoder_layers", ctx.config.num_layers);
        int32_t decoder_layers = extract_json_int(json, "decoder_layers", ctx.config.num_layers);
        int32_t dl = (decoder_layers > 0) ? decoder_layers : ctx.config.num_layers;
        WhisperConfig wc;
        wc.num_mel_bins = extract_json_int(json, "num_mel_bins", 80);
        wc.max_source_positions = extract_json_int(json, "max_source_positions", 1500);
        wc.max_target_positions = extract_json_int(json, "max_target_positions", 448);
        wc.encoder_layers = encoder_layers;
        wc.decoder_layers = dl;
        int32_t eot_token_id = extract_json_int(json, "eot_token_id", -1);
        wc.eot_token_id = (eot_token_id >= 0) ? eot_token_id : ctx.config.id_eos;
        wc.mel_length = extract_json_int(json, "mel_length", 0);
        wc.decoder_start_token_ids = extract_json_int_array(json, "decoder_start_token_ids");

        // Create KvCache for decoder self-attention
        cudaStream_t stream = dec_loaded.module->stream();
        int32_t kv_dim = compute_kv_dim(ctx.config);
        int32_t max_cache = ctx.config.max_cache_length;
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        std::unique_ptr<IInferenceState> state =
            std::make_unique<KvCache>(dl, max_cache, kv_dim, stream, cache_dtype);
        if (!state->ok())
            throw std::runtime_error("Failed to create KvCache for Whisper decoder");

        // Load mel filterbank + tokenizer
        auto mel_fb = load_mel_filterbank(ctx.bundle);
        auto tok = create_tokenizer_from_bundle(ctx.bundle);

        int32_t mel_n_fft = extract_json_int(json, "mel_n_fft", 400);
        int32_t mel_hop_length = extract_json_int(json, "mel_hop_length", 160);
        int32_t mel_chunk_length = extract_json_int(json, "mel_chunk_length", 30);
        int32_t mel_sampling_rate = extract_json_int(json, "mel_sampling_rate", 16000);

        return std::make_unique<WhisperPipeline>(
            std::move(enc_loaded.module), std::move(dec_loaded.module), std::move(state),
            std::move(wc), ctx.config.hidden_size, dl, std::move(mel_fb), mel_n_fft, mel_hop_length,
            mel_chunk_length, mel_sampling_rate, stream, std::move(tok), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_whisper_plugin, WhisperPlugin, "speech_to_text");

} // namespace trtmc
