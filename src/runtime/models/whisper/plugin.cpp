// WhisperPlugin: handles "whisper_speech_to_text" strategy.
// Whisper encoder-decoder pipeline with mel spectrogram input.

#include "plugin_helpers.h"
#include "runtime/models/whisper/pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <limits>
#include <string>
#include <vector>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

struct SpeechMelRuntimeConfig {
    int32_t n_fft{400};
    int32_t hop_length{160};
    int32_t chunk_length{30};
    int32_t sampling_rate{16000};
    int32_t win_length{400};
    float preemph{0.0F};
    bool normalize_per_feature{false};
    std::string frontend{"whisper"};
};

TensorParallelRuntimeConfig parse_tensor_parallel_runtime_config(const std::string& config_json) {
    TensorParallelRuntimeConfig cfg;
    cfg.tp_size = extract_json_int(config_json, "tensor_parallel_size", 1);
    const auto mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    cfg.enabled = (mode == "tensor_parallel" && cfg.tp_size > 1);
    return cfg;
}

SpeechMelRuntimeConfig parse_speech_mel_runtime_config(const std::string& json,
                                                       const std::string& bundle_model_type,
                                                       const std::string& bundle_family) {
    SpeechMelRuntimeConfig cfg;
    cfg.n_fft = extract_json_int(json, "mel_n_fft", 400);
    cfg.hop_length = extract_json_int(json, "mel_hop_length", 160);
    cfg.chunk_length = extract_json_int(json, "mel_chunk_length", 30);
    cfg.sampling_rate = extract_json_int(json, "mel_sampling_rate", 16000);

    const std::string model_type = extract_json_string(json, "model_type", bundle_model_type);
    const bool is_canary = (model_type == "canary" || bundle_family == "canary");
    cfg.frontend = extract_json_string(json, "mel_frontend", is_canary ? "nemo" : "whisper");
    cfg.win_length =
        extract_json_int(json, "mel_win_length", (cfg.frontend == "nemo") ? 400 : cfg.n_fft);
    cfg.preemph = extract_json_float(json, "mel_preemph", (cfg.frontend == "nemo") ? 0.97F : 0.0F);

    const std::string normalize =
        extract_json_string(json, "mel_normalize", is_canary ? "per_feature" : "");
    cfg.normalize_per_feature = (cfg.frontend == "nemo" && normalize == "per_feature");
    return cfg;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine_plan_tp_rank" + std::to_string(rank);
}

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const auto value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t decoder_cache_row_width(const TrtModule& module, const BaseConfig& config) {
    const int32_t from_engine = dim_at(module.tensor_shape("cache_k_0"), 1);
    return from_engine > 0 ? from_engine : compute_kv_dim(config);
}

} // namespace

class WhisperPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto& json = ctx.config_json;
        const auto tp_config = parse_tensor_parallel_runtime_config(json);
        DistributedRuntimeGroup tp_group;
        if (tp_config.enabled)
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);

        // Load encoder (stored as vision_engine_plan in Whisper bundles)
        const auto* enc_plan = find_section(ctx.bundle, "vision_engine_plan");
        if (!enc_plan || enc_plan->empty())
            enc_plan = find_section(ctx.bundle, "coarse_engine_plan");
        auto enc_loaded = load_trt_module_from_plan(ctx.backend, enc_plan, "whisper encoder", opts);

        if (tp_config.enabled) {
            opts.distributed_communicator = tp_group.communicator;
            opts.distributed_owner = tp_group.owner;
        }

        // Load decoder (main engine_plan or rank-local TP section)
        const std::string decoder_section =
            tp_config.enabled ? tp_engine_section_name(tp_group.rank) : std::string("engine_plan");
        auto dec_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, decoder_section), "whisper decoder", opts);

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

        // Create WhisperKvCache for decoder self-attention
        cudaStream_t stream = dec_loaded.module->stream();
        int32_t kv_dim = decoder_cache_row_width(*dec_loaded.module, ctx.config);
        int32_t max_cache = ctx.config.max_cache_length;
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        std::unique_ptr<WhisperInferenceState> state =
            std::make_unique<WhisperKvCache>(dl, max_cache, kv_dim, stream, cache_dtype);
        if (!state->ok())
            throw std::runtime_error("Failed to create WhisperKvCache for Whisper decoder");

        // Load mel filterbank + tokenizer
        auto mel_fb = load_mel_filterbank(ctx.bundle);
        auto tok = create_tokenizer_from_bundle(ctx.bundle);

        SpeechMelRuntimeConfig mel_cfg = parse_speech_mel_runtime_config(
            json, ctx.bundle.info.model_type, ctx.bundle.info.family);

        return std::make_unique<WhisperPipeline>(
            std::move(enc_loaded.module), std::move(dec_loaded.module), std::move(state),
            std::move(wc), ctx.config.hidden_size, dl, std::move(mel_fb), mel_cfg.n_fft,
            mel_cfg.hop_length, mel_cfg.chunk_length, mel_cfg.sampling_rate, mel_cfg.win_length,
            mel_cfg.preemph, mel_cfg.normalize_per_feature, std::move(mel_cfg.frontend), stream,
            std::move(tok), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_whisper_plugin, WhisperPlugin,
                                       "whisper_speech_to_text");

} // namespace trtmc
