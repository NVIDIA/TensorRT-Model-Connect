#include "runtime/plugins/shared/audio_helpers.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtmc {

namespace {

// Parse speech_text_prompt_ids from JSON (array of ints).
// Mirrors the logic from fast_path_config.cpp's parse_speech_text_prompt_ids.
std::vector<int32_t> parse_speech_text_prompt_ids(const std::string& config_text) {
    std::vector<int32_t> prompt_ids;
    const std::size_t pos = config_text.find("\"speech_text_prompt_ids\"");
    if (pos == std::string::npos)
        return prompt_ids;

    const std::size_t bracket = config_text.find('[', pos);
    const std::size_t end_bracket = config_text.find(']', bracket);
    if (bracket == std::string::npos || end_bracket == std::string::npos)
        return prompt_ids;

    std::string arr = config_text.substr(bracket + 1, end_bracket - bracket - 1);
    std::size_t p = 0;
    while (p < arr.size()) {
        const std::size_t num_start = arr.find_first_of("0123456789", p);
        if (num_start == std::string::npos)
            break;
        prompt_ids.push_back(std::stoi(arr.substr(num_start)));
        p = arr.find_first_of(",]", num_start);
        if (p == std::string::npos)
            break;
        ++p;
    }
    return prompt_ids;
}

} // namespace

std::unique_ptr<KvCache> make_coarse_kv_cache(const std::string& json, const BaseConfig& base,
                                              cudaStream_t stream, DType cache_dtype) {
    int32_t hidden = extract_json_int(json, "coarse_hidden_size", base.hidden_size);
    int32_t layers = extract_json_int(json, "coarse_num_layers", base.num_layers);
    int32_t heads = extract_json_int(json, "coarse_num_heads", base.num_heads);
    int32_t hd = (heads > 0) ? hidden / heads : 128;
    int32_t max_cache = extract_json_int(json, "coarse_max_cache_length", base.max_cache_length);
    return std::make_unique<KvCache>(layers, max_cache, heads * hd, stream, cache_dtype);
}

MagpieTTSConfig build_magpie_config(const std::string& json, const BaseConfig& base) {
    MagpieTTSConfig magpie_cfg;
    magpie_cfg.sample_rate = extract_json_int(json, "sample_rate", 22050);
    int32_t mh = extract_json_int(json, "magpie_hidden_size", base.hidden_size);
    magpie_cfg.hidden_size = mh > 0 ? mh : base.hidden_size;
    magpie_cfg.num_codebooks = extract_json_int(json, "magpie_num_codebooks", 8);
    magpie_cfg.codebook_size = extract_json_int(json, "magpie_codebook_size", 2024);
    magpie_cfg.frames_per_second = extract_json_float(json, "magpie_fps", 21.5F);
    magpie_cfg.num_speakers = extract_json_int(json, "magpie_num_speakers", 5);
    magpie_cfg.encoder_layers = extract_json_int(json, "magpie_encoder_layers", 6);
    magpie_cfg.decoder_layers = extract_json_int(json, "magpie_decoder_layers", 12);
    magpie_cfg.text_vocab_size = extract_json_int(json, "magpie_text_vocab_size", 0);
    magpie_cfg.max_source_positions = extract_json_int(json, "magpie_max_source_positions", 2048);
    magpie_cfg.xa_n_heads = extract_json_int(json, "magpie_xa_n_heads", 1);
    magpie_cfg.xa_d_head = extract_json_int(json, "magpie_xa_d_head", 128);
    magpie_cfg.temperature = extract_json_float(json, "magpie_temperature", 0.6F);
    magpie_cfg.cfg_scale = extract_json_float(json, "magpie_cfg_scale", 2.5F);
    magpie_cfg.finished_limit_with_eot =
        extract_json_int(json, "magpie_finished_limit_with_eot", 0);
    return magpie_cfg;
}

int32_t compute_kv_dim_kv_heads(const BaseConfig& base, int32_t default_dim) {
    if (base.num_kv_heads > 0 && base.head_dim > 0)
        return base.num_kv_heads * base.head_dim;
    if (base.attention_size > 0)
        return base.attention_size;
    return default_dim;
}

void allocate_cross_kv_buffers(int32_t num_layers, std::size_t buf_size,
                               std::vector<CudaBuffer>& cross_k, std::vector<CudaBuffer>& cross_v) {
    cross_k.reserve(static_cast<std::size_t>(num_layers));
    cross_v.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        cross_k.emplace_back(buf_size);
        cross_v.emplace_back(buf_size);
    }
}

std::vector<std::unique_ptr<TrtModule>> load_depth_engines(IBackend* backend,
                                                           const BundleFile& bundle,
                                                           const ModuleCreateOptions& options) {
    std::vector<std::unique_ptr<TrtModule>> depth_engines;
    auto depth_plans = find_sections_by_prefix(bundle, "depth_engine_plan_");
    if (!depth_plans.empty()) {
        for (std::size_t i = 0; i < depth_plans.size(); ++i) {
            auto m = extract_optional_module(
                backend, depth_plans[i], ("speech depth_" + std::to_string(i)).c_str(), options);
            if (m)
                depth_engines.push_back(std::move(m));
        }
    }
    if (depth_engines.empty()) {
        auto* fallback = find_section(bundle, "depth_engine_plan");
        auto m = extract_optional_module(backend, fallback, "speech depth", options);
        if (m)
            depth_engines.push_back(std::move(m));
    }
    return depth_engines;
}

std::shared_ptr<ITokenizer> make_ipa_tok(const BundleFile& bundle) {
    auto* phoneme = find_section(bundle, "magpie_ipa_phoneme_dict");
    auto* vocab = find_section(bundle, "magpie_ipa_vocab");
    auto* heteronyms = find_section(bundle, "magpie_ipa_heteronyms");
    auto* config = find_section(bundle, "magpie_ipa_config");
    if (!has_section_data(phoneme) || !has_section_data(vocab)) {
        throw std::runtime_error(
            "Bundle missing IPA tokenizer sections (magpie_ipa_phoneme_dict, "
            "magpie_ipa_vocab). Rebuild the bundle with the latest trtmc build.");
    }
    return CreateIpaTokenizer(phoneme->data(), phoneme->size(),
                              has_section_data(heteronyms) ? heteronyms->data() : nullptr,
                              has_section_data(heteronyms) ? heteronyms->size() : 0, vocab->data(),
                              vocab->size(), has_section_data(config) ? config->data() : nullptr,
                              has_section_data(config) ? config->size() : 0);
}

SpeechConfig build_speech_config_from_bundle(const BundleFile& bundle, const std::string& json,
                                             const BaseConfig& base, const std::string& hf_python) {
    SpeechConfig sc;
    sc.sample_rate = extract_json_int(json, "sample_rate", 24000);
    sc.temporal_hidden_size = base.hidden_size;
    sc.temporal_num_layers = base.num_layers;
    sc.num_codebooks = extract_json_int(json, "num_codebooks", 8);
    sc.codebook_size = extract_json_int(json, "codebook_size", 2048);
    sc.frame_rate = extract_json_float(json, "frame_rate", 12.5F);
    int32_t depth_num_layers = extract_json_int(json, "depth_num_layers", 6);
    sc.depth_num_layers = extract_json_int(json, "fine_num_layers", depth_num_layers);
    int32_t depth_hidden_size = extract_json_int(json, "depth_hidden_size", base.hidden_size);
    sc.depth_hidden_size = extract_json_int(json, "fine_hidden_size", depth_hidden_size);
    int32_t depth_num_heads = extract_json_int(json, "depth_num_attention_heads", base.num_heads);
    sc.depth_num_heads = extract_json_int(json, "fine_num_heads", depth_num_heads);
    sc.depth_num_kv_heads = extract_json_int(json, "depth_num_key_value_heads", sc.depth_num_heads);
    sc.delays = extract_json_int_array(json, "delays", 32);
    sc.text_initial_token_id = extract_json_int(json, "text_initial_token_id", 32000);
    sc.audio_initial_token_id = extract_json_int(json, "audio_initial_token_id", 2048);
    sc.text_padding_id = extract_json_int(json, "text_padding_id", 3);
    sc.depth_temperature = extract_json_float(json, "speech_depth_temperature", 0.0F);
    sc.depth_top_k = extract_json_int(json, "speech_depth_top_k", 0);
    sc.text_eos_token_id = base.id_eos;
    sc.system_prompt = extract_json_string(json, "speech_system_prompt", "");
    sc.text_prompt_ids = parse_speech_text_prompt_ids(json);
    sc.hf_python = hf_python;
    sc.audio_embeddings = section_to_floats(find_section(bundle, "audio_embeddings"));
    sc.temporal_text_embedding = section_to_floats(find_section(bundle, "temporal_text_embedding"));
    sc.depth_text_embedding = section_to_floats(find_section(bundle, "depth_text_embedding"));
    sc.depth_audio_embeddings = section_to_floats(find_section(bundle, "depth_audio_embeddings"));
    sc.depth_projection = section_to_floats(find_section(bundle, "depth_projection"));
    return sc;
}

int32_t safe_embed_dim(const std::vector<float>& data, int32_t divisor) {
    return (divisor > 0 && !data.empty()) ? static_cast<int32_t>(data.size()) / divisor : 0;
}

void infer_speech_vocab_sizes(SpeechConfig& sc, const std::string& json, const BaseConfig& base) {
    const int32_t h = base.hidden_size;
    int32_t depth_hidden = extract_json_int(json, "depth_hidden_size", base.hidden_size);
    const int32_t dh = extract_json_int(json, "fine_hidden_size", depth_hidden);
    int32_t n_codebooks = extract_json_int(json, "num_codebooks", 8);
    sc.audio_vocab_size = safe_embed_dim(sc.audio_embeddings, n_codebooks * h);
    sc.temporal_text_vocab = safe_embed_dim(sc.temporal_text_embedding, h);
    sc.depth_text_vocab = safe_embed_dim(sc.depth_text_embedding, dh);
    sc.num_depformer_emb = safe_embed_dim(sc.depth_audio_embeddings, sc.audio_vocab_size * dh);
    sc.temporal_hidden_for_proj = (!sc.depth_projection.empty() && h > 0) ? h : 0;
}

BaseConfig make_depth_engine_config(const std::string& json, const BaseConfig& base) {
    // Resolve fine_* fields first (used as fallbacks for speech_depth_*)
    int32_t fine_num_layers =
        extract_json_int(json, "fine_num_layers", extract_json_int(json, "depth_num_layers", 6));
    int32_t fine_hidden_size = extract_json_int(
        json, "fine_hidden_size", extract_json_int(json, "depth_hidden_size", base.hidden_size));
    int32_t fine_num_heads =
        extract_json_int(json, "fine_num_heads",
                         extract_json_int(json, "depth_num_attention_heads", base.num_heads));

    BaseConfig dc;
    dc.num_layers = extract_json_int(json, "speech_depth_num_layers", fine_num_layers);
    if (dc.num_layers <= 0)
        dc.num_layers = fine_num_layers;
    dc.hidden_size = extract_json_int(json, "speech_depth_hidden_size", fine_hidden_size);
    if (dc.hidden_size <= 0)
        dc.hidden_size = fine_hidden_size;
    dc.num_heads = extract_json_int(json, "speech_depth_num_heads", fine_num_heads);
    if (dc.num_heads <= 0)
        dc.num_heads = fine_num_heads;
    dc.num_kv_heads =
        extract_json_int(json, "speech_depth_num_kv_heads",
                         extract_json_int(json, "depth_num_key_value_heads", dc.num_heads));
    if (dc.num_kv_heads <= 0)
        dc.num_kv_heads = dc.num_heads;
    dc.vocab_size = extract_json_int(json, "speech_codebook_size",
                                     extract_json_int(json, "codebook_size", 2048));
    dc.head_dim = dc.hidden_size / std::max(dc.num_heads, 1);
    dc.attention_size = dc.num_heads * dc.head_dim;
    int32_t n_codebooks =
        extract_json_int(json, "speech_num_codebooks", extract_json_int(json, "num_codebooks", 8));
    dc.max_cache_length = n_codebooks + 2;
    return dc;
}

} // namespace trtmc
