#include "audio_helpers.h"

#include <stdexcept>

namespace trtmc {

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

void allocate_cross_kv_buffers(int32_t num_layers, std::size_t buf_size,
                               std::vector<CudaBuffer>& cross_k, std::vector<CudaBuffer>& cross_v) {
    cross_k.reserve(static_cast<std::size_t>(num_layers));
    cross_v.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        cross_k.emplace_back(buf_size);
        cross_v.emplace_back(buf_size);
    }
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

} // namespace trtmc
