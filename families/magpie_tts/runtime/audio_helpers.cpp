/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_helpers.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc {

MagpieTTSConfig build_magpie_config(const std::string& json) {
    const auto document = nlohmann::json::parse(json);
    MagpieTTSConfig magpie_cfg;
    magpie_cfg.sample_rate = document.at("sample_rate").get<int32_t>();
    magpie_cfg.hidden_size = document.at("hidden_size").get<int32_t>();
    magpie_cfg.num_codebooks = document.at("num_codebooks").get<int32_t>();
    magpie_cfg.codebook_size = document.at("codebook_size").get<int32_t>();
    magpie_cfg.frames_per_second = document.at("frames_per_second").get<float>();
    magpie_cfg.num_speakers = document.at("num_speakers").get<int32_t>();
    magpie_cfg.encoder_layers = document.at("encoder_layers").get<int32_t>();
    magpie_cfg.decoder_layers = document.at("decoder_layers").get<int32_t>();
    magpie_cfg.text_vocab_size = document.at("text_vocab_size").get<int32_t>();
    magpie_cfg.max_source_positions = document.at("max_source_positions").get<int32_t>();
    magpie_cfg.xa_n_heads = document.at("xa_n_heads").get<int32_t>();
    magpie_cfg.xa_d_head = document.at("xa_d_head").get<int32_t>();
    magpie_cfg.temperature = document.at("temperature").get<float>();
    magpie_cfg.top_k = document.at("top_k").get<int32_t>();
    magpie_cfg.greedy = document.at("greedy").get<bool>();
    magpie_cfg.cfg_scale = document.at("cfg_scale").get<float>();
    magpie_cfg.finished_limit_with_eot = document.at("finished_limit_with_eot").get<int32_t>();
    magpie_cfg.enable_finished_limit_stop = document.at("enable_finished_limit_stop").get<bool>();
    magpie_cfg.seed = document.at("seed").get<int64_t>();
    if (magpie_cfg.sample_rate <= 0 || magpie_cfg.hidden_size <= 0 ||
        magpie_cfg.num_codebooks <= 0 || magpie_cfg.codebook_size <= 0 ||
        magpie_cfg.encoder_layers <= 0 || magpie_cfg.decoder_layers <= 0 ||
        magpie_cfg.max_source_positions <= 0)
        throw std::runtime_error("Magpie runtime.json does not match its runtime contract");
    return magpie_cfg;
}

void allocate_cross_kv_buffers(int32_t num_layers, std::size_t buf_size,
                               std::vector<MagpieCudaBuffer>& cross_k,
                               std::vector<MagpieCudaBuffer>& cross_v) {
    cross_k.reserve(static_cast<std::size_t>(num_layers));
    cross_v.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        cross_k.emplace_back(buf_size);
        cross_v.emplace_back(buf_size);
    }
}

std::shared_ptr<ITokenizer> make_ipa_tok(const BundleReader& bundle) {
    const auto require = [&bundle](const char* name) {
        const auto* section = bundle.find_section(name);
        if (section == nullptr || section->length == 0)
            throw std::runtime_error("Magpie bundle section is missing: " + std::string(name));
        return bundle.read_section(name);
    };
    const auto& phoneme = require("ipa.phonemes");
    const auto& vocab = require("ipa.vocab");
    const auto& heteronyms = require("ipa.heteronyms");
    const auto& config = require("ipa.config");
    auto tokenizer =
        CreateIpaTokenizer(phoneme.data(), phoneme.size(), heteronyms.data(), heteronyms.size(),
                           vocab.data(), vocab.size(), config.data(), config.size());
    if (!tokenizer)
        throw std::runtime_error("Magpie IPA tokenizer sections are invalid");
    return tokenizer;
}

} // namespace trtmc
