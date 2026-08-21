/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/nemotron_voicechat/pipeline.h"
#include "runtime/models/nemotron_voicechat/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

std::string section_text(const BundleFile& bundle, const char* name) {
    const auto* section = find_section(bundle, name);
    if (!section || section->empty())
        throw std::runtime_error(std::string("VoiceChat bundle is missing ") + name);
    return std::string(section->begin(), section->end());
}

std::vector<std::string> load_rnnt_vocabulary(const BundleFile& bundle) {
    const auto document = nlohmann::json::parse(section_text(bundle, "rnnt_vocab.json"));
    if (!document.is_array())
        throw std::runtime_error("VoiceChat rnnt_vocab.json must be a JSON array");
    return document.get<std::vector<std::string>>();
}

void validate_tts_prompt_prefix(const VoiceChatTtsPrompt& prompt, std::size_t steps) {
    if (prompt.warmup_steps <= 0) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
    if (prompt.first_generation_position != prompt.warmup_steps) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
    if (prompt.subword_ids.size() != steps) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
    if (prompt.subword_mask.size() != steps) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
    if (prompt.audio_prompt_mode.size() != steps) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
    if (prompt.bos_flags.size() != steps) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
    if (prompt.position_ids.size() != steps) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
}

void validate_tts_prompt_assets(const VoiceChatTtsPrompt& prompt,
                                const nemotron_voicechat::Config& config,
                                std::size_t expected_embeddings) {
    if (prompt.aria_embeddings.size() != expected_embeddings) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
    if (prompt.first_codes.size() != static_cast<std::size_t>(config.tts_num_quantizers)) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
    if (prompt.silence_codes.size() != static_cast<std::size_t>(config.tts_num_quantizers)) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
    if (prompt.control_codes.size() != 3U) {
        throw std::runtime_error("VoiceChat TTS warmup assets do not match the runtime contract");
    }
}

VoiceChatTtsPrompt load_tts_prompt(const BundleFile& bundle, nemotron_voicechat::Config& config) {
    VoiceChatTtsPrompt prompt;
    const auto recipe = nlohmann::json::parse(section_text(bundle, "tts_prompt_config.json"));
    prompt.warmup_steps = recipe.at("num_steps").get<int32_t>();
    prompt.first_generation_position = recipe.at("first_generation_position_id").get<int32_t>();
    prompt.subword_ids = recipe.at("subword_ids").get<std::vector<int32_t>>();
    prompt.subword_mask = recipe.at("subword_mask").get<std::vector<float>>();
    prompt.audio_prompt_mode = recipe.at("audio_prompt_mode").get<std::vector<float>>();
    prompt.bos_flags = recipe.at("bos_flags").get<std::vector<float>>();
    prompt.position_ids = recipe.at("position_ids").get<std::vector<int32_t>>();
    config.tts_max_cache_length = recipe.at("tts_max_cache_length").get<int32_t>();

    const std::string embedding_section = recipe.at("embedding_section").get<std::string>();
    prompt.aria_embeddings = section_to_floats(find_section(bundle, embedding_section));
    prompt.first_codes = section_to_int32s(find_section(bundle, "tts_first_code_input"));
    prompt.silence_codes = section_to_int32s(find_section(bundle, "tts_silence_codes"));
    prompt.control_codes = section_to_int32s(find_section(bundle, "tts_control_codes"));

    const auto steps = static_cast<std::size_t>(prompt.warmup_steps);
    const auto expected_embeddings = steps * static_cast<std::size_t>(config.tts_hidden_size);
    validate_tts_prompt_prefix(prompt, steps);
    validate_tts_prompt_assets(prompt, config, expected_embeddings);
    return prompt;
}

nemotron_voicechat::Config parse_config(const PipelineContext& context) {
    const auto& json = context.config_json;
    nemotron_voicechat::Config config;
    config.vocab_size = extract_json_int(json, "vocab_size", context.config.vocab_size);
    config.hidden_size = extract_json_int(json, "hidden_size", context.config.hidden_size);
    config.num_attention_heads =
        extract_json_int(json, "num_attention_heads", context.config.num_heads);
    config.num_key_value_heads =
        extract_json_int(json, "num_key_value_heads", context.config.num_kv_heads);
    config.head_dim = extract_json_int(json, "head_dim", context.config.head_dim);
    config.max_cache_length =
        extract_json_int(json, "max_cache_length", context.config.max_cache_length);
    config.num_attention_layers = extract_json_int(json, "num_attention_layers", 4);
    config.num_mamba_layers = extract_json_int(json, "num_mamba_layers", 27);
    config.d_inner = extract_json_int(json, "d_inner", 10240);
    config.mamba_d_state = extract_json_int(json, "mamba_d_state", 128);
    config.mamba_d_conv = extract_json_int(json, "mamba_d_conv", 4);
    config.mamba_nheads = extract_json_int(json, "mamba_nheads", 128);
    config.mamba_head_dim = extract_json_int(json, "mamba_head_dim", 80);
    config.conv_dim = extract_json_int(json, "conv_dim", 12288);

    config.bos_token_id = extract_json_int(json, "bos_token_id", 1);
    config.eos_token_id = extract_json_int(json, "eos_token_id", 2);
    config.pad_token_id = extract_json_int(json, "pad_token_id", 12);
    config.listening_silence_token_id =
        extract_json_int(json, "voicechat_listening_silence_token_id", 11);

    config.input_sample_rate = extract_json_int(json, "input_sample_rate", 16000);
    config.output_sample_rate = extract_json_int(json, "output_sample_rate", 22050);
    config.input_samples_per_frame =
        extract_json_int(json, "voicechat_input_samples_per_frame", 1280);
    config.output_samples_per_frame =
        extract_json_int(json, "voicechat_output_samples_per_frame", 1764);
    config.mel_n_fft = extract_json_int(json, "mel_n_fft", 512);
    config.mel_win_length = extract_json_int(json, "mel_win_length", 400);
    config.mel_hop_length = extract_json_int(json, "mel_hop_length", 160);
    config.mel_num_bins = extract_json_int(json, "mel_num_bins", 128);
    config.mel_length = extract_json_int(json, "mel_length", 3000);
    config.mel_preemphasis = extract_json_float(json, "mel_preemphasis", 0.97F);
    config.perception_hidden_size = extract_json_int(json, "perception_hidden_size", 1024);
    config.perception_num_layers = extract_json_int(json, "perception_num_layers", 24);
    config.perception_num_heads = extract_json_int(json, "perception_num_heads", 8);
    config.perception_att_context_left = extract_json_int(json, "perception_att_context_left", 70);
    config.perception_att_context_right = extract_json_int(json, "perception_att_context_right", 0);

    config.rnnt_pred_hidden_size = extract_json_int(json, "rnnt_pred_hidden_size", 640);
    config.rnnt_pred_num_layers = extract_json_int(json, "rnnt_pred_num_layers", 2);
    config.rnnt_vocab_size = extract_json_int(json, "rnnt_vocab_size", 1024);
    config.rnnt_blank_id = extract_json_int(json, "rnnt_blank_id", 1024);
    config.rnnt_max_symbols_per_step = extract_json_int(json, "rnnt_max_symbols_per_step", 10);

    config.tts_hidden_size = extract_json_int(json, "tts_hidden_size", 1152);
    config.tts_num_layers = extract_json_int(json, "tts_num_layers", 28);
    config.tts_num_heads = extract_json_int(json, "tts_num_heads", 16);
    config.tts_num_key_value_heads =
        extract_json_int(json, "tts_num_key_value_heads", config.tts_num_heads);
    config.tts_head_dim = extract_json_int(json, "tts_head_dim", 72);
    config.tts_kv_width = config.tts_num_key_value_heads * config.tts_head_dim;
    config.tts_num_quantizers = extract_json_int(json, "tts_num_quantizers", 31);
    config.tts_codebook_size = extract_json_int(json, "tts_codebook_size", 1024);
    config.tts_guidance_scale = extract_json_float(json, "tts_guidance_scale", 0.2F);
    config.tts_top_p = extract_json_float(json, "tts_top_p", 0.95F);
    config.tts_noise_scale = extract_json_float(json, "tts_noise_scale", 0.001F);
    config.tts_num_refinement_steps = extract_json_int(json, "tts_num_refinement_steps", 8);
    config.codec_latent_size = extract_json_int(json, "codec_latent_size", 512);
    config.codec_wav_to_token_ratio = extract_json_int(json, "codec_wav_to_token_ratio", 1764);
    return config;
}

} // namespace

class NemotronVoiceChatPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& context) override {
        auto config = parse_config(context);

        ModuleCreateOptions options;
        options.runtime_cache_path = context.runtime_cache_path.c_str();
        options.cuda_graphs = context.cuda_graphs;
        auto thinker =
            load_trt_module_from_plan(context.backend, find_section(context.bundle, "engine_plan"),
                                      "VoiceChat thinker", options);
        ModuleCreateOptions chained = options;
        chained.stream = thinker.module->stream();
        auto perception_first = load_trt_module_from_plan(
            context.backend, find_section(context.bundle, "perception_stream_first_plan"),
            "VoiceChat streaming perception first", chained);
        auto perception = load_trt_module_from_plan(
            context.backend, find_section(context.bundle, "perception_stream_plan"),
            "VoiceChat streaming perception", chained);
        auto rnnt_predictor = load_trt_module_from_plan(
            context.backend, find_section(context.bundle, "rnnt_predictor_plan"),
            "VoiceChat RNNT predictor", chained);
        auto rnnt_joint = load_trt_module_from_plan(context.backend,
                                                    find_section(context.bundle, "rnnt_joint_plan"),
                                                    "VoiceChat RNNT joint", chained);
        auto tts = load_trt_module_from_plan(context.backend,
                                             find_section(context.bundle, "tts_engine_plan"),
                                             "VoiceChat EAR-TTS", chained);
        auto codec = load_trt_module_from_plan(context.backend,
                                               find_section(context.bundle, "codec_engine_plan"),
                                               "VoiceChat RVQ codec", chained);

        const auto mel = load_mel_filterbank(context.bundle);
        VoiceChatAssets assets;
        assets.mel_filterbank = mel.data;
        assets.mel_freq_bins = mel.n_freq_bins;
        assets.mel_bins = mel.n_mel_bins;
        assets.mel_window = section_to_floats(find_section(context.bundle, "mel_window"));
        if (assets.mel_window.size() != static_cast<std::size_t>(config.mel_win_length))
            throw std::runtime_error(
                "VoiceChat bundle mel_window must contain exactly mel_win_length FP32 values");
        assets.rnnt_vocabulary = load_rnnt_vocabulary(context.bundle);
        assets.tts_prompt = load_tts_prompt(context.bundle, config);
        auto tokenizer = try_create_native_tokenizer(context.bundle, false);
        if (!tokenizer)
            throw std::runtime_error(
                "VoiceChat requires the bundled native Nemotron BPE tokenizer");

        return std::make_unique<NemotronVoiceChatPipeline>(
            std::move(thinker.module), std::move(perception_first.module),
            std::move(perception.module), std::move(rnnt_predictor.module),
            std::move(rnnt_joint.module), std::move(tts.module), std::move(codec.module),
            std::move(config), std::move(assets), std::move(tokenizer),
            context.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_nemotron_voicechat_plugin, NemotronVoiceChatPlugin,
                                       "nemotron_voicechat_full_duplex");

} // namespace trtmc
