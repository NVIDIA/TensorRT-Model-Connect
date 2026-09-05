/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_voicechat/runtime/pipeline.h"
#include "families/nemotron_voicechat/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::voicechat_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::string section_text(const BundleReader& bundle, const char* name) {
    const auto& section = require_section(bundle, name);
    return std::string(section.begin(), section.end());
}

nemotron_voicechat::Config parse_config(const nlohmann::json& json) {
    nemotron_voicechat::Config config;
#define VC_INT(field) config.field = json.at(#field).get<std::int32_t>()
    VC_INT(vocab_size);
    VC_INT(hidden_size);
    VC_INT(num_attention_heads);
    VC_INT(num_key_value_heads);
    VC_INT(head_dim);
    VC_INT(max_cache_length);
    VC_INT(num_attention_layers);
    VC_INT(num_mamba_layers);
    VC_INT(d_inner);
    VC_INT(mamba_d_state);
    VC_INT(mamba_d_conv);
    VC_INT(mamba_nheads);
    VC_INT(mamba_head_dim);
    VC_INT(conv_dim);
    VC_INT(bos_token_id);
    VC_INT(eos_token_id);
    VC_INT(pad_token_id);
    VC_INT(input_sample_rate);
    VC_INT(output_sample_rate);
    VC_INT(input_samples_per_frame);
    VC_INT(mel_n_fft);
    VC_INT(mel_win_length);
    VC_INT(mel_hop_length);
    VC_INT(mel_num_bins);
    VC_INT(mel_length);
    VC_INT(perception_hidden_size);
    VC_INT(perception_num_layers);
    VC_INT(perception_num_heads);
    VC_INT(perception_att_context_left);
    VC_INT(perception_att_context_right);
    VC_INT(rnnt_pred_hidden_size);
    VC_INT(rnnt_pred_num_layers);
    VC_INT(rnnt_vocab_size);
    VC_INT(rnnt_blank_id);
    VC_INT(rnnt_max_symbols_per_step);
    VC_INT(rnnt_eou_frames);
    VC_INT(rnnt_bou_frames);
    VC_INT(rnnt_min_speech_frames);
    VC_INT(rnnt_min_speech_frames_first_turn);
    VC_INT(function_max_call_tokens);
    VC_INT(function_max_response_tokens);
    VC_INT(function_max_async_steps);
    VC_INT(function_tool_timeout_ms);
    VC_INT(function_on_hold_min_pad_frames);
    VC_INT(tts_hidden_size);
    VC_INT(tts_num_layers);
    VC_INT(tts_num_heads);
    VC_INT(tts_num_key_value_heads);
    VC_INT(tts_head_dim);
    VC_INT(tts_kv_width);
    VC_INT(tts_max_cache_length);
    VC_INT(tts_num_quantizers);
    VC_INT(tts_codebook_size);
    VC_INT(tts_mog_num_predictions);
    VC_INT(tts_num_refinement_steps);
    VC_INT(codec_latent_size);
    VC_INT(codec_wav_to_token_ratio);
    VC_INT(max_response_frames);
    VC_INT(tts_text_token_ratio_cap);
    VC_INT(tts_text_token_ratio_min_tokens);
    VC_INT(max_pending_input_ms);
    VC_INT(max_pending_events);
    VC_INT(stream_tick_ms);
#undef VC_INT
    config.mel_preemphasis = json.at("mel_preemphasis").get<float>();
    config.tts_guidance_scale = json.at("tts_guidance_scale").get<float>();
    config.tts_top_p = json.at("tts_top_p").get<float>();
    config.tts_noise_scale = json.at("tts_noise_scale").get<float>();
    config.default_system_prompt = json.at("default_system_prompt").get<std::string>();
    if (config.hidden_size <= 0 || config.max_cache_length <= 0 || config.input_sample_rate <= 0 ||
        config.output_sample_rate <= 0 || config.tts_hidden_size <= 0 ||
        config.tts_num_layers <= 0 || config.default_system_prompt.empty())
        throw std::runtime_error("VoiceChat runtime.json does not match its runtime contract");
    return config;
}

VoiceChatTtsPrompt load_tts_prompt(const BundleReader& bundle,
                                   const nemotron_voicechat::Config& config) {
    const auto recipe = nlohmann::json::parse(section_text(bundle, "tts_prompt.json"));
    VoiceChatTtsPrompt prompt;
    prompt.warmup_steps = recipe.at("num_steps").get<std::int32_t>();
    prompt.first_generation_position =
        recipe.at("first_generation_position_id").get<std::int32_t>();
    prompt.subword_ids = recipe.at("subword_ids").get<std::vector<std::int32_t>>();
    prompt.subword_mask = recipe.at("subword_mask").get<std::vector<float>>();
    prompt.audio_prompt_mode = recipe.at("audio_prompt_mode").get<std::vector<float>>();
    prompt.bos_flags = recipe.at("bos_flags").get<std::vector<float>>();
    prompt.position_ids = recipe.at("position_ids").get<std::vector<std::int32_t>>();
    if (recipe.at("tts_max_cache_length").get<std::int32_t>() != config.tts_max_cache_length)
        throw std::runtime_error("VoiceChat TTS prompt cache length does not match runtime.json");
    const auto embeddings = require_section(bundle, "tts_prompt.embeddings");
    const auto first_codes = require_section(bundle, "tts_prompt.first_codes");
    const auto silence_codes = require_section(bundle, "tts_prompt.silence_codes");
    const auto control_codes = require_section(bundle, "tts_prompt.control_codes");
    prompt.aria_embeddings = section_to_floats(&embeddings);
    prompt.first_codes = section_to_int32s(&first_codes);
    prompt.silence_codes = section_to_int32s(&silence_codes);
    prompt.control_codes = section_to_int32s(&control_codes);
    const auto steps = static_cast<std::size_t>(prompt.warmup_steps);
    if (prompt.warmup_steps <= 0 || prompt.first_generation_position != prompt.warmup_steps ||
        prompt.subword_ids.size() != steps || prompt.subword_mask.size() != steps ||
        prompt.audio_prompt_mode.size() != steps || prompt.bos_flags.size() != steps ||
        prompt.position_ids.size() != steps ||
        prompt.aria_embeddings.size() != steps * static_cast<std::size_t>(config.tts_hidden_size) ||
        prompt.first_codes.size() != static_cast<std::size_t>(config.tts_num_quantizers) ||
        prompt.silence_codes.size() != static_cast<std::size_t>(config.tts_num_quantizers) ||
        prompt.control_codes.size() != 3U)
        throw std::runtime_error("VoiceChat TTS prompt sections are inconsistent");
    return prompt;
}

} // namespace
} // namespace trtmc::voicechat_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("nemotron_voicechat does not support --kv-cache-size");
    using namespace trtmc;
    const auto runtime_text = voicechat_factory::section_text(context.reader, "runtime.json");
    auto config = voicechat_factory::parse_config(nlohmann::json::parse(runtime_text));
    ModuleCreateOptions options{};
    const auto load = [&](const char* name, cudaStream_t stream = nullptr) {
        auto module_options = options;
        module_options.stream = stream;
        const auto& plan = voicechat_factory::require_section(context.reader, name);
        return load_trt_module_from_plan(&context.backend, &plan, name, module_options).module;
    };
    auto thinker = load("engine.plan");
    const auto stream = thinker->stream();
    auto perception_first = load("perception.first.plan", stream);
    auto perception = load("perception.plan", stream);
    auto rnnt_predictor = load("rnnt.predictor.plan", stream);
    auto rnnt_joint = load("rnnt.joint.plan", stream);
    auto tts = load("tts.plan", stream);
    auto codec = load("codec.plan", stream);

    const auto mel = load_mel_filterbank(context.reader);
    VoiceChatAssets assets;
    assets.mel_filterbank = mel.data;
    assets.mel_freq_bins = mel.n_freq_bins;
    assets.mel_bins = mel.n_mel_bins;
    const auto mel_window = voicechat_factory::require_section(context.reader, "mel.window");
    assets.mel_window = section_to_floats(&mel_window);
    assets.rnnt_vocabulary =
        nlohmann::json::parse(voicechat_factory::section_text(context.reader, "rnnt.vocab.json"))
            .get<std::vector<std::string>>();
    assets.tts_prompt = voicechat_factory::load_tts_prompt(context.reader, config);
    if (assets.mel_filterbank.empty() ||
        assets.mel_window.size() != static_cast<std::size_t>(config.mel_win_length) ||
        assets.rnnt_vocabulary.empty())
        throw std::runtime_error("VoiceChat audio assets are invalid");
    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("VoiceChat bundle does not contain its required tokenizer");
    return new NemotronVoiceChatPipeline(
        std::move(thinker), std::move(perception_first), std::move(perception),
        std::move(rnnt_predictor), std::move(rnnt_joint), std::move(tts), std::move(codec),
        std::move(config), std::move(assets), std::move(tokenizer), "");
}
