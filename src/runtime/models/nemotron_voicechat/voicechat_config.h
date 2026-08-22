/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc::nemotron_voicechat {

struct Config {
    int32_t vocab_size{131072};
    int32_t hidden_size{4480};
    int32_t num_attention_heads{40};
    int32_t num_key_value_heads{8};
    int32_t head_dim{128};
    int32_t max_cache_length{8192};
    int32_t num_attention_layers{4};
    int32_t num_mamba_layers{27};
    int32_t d_inner{10240};
    int32_t mamba_d_state{128};
    int32_t mamba_d_conv{4};
    int32_t mamba_nheads{128};
    int32_t mamba_head_dim{80};
    int32_t conv_dim{12288};

    int32_t bos_token_id{1};
    int32_t eos_token_id{2};
    int32_t pad_token_id{12};

    int32_t input_sample_rate{16000};
    int32_t output_sample_rate{22050};
    int32_t input_samples_per_frame{1280};
    int32_t mel_n_fft{512};
    int32_t mel_win_length{400};
    int32_t mel_hop_length{160};
    int32_t mel_num_bins{128};
    int32_t mel_length{3000};
    float mel_preemphasis{0.97F};
    int32_t perception_hidden_size{1024};
    int32_t perception_num_layers{24};
    int32_t perception_num_heads{8};
    int32_t perception_att_context_left{70};
    int32_t perception_att_context_right{0};

    int32_t rnnt_pred_hidden_size{640};
    int32_t rnnt_pred_num_layers{2};
    int32_t rnnt_vocab_size{1024};
    int32_t rnnt_blank_id{1024};
    int32_t rnnt_max_symbols_per_step{10};
    // Model-owned live turn-taking policy. The defaults follow the public
    // NeMo wrapper's low-latency RNNT policy: confirm the first utterance with
    // two speech frames, subsequent utterances with three, end an utterance
    // after ten blank frames, and confirm barge-in after three consecutive
    // non-unknown RNNT speech frames while an agent turn is active.
    int32_t rnnt_eou_frames{10};
    int32_t rnnt_bou_frames{3};
    int32_t rnnt_min_speech_frames{3};
    int32_t rnnt_min_speech_frames_first_turn{2};

    int32_t function_max_call_tokens{512};
    int32_t function_max_response_tokens{1024};
    int32_t function_max_async_steps{2048};
    int32_t function_tool_timeout_ms{15000};
    int32_t function_on_hold_min_pad_frames{17};

    int32_t tts_hidden_size{1152};
    int32_t tts_num_layers{28};
    int32_t tts_num_heads{16};
    int32_t tts_num_key_value_heads{16};
    int32_t tts_head_dim{72};
    int32_t tts_kv_width{1152};
    int32_t tts_max_cache_length{7500};
    int32_t tts_num_quantizers{31};
    int32_t tts_codebook_size{1024};
    int32_t tts_mog_num_predictions{1024};
    int32_t tts_num_refinement_steps{8};
    float tts_guidance_scale{0.2F};
    float tts_top_p{0.95F};
    float tts_noise_scale{0.001F};

    int32_t codec_latent_size{512};
    int32_t codec_wav_to_token_ratio{1764};

    // Native full-duplex policy. append_audio() stays enqueue-only. After
    // finish_input(), the worker may pump this bounded number of silence
    // frames while an agent response remains active; offline speak() supplies
    // its own explicit tail and disables this implicit live-session tail.
    int32_t max_response_frames{256};
    int32_t tts_text_token_ratio_cap{16};
    int32_t tts_text_token_ratio_min_tokens{5};

    // Bound queued live input and undrained public events. Offline convenience
    // paths bypass the input bound because they intentionally enqueue a whole
    // recording. Live sessions remain bounded even when a producer or consumer
    // runs faster or slower than inference.
    int32_t max_pending_input_ms{30000};
    int32_t max_pending_events{4096};
    int32_t stream_tick_ms{80};

    std::string default_system_prompt{
        "You are an AI voice assistant developed by NVIDIA. Your name is NVIDIA Voice Chat. "
        "Answer in a spoken, conversational style rather than a written one. Do not repeat "
        "the same sentence over and over again. Start the conversation by greeting the user."};
};

} // namespace trtmc::nemotron_voicechat
