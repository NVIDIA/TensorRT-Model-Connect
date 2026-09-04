/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/qwen3_omni/runtime/kv_cache.h"
#include "families/qwen3_omni/runtime/sampler.h"
#include "families/qwen3_omni/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct Qwen3OmniRuntimeConfig {
    std::string precision;
    std::int32_t sample_rate{24000};

    std::int32_t thinker_hidden_size{0};
    std::int32_t thinker_num_layers{0};
    std::int32_t thinker_num_attention_heads{0};
    std::int32_t thinker_num_key_value_heads{0};
    std::int32_t thinker_head_dim{0};
    std::int32_t thinker_vocab_size{0};
    std::int32_t thinker_max_cache_length{0};
    std::int32_t thinker_eos_token_id{-1};

    std::int32_t talker_hidden_size{0};
    std::int32_t talker_num_layers{0};
    std::int32_t talker_num_attention_heads{0};
    std::int32_t talker_num_key_value_heads{0};
    std::int32_t talker_head_dim{0};
    std::int32_t talker_vocab_size{0};
    std::int32_t talker_max_cache_length{0};

    std::int32_t predictor_hidden_size{0};
    std::int32_t predictor_num_layers{0};
    std::int32_t predictor_num_attention_heads{0};
    std::int32_t predictor_num_key_value_heads{0};
    std::int32_t predictor_head_dim{0};
    std::int32_t predictor_vocab_size{0};
    std::int32_t predictor_max_cache_length{0};

    std::int32_t num_codebooks{0};
    std::int32_t codebook_size{0};
    std::int32_t talker_max_frames{0};

    std::int32_t im_start_token_id{-1};
    std::int32_t system_token_id{-1};
    std::int32_t user_token_id{-1};
    std::int32_t assistant_token_id{-1};
    std::int32_t tts_bos_token_id{-1};
    std::int32_t tts_eos_token_id{-1};
    std::int32_t tts_pad_token_id{-1};
    std::int32_t codec_bos_id{-1};
    std::int32_t codec_eos_token_id{-1};
    std::int32_t codec_nothink_id{-1};
    std::int32_t codec_pad_id{-1};
    std::int32_t codec_think_bos_id{-1};
    std::int32_t codec_think_eos_id{-1};
    std::int32_t speaker_id{-1};

    std::int32_t code2wav_max_frames{0};
    std::int32_t code2wav_upsample_factor{0};
    std::int32_t code2wav_output_delay{0};
    std::int32_t code2wav_num_quantizers{0};
};

class Qwen3OmniAudioPipeline final : public IAudioGeneration, public ITextGeneration {
  public:
    Qwen3OmniAudioPipeline(
        std::unique_ptr<ITrtModule> thinker_prefill, std::unique_ptr<ITrtModule> thinker_decode,
        std::unique_ptr<Qwen3OmniKvCache> thinker_state,
        std::unique_ptr<ITrtModule> text_projection, std::unique_ptr<ITrtModule> talker_prefill,
        std::unique_ptr<ITrtModule> talker_decode, std::unique_ptr<Qwen3OmniKvCache> talker_state,
        std::unique_ptr<ITrtModule> predictor_prefill, std::unique_ptr<ITrtModule> predictor_decode,
        std::unique_ptr<Qwen3OmniKvCache> predictor_state, std::unique_ptr<ITrtModule> code2wav,
        std::vector<float> talker_codec_embedding, std::vector<float> predictor_codec_embeddings,
        Qwen3OmniRuntimeConfig config, std::shared_ptr<ITokenizer> tokenizer);

    AudioResult generate_audio(const std::string& prompt,
                               const AudioGenerationConfig& config = {}) override;
    const char* task() const noexcept override { return IAudioGeneration::kTask; }
    std::int32_t default_max_new_tokens() const override { return 128; }
    TextResult generate(const std::string& prompt,
                        const TextGenerationConfig& config = {}) override;

  private:
    struct DecoderOutput {
        std::vector<float> logits;
        std::vector<float> hidden;
    };

    struct TalkerInputs {
        std::vector<float> initial;
        std::vector<float> trailing;
        std::int32_t trailing_rows{0};
        std::vector<float> pad;
    };

    DecoderOutput run_token_prefill(const std::vector<std::int32_t>& token_ids);
    DecoderOutput run_embed_prefill(ITrtModule& module, Qwen3OmniKvCache& state,
                                    const std::vector<float>& embeddings, std::int32_t hidden_size,
                                    const std::string& logits_name, std::int32_t logits_size);
    DecoderOutput run_token_step(std::int32_t token_id);
    DecoderOutput run_embed_step(ITrtModule& module, Qwen3OmniKvCache& state,
                                 const float* embedding, std::int32_t hidden_size,
                                 const std::string& logits_name, std::int32_t logits_size);
    std::vector<std::int32_t> run_thinker(const std::string& prompt, std::int32_t max_new_tokens);
    std::vector<float> project_tokens(const std::vector<std::int32_t>& token_ids);
    TalkerInputs prepare_talker_inputs(const std::string& prompt,
                                       const std::string& assistant_text);
    std::vector<std::int32_t> run_code_predictor(const std::vector<float>& talker_hidden,
                                                 std::int32_t coarse_code,
                                                 std::vector<float>& next_embedding,
                                                 qwen3_omni::ResidualCodeSampler& sampler);
    std::vector<std::int32_t> run_talker(const TalkerInputs& inputs,
                                         qwen3_omni::ResidualCodeSampler& sampler,
                                         std::int32_t max_frames);
    std::vector<float> run_code2wav(const std::vector<std::int32_t>& frame_major_codes);

    const float* talker_embedding_row(std::int32_t token_id) const;
    const float* predictor_embedding_row(std::int32_t group, std::int32_t token_id) const;

    std::unique_ptr<ITrtModule> thinker_prefill_;
    std::unique_ptr<ITrtModule> thinker_decode_;
    std::unique_ptr<Qwen3OmniKvCache> thinker_state_;
    std::unique_ptr<ITrtModule> text_projection_;
    std::unique_ptr<ITrtModule> talker_prefill_;
    std::unique_ptr<ITrtModule> talker_decode_;
    std::unique_ptr<Qwen3OmniKvCache> talker_state_;
    std::unique_ptr<ITrtModule> predictor_prefill_;
    std::unique_ptr<ITrtModule> predictor_decode_;
    std::unique_ptr<Qwen3OmniKvCache> predictor_state_;
    std::unique_ptr<ITrtModule> code2wav_;
    std::vector<float> talker_codec_embedding_;
    std::vector<float> predictor_codec_embeddings_;
    Qwen3OmniRuntimeConfig config_;
    std::shared_ptr<ITokenizer> tokenizer_;
};

} // namespace trtmc
