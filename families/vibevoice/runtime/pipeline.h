/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/vibevoice/runtime/noise.h"
#include "families/vibevoice/runtime/scheduler.h"
#include "families/vibevoice/runtime/script.h"
#include "families/vibevoice/runtime/tokenizer.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trtmc::vibevoice {

struct RuntimeConfig {
    bool bfloat16{true};
    std::int32_t hidden_size{0};
    std::int32_t num_layers{0};
    std::int32_t num_kv_heads{0};
    std::int32_t head_dim{0};
    std::int32_t capacity{0};
    std::int32_t prefill_chunk{0};
    std::vector<float> rope_inv_freq;
    std::vector<std::int32_t> control_token_ids; // ascending: ties resolve to the lowest id
    std::int32_t eos_token_id{0};
    std::int32_t audio_bos_token_id{0};
    std::int32_t audio_eos_token_id{0};
    std::int32_t audio_token_id{0};
    PromptFrame frame;
    std::int32_t sample_rate{0};
    std::int64_t hop_length{0};
    std::int32_t latent_size{0};
    float target_dbfs{0.0F};
    float normalize_eps{0.0F};
    std::int64_t max_voice_samples{0};
    std::int64_t stream_state_size{0};
    std::vector<float> train_sigmas;
    double default_guidance_scale{1.0};
    std::int64_t default_diffusion_steps{10};
};

RuntimeConfig parse_runtime_config(const nlohmann::json& json);

struct Engines {
    std::unique_ptr<ITrtModule> lm_prefill;
    std::unique_ptr<ITrtModule> lm_decode;
    std::unique_ptr<ITrtModule> embed;
    std::unique_ptr<ITrtModule> head;
    std::unique_ptr<ITrtModule> feedback;
    std::unique_ptr<ITrtModule> voice;
};

// One KV cache: [layer][key|value][position][kv_heads * head_dim], so the
// first P positions of a layer form the engine's contiguous past tensor.
class Sequence {
  public:
    Sequence(const RuntimeConfig& config, cudaStream_t stream);
    void* key(std::int32_t layer, std::int32_t position);
    void* value(std::int32_t layer, std::int32_t position);
    std::int32_t length{0};

  private:
    void* row(std::int32_t layer, std::int32_t kind, std::int32_t position);
    DeviceTensor storage_;
    std::size_t row_bytes_{0};
    std::int32_t capacity_{0};
};

class VibeVoicePipeline final : public internal::IModel, public internal::ITextToSpeech {
  public:
    VibeVoicePipeline(Engines engines, RuntimeConfig config, std::unique_ptr<ITokenizer> tokenizer,
                      std::vector<std::string> special_tokens);

    const char* task() const noexcept override { return internal::ITextToSpeech::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::AudioResult run(const internal::TextToSpeechRequest& request,
                              internal::ConfigView config) override;

  private:
    struct Decision {
        std::vector<float> logits;
        float normalizer{0.0F};
    };
    struct Options {
        std::uint64_t seed{0};
        std::int64_t max_new_tokens{-1};
        double guidance_scale{1.0};
        std::int64_t diffusion_steps{10};
        std::vector<std::string> voice_prompts;
    };

    Options resolve_options(internal::ConfigView config) const;
    std::int64_t element_bytes() const noexcept;
    void embed_tokens(const std::vector<std::int32_t>& ids, void* destination);
    void encode_voice(const std::vector<float>& samples, NoiseStream& noise, void* destination);
    // Append ``rows`` input rows at the sequence's end; the decision is the last row's.
    Decision forward(ITrtModule& module, Sequence& sequence, const void* inputs, std::int32_t rows);
    void copy_hidden(ITrtModule& module, std::int32_t condition_row);
    const void* special_row(std::int32_t token) const;
    std::int32_t choose(const Decision& conditional, const Decision* unconditional,
                        double guidance_scale) const;
    std::vector<float> sample_latent(NoiseStream& noise, DpmSolver& solver, double guidance_scale);
    void feed_back(const std::vector<float>& latent, std::vector<float>& audio, int& state);

    Engines engines_;
    RuntimeConfig config_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::vector<std::string> special_tokens_;
    cudaStream_t stream_{nullptr};
    std::vector<internal::ConfigField> fields_;
    Sequence positive_;
    Sequence negative_;
    Sequence unconditional_;
    DeviceTensor prompt_;         // [capacity, hidden]
    DeviceTensor special_embeds_; // [4, hidden] in control_token_ids order
    DeviceTensor step_input_;     // [1, hidden]
    DeviceTensor conditions_;     // [2, hidden]
    DeviceTensor rope_cos_;       // [prefill_chunk, head_dim]
    DeviceTensor rope_sin_;
    DeviceTensor states_[2]; // feedback stream states
};

} // namespace trtmc::vibevoice
