/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// ElfFlowPipeline: numeric API path for GitHub ELF denoiser/decoder bundles.

#include "families/elf_flow/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trtmc {

class ElfFlowPipeline final : public ITextGeneration, public INeuralOperator {
  public:
    ElfFlowPipeline(std::unique_ptr<ITrtModule> model, int32_t max_length, int32_t max_input_length,
                    int32_t input_dim, int32_t text_dim, int32_t vocab_size,
                    float denoiser_noise_scale, float denoiser_p_mean, float denoiser_p_std,
                    float t_eps, std::shared_ptr<ITokenizer> tokenizer = nullptr,
                    std::string model_id_str = "",
                    std::unique_ptr<ITrtModule> text_encoder = nullptr, float latent_mean = 0.0F,
                    float latent_std = 1.0F, int32_t encoder_pad_token_id = 0);
    ~ElfFlowPipeline() override;

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg = {}) override;
    int32_t default_max_new_tokens() const override { return 0; }
    const char* task() const noexcept override { return ITextGeneration::kTask; }

    EmbeddingResult solve(const float* branch_input, int32_t branch_len, const float* trunk_input,
                          int32_t trunk_len) override;

  private:
    struct ForwardOutput {
        std::vector<float> data;
        int32_t row_dim{0};
    };
    struct DenoiseOutput {
        std::vector<float> v;
        std::vector<float> x;
    };
    struct PromptCondition {
        std::vector<float> latents;
        std::vector<float> mask;
    };
    struct PromptTokenSpan {
        std::size_t begin{0};
        std::size_t end{0};
        int32_t length{0};
    };
    struct PromptEncoderInputs {
        std::vector<int32_t> input_ids;
        std::vector<float> attention_mask;
    };
    struct ConditionState {
        std::vector<float> latents;
        std::vector<float> mask;
    };
    struct SamplingOptions {
        int32_t seed{42};
        int32_t num_steps{32};
        float self_cond_cfg_scale{3.0F};
        float cfg_scale{1.0F};
        float sde_gamma{1.5F};
    };
    struct SamplingWorkspace {
        std::vector<float> z;
        std::vector<float> x_pred;
    };
    struct DeviceSamplingWorkspace;

    static const Tensor* select_output(const TensorMap& outputs, bool decoder_mode);
    static std::vector<float> copy_tensor_data(const Tensor& tensor);
    void configure_model_dimensions();
    void configure_text_encoder();
    void validate_generate_config(const TextGenerationConfig& cfg) const;
    void add_self_cond_cfg_input(TensorMap& inputs, Tensor& tensor) const;
    int32_t resolve_forward_row_dim(const Tensor& tensor, bool decoder_mode) const;
    ForwardOutput forward_model(const std::vector<float>& latent, float timestep,
                                float self_cond_cfg_scale, bool decoder_mode);
    DenoiseOutput denoise_pass(const std::vector<float>& z, float timestep,
                               const std::vector<float>& x_pred_prev, float self_cond_cfg_scale,
                               const std::vector<float>& cond_seq,
                               const std::vector<float>& cond_mask);
    DenoiseOutput denoise_with_cfg(const std::vector<float>& z, float timestep,
                                   const std::vector<float>& x_pred_prev, float cfg_scale,
                                   float self_cond_cfg_scale, const std::vector<float>& cond_seq,
                                   const std::vector<float>& cond_mask);
    std::vector<float> build_model_latent(const std::vector<float>& z,
                                          const std::vector<float>& self_cond) const;
    std::vector<float> make_sampling_steps(const TextGenerationConfig& cfg, int32_t num_steps,
                                           int32_t seed) const;
    std::vector<float> make_initial_latent(const TextGenerationConfig& cfg, int32_t seed) const;
    std::vector<int32_t> decode_tokens(const std::vector<float>& latent, float self_cond_cfg_scale,
                                       int32_t eos_token_id, int32_t prefix_tokens_to_drop,
                                       int32_t max_output_tokens);
    std::vector<float> make_condition_latents(const TextGenerationConfig& cfg) const;
    std::vector<float> make_condition_mask(const TextGenerationConfig& cfg) const;
    PromptCondition make_prompt_condition(const std::string& prompt, int32_t eos_token_id);
    PromptTokenSpan trim_prompt_tokens(const std::vector<int32_t>& encoded,
                                       int32_t eos_token_id) const;
    PromptEncoderInputs make_prompt_encoder_inputs(const std::vector<int32_t>& encoded,
                                                   const PromptTokenSpan& span) const;
    std::vector<float> run_prompt_encoder(const PromptEncoderInputs& inputs) const;
    std::vector<float> normalize_prompt_latents(const Tensor& embeddings) const;
    ConditionState make_condition_state(const std::string& prompt, const TextGenerationConfig& cfg,
                                        int32_t eos_token_id);
    SamplingOptions resolve_sampling_options(const TextGenerationConfig& cfg,
                                             bool has_condition) const;
    SamplingWorkspace make_sampling_workspace(const TextGenerationConfig& cfg,
                                              const ConditionState& cond, int32_t seed) const;
    void validate_sde_noises(const TextGenerationConfig& cfg,
                             const std::vector<float>& steps) const;
    void apply_sde_perturbation(std::vector<float>& z_eval, const std::vector<float>& z,
                                const TextGenerationConfig& cfg, const ConditionState& cond,
                                float sde_gamma, float t, float t_next, std::size_t step_idx,
                                float& t_eval, std::mt19937& rng,
                                std::normal_distribution<float>& normal) const;
    void run_intermediate_step(SamplingWorkspace& workspace, const TextGenerationConfig& cfg,
                               const ConditionState& cond, const SamplingOptions& options,
                               const std::vector<float>& steps, std::size_t step_idx,
                               std::mt19937& rng, std::normal_distribution<float>& normal);
    void run_final_step(SamplingWorkspace& workspace, const ConditionState& cond,
                        const SamplingOptions& options, const std::vector<float>& steps);
    void run_sampling(SamplingWorkspace& workspace, const TextGenerationConfig& cfg,
                      const ConditionState& cond, const SamplingOptions& options,
                      const std::vector<float>& steps);
    bool supports_device_sampling() const;
    DeviceSamplingWorkspace&
    prepare_device_sampling_workspace(const SamplingWorkspace& host_workspace,
                                      const ConditionState& cond,
                                      const std::vector<float>& sde_noises);
    std::vector<float> make_device_sde_noises(const TextGenerationConfig& cfg,
                                              const SamplingOptions& options,
                                              const std::vector<float>& steps) const;
    void enqueue_device_forward(DeviceSamplingWorkspace& workspace, float timestep,
                                float self_cond_cfg_scale, bool decoder_mode, bool zero_condition,
                                bool zero_self_condition);
    void run_device_denoise_step(DeviceSamplingWorkspace& workspace, const SamplingOptions& options,
                                 float timestep, float next_timestep);
    void run_device_sampling(DeviceSamplingWorkspace& workspace, const SamplingOptions& options,
                             const std::vector<float>& steps);
    std::vector<int32_t> decode_device_tokens(DeviceSamplingWorkspace& workspace,
                                              const SamplingOptions& options, int32_t eos_token_id,
                                              int32_t prefix_tokens_to_drop,
                                              int32_t max_output_tokens);
    TextResult make_device_text_result(DeviceSamplingWorkspace& workspace,
                                       const TextGenerationConfig& cfg,
                                       const SamplingOptions& options, bool has_condition,
                                       int32_t eos_token_id, int32_t cond_prefix_len,
                                       double sampling_ms);
    TextResult make_text_result(const SamplingWorkspace& workspace, const TextGenerationConfig& cfg,
                                const SamplingOptions& options, bool has_condition,
                                int32_t eos_token_id, int32_t cond_prefix_len, double sampling_ms);
    int32_t condition_prefix_tokens(const std::vector<float>& cond_mask) const;
    bool has_active_condition(const std::vector<float>& cond_mask) const;
    void restore_condition(std::vector<float>& values, const std::vector<float>& cond_seq,
                           const std::vector<float>& cond_mask) const;
    void zero_condition(std::vector<float>& values, const std::vector<float>& cond_mask) const;
    int32_t resolve_max_output_tokens(const TextGenerationConfig& cfg, bool has_condition) const;
    int32_t resolve_eos_token_id(const TextGenerationConfig& cfg) const;

    std::unique_ptr<ITrtModule> model_;
    std::unique_ptr<ITrtModule> text_encoder_;
    int32_t max_length_{0};
    int32_t max_input_length_{0};
    int32_t encoder_seq_length_{0};
    int32_t input_dim_{0};
    int32_t text_dim_{0};
    int32_t vocab_size_{0};
    float denoiser_noise_scale_{1.0F};
    float denoiser_p_mean_{-1.5F};
    float denoiser_p_std_{0.8F};
    float t_eps_{5e-2F};
    float latent_mean_{0.0F};
    float latent_std_{1.0F};
    int32_t encoder_pad_token_id_{0};
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<DeviceSamplingWorkspace> device_sampling_workspace_;
};

} // namespace trtmc
