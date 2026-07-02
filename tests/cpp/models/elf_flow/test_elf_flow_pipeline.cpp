/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// Test suite: ELF flow pipeline generation loop
// =============================================================================
//
// Purpose:
//   Validates the C++ ELF pipeline's end-to-end generate() path without
//   requiring a TensorRT SDK. A fake module emulates the GitHub ELF engine
//   contract: denoise latent embeddings, then decode final embeddings to token
//   logits.
//
// Dependencies:
//   - runtime/models/elf_flow/pipeline.h
//   - trtmc/tokenizer.h
//   - No GPU, TensorRT headers, or model files required.
// =============================================================================

#include "runtime/models/elf_flow/pipeline.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

bool close_enough(float actual, float expected) {
    return std::fabs(actual - expected) <= 1e-6F;
}

float scalar_input(const trtmc::TensorMap& inputs, const std::string& name) {
    const auto it = inputs.find(name);
    if (it == inputs.end() || !it->second.data)
        throw std::runtime_error("missing scalar input: " + name);
    return *static_cast<float*>(it->second.data);
}

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& /*text*/) const override { return encoded_ids; }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string out;
        for (int32_t id : ids)
            out += token_for_id(id);
        return out;
    }

    int32_t id_for_token(std::string_view token) const override {
        if (token == "</s>")
            return eos_id;
        return -1;
    }

    std::string token_for_id(int32_t id) const override {
        if (id == 1)
            return "A";
        if (id == 2)
            return "B";
        if (id == 3)
            return "C";
        return "?";
    }

    int32_t eos_id{-1};
    std::vector<int32_t> encoded_ids;
};

class FakeTextEncoderModule final : public trtmc::TrtModule {
  public:
    bool ok() const override { return true; }

    std::vector<trtmc::TensorInfo> input_info() const override {
        return {
            trtmc::TensorInfo{"input_ids", {1, kMaxLength}, trtmc::DType::kInt32, true},
            trtmc::TensorInfo{"attention_mask", {1, kMaxLength}, trtmc::DType::kFloat32, true},
        };
    }

    std::vector<trtmc::TensorInfo> output_info() const override {
        return {trtmc::TensorInfo{
            "text_embeddings", {1, kMaxLength, kTextDim}, trtmc::DType::kFloat32, false}};
    }

    bool has_input(const std::string& name) const override {
        return name == "input_ids" || name == "attention_mask";
    }

    bool has_output(const std::string& name) const override { return name == "text_embeddings"; }

    trtmc::DType tensor_dtype(const std::string& name) const override {
        if (name == "input_ids")
            return trtmc::DType::kInt32;
        return trtmc::DType::kFloat32;
    }

    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "input_ids" || name == "attention_mask")
            return {1, kMaxLength};
        if (name == "text_embeddings")
            return {1, kMaxLength, kTextDim};
        return {1};
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto ids_it = inputs.find("input_ids");
        const auto mask_it = inputs.find("attention_mask");
        if (ids_it == inputs.end() || mask_it == inputs.end() || !ids_it->second.data ||
            !mask_it->second.data) {
            throw std::runtime_error("missing T5 encoder input");
        }
        const auto* ids = static_cast<const int32_t*>(ids_it->second.data);
        const auto* mask = static_cast<const float*>(mask_it->second.data);
        saw_prompt_ids = ids[0] == 5 && ids[1] == 1 && ids[2] == 1;
        saw_prompt_mask = close_enough(mask[0], 0.0F) && mask[1] < -1e8F && mask[2] < -1e8F;

        output_.assign(kMaxLength * kTextDim, 0.0F);
        output_[0] = 19.0F;
        output_[1] = 17.0F;
        trtmc::Tensor tensor;
        tensor.data = output_.data();
        tensor.shape = {1, kMaxLength, kTextDim};
        tensor.dtype = trtmc::DType::kFloat32;
        return {{"text_embeddings", tensor}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& /*inputs*/) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<int64_t>
    input_profile_shape(const std::string& name, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& /*name*/) const override { return nullptr; }
    void bind_external(const std::string& /*name*/, void* /*ptr*/) override {}
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }

    bool saw_prompt_ids{false};
    bool saw_prompt_mask{false};

  private:
    static constexpr int32_t kMaxLength = 3;
    static constexpr int32_t kTextDim = 2;

    std::vector<float> output_;
    std::shared_ptr<void> keep_alive_;
};

class FakeElfModule final : public trtmc::TrtModule {
  public:
    bool ok() const override { return true; }

    std::vector<trtmc::TensorInfo> input_info() const override {
        return {trtmc::TensorInfo{"latent", {kMaxLength, kInputDim}, trtmc::DType::kFloat32, true},
                trtmc::TensorInfo{"timestep", {1}, trtmc::DType::kFloat32, true},
                trtmc::TensorInfo{"decoder_mode", {1}, trtmc::DType::kFloat32, true},
                trtmc::TensorInfo{"self_cond_cfg_scale", {1}, trtmc::DType::kFloat32, true}};
    }

    std::vector<trtmc::TensorInfo> output_info() const override {
        return {
            trtmc::TensorInfo{"denoised", {kMaxLength, kTextDim}, trtmc::DType::kFloat32, false},
            trtmc::TensorInfo{
                "decoder_logits", {kMaxLength, kVocabSize}, trtmc::DType::kFloat32, false}};
    }

    bool has_input(const std::string& name) const override {
        return name == "latent" || name == "timestep" || name == "decoder_mode" ||
               name == "self_cond_cfg_scale";
    }

    bool has_output(const std::string& name) const override {
        return name == "denoised" || name == "decoder_logits";
    }

    trtmc::DType tensor_dtype(const std::string& /*name*/) const override {
        return trtmc::DType::kFloat32;
    }

    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "latent")
            return {kMaxLength, kInputDim};
        if (name == "denoised")
            return {kMaxLength, kTextDim};
        if (name == "decoder_logits")
            return {kMaxLength, kVocabSize};
        return {1};
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const float decoder_mode = scalar_input(inputs, "decoder_mode");
        const float self_cond_cfg = scalar_input(inputs, "self_cond_cfg_scale");
        const auto latent_it = inputs.find("latent");
        if (latent_it == inputs.end() || !latent_it->second.data)
            throw std::runtime_error("missing latent input");
        const auto* latent = static_cast<const float*>(latent_it->second.data);

        if (decoder_mode < 0.5F)
            return forward_denoise(inputs, latent, self_cond_cfg);
        return forward_decode(inputs, latent, self_cond_cfg);
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& /*inputs*/) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<int64_t>
    input_profile_shape(const std::string& name, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& /*name*/) const override { return nullptr; }
    void bind_external(const std::string& /*name*/, void* /*ptr*/) override {}
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }

    bool saw_denoise_call{false};
    bool saw_decode_call{false};
    bool decode_received_final_latent{false};
    bool expect_condition{false};
    bool require_zero_denoise_timestep{true};
    bool check_first_cond_uses_initial_latent{false};
    bool saw_cond_cfg_call{false};
    bool saw_uncond_cfg_call{false};
    bool saw_first_cond_initial_latent{false};
    bool check_first_denoise_latent{false};
    float expected_self_cond_cfg{1.0F};
    int32_t denoise_calls{0};
    std::vector<float> expected_initial_latent;
    std::vector<float> expected_first_denoise_latent;
    std::vector<float> denoise_timesteps;

  private:
    trtmc::TensorMap forward_denoise(const trtmc::TensorMap& inputs, const float* latent,
                                     float self_cond_cfg) {
        saw_denoise_call = true;
        ++denoise_calls;
        denoise_timesteps.push_back(scalar_input(inputs, "timestep"));
        if (require_zero_denoise_timestep) {
            check(close_enough(scalar_input(inputs, "timestep"), 0.0F),
                  "denoise call uses first upstream ODE timestep");
        }
        check(close_enough(self_cond_cfg, expected_self_cond_cfg),
              "denoise call forwards self-cond CFG scale");

        if (expect_condition) {
            const bool cond_call = close_enough(latent[0], 9.0F) && close_enough(latent[1], 8.0F) &&
                                   close_enough(latent[2], 9.0F) && close_enough(latent[3], 8.0F);
            const bool uncond_call = close_enough(latent[0], 0.0F) &&
                                     close_enough(latent[1], 0.0F) &&
                                     close_enough(latent[2], 0.0F) && close_enough(latent[3], 0.0F);
            saw_cond_cfg_call = saw_cond_cfg_call || cond_call;
            saw_uncond_cfg_call = saw_uncond_cfg_call || uncond_call;
            check(cond_call || uncond_call,
                  "conditional denoise receives either cond or zeroed CFG prefix");
            if (cond_call && check_first_cond_uses_initial_latent &&
                !saw_first_cond_initial_latent) {
                bool matches_initial = expected_initial_latent.size() == kMaxLength * kTextDim;
                for (int32_t row = 1; row < kMaxLength && matches_initial; ++row) {
                    for (int32_t col = 0; col < kTextDim; ++col) {
                        const auto latent_idx = row * kInputDim + col;
                        const auto expected_idx = row * kTextDim + col;
                        matches_initial =
                            matches_initial &&
                            close_enough(latent[latent_idx], expected_initial_latent[expected_idx]);
                    }
                }
                saw_first_cond_initial_latent = matches_initial;
                check(matches_initial,
                      "conditional default sampling starts with ODE latent, not SDE backstep");
            }
        }
        if (check_first_denoise_latent && denoise_calls == 1) {
            bool matches = expected_first_denoise_latent.size() == kMaxLength * kTextDim;
            for (int32_t row = 0; row < kMaxLength && matches; ++row) {
                for (int32_t col = 0; col < kTextDim; ++col) {
                    matches = matches &&
                              close_enough(latent[row * kInputDim + col],
                                           expected_first_denoise_latent[row * kTextDim + col]);
                }
            }
            check(matches, "raw SDE noise controls first denoise latent");
        }

        output_.resize(kMaxLength * kTextDim);
        last_denoised_.resize(kMaxLength * kTextDim);
        for (int32_t row = 0; row < kMaxLength; ++row) {
            for (int32_t col = 0; col < kTextDim; ++col) {
                const float value = latent[row * kInputDim + col] + 0.25F;
                output_[row * kTextDim + col] = value;
                last_denoised_[row * kTextDim + col] = value;
            }
            for (int32_t col = kTextDim; col < kInputDim; ++col) {
                const int32_t first_step_calls = expect_condition ? 2 : 1;
                if (denoise_calls <= first_step_calls && (!expect_condition || row != 0)) {
                    check(close_enough(latent[row * kInputDim + col], 0.0F),
                          "first denoise call receives zero self-conditioning");
                }
            }
        }

        trtmc::Tensor tensor;
        tensor.data = output_.data();
        tensor.shape = {kMaxLength, kTextDim};
        tensor.dtype = trtmc::DType::kFloat32;
        return {{"denoised", tensor}};
    }

    trtmc::TensorMap forward_decode(const trtmc::TensorMap& inputs, const float* latent,
                                    float self_cond_cfg) {
        saw_decode_call = true;
        check(close_enough(scalar_input(inputs, "timestep"), 1.0F),
              "decode call uses upstream final decoder timestep");
        check(close_enough(self_cond_cfg, 1.0F), "decode call forwards self-cond CFG scale");

        decode_received_final_latent = true;
        for (int32_t row = 0; row < kMaxLength; ++row) {
            for (int32_t col = 0; col < kTextDim; ++col) {
                float expected = last_denoised_[row * kTextDim + col];
                if (expect_condition && row == 0)
                    expected = col == 0 ? 9.0F : 8.0F;
                decode_received_final_latent &=
                    close_enough(latent[row * kInputDim + col], expected);
            }
            for (int32_t col = kTextDim; col < kInputDim; ++col) {
                decode_received_final_latent &= close_enough(latent[row * kInputDim + col], 0.0F);
            }
        }

        output_.assign(kMaxLength * kVocabSize, -100.0F);
        for (int32_t row = 0; row < kMaxLength; ++row)
            output_[row * kVocabSize + row + 1] = 100.0F;

        trtmc::Tensor tensor;
        tensor.data = output_.data();
        tensor.shape = {kMaxLength, kVocabSize};
        tensor.dtype = trtmc::DType::kFloat32;
        return {{"decoder_logits", tensor}};
    }

    static constexpr int32_t kMaxLength = 3;
    static constexpr int32_t kTextDim = 2;
    static constexpr int32_t kInputDim = 4;
    static constexpr int32_t kVocabSize = 4;

    std::vector<float> output_;
    std::vector<float> last_denoised_;
    std::shared_ptr<void> keep_alive_;
};

void test_generate_runs_upstream_shape_contract() {
    auto module = std::make_unique<FakeElfModule>();
    auto* module_ptr = module.get();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    trtmc::ElfFlowPipeline pipeline(std::move(module), 3, 0, 4, 2, 4, 2.0F, -1.5F, 0.8F, 0.05F,
                                    tokenizer, "fake-elf");

    trtmc::GenerateConfig cfg;
    cfg.num_steps = 1;
    cfg.guidance_scale = 1.0F;
    cfg.sde_gamma = 0.0F;
    cfg.initial_latents = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};

    const auto result = pipeline.generate("", cfg);
    check(result.text == "ABC", "ELF generate decodes argmax tokens to text");
    check((result.token_ids == std::vector<int32_t>{1, 2, 3}),
          "ELF generate returns decoded token ids");
    check(module_ptr->saw_denoise_call, "ELF generate runs denoiser");
    check(module_ptr->saw_decode_call, "ELF generate runs final decoder head");
    check(module_ptr->decode_received_final_latent,
          "ELF final decoder receives denoised latent and zero self-conditioning");
}

void test_conditional_generate_restores_prefix_and_strips_decoded_condition() {
    auto module = std::make_unique<FakeElfModule>();
    module->expect_condition = true;
    auto* module_ptr = module.get();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    trtmc::ElfFlowPipeline pipeline(std::move(module), 3, 0, 4, 2, 4, 2.0F, -1.5F, 0.8F, 0.05F,
                                    tokenizer, "fake-elf");

    trtmc::GenerateConfig cfg;
    cfg.num_steps = 1;
    cfg.guidance_scale = 1.0F;
    cfg.cfg_scale = 2.0F;
    cfg.sde_gamma = 0.0F;
    cfg.initial_latents = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
    cfg.condition_latents = {9.0F, 8.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    cfg.condition_mask = {1.0F, 0.0F, 0.0F};

    const auto result = pipeline.generate("", cfg);
    check(result.text == "BC", "conditional ELF strips decoded condition prefix");
    check((result.token_ids == std::vector<int32_t>{2, 3}),
          "conditional ELF returns only generated token ids");
    check(module_ptr->denoise_calls == 2, "conditional CFG runs cond and uncond denoise passes");
    check(module_ptr->saw_cond_cfg_call, "conditional CFG pass receives condition latents");
    check(module_ptr->saw_uncond_cfg_call, "unconditional CFG pass zeroes condition latents");
    check(module_ptr->decode_received_final_latent,
          "conditional final decoder receives restored condition prefix");
}

void test_conditional_defaults_match_upstream_sampling_config() {
    auto module = std::make_unique<FakeElfModule>();
    module->expect_condition = true;
    module->require_zero_denoise_timestep = false;
    module->check_first_cond_uses_initial_latent = true;
    auto* module_ptr = module.get();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    trtmc::ElfFlowPipeline pipeline(std::move(module), 3, 1, 4, 2, 4, 2.0F, -1.5F, 0.8F, 0.05F,
                                    tokenizer, "fake-elf");

    trtmc::GenerateConfig cfg;
    cfg.initial_latents = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
    cfg.condition_latents = {9.0F, 8.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    cfg.condition_mask = {1.0F, 0.0F, 0.0F};
    module_ptr->expected_initial_latent = cfg.initial_latents;

    const auto result = pipeline.generate("", cfg);
    check(result.text == "BC", "conditional default ELF strips decoded condition prefix");
    check(module_ptr->denoise_calls == 128,
          "conditional ELF defaults to upstream 64 ODE steps with CFG scale 2");
    check(module_ptr->saw_cond_cfg_call, "conditional default CFG receives condition latents");
    check(module_ptr->saw_uncond_cfg_call, "conditional default CFG zeroes condition latents");
    check(module_ptr->saw_first_cond_initial_latent,
          "conditional default uses upstream ODE sampler with no SDE churn");
}

void test_eos_is_stripped_before_returning_token_ids() {
    auto module = std::make_unique<FakeElfModule>();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    tokenizer->eos_id = 2;
    trtmc::ElfFlowPipeline pipeline(std::move(module), 3, 0, 4, 2, 4, 2.0F, -1.5F, 0.8F, 0.05F,
                                    tokenizer, "fake-elf");

    trtmc::GenerateConfig cfg;
    cfg.num_steps = 1;
    cfg.guidance_scale = 1.0F;
    cfg.sde_gamma = 0.0F;
    cfg.initial_latents = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};

    const auto result = pipeline.generate("", cfg);
    check(result.text == "A", "ELF decode stops before EOS text");
    check((result.token_ids == std::vector<int32_t>{1}),
          "ELF decode stops before returning EOS token id");
}

void test_raw_sampling_steps_and_sde_noise_replay_upstream_inputs() {
    auto module = std::make_unique<FakeElfModule>();
    module->require_zero_denoise_timestep = false;
    module->check_first_denoise_latent = true;
    module->expected_first_denoise_latent = {1.05F, 1.10F, 1.15F, 1.20F, 1.25F, 1.30F};
    auto* module_ptr = module.get();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    trtmc::ElfFlowPipeline pipeline(std::move(module), 3, 0, 4, 2, 4, 2.0F, -1.5F, 0.8F, 0.05F,
                                    tokenizer, "fake-elf");

    trtmc::GenerateConfig cfg;
    cfg.guidance_scale = 1.0F;
    cfg.sde_gamma = 1.0F;
    cfg.initial_latents = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
    cfg.sampling_steps = {0.0F, 0.5F, 1.0F};
    cfg.sde_noises = {2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F};

    const auto result = pipeline.generate("", cfg);
    check(result.text == "ABC", "raw replay still decodes text");
    check(module_ptr->denoise_timesteps.size() == 2, "raw replay runs one SDE and final ODE step");
    check(close_enough(module_ptr->denoise_timesteps[0], 0.0F),
          "raw replay first timestep uses supplied start");
    check(close_enough(module_ptr->denoise_timesteps[1], 0.5F),
          "raw replay final ODE uses supplied penultimate timestep");
}

void test_prompt_condition_uses_bundled_t5_encoder() {
    auto module = std::make_unique<FakeElfModule>();
    module->expect_condition = true;
    auto* module_ptr = module.get();
    auto text_encoder = std::make_unique<FakeTextEncoderModule>();
    auto* text_encoder_ptr = text_encoder.get();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    tokenizer->eos_id = 1;
    tokenizer->encoded_ids = {1, 5, 1};
    trtmc::ElfFlowPipeline pipeline(std::move(module), 3, 1, 4, 2, 4, 2.0F, -1.5F, 0.8F, 0.05F,
                                    tokenizer, "fake-elf", std::move(text_encoder), 1.0F, 2.0F, 1);

    trtmc::GenerateConfig cfg;
    cfg.num_steps = 1;
    cfg.guidance_scale = 1.0F;
    cfg.cfg_scale = 2.0F;
    cfg.sde_gamma = 0.0F;
    cfg.initial_latents = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};

    const auto result = pipeline.generate("Quelle", cfg);
    check(result.text == "BC", "prompt-conditioned ELF strips source-token prefix");
    check(text_encoder_ptr->saw_prompt_ids,
          "prompt-conditioned ELF strips tokenizer-added EOS before T5 encoding");
    check(text_encoder_ptr->saw_prompt_mask,
          "prompt-conditioned ELF builds additive T5 attention mask");
    check(module_ptr->saw_cond_cfg_call,
          "prompt-conditioned ELF normalizes T5 embeddings into condition latents");
    check(module_ptr->saw_uncond_cfg_call,
          "prompt-conditioned ELF still runs upstream conditional CFG");
}

void test_prompt_condition_requires_bundled_t5_encoder() {
    auto module = std::make_unique<FakeElfModule>();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    tokenizer->encoded_ids = {5};
    trtmc::ElfFlowPipeline pipeline(std::move(module), 3, 1, 4, 2, 4, 2.0F, -1.5F, 0.8F, 0.05F,
                                    tokenizer, "fake-elf");

    trtmc::GenerateConfig cfg;
    cfg.num_steps = 1;
    cfg.guidance_scale = 1.0F;
    cfg.sde_gamma = 0.0F;
    cfg.initial_latents = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};

    bool threw = false;
    try {
        (void)pipeline.generate("Quelle", cfg);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("elf_text_encoder_plan") != std::string::npos;
    }
    check(threw, "ELF rejects text prompt when the bundle has no T5 encoder");
}

void test_condition_api_requires_latents_and_mask() {
    {
        auto module = std::make_unique<FakeElfModule>();
        auto tokenizer = std::make_shared<FakeTokenizer>();
        trtmc::ElfFlowPipeline pipeline(std::move(module), 3, 0, 4, 2, 4, 2.0F, -1.5F, 0.8F, 0.05F,
                                        tokenizer, "fake-elf");

        trtmc::GenerateConfig cfg;
        cfg.num_steps = 1;
        cfg.guidance_scale = 1.0F;
        cfg.sde_gamma = 0.0F;
        cfg.initial_latents = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
        cfg.condition_latents = {9.0F, 8.0F, 0.0F, 0.0F, 0.0F, 0.0F};

        bool threw = false;
        try {
            (void)pipeline.generate("ignored", cfg);
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("condition_latents") != std::string::npos;
        }
        check(threw, "ELF rejects condition latents without condition mask");
    }
    {
        auto module = std::make_unique<FakeElfModule>();
        auto tokenizer = std::make_shared<FakeTokenizer>();
        trtmc::ElfFlowPipeline pipeline(std::move(module), 3, 0, 4, 2, 4, 2.0F, -1.5F, 0.8F, 0.05F,
                                        tokenizer, "fake-elf");

        trtmc::GenerateConfig cfg;
        cfg.num_steps = 1;
        cfg.guidance_scale = 1.0F;
        cfg.sde_gamma = 0.0F;
        cfg.initial_latents = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
        cfg.condition_mask = {1.0F, 0.0F, 0.0F};

        bool threw = false;
        try {
            (void)pipeline.generate("ignored", cfg);
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("condition_mask") != std::string::npos;
        }
        check(threw, "ELF rejects condition mask without condition latents");
    }
}

} // namespace

int main() {
    test_generate_runs_upstream_shape_contract();
    test_conditional_generate_restores_prefix_and_strips_decoded_condition();
    test_conditional_defaults_match_upstream_sampling_config();
    test_eos_is_stripped_before_returning_token_ids();
    test_raw_sampling_steps_and_sde_noise_replay_upstream_inputs();
    test_prompt_condition_uses_bundled_t5_encoder();
    test_prompt_condition_requires_bundled_t5_encoder();
    test_condition_api_requires_latents_and_mask();
    return g_failures == 0 ? 0 : 1;
}
