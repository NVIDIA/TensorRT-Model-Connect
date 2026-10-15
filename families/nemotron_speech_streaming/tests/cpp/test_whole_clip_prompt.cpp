/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_speech_streaming/runtime/pipeline.h"
#include "families/nemotron_speech_streaming/runtime/plugin_helpers.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
using namespace trtmc;

class Module final : public ITrtModule {
  public:
    explicit Module(std::function<TensorMap(const TensorMap&)> invoke)
        : invoke_(std::move(invoke)) {}
    TensorMap forward(const TensorMap& inputs) override { return invoke_(inputs); }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const DeviceTensorMap&) override { throw std::logic_error("unused"); }
    void forward_async(const TensorMap&) override { throw std::logic_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string&) const override { return DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {1, 2}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {1, 2};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 2; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::function<TensorMap(const TensorMap&)> invoke_;
};

class Backend final : public IBackend {
  public:
    explicit Backend(std::function<std::unique_ptr<ITrtModule>()> create)
        : create_(std::move(create)) {}
    std::unique_ptr<ITrtModule> create_module(const void*, size_t,
                                              const ModuleCreateOptions&) override {
        return create_();
    }
    std::unique_ptr<ITrtModule>
    create_module_prebound(const void*, size_t, const ModuleCreateOptions&,
                           const std::vector<ModuleExternalBinding>&) override {
        throw std::logic_error("unused");
    }
    BackendDualProfileModules create_dual_profile_modules(const void*, size_t,
                                                          const ModuleCreateOptions&) override {
        throw std::logic_error("unused");
    }
    const char* name() const override { return "test"; }

  private:
    std::function<std::unique_ptr<ITrtModule>()> create_;
};

Tensor tensor(std::vector<float>& values) {
    Tensor result;
    result.data = values.data();
    result.shape = {1, static_cast<int64_t>(values.size())};
    result.dtype = DType::kFloat32;
    return result;
}

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        std::vector<float> features{1.0F, 2.0F}, state(2, 0.0F), logits{1.0F, -1.0F};
        std::vector<std::vector<float>> prompts;
        auto encoder = std::make_unique<Module>(
            [&](const TensorMap&) { return TensorMap{{"encoder_output", tensor(features)}}; });
        auto predictor = std::make_unique<Module>([&](const TensorMap&) {
            return TensorMap{{"pred_output", tensor(state)},
                             {"next_h_0", tensor(state)},
                             {"next_c_0", tensor(state)}};
        });
        auto joint = std::make_unique<Module>(
            [&](const TensorMap&) { return TensorMap{{"logits", tensor(logits)}}; });
        auto prompt = std::make_unique<Module>([&](const TensorMap& inputs) {
            const auto& value = inputs.at("prompt_onehot");
            check(value.data != nullptr && value.shape == std::vector<int64_t>({1, 3}),
                  "whole-clip decoding must supply an initialized prompt tensor");
            const auto* data = static_cast<const float*>(value.data);
            prompts.emplace_back(data, data + 3);
            return TensorMap{{"prompt_kernel_output", tensor(features)}};
        });
        RnntConfig config;
        config.encoder_hidden_size = config.pred_hidden_size = 2;
        config.has_prompt_kernel = true;
        config.num_prompts = 3;
        config.prompt_dictionary = {{"en-US", 2}, {"de-DE", 1}};
        config.num_mel_bins = 2;
        config.mel_n_fft = config.mel_win_length = 16;
        config.mel_hop_length = 8;
        config.mel_chunk_length = 1;
        config.mel_length = 32;
        config.encoder_layers = 1;
        config.att_context_right = 0;
        config.supported_right_contexts = {0};
        MelFilterbank bank;
        bank.n_freq_bins = 9;
        bank.n_mel_bins = 2;
        bank.data.assign(18, 1.0F);
        Backend backend([&] {
            return std::make_unique<Module>([&](const TensorMap&) {
                return TensorMap{{"encoder_output", tensor(features)},
                                 {"cache_last_channel_next", tensor(state)},
                                 {"cache_last_time_next", tensor(state)}};
            });
        });
        RnntPipeline pipeline(std::move(encoder), std::move(predictor), std::move(joint),
                              std::move(prompt), {{0, {'p'}}}, &backend, {}, {{0, {'p'}}}, config,
                              bank, nullptr);
        const std::vector<float> audio(128, 0.1F);
        pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()));
        check(prompts.size() == 1 && prompts.back() == std::vector<float>({0, 0, 1}),
              "default English uses the dictionary's en-US index rather than assuming zero");
        TranscriptionConfig request;
        request.source_language = "de-DE";
        pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()), request);
        check(prompts.size() == 2 && prompts.back() == std::vector<float>({0, 1, 0}),
              "an explicit language changes the prompt used for whole-clip decoding");
        pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()));
        check(prompts.size() == 3 && prompts.back() == std::vector<float>({0, 0, 1}),
              "each request resets the prompt instead of reusing the previous language");
        TranscriptionStreamConfig stream_config;
        stream_config.att_context_right = 0;
        stream_config.language = "de-DE";
        auto german = pipeline.create_transcription_stream(stream_config);
        pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()));
        check(prompts.back() == std::vector<float>({0, 0, 1}),
              "whole-clip decoding uses English while a German stream exists");
        auto before = prompts.size();
        german->accept_audio(audio.data(), static_cast<int32_t>(audio.size()), true);
        check(prompts.size() > before, "the German stream actually executes prompt decoding");
        for (auto index = before; index < prompts.size(); ++index)
            check(prompts[index] == std::vector<float>({0, 1, 0}),
                  "whole-clip English does not overwrite an existing stream's German prompt");
        german->reset();
        stream_config.language = "en-US";
        auto english = pipeline.create_transcription_stream(stream_config);
        english->accept_audio(audio.data(), static_cast<int32_t>(audio.size()), true);
        check(prompts.back() == std::vector<float>({0, 0, 1}),
              "a second stream owns its English prompt");
        before = prompts.size();
        german->accept_audio(audio.data(), static_cast<int32_t>(audio.size()), true);
        check(prompts.size() > before, "the reset German stream executes prompt decoding");
        for (auto index = before; index < prompts.size(); ++index)
            check(prompts[index] == std::vector<float>({0, 1, 0}),
                  "stream reset and another stream preserve the original German prompt");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
