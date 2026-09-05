/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/kv_cache.h"
#include "families/qwen3_omni/runtime/pipeline.h"
#include "families/qwen3_omni/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class OmniFixedTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string& text) const override {
        if (text.find("hello<|im_end|>") != std::string::npos) {
            return {
                8, 9,  20,            // system
                8, 10, 21,            // user
                8, 11, 22, 23, 24, 3, // assistant and final <|im_end|>
            };
        }
        return {4, 5, 6};
    }

    std::string decode(const std::vector<std::int32_t>&) const override { return "hello"; }

    std::int32_t id_for_token(std::string_view token) const override {
        return token == "<|endoftext|>" ? 7 : -1;
    }

    std::string token_for_id(std::int32_t id) const override { return id == 1 ? "hello" : ""; }
};

struct ModuleStats {
    std::int32_t launches{0};
    std::unordered_map<std::string, std::vector<std::int64_t>> input_shapes;
};

enum class ModuleKind {
    kThinkerPrefill,
    kThinkerDecode,
    kProjection,
    kTalkerPrefill,
    kTalkerDecode,
    kPredictorPrefill,
    kPredictorDecode,
    kCode2Wav,
};

class FakeOmniModule final : public trtmc::ITrtModule {
  public:
    FakeOmniModule(ModuleKind kind, std::shared_ptr<ModuleStats> stats, cudaStream_t stream,
                   std::int32_t max_length, std::int32_t vocab_size, std::int32_t hidden_size,
                   std::int32_t waveform_samples = 0)
        : kind_(kind), stats_(std::move(stats)), stream_(stream), max_length_(max_length),
          vocab_size_(vocab_size), hidden_size_(hidden_size),
          present_k_(std::vector<std::int64_t>{max_length, 1}, trtmc::DType::kFloat32, stream),
          present_v_(std::vector<std::int64_t>{max_length, 1}, trtmc::DType::kFloat32, stream) {
        logits_.assign(static_cast<std::size_t>(std::max(vocab_size_, 0)), -100.0F);
        if (kind_ == ModuleKind::kThinkerPrefill && logits_.size() > 2) {
            logits_[1] = 10.0F;
            logits_[2] = 10.0F;
        } else if (kind_ == ModuleKind::kThinkerDecode && logits_.size() > 3) {
            logits_[3] = 10.0F;
        } else if ((kind_ == ModuleKind::kTalkerPrefill || kind_ == ModuleKind::kTalkerDecode) &&
                   logits_.size() > 1) {
            logits_[1] = 100.0F;
        } else if ((kind_ == ModuleKind::kPredictorPrefill ||
                    kind_ == ModuleKind::kPredictorDecode) &&
                   logits_.size() > 2) {
            logits_[2] = 100.0F;
        }
        hidden_.assign(static_cast<std::size_t>(std::max(hidden_size_, 0)), 0.25F);
        waveform_.assign(static_cast<std::size_t>(std::max(waveform_samples, 0)), 0.125F);
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++stats_->launches;
        stats_->input_shapes.clear();
        for (const auto& [name, tensor] : inputs)
            stats_->input_shapes[name] = tensor.shape;

        if (kind_ == ModuleKind::kProjection) {
            const auto rows = inputs.at("token_id").shape.at(0);
            embeddings_.assign(static_cast<std::size_t>(rows) * hidden_size_, 0.5F);
            return {
                {"embeddings",
                 trtmc::Tensor{embeddings_.data(), {rows, hidden_size_}, trtmc::DType::kFloat32}}};
        }
        if (kind_ == ModuleKind::kCode2Wav) {
            return {{"waveform", trtmc::Tensor{waveform_.data(),
                                               {1, 1, static_cast<std::int64_t>(waveform_.size())},
                                               trtmc::DType::kFloat32}}};
        }

        const auto token = inputs.find("token_id");
        const auto embedding = inputs.find("input_embed");
        const std::int64_t rows =
            token != inputs.end() ? token->second.shape.at(0) : embedding->second.shape.at(0);
        present_host_.assign(static_cast<std::size_t>(rows), 0.0F);
        trtmc::TensorMap outputs{
            {"present_k_0", trtmc::Tensor{present_host_.data(), {rows, 1}, trtmc::DType::kFloat32}},
            {"present_v_0", trtmc::Tensor{present_host_.data(), {rows, 1}, trtmc::DType::kFloat32}},
        };
        if (kind_ == ModuleKind::kPredictorPrefill || kind_ == ModuleKind::kPredictorDecode) {
            for (std::int32_t group = 0; group < 15; ++group) {
                outputs["logits_" + std::to_string(group)] =
                    trtmc::Tensor{logits_.data(), {1, vocab_size_}, trtmc::DType::kFloat32};
            }
        } else {
            outputs["logits"] =
                trtmc::Tensor{logits_.data(), {1, vocab_size_}, trtmc::DType::kFloat32};
        }
        if (kind_ == ModuleKind::kTalkerPrefill || kind_ == ModuleKind::kTalkerDecode ||
            kind_ == ModuleKind::kPredictorPrefill || kind_ == ModuleKind::kPredictorDecode) {
            outputs["hidden_state"] =
                trtmc::Tensor{hidden_.data(), {1, hidden_size_}, trtmc::DType::kFloat32};
        }
        return outputs;
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override { cudaStreamSynchronize(stream_); }
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }

    bool has_input(const std::string& name) const override {
        if (kind_ == ModuleKind::kProjection)
            return name == "token_id";
        if (kind_ == ModuleKind::kCode2Wav)
            return name == "codec_tokens";
        return name == "token_id" || name == "input_embed" || name == "position_id" ||
               name == "attention_mask" || name == "cache_k_0" || name == "cache_v_0";
    }

    bool has_output(const std::string& name) const override {
        return name == "logits" || name == "hidden_state" || name == "present_k_0" ||
               name == "present_v_0" || name == "waveform" || name.rfind("logits_", 0) == 0;
    }

    trtmc::DType tensor_dtype(const std::string& name) const override {
        return name == "position_id" || name == "codec_tokens" ? trtmc::DType::kInt32
                                                               : trtmc::DType::kFloat32;
    }

    std::vector<std::int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_k_0" || name == "cache_v_0")
            return {max_length_, 1};
        if (name == "present_k_0" || name == "present_v_0")
            return {1, 1};
        if (name == "position_id")
            return {1};
        if (name == "attention_mask")
            return {1, max_length_ + 1};
        if (name == "logits" || name.rfind("logits_", 0) == 0)
            return {1, vocab_size_};
        return {};
    }

    std::vector<std::int64_t> input_profile_shape(const std::string&, std::int32_t,
                                                  trtmc::ProfileShapeSelector) const override {
        return {};
    }

    std::int32_t optimization_profile_count() const override { return 1; }

    void* device_ptr(const std::string& name) const override {
        const auto bound = bindings_.find(name);
        if (bound != bindings_.end())
            return bound->second;
        if (name == "present_k_0")
            return const_cast<void*>(present_k_.data());
        if (name == "present_v_0")
            return const_cast<void*>(present_v_.data());
        return nullptr;
    }

    void bind_external(const std::string& name, void* pointer) override {
        bindings_[name] = pointer;
    }

    void bind_external(const std::string& name, void* pointer,
                       const std::vector<std::int64_t>&) override {
        bind_external(name, pointer);
    }

    std::int32_t input_rank(const std::string& name) const override {
        return name == "token_id" || name == "position_id" ? 1 : 2;
    }

    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return present_k_.ok() && present_v_.ok(); }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    ModuleKind kind_;
    std::shared_ptr<ModuleStats> stats_;
    cudaStream_t stream_{nullptr};
    std::int32_t max_length_{0};
    std::int32_t vocab_size_{0};
    std::int32_t hidden_size_{0};
    mutable std::unordered_map<std::string, void*> bindings_;
    mutable trtmc::DeviceTensor present_k_;
    mutable trtmc::DeviceTensor present_v_;
    std::vector<float> logits_;
    std::vector<float> hidden_;
    std::vector<float> waveform_;
    std::vector<float> embeddings_;
    std::vector<float> present_host_;
};

struct PipelineStats {
    std::shared_ptr<ModuleStats> thinker_prefill = std::make_shared<ModuleStats>();
    std::shared_ptr<ModuleStats> thinker_decode = std::make_shared<ModuleStats>();
    std::shared_ptr<ModuleStats> code2wav = std::make_shared<ModuleStats>();
};

trtmc::Qwen3OmniRuntimeConfig make_config() {
    trtmc::Qwen3OmniRuntimeConfig config;
    config.precision = "bf16";
    config.sample_rate = 24000;
    config.thinker_hidden_size = 4;
    config.thinker_num_layers = 1;
    config.thinker_vocab_size = 8;
    config.thinker_max_cache_length = 32;
    config.thinker_eos_token_id = 3;
    config.talker_hidden_size = 4;
    config.talker_num_layers = 1;
    config.talker_vocab_size = 2151;
    config.talker_max_cache_length = 64;
    config.predictor_hidden_size = 4;
    config.predictor_num_layers = 1;
    config.predictor_vocab_size = 2048;
    config.predictor_max_cache_length = 32;
    config.num_codebooks = 16;
    config.codebook_size = 2048;
    config.talker_max_frames = 32;
    config.im_start_token_id = 8;
    config.system_token_id = 9;
    config.user_token_id = 10;
    config.assistant_token_id = 11;
    config.tts_bos_token_id = 12;
    config.tts_eos_token_id = 13;
    config.tts_pad_token_id = 14;
    config.codec_bos_id = 4;
    config.codec_eos_token_id = 2150;
    config.codec_nothink_id = 6;
    config.codec_pad_id = 7;
    config.codec_think_bos_id = 8;
    config.codec_think_eos_id = 9;
    config.speaker_id = 10;
    config.code2wav_max_frames = 32;
    config.code2wav_upsample_factor = 1920;
    config.code2wav_output_delay = 555;
    config.code2wav_num_quantizers = 16;
    return config;
}

std::unique_ptr<trtmc::Qwen3OmniKvCache> make_cache(std::int32_t max_length, cudaStream_t stream) {
    return std::make_unique<trtmc::Qwen3OmniKvCache>(1, max_length, 1, stream,
                                                     trtmc::DType::kFloat32);
}

std::unique_ptr<trtmc::Qwen3OmniAudioPipeline>
make_pipeline(cudaStream_t stream, PipelineStats& stats, bool omit_thinker = false) {
    auto fresh_stats = [] { return std::make_shared<ModuleStats>(); };
    auto thinker_prefill =
        omit_thinker ? std::unique_ptr<FakeOmniModule>{}
                     : std::make_unique<FakeOmniModule>(ModuleKind::kThinkerPrefill,
                                                        stats.thinker_prefill, stream, 32, 8, 0);
    auto thinker_decode = std::make_unique<FakeOmniModule>(ModuleKind::kThinkerDecode,
                                                           stats.thinker_decode, stream, 32, 8, 0);
    auto projection =
        std::make_unique<FakeOmniModule>(ModuleKind::kProjection, fresh_stats(), stream, 32, 0, 4);
    auto talker_prefill = std::make_unique<FakeOmniModule>(ModuleKind::kTalkerPrefill,
                                                           fresh_stats(), stream, 64, 2151, 4);
    auto talker_decode = std::make_unique<FakeOmniModule>(ModuleKind::kTalkerDecode, fresh_stats(),
                                                          stream, 64, 2151, 4);
    auto predictor_prefill = std::make_unique<FakeOmniModule>(ModuleKind::kPredictorPrefill,
                                                              fresh_stats(), stream, 32, 2048, 4);
    auto predictor_decode = std::make_unique<FakeOmniModule>(ModuleKind::kPredictorDecode,
                                                             fresh_stats(), stream, 32, 2048, 4);
    constexpr std::int32_t kCode2WavSamples = 32 * 1920 - 555;
    auto code2wav = std::make_unique<FakeOmniModule>(ModuleKind::kCode2Wav, stats.code2wav, stream,
                                                     32, 0, 0, kCode2WavSamples);
    std::vector<float> talker_embedding(2151U * 4U, 0.25F);
    std::vector<float> predictor_embeddings(15U * 2048U * 4U, 0.125F);
    return std::make_unique<trtmc::Qwen3OmniAudioPipeline>(
        std::move(thinker_prefill), std::move(thinker_decode), make_cache(32, stream),
        std::move(projection), std::move(talker_prefill), std::move(talker_decode),
        make_cache(64, stream), std::move(predictor_prefill), std::move(predictor_decode),
        make_cache(32, stream), std::move(code2wav), std::move(talker_embedding),
        std::move(predictor_embeddings), make_config(), std::make_shared<OmniFixedTokenizer>());
}

class StreamFixture {
  public:
    StreamFixture() { cudaStreamCreate(&stream); }
    ~StreamFixture() { cudaStreamDestroy(stream); }
    cudaStream_t stream{nullptr};
};

void test_omni_pipeline_construction() {
    StreamFixture fixture;
    PipelineStats stats;
    auto pipeline = make_pipeline(fixture.stream, stats);
    check(std::string(pipeline->task()) == trtmc::IAudioGeneration::kTask,
          "Qwen3-Omni exposes the audio-generation Task API");
    check(pipeline->default_max_new_tokens() == 128,
          "Qwen3-Omni preserves its default text token limit");
}

void test_omni_generate_audio() {
    StreamFixture fixture;
    PipelineStats stats;
    auto pipeline = make_pipeline(fixture.stream, stats);
    trtmc::AudioGenerationConfig config;
    config.max_new_tokens = 1;
    config.talker_max_new_tokens = 2;
    config.seed = 7;
    const auto result = pipeline->generate_audio("say hello", config);
    check(result.num_samples == 1920 - 555,
          "Qwen3-Omni trims strict Code2Wav output to the generated frame count");
    check(result.samples.size() == static_cast<std::size_t>(result.num_samples) &&
              result.samples.front() == 0.125F,
          "Qwen3-Omni returns the Code2Wav waveform");
    check(result.sample_rate == 24000, "Qwen3-Omni preserves the checkpoint sample rate");
    check(stats.code2wav->launches == 1 && stats.code2wav->input_shapes["codec_tokens"] ==
                                               std::vector<std::int64_t>({1, 16, 32}),
          "Qwen3-Omni invokes Code2Wav with the strict padded codec layout");
}

void test_omni_validates_thinker() {
    StreamFixture fixture;
    PipelineStats stats;
    bool threw = false;
    try {
        auto pipeline = make_pipeline(fixture.stream, stats, true);
        (void)pipeline;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "Qwen3-Omni rejects a missing required Thinker component");
}

void test_omni_batched_prefill_and_argmax() {
    StreamFixture fixture;
    PipelineStats stats;
    auto pipeline = make_pipeline(fixture.stream, stats);
    trtmc::TextGenerationConfig config;
    config.max_new_tokens = 2;
    const auto result = pipeline->generate("question", config);
    check(result.text == "hello" && result.token_ids == std::vector<std::int32_t>({1}),
          "Qwen3-Omni keeps lowest-index argmax and stops on the decode EOS");
    check(stats.thinker_prefill->launches == 1 && stats.thinker_decode->launches == 1,
          "Qwen3-Omni uses one batched prefill and one decode launch");
    check(stats.thinker_prefill->input_shapes["token_id"] == std::vector<std::int64_t>({3}) &&
              stats.thinker_prefill->input_shapes["position_id"] ==
                  std::vector<std::int64_t>({3}) &&
              stats.thinker_prefill->input_shapes["attention_mask"] ==
                  std::vector<std::int64_t>({3, 35}),
          "Qwen3-Omni preserves batched prefill token, position, and causal-mask shapes");
}

} // namespace

int main() {
    test_omni_pipeline_construction();
    test_omni_generate_audio();
    test_omni_validates_thinker();
    test_omni_batched_prefill_and_argmax();
    if (failures != 0)
        std::cerr << failures << " Qwen3-Omni pipeline test(s) failed\n";
    return failures;
}
