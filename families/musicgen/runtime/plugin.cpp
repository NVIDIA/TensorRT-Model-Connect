/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/musicgen/runtime/tokenizer.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::musicgen {
using namespace trtmc::internal;

constexpr ConfigField fields[] = {
    {"do_sample", ConfigKind::Bool, ConfigValue{true},
     "Sample audio tokens; false selects greedy decoding."},
    {"seed", ConfigKind::I64, ConfigValue{std::int64_t{0}}, "Request-local sampling seed."},
    {"guidance_scale", ConfigKind::F64, ConfigValue{3.0}, "Classifier-free guidance scale."},
    {"top_k", ConfigKind::I64, ConfigValue{std::int64_t{50}},
     "Number of tokens retained for sampling."},
    {"temperature", ConfigKind::F64, ConfigValue{1.0}, "Positive sampling temperature."},
};

class Pipeline final : public IModel, public ITextToAudio {
  public:
    Pipeline(std::unique_ptr<ITrtModule> text, std::unique_ptr<ITrtModule> decoder,
             std::unique_ptr<ITrtModule> codec, std::unique_ptr<ITokenizer> tokenizer,
             const nlohmann::json& config)
        : text_(std::move(text)), decoder_(std::move(decoder)), codec_(std::move(codec)),
          tokenizer_(std::move(tokenizer)), length_(config.at("sequence_length")),
          frames_(config.at("audio_frames")), text_length_(config.at("text_length")),
          hidden_(config.at("hidden_size")) {
        if (length_ < 11 || length_ > 1504 || frames_ != length_ - 4 || text_length_ != 64 ||
            hidden_ != 1536 || config.at("codebooks") != 4 || config.at("vocab_size") != 2048 ||
            config.at("pad_token_id") != 2048 || config.at("sample_rate") != 32000 ||
            config.at("samples_per_frame") != 640)
            throw std::runtime_error("Invalid MusicGen bundle runtime contract");
        if (decoder_->tensor_shape("audio_ids") != std::vector<int64_t>{2, 4, length_} ||
            codec_->tensor_shape("audio_codes") != std::vector<int64_t>{4, frames_})
            throw std::runtime_error("MusicGen engine shapes disagree with runtime.json");
    }

    const char* task() const noexcept override { return "text_to_audio"; }
    std::vector<TaskInstance> task_bindings() override {
        return {bind<ITextToAudio>(*this, fields)};
    }

    internal::AudioResult run(const TextToAudioRequest& request, ConfigView config) override {
        validate_config(fields, config);
        const bool sample = *config_get<bool>(config, fields, "do_sample");
        const auto seed = *config_get<std::int64_t>(config, fields, "seed");
        const auto top_k = *config_get<std::int64_t>(config, fields, "top_k");
        const double temperature = *config_get<double>(config, fields, "temperature");
        const double guidance = *config_get<double>(config, fields, "guidance_scale");
        if (seed < 0 || top_k < 1 || top_k > 2048 || !std::isfinite(temperature) ||
            temperature <= 0 || !std::isfinite(guidance) || guidance < 1)
            throw ConfigError("MusicGen requires seed >= 0, top_k in [1,2048], temperature > 0, "
                              "guidance_scale >= 1");
        auto token_ids = tokenizer_->encode(std::string(request.prompt));
        if (token_ids.empty() || token_ids.size() > static_cast<std::size_t>(text_length_))
            throw std::invalid_argument("MusicGen prompt must fit in 64 T5 tokens including EOS");
        // Each call starts from BOS and owns its RNG, including after a failed call.
        std::lock_guard<std::mutex> lock(mutex_);
        const auto started = std::chrono::steady_clock::now();
        std::vector<int32_t> input_ids(text_length_, 0);
        std::copy(token_ids.begin(), token_ids.end(), input_ids.begin());
        std::vector<float> mask(text_length_, std::numeric_limits<float>::lowest());
        std::fill_n(mask.begin(), token_ids.size(), 0.0f);
        auto encoded = text_->forward({
            {"input_ids", {input_ids.data(), {1, text_length_}, DType::kInt32}},
            {"attention_mask", {mask.data(), {1, 1, 1, text_length_}, DType::kFloat32}},
        });
        const auto& text_output = encoded.at("encoder_hidden_states");
        if (text_output.dtype != DType::kFloat32 ||
            text_output.numel() != static_cast<std::size_t>(text_length_) * hidden_)
            throw std::runtime_error("Invalid MusicGen text engine output");
        std::vector<float> encoder(2 * text_length_ * hidden_, 0.0f);
        std::copy_n(static_cast<float*>(text_output.data), text_length_ * hidden_, encoder.data());
        // CFG's unconditional encoder states are zero AFTER enc_to_dec_proj.
        std::vector<float> encoder_mask(2 * text_length_, std::numeric_limits<float>::lowest());
        std::copy(mask.begin(), mask.end(), encoder_mask.begin());
        std::vector<int32_t> ids(2 * 4 * length_, 2048);
        std::mt19937_64 random(static_cast<std::uint64_t>(seed));
        std::vector<float> scores(2048);
        std::vector<int32_t> order(2048);
        std::vector<double> probabilities;
        for (int32_t position = 0; position < length_ - 1; ++position) {
            auto outputs = decoder_->forward({
                {"audio_ids", {ids.data(), {2, 4, length_}, DType::kInt32}},
                {"encoder_hidden_states",
                 {encoder.data(), {2, text_length_, hidden_}, DType::kFloat32}},
                {"encoder_mask", {encoder_mask.data(), {2, 1, 1, text_length_}, DType::kFloat32}},
                {"position", {&position, {1}, DType::kInt32}},
            });
            const auto& logits_tensor = outputs.at("logits");
            if (logits_tensor.dtype != DType::kFloat32 || logits_tensor.numel() != 2 * 4 * 2048)
                throw std::runtime_error("Invalid MusicGen decoder output");
            const float* logits = static_cast<float*>(logits_tensor.data);
            const auto next = position + 1;
            for (int cb = 0; cb < 4; ++cb) {
                // Preserve the BOS/EOS delay mask instead of treating pad as a sample.
                if (next <= cb || next >= length_ - 4 + 1 + cb)
                    continue;
                for (int token = 0; token < 2048; ++token) {
                    const float conditioned = logits[cb * 2048 + token];
                    const float unconditional = logits[(4 + cb) * 2048 + token];
                    scores[token] = unconditional +
                                    static_cast<float>(guidance) * (conditioned - unconditional);
                    if (!std::isfinite(scores[token]))
                        throw std::runtime_error("MusicGen produced non-finite logits");
                }
                int32_t token = static_cast<int32_t>(
                    std::max_element(scores.begin(), scores.end()) - scores.begin());
                if (sample) {
                    std::iota(order.begin(), order.end(), 0);
                    std::partial_sort(
                        order.begin(), order.begin() + top_k, order.end(), [&](int a, int b) {
                            return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
                        });
                    probabilities.resize(top_k);
                    for (int i = 0; i < top_k; ++i)
                        probabilities[i] =
                            std::exp((scores[order[i]] - scores[token]) / temperature);
                    std::discrete_distribution<int> draw(probabilities.begin(),
                                                         probabilities.end());
                    token = order[draw(random)];
                }
                ids[cb * length_ + next] = token;
                ids[(4 + cb) * length_ + next] = token;
            }
        }
        std::vector<int32_t> codes(4 * frames_);
        for (int cb = 0; cb < 4; ++cb)
            std::copy_n(ids.data() + cb * length_ + cb + 1, frames_, codes.data() + cb * frames_);
        auto decoded = codec_->forward({
            {"audio_codes", {codes.data(), {4, frames_}, DType::kInt32}},
        });
        const auto& waveform = decoded.at("waveform");
        if (waveform.dtype != DType::kFloat32 ||
            waveform.numel() != static_cast<std::size_t>(frames_) * 640)
            throw std::runtime_error("MusicGen codec returned the wrong waveform shape");
        const float* samples = static_cast<float*>(waveform.data);
        internal::AudioResult result;
        result.samples.assign(samples, samples + waveform.numel());
        if (!std::all_of(result.samples.begin(), result.samples.end(),
                         [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("MusicGen codec returned non-finite PCM");
        result.sample_rate = 32000;
        result.channels = 1;
        result.inference_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count();
        return result;
    }

  private:
    std::unique_ptr<ITrtModule> text_, decoder_, codec_;
    std::unique_ptr<ITokenizer> tokenizer_;
    int32_t length_, frames_, text_length_, hidden_;
    std::mutex mutex_;
};
} // namespace trtmc::musicgen

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("musicgen does not support --kv-cache-size");
    if (context.reader.info().task != "text_to_audio")
        throw std::invalid_argument("MusicGen implements text_to_audio only");
    const auto require = [&](const char* name) {
        const auto* section = context.reader.find_section(name);
        if (section == nullptr || section->length == 0)
            throw std::runtime_error(std::string("Missing MusicGen bundle section: ") + name);
        return context.reader.read_section(name);
    };
    const auto runtime = require("runtime.json");
    const auto config = nlohmann::json::parse(runtime.begin(), runtime.end());
    const auto tokenizer_json = require("tokenizer.json");
    auto tokenizer =
        musicgen::CreateUnigramTokenizer(tokenizer_json.data(), tokenizer_json.size(), true);
    const auto load = [&](const char* name) {
        const auto plan = require(name);
        ModuleCreateOptions options{};
        auto loaded = context.backend.create_module(plan.data(), plan.size(), options);
        if (!loaded || !loaded->ok())
            throw std::runtime_error(std::string("Cannot load MusicGen engine: ") + name);
        loaded->set_timing_label(name);
        return loaded;
    };
    return new musicgen::Pipeline(load("text.plan"), load("decoder.plan"), load("codec.plan"),
                                  std::move(tokenizer), config);
}
