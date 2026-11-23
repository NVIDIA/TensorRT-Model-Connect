/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/granite_speech/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::granite_speech {
namespace {

const internal::ConfigField fields[] = {
    {"max_new_tokens", internal::ConfigKind::I64, internal::ConfigValue{std::int64_t{200}},
     "Most tokens generated after the prompt, the final EOS included"},
};

std::int32_t positive(const nlohmann::json& json, const char* key) {
    const auto value = json.at(key).get<std::int64_t>();
    if (value <= 0 || value > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error(std::string("granite_speech runtime.json needs a positive ") +
                                 key);
    return static_cast<std::int32_t>(value);
}

void require_io(const ITrtModule& module, const std::string& name, bool input, DType dtype,
                const char* engine) {
    const bool present = input ? module.has_input(name) : module.has_output(name);
    if (!present || module.tensor_dtype(name) != dtype)
        throw std::runtime_error(std::string("granite_speech ") + engine + " engine lacks " +
                                 (input ? "input " : "output ") + name + " of the bundle's type");
}

std::vector<float> take_logits(const TensorMap& outputs, std::int32_t vocab) {
    const auto found = outputs.find("logits");
    if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
        found->second.data == nullptr || found->second.numel() != static_cast<std::size_t>(vocab))
        throw std::runtime_error("granite_speech decoder returned no [1, vocab] FP32 logits");
    const auto* begin = static_cast<const float*>(found->second.data);
    return {begin, begin + vocab};
}

} // namespace

RuntimeConfig RuntimeConfig::from_json(std::string_view json) {
    const auto parsed = nlohmann::json::parse(json.begin(), json.end());
    RuntimeConfig config;
    const auto precision = parsed.at("precision").get<std::string>();
    if (precision == "bf16")
        config.work = DType::kBFloat16;
    else if (precision == "fp32")
        config.work = DType::kFloat32;
    else
        throw std::runtime_error("granite_speech runtime.json states an unknown precision: " +
                                 precision);
    config.sample_rate = positive(parsed, "sample_rate");
    config.n_fft = positive(parsed, "n_fft");
    config.hop_length = positive(parsed, "hop_length");
    config.window_size = positive(parsed, "window_size");
    config.num_queries = positive(parsed, "num_queries");
    config.min_samples = parsed.at("min_samples").get<std::int64_t>();
    config.max_samples = parsed.at("max_samples").get<std::int64_t>();
    config.capacity = positive(parsed, "capacity");
    config.hidden_size = positive(parsed, "hidden_size");
    config.num_layers = positive(parsed, "num_layers");
    config.kv_width = positive(parsed, "kv_width");
    config.vocab_size = positive(parsed, "vocab_size");
    config.audio_token_id = parsed.at("audio_token_id").get<std::int32_t>();
    config.eos_token_id = parsed.at("eos_token_id").get<std::int32_t>();
    config.prefix_ids = parsed.at("prompt_prefix_ids").get<std::vector<std::int32_t>>();
    config.suffix_ids = parsed.at("prompt_suffix_ids").get<std::vector<std::int32_t>>();
    if (config.min_samples <= config.n_fft / 2 || config.max_samples < config.min_samples)
        throw std::runtime_error("granite_speech runtime.json states an empty sample range");
    const auto fixed =
        static_cast<std::int64_t>(config.prefix_ids.size() + config.suffix_ids.size());
    if (fixed + config.audio_tokens(config.max_samples) >= config.capacity)
        throw std::runtime_error(
            "granite_speech runtime.json sample range exceeds the token capacity");
    for (const auto id : config.prefix_ids)
        if (id < 0 || id >= config.vocab_size)
            throw std::runtime_error("granite_speech prompt holds an id outside the vocabulary");
    for (const auto id : config.suffix_ids)
        if (id < 0 || id >= config.vocab_size)
            throw std::runtime_error("granite_speech prompt holds an id outside the vocabulary");
    if (config.audio_token_id < 0 || config.audio_token_id >= config.vocab_size ||
        config.eos_token_id < 0 || config.eos_token_id >= config.vocab_size)
        throw std::runtime_error("granite_speech special token ids are outside the vocabulary");
    return config;
}

std::int64_t RuntimeConfig::audio_tokens(std::int64_t samples) const {
    const std::int64_t frames = (samples / hop_length + 1) / 2;
    const std::int64_t windows = (frames + window_size - 1) / window_size;
    return windows * num_queries;
}

std::vector<float> reflect_pad(const float* samples, std::size_t count, std::size_t half) {
    if (count <= half)
        throw std::invalid_argument("reflect padding needs more samples than the pad width");
    std::vector<float> out(count + 2 * half);
    for (std::size_t i = 0; i < half; ++i) {
        out[i] = samples[half - i];
        out[half + count + i] = samples[count - 2 - i];
    }
    std::memcpy(out.data() + half, samples, count * sizeof(float));
    return out;
}

std::int32_t first_argmax(const float* values, std::size_t count) {
    if (count == 0)
        throw std::invalid_argument("argmax of no values");
    return static_cast<std::int32_t>(std::max_element(values, values + count) - values);
}

Pipeline::Pipeline(std::unique_ptr<ITrtModule> audio, std::unique_ptr<ITrtModule> prefill,
                   std::unique_ptr<ITrtModule> decode, RuntimeConfig config, Vocabulary vocabulary)
    : audio_(std::move(audio)), prefill_(std::move(prefill)), decode_(std::move(decode)),
      config_(std::move(config)), vocabulary_(std::move(vocabulary)) {
    if (!audio_ || !audio_->ok() || !prefill_ || !prefill_->ok() || !decode_ || !decode_->ok())
        throw std::runtime_error(
            "granite_speech needs a loaded audio engine and both decoder profiles");
    if (vocabulary_.size() != static_cast<std::size_t>(config_.vocab_size))
        throw std::runtime_error(
            "granite_speech vocabulary does not match the decoder's vocabulary");
    const auto work = config_.work;
    require_io(*audio_, "waveform", true, DType::kFloat32, "audio");
    require_io(*audio_, "audio_embeds", false, work, "audio");
    for (ITrtModule* module : {prefill_.get(), decode_.get()}) {
        require_io(*module, "input_ids", true, DType::kInt32, "decoder");
        require_io(*module, "audio_mask", true, DType::kInt32, "decoder");
        require_io(*module, "audio_embeds", true, work, "decoder");
        require_io(*module, "logits", false, DType::kFloat32, "decoder");
        for (std::int32_t index = 0; index < config_.num_layers; ++index) {
            for (const char* stem : {"past_key_", "past_value_"})
                require_io(*module, stem + std::to_string(index), true, work, "decoder");
            for (const char* stem : {"present_key_", "present_value_"})
                require_io(*module, stem + std::to_string(index), false, work, "decoder");
        }
    }
    const auto stream = prefill_->stream();
    const std::vector<std::int64_t> cache_shape{config_.capacity, config_.kv_width};
    keys_.reserve(static_cast<std::size_t>(config_.num_layers));
    values_.reserve(static_cast<std::size_t>(config_.num_layers));
    for (std::int32_t index = 0; index < config_.num_layers; ++index) {
        keys_.emplace_back(cache_shape, work, stream);
        values_.emplace_back(cache_shape, work, stream);
        if (!keys_.back().ok() || !values_.back().ok())
            throw std::runtime_error("granite_speech could not allocate the KV cache");
    }
    prompt_embeds_ = DeviceTensor::zeros({config_.capacity, config_.hidden_size}, work, stream);
    zero_row_ = DeviceTensor::zeros({1, config_.hidden_size}, work, stream);
    if (!prompt_embeds_.ok() || !zero_row_.ok())
        throw std::runtime_error("granite_speech could not allocate the prompt embeddings");
    cudaStreamSynchronize(stream);
    // The audio engine writes its rows straight after the prompt prefix; the decode profile never
    // selects audio rows, so it reads one zero row.
    const auto row_bytes = static_cast<std::size_t>(config_.hidden_size) * dtype_size(work);
    audio_->bind_external("audio_embeds", static_cast<char*>(prompt_embeds_.data()) +
                                              config_.prefix_ids.size() * row_bytes);
    decode_->bind_external("audio_embeds", zero_row_.data(), {1, config_.hidden_size});
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::ISpeechTranscription>(*this, fields)};
}

void Pipeline::bind_cache(ITrtModule& module, std::int64_t cached_rows) {
    const auto row_bytes = static_cast<std::size_t>(config_.kv_width) * dtype_size(config_.work);
    const auto offset = static_cast<std::size_t>(cached_rows) * row_bytes;
    const std::vector<std::int64_t> past{cached_rows, config_.kv_width};
    for (std::int32_t index = 0; index < config_.num_layers; ++index) {
        const auto layer = std::to_string(index);
        auto* key = static_cast<char*>(keys_[static_cast<std::size_t>(index)].data());
        auto* value = static_cast<char*>(values_[static_cast<std::size_t>(index)].data());
        module.bind_external("past_key_" + layer, key, past);
        module.bind_external("past_value_" + layer, value, past);
        module.bind_external("present_key_" + layer, key + offset);
        module.bind_external("present_value_" + layer, value + offset);
    }
}

std::vector<float> Pipeline::prefill(const std::vector<float>& padded, std::int64_t audio_rows,
                                     std::int64_t& prompt_rows) {
    Tensor waveform{const_cast<float*>(padded.data()),
                    {static_cast<std::int64_t>(padded.size())},
                    DType::kFloat32};
    audio_->forward({{"waveform", waveform}});

    std::vector<std::int32_t> ids(config_.prefix_ids);
    ids.insert(ids.end(), static_cast<std::size_t>(audio_rows), config_.audio_token_id);
    ids.insert(ids.end(), config_.suffix_ids.begin(), config_.suffix_ids.end());
    std::vector<std::int32_t> mask(ids.size(), 0);
    std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(config_.prefix_ids.size()), audio_rows,
                1);
    prompt_rows = static_cast<std::int64_t>(ids.size());

    bind_cache(*prefill_, 0);
    prefill_->bind_external("audio_embeds", prompt_embeds_.data(),
                            {prompt_rows, config_.hidden_size});
    const auto outputs =
        prefill_->forward({{"input_ids", Tensor{ids.data(), {prompt_rows}, DType::kInt32}},
                           {"audio_mask", Tensor{mask.data(), {prompt_rows}, DType::kInt32}}});
    return take_logits(outputs, config_.vocab_size);
}

std::vector<float> Pipeline::step(std::int32_t token, std::int64_t cached_rows) {
    bind_cache(*decode_, cached_rows);
    // A generated audio placeholder is embedded as id 0, as the reference does for every
    // placeholder it embeds without audio features.
    std::int32_t id = token == config_.audio_token_id ? 0 : token;
    std::int32_t mask = 0;
    const auto outputs = decode_->forward({{"input_ids", Tensor{&id, {1}, DType::kInt32}},
                                           {"audio_mask", Tensor{&mask, {1}, DType::kInt32}}});
    return take_logits(outputs, config_.vocab_size);
}

TextResult Pipeline::run(const internal::SpeechTranscriptionRequest& request,
                         internal::ConfigView options) {
    internal::validate_config(fields, options);
    const auto limit =
        internal::config_get<std::int64_t>(options, fields, "max_new_tokens").value();
    if (limit < 1 || limit > config_.capacity)
        throw internal::ConfigError("granite_speech max_new_tokens must be between 1 and " +
                                    std::to_string(config_.capacity));
    if (request.source_language)
        throw std::invalid_argument(
            "granite_speech transcribes the spoken language without a language hint; "
            "omit source_language");
    const internal::AudioView& audio = request.audio;
    if (audio.samples.data() == nullptr || audio.samples.empty())
        throw std::invalid_argument("granite_speech requires float32 PCM input");
    const auto rate = audio.sample_rate.value_or(static_cast<std::uint32_t>(config_.sample_rate));
    if (rate != static_cast<std::uint32_t>(config_.sample_rate))
        throw std::invalid_argument("granite_speech requires " +
                                    std::to_string(config_.sample_rate) + " Hz audio, not " +
                                    std::to_string(rate) + " Hz; it ships no resampler");
    if ((audio.channels == 0 ? 1U : audio.channels) != 1U)
        throw std::invalid_argument("granite_speech requires mono audio, not " +
                                    std::to_string(audio.channels) + " channels");
    const auto count = static_cast<std::int64_t>(audio.samples.size());
    if (count < config_.min_samples || count > config_.max_samples)
        throw std::invalid_argument(
            "granite_speech accepts " + std::to_string(config_.min_samples) + " to " +
            std::to_string(config_.max_samples) + " samples, not " + std::to_string(count));
    if (!std::all_of(audio.samples.begin(), audio.samples.end(),
                     [](float v) { return std::isfinite(v); }))
        throw std::invalid_argument("granite_speech received a non-finite audio sample");
    const auto audio_rows = config_.audio_tokens(count);
    const auto fixed =
        static_cast<std::int64_t>(config_.prefix_ids.size() + config_.suffix_ids.size());
    if (fixed + audio_rows + limit > config_.capacity)
        throw std::invalid_argument(
            "granite_speech prompt of " + std::to_string(fixed + audio_rows) +
            " tokens plus max_new_tokens " + std::to_string(limit) +
            " exceeds the bundle's capacity of " + std::to_string(config_.capacity));

    std::lock_guard<std::mutex> lock(mutex_);
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto padded = reflect_pad(audio.samples.data(), audio.samples.size(),
                                    static_cast<std::size_t>(config_.n_fft / 2));
    std::int64_t prompt_rows = 0;
    auto logits = prefill(padded, audio_rows, prompt_rows);
    const auto after_prefill = Clock::now();

    std::vector<std::int32_t> generated;
    while (true) {
        const auto token = first_argmax(logits.data(), logits.size());
        generated.push_back(token);
        if (token == config_.eos_token_id || static_cast<std::int64_t>(generated.size()) == limit)
            break;
        logits = step(token, prompt_rows + static_cast<std::int64_t>(generated.size()) - 1);
    }
    TextResult result;
    result.text = vocabulary_.decode(generated);
    result.token_ids = std::move(generated);
    result.prefill_ms = std::chrono::duration<double, std::milli>(after_prefill - start).count();
    result.decode_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - after_prefill).count();
    return result;
}

} // namespace trtmc::granite_speech
