/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/voxtral/runtime/pipeline.h"

#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::voxtral {
namespace {

using Clock = std::chrono::steady_clock;

const internal::ConfigField kFields[] = {
    {"max_new_tokens", internal::ConfigKind::I64, std::nullopt,
     "Maximum generated tokens; by default every KV-cache position the prompt leaves free"},
};

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void checked(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("voxtral ") + what + ": " +
                                 cudaGetErrorString(status));
}

std::vector<char> require_section(const BundleReader& reader, const char* name) {
    const auto* section = reader.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error(std::string("voxtral bundle section is missing or empty: ") +
                                 name);
    return reader.read_section(name);
}

std::vector<std::int32_t> ids(const nlohmann::json& value) {
    return value.get<std::vector<std::int32_t>>();
}

std::size_t element_size(DType dtype) {
    return dtype == DType::kFloat32 ? 4U : 2U;
}

std::string cache_name(const char* role, const char* stem, std::int64_t layer) {
    return std::string(role) + "_" + stem + "_" + std::to_string(layer);
}

std::vector<float> read_filters(const BundleReader& reader) {
    const auto bytes = require_section(reader, "mel_filters.f32");
    if (bytes.size() % sizeof(float) != 0)
        throw std::runtime_error("voxtral mel filter section is not float32");
    std::vector<float> filters(bytes.size() / sizeof(float));
    std::memcpy(filters.data(), bytes.data(), bytes.size());
    return filters;
}

std::int32_t argmax(const std::vector<float>& logits) {
    return static_cast<std::int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace

RuntimeConfig parse_runtime_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    if (json.at("version").get<int>() != 1)
        throw std::invalid_argument("voxtral runtime.json has an unsupported version");
    RuntimeConfig config;
    const auto precision = json.at("precision").get<std::string>();
    if (precision == "fp32")
        config.dtype = DType::kFloat32;
    else if (precision == "fp16")
        config.dtype = DType::kFloat16;
    else if (precision == "bf16")
        config.dtype = DType::kBFloat16;
    else
        throw std::invalid_argument("voxtral runtime.json names an unsupported precision");
    auto& frontend = config.frontend;
    frontend.sample_rate = json.at("sample_rate").get<std::int32_t>();
    frontend.chunk_samples = json.at("chunk_samples").get<std::int64_t>();
    frontend.n_fft = json.at("n_fft").get<std::int32_t>();
    frontend.hop_length = json.at("hop_length").get<std::int32_t>();
    frontend.num_mel_bins = json.at("num_mel_bins").get<std::int32_t>();
    frontend.mel_frames_per_chunk = json.at("mel_frames_per_chunk").get<std::int32_t>();
    auto& prompt = config.prompt;
    prompt.prefix = ids(json.at("prompt_prefix_ids"));
    prompt.suffix = ids(json.at("prompt_suffix_ids"));
    for (const auto& [language, suffix] : json.at("language_suffix_ids").items())
        prompt.language_suffix.emplace(language, ids(suffix));
    prompt.audio_token_id = json.at("audio_token_id").get<std::int32_t>();
    prompt.audio_tokens_per_chunk = json.at("audio_tokens_per_chunk").get<std::int32_t>();
    config.max_audio_chunks = json.at("max_audio_chunks").get<std::int64_t>();
    config.hidden_size = json.at("hidden_size").get<std::int64_t>();
    config.vocab_size = json.at("vocab_size").get<std::int64_t>();
    config.num_hidden_layers = json.at("num_hidden_layers").get<std::int64_t>();
    config.kv_width = json.at("kv_width").get<std::int64_t>();
    config.capacity = json.at("max_cache_length").get<std::int64_t>();
    config.prefill_rows = json.at("prefill_max_rows").get<std::int64_t>();
    config.eos_token_id = json.at("eos_token_id").get<std::int32_t>();
    config.num_special_tokens = json.at("num_special_tokens").get<std::int32_t>();

    auto valid_id = [&](std::int32_t id) { return id >= 0 && id < config.vocab_size; };
    auto valid_ids = [&](const std::vector<std::int32_t>& values) {
        return std::all_of(values.begin(), values.end(), valid_id);
    };
    bool suffixes_valid = valid_ids(prompt.suffix) && !prompt.suffix.empty();
    for (const auto& item : prompt.language_suffix)
        suffixes_valid = suffixes_valid && valid_ids(item.second) && !item.second.empty();
    if (config.max_audio_chunks <= 0 || config.hidden_size <= 0 || config.vocab_size <= 0 ||
        config.num_hidden_layers <= 0 || config.kv_width <= 0 || config.capacity <= 0 ||
        config.prefill_rows <= 0 || config.prefill_rows > config.capacity ||
        prompt.audio_tokens_per_chunk <= 0 || !valid_id(prompt.audio_token_id) ||
        !valid_id(config.eos_token_id) || prompt.prefix.empty() || !valid_ids(prompt.prefix) ||
        !suffixes_valid || config.num_special_tokens < 0 ||
        config.num_special_tokens > config.vocab_size)
        throw std::invalid_argument("voxtral runtime.json holds invalid dimensions or token ids");
    return config;
}

VoxtralPipeline::VoxtralPipeline(const FamilyContext& context)
    : config_(parse_runtime_config(require_section(context.reader, "runtime.json"))),
      embeddings_(require_section(context.reader, "embed_tokens.bin")),
      mel_filters_(read_filters(context.reader)),
      vocabulary_(require_section(context.reader, "vocabulary.bin"), config_.num_special_tokens) {
    if (vocabulary_.size() != config_.vocab_size)
        throw std::invalid_argument("voxtral vocabulary size differs from runtime.json");
    if (embeddings_.size() != static_cast<std::size_t>(config_.vocab_size) * row_bytes())
        throw std::invalid_argument("voxtral embedding table size differs from runtime.json");
    const auto encoder_plan = require_section(context.reader, "encoder.plan");
    encoder_ = context.backend.create_module(encoder_plan.data(), encoder_plan.size(), {});
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("voxtral encoder engine failed to load");
    encoder_->set_timing_label("voxtral_encoder");
    const auto decoder_plan = require_section(context.reader, "decoder.plan");
    auto modules =
        context.backend.create_dual_profile_modules(decoder_plan.data(), decoder_plan.size(), {});
    if (!modules.prefill || !modules.prefill->ok() || !modules.decode || !modules.decode->ok())
        throw std::runtime_error("voxtral decoder engine failed to load");
    prefill_ = std::move(modules.prefill);
    decode_ = std::move(modules.decode);
    prefill_->set_timing_label("voxtral_prefill");
    decode_->set_timing_label("voxtral_decode");
    validate_engines();
    // Both decoder contexts read and append to the decode context's caches.
    for (std::int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
        for (const char* stem : {"k", "v"}) {
            const auto name = cache_name("cache", stem, layer);
            prefill_->bind_external(name, decode_->device_ptr(name));
        }
    }
    reset_caches();
}

std::size_t VoxtralPipeline::row_bytes() const noexcept {
    return static_cast<std::size_t>(config_.hidden_size) * element_size(config_.dtype);
}

void VoxtralPipeline::validate_engines() const {
    // Dynamic dimensions are checked against the optimization profiles; module
    // shapes of dynamic tensors are runtime shapes, not the engine declaration.
    auto require = [](bool condition, const std::string& name) {
        if (!condition)
            throw std::invalid_argument("voxtral engine binding differs from runtime.json: " +
                                        name);
    };
    auto profile_max = [](const ITrtModule& module, const std::string& name) {
        return module.input_profile_shape(name, module.profile_idx(), ProfileShapeSelector::kMax);
    };
    const auto& frontend = config_.frontend;
    const auto audio_rows = config_.max_audio_chunks * config_.prompt.audio_tokens_per_chunk;
    require(encoder_->tensor_dtype("input_features") == DType::kFloat32 &&
                profile_max(*encoder_, "input_features") ==
                    std::vector<int64_t>{config_.max_audio_chunks, frontend.num_mel_bins,
                                         frontend.mel_frames_per_chunk},
            "input_features");
    require(encoder_->tensor_dtype("audio_embeds") == config_.dtype &&
                encoder_->tensor_shape("audio_embeds") ==
                    std::vector<int64_t>{audio_rows, config_.hidden_size},
            "audio_embeds");
    for (const auto& [module, rows] :
         {std::pair<const ITrtModule*, std::int64_t>{prefill_.get(), config_.prefill_rows},
          std::pair<const ITrtModule*, std::int64_t>{decode_.get(), 1}}) {
        require(module->tensor_dtype("inputs_embeds") == config_.dtype &&
                    profile_max(*module, "inputs_embeds") ==
                        std::vector<int64_t>{rows, config_.hidden_size},
                "inputs_embeds");
        require(module->tensor_dtype("position_id") == DType::kInt32 &&
                    profile_max(*module, "position_id") == std::vector<int64_t>{rows},
                "position_id");
        require(module->tensor_dtype("attention_mask") == DType::kFloat32 &&
                    profile_max(*module, "attention_mask") ==
                        std::vector<int64_t>{rows, config_.capacity},
                "attention_mask");
        require(module->tensor_dtype("logits") == DType::kFloat32 &&
                    module->tensor_shape("logits") == std::vector<int64_t>{1, config_.vocab_size},
                "logits");
        for (std::int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
            for (const char* stem : {"k", "v"}) {
                const auto cache = cache_name("cache", stem, layer);
                const auto present = cache_name("present", stem, layer);
                require(module->tensor_dtype(cache) == config_.dtype &&
                            module->tensor_shape(cache) ==
                                std::vector<int64_t>{config_.capacity, config_.kv_width},
                        cache);
                require(module->tensor_dtype(present) == config_.dtype &&
                            module->tensor_shape(present) ==
                                std::vector<int64_t>{rows, config_.kv_width},
                        present);
            }
        }
    }
    for (std::int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
        for (const char* stem : {"k", "v"}) {
            if (decode_->device_ptr(cache_name("cache", stem, layer)) == nullptr)
                throw std::runtime_error("voxtral decoder owns no KV-cache buffer");
        }
    }
}

std::vector<internal::TaskInstance> VoxtralPipeline::task_bindings() {
    return {internal::bind<internal::ISpeechTranscription>(*this, kFields)};
}

void VoxtralPipeline::reset_caches() {
    const auto bytes =
        static_cast<std::size_t>(config_.capacity * config_.kv_width) * element_size(config_.dtype);
    for (std::int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
        for (const char* stem : {"k", "v"}) {
            checked(cudaMemsetAsync(decode_->device_ptr(cache_name("cache", stem, layer)), 0, bytes,
                                    decode_->stream()),
                    "KV-cache reset");
        }
    }
    decode_->sync();
}

// Runs `rows` prompt or generated rows at positions [position, position + rows),
// appends their keys and values to the caches and returns the last row's logits.
std::vector<float> VoxtralPipeline::run_rows(ITrtModule& module, const char* embeds,
                                             std::int64_t rows, std::int64_t position) {
    if (rows < 1 || position < 0 || position + rows > config_.capacity)
        throw std::out_of_range("voxtral KV-cache position is out of range");
    std::vector<std::int32_t> positions(static_cast<std::size_t>(rows));
    std::iota(positions.begin(), positions.end(), static_cast<std::int32_t>(position));
    std::vector<float> mask(static_cast<std::size_t>(rows * config_.capacity),
                            -std::numeric_limits<float>::infinity());
    for (std::int64_t row = 0; row < rows; ++row) {
        auto begin = mask.begin() + static_cast<std::ptrdiff_t>(row * config_.capacity);
        std::fill(begin, begin + position + row + 1, 0.0F);
    }
    TensorMap inputs{
        {"inputs_embeds", {const_cast<char*>(embeds), {rows, config_.hidden_size}, config_.dtype}},
        {"position_id", {positions.data(), {rows}, DType::kInt32}},
        {"attention_mask", {mask.data(), {rows, config_.capacity}, DType::kFloat32}}};
    module.forward_async(inputs);
    const auto kv_row = static_cast<std::size_t>(config_.kv_width) * element_size(config_.dtype);
    for (std::int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
        for (const char* stem : {"k", "v"}) {
            auto* cache = static_cast<char*>(decode_->device_ptr(cache_name("cache", stem, layer)));
            checked(cudaMemcpyAsync(cache + static_cast<std::size_t>(position) * kv_row,
                                    module.device_ptr(cache_name("present", stem, layer)),
                                    static_cast<std::size_t>(rows) * kv_row,
                                    cudaMemcpyDeviceToDevice, module.stream()),
                    "KV-cache append");
        }
    }
    std::vector<float> logits(static_cast<std::size_t>(config_.vocab_size));
    checked(cudaMemcpyAsync(logits.data(), module.device_ptr("logits"),
                            logits.size() * sizeof(float), cudaMemcpyDeviceToHost, module.stream()),
            "logits copy");
    module.sync();
    if (!std::all_of(logits.begin(), logits.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("voxtral decoder produced non-finite logits at position " +
                                 std::to_string(position + rows - 1));
    return logits;
}

TextResult VoxtralPipeline::run(const internal::SpeechTranscriptionRequest& request,
                                internal::ConfigView options) {
    internal::validate_config(kFields, options);
    const auto requested = internal::config_get<std::int64_t>(options, kFields, "max_new_tokens");
    const auto& frontend = config_.frontend;
    const auto& audio = request.audio;
    const auto rate = audio.sample_rate.value_or(static_cast<std::uint32_t>(frontend.sample_rate));
    if (rate != static_cast<std::uint32_t>(frontend.sample_rate))
        throw std::invalid_argument("voxtral requires " + std::to_string(frontend.sample_rate) +
                                    " Hz audio, not " + std::to_string(rate) + " Hz");

    const auto setup_start = Clock::now();
    const auto mono = downmix(audio.samples.data(), audio.samples.size(), audio.channels);
    const auto chunks = chunk_count(mono.size(), frontend.chunk_samples);
    if (chunks > config_.max_audio_chunks) {
        const auto seconds =
            config_.max_audio_chunks * frontend.chunk_samples / frontend.sample_rate;
        throw std::invalid_argument("voxtral bundle transcribes at most " +
                                    std::to_string(seconds) +
                                    " s of audio; rebuild with a larger max_sequence_length");
    }
    const auto prompt = transcription_prompt(config_.prompt, chunks, request.source_language);
    const auto prompt_length = static_cast<std::int64_t>(prompt.size());
    const std::int64_t room = config_.capacity - prompt_length;
    if (room < 1)
        throw std::invalid_argument("voxtral prompt fills the bundle's KV cache");
    std::int64_t limit = room;
    if (requested) {
        if (*requested <= 0 || *requested > room)
            throw std::invalid_argument("voxtral max_new_tokens must be between 1 and " +
                                        std::to_string(room) + " for this audio and bundle");
        limit = *requested;
    }
    auto features = log_mel_features(mono, mel_filters_, frontend);
    const double setup_ms = milliseconds_since(setup_start);

    std::lock_guard<std::mutex> lock(mutex_);
    const auto prefill_start = Clock::now();
    Tensor input{features.data(),
                 {chunks, frontend.num_mel_bins, frontend.mel_frames_per_chunk},
                 DType::kFloat32};
    auto outputs = encoder_->forward({{"input_features", input}});
    const auto found = outputs.find("audio_embeds");
    const auto audio_rows = chunks * config_.prompt.audio_tokens_per_chunk;
    if (found == outputs.end() || found->second.data == nullptr ||
        found->second.dtype != config_.dtype ||
        found->second.numel() != static_cast<std::size_t>(audio_rows * config_.hidden_size))
        throw std::runtime_error("voxtral encoder returned unusable audio embeddings");
    const auto* audio_embeds = static_cast<const char*>(found->second.data);

    const std::size_t width = row_bytes();
    std::vector<char> embeds(static_cast<std::size_t>(prompt_length) * width);
    std::int64_t audio_row = 0;
    for (std::int64_t index = 0; index < prompt_length; ++index) {
        const auto token = prompt[static_cast<std::size_t>(index)];
        const char* source = token == config_.prompt.audio_token_id
                                 ? audio_embeds + static_cast<std::size_t>(audio_row++) * width
                                 : embeddings_.data() + static_cast<std::size_t>(token) * width;
        std::memcpy(embeds.data() + static_cast<std::size_t>(index) * width, source, width);
    }
    if (audio_row != audio_rows)
        throw std::logic_error("voxtral prompt and audio embeddings disagree");

    reset_caches();
    std::vector<float> logits;
    for (std::int64_t start = 0; start < prompt_length; start += config_.prefill_rows) {
        const auto rows = std::min(config_.prefill_rows, prompt_length - start);
        logits = run_rows(*prefill_, embeds.data() + static_cast<std::size_t>(start) * width, rows,
                          start);
    }
    const double prefill_ms = milliseconds_since(prefill_start);

    const auto decode_start = Clock::now();
    std::vector<std::int32_t> tokens;
    std::int64_t position = prompt_length;
    while (true) {
        const auto token = argmax(logits);
        if (token == config_.eos_token_id)
            break;
        tokens.push_back(token);
        if (static_cast<std::int64_t>(tokens.size()) == limit)
            break;
        logits = run_rows(*decode_, embeddings_.data() + static_cast<std::size_t>(token) * width, 1,
                          position++);
    }
    TextResult result(vocabulary_.decode(tokens), tokens, prefill_ms,
                      milliseconds_since(decode_start));
    result.setup_ms = setup_ms;
    return result;
}

} // namespace trtmc::voxtral
