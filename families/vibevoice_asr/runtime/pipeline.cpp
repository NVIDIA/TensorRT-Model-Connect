/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vibevoice_asr/runtime/pipeline.h"

#include "families/vibevoice_asr/runtime/audio_frontend.h"
#include "families/vibevoice_asr/runtime/transcript.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::vibevoice_asr {
namespace {

const internal::ConfigField kFields[] = {
    {"max_output_tokens", internal::ConfigKind::I64, std::nullopt,
     "Maximum generated tokens; by default generation ends at the end-of-text token or when "
     "the bundle's sequence length is full"},
    {"context_info", internal::ConfigKind::String, internal::ConfigValue{std::string_view{}},
     "Hotwords or background text the model is told about (the official context_info)"},
    {"raw_output", internal::ConfigKind::Bool, internal::ConfigValue{false},
     "Return the decoded model output, the JSON segment list with speaker ids, as text"},
};

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

std::string layer_name(const char* base, std::int32_t index) {
    return std::string(base) + "_" + std::to_string(index);
}

void require_io(const ITrtModule& module, const char* engine, const char* name, DType dtype,
                bool input) {
    if ((input ? !module.has_input(name) : !module.has_output(name)) ||
        module.tensor_dtype(name) != dtype)
        throw std::invalid_argument(std::string("VibeVoice-ASR ") + engine +
                                    " engine lacks the expected tensor " + name);
}

} // namespace

VibeVoiceAsrModel::VibeVoiceAsrModel(RuntimeConfig config, std::unique_ptr<ITokenizer> tokenizer,
                                     std::unique_ptr<ITrtModule> encoder,
                                     std::unique_ptr<ITrtModule> prefill,
                                     std::unique_ptr<ITrtModule> decode)
    : config_(std::move(config)), tokenizer_(std::move(tokenizer)), encoder_(std::move(encoder)),
      prefill_(std::move(prefill)), decode_(std::move(decode)) {
    if (!tokenizer_ || !encoder_ || !encoder_->ok() || !prefill_ || !prefill_->ok() || !decode_ ||
        !decode_->ok())
        throw std::invalid_argument("VibeVoice-ASR needs a tokenizer and three loaded modules");
    require_io(*encoder_, "encoder", "audio", DType::kFloat32, true);
    require_io(*encoder_, "encoder", "audio_features", DType::kBFloat16, false);
    for (auto* module : {prefill_.get(), decode_.get()}) {
        require_io(*module, "decoder", "input_ids", DType::kInt32, true);
        require_io(*module, "decoder", "audio_embeds", DType::kBFloat16, true);
        require_io(*module, "decoder", "audio_mask", DType::kInt32, true);
        require_io(*module, "decoder", "cache_write_index", DType::kInt32, true);
        require_io(*module, "decoder", "logits", DType::kFloat32, false);
    }
    const std::vector<std::int64_t> shape{1, config_.num_kv_heads, config_.max_sequence_length,
                                          config_.head_dim};
    for (std::int32_t layer = 0; layer < config_.num_layers; ++layer) {
        for (const char* base : {"cache_k", "cache_v"}) {
            const auto name = layer_name(base, layer);
            require_io(*prefill_, "decoder", name.c_str(), DType::kFloat32, true);
            if (prefill_->tensor_shape(name) != shape)
                throw std::invalid_argument("VibeVoice-ASR KV cache shape differs from runtime");
            cache_.emplace_back(shape, DType::kFloat32, prefill_->stream());
            if (!cache_.back().ok())
                throw std::runtime_error("VibeVoice-ASR could not allocate its KV cache");
            prefill_->bind_external(name, cache_.back().data());
            decode_->bind_external(name, cache_.back().data());
        }
    }
    zero_row_.assign(static_cast<std::size_t>(config_.hidden_size), 0);
}

std::vector<internal::TaskInstance> VibeVoiceAsrModel::task_bindings() {
    return {internal::bind<internal::ISpeechTranscription>(*this, kFields)};
}

std::vector<float> VibeVoiceAsrModel::prepare_audio(const internal::AudioView& audio) const {
    if (audio.channels == 0 || audio.samples.empty() || audio.samples.data() == nullptr ||
        audio.samples.size() % audio.channels != 0)
        throw std::invalid_argument("VibeVoice-ASR requires complete nonempty interleaved PCM");
    const auto rate = audio.sample_rate.value_or(static_cast<std::uint32_t>(config_.sample_rate));
    if (rate == 0 || rate > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("VibeVoice-ASR input sample rate is out of range");
    auto samples =
        downmix(audio.samples.data(), audio.samples.size() / audio.channels, audio.channels);
    samples = resample(samples, static_cast<std::int32_t>(rate), config_.sample_rate);
    normalize_loudness(samples, config_.normalize_target_dbfs, config_.normalize_eps);
    return samples;
}

std::vector<std::uint16_t> VibeVoiceAsrModel::encode_audio(const std::vector<float>& samples) {
    const std::int64_t hop = config_.hop_length;
    const std::int64_t count = static_cast<std::int64_t>(samples.size());
    const std::int64_t total = (count + hop - 1) / hop;
    const std::size_t hidden = static_cast<std::size_t>(config_.hidden_size);
    std::vector<std::uint16_t> features(static_cast<std::size_t>(total) * hidden);
    for (const auto& chunk :
         plan_encoder_chunks(total, config_.encoder_chunk_frames, config_.encoder_context_frames)) {
        const std::int64_t begin = (chunk.first_frame - chunk.context_frames) * hop;
        const std::int64_t end = std::min((chunk.first_frame + chunk.frames) * hop, count);
        TensorMap inputs;
        inputs["audio"] = Tensor{
            const_cast<float*>(samples.data() + begin), {1, 1, 1, end - begin}, DType::kFloat32};
        const auto outputs = encoder_->forward(inputs);
        const auto& rows = outputs.at("audio_features");
        const std::vector<std::int64_t> expected{chunk.context_frames + chunk.frames,
                                                 config_.hidden_size};
        if (rows.dtype != DType::kBFloat16 || rows.shape != expected || rows.data == nullptr)
            throw std::runtime_error("VibeVoice-ASR encoder returned unexpected features");
        const auto* source = static_cast<const std::uint16_t*>(rows.data) +
                             static_cast<std::size_t>(chunk.context_frames) * hidden;
        std::memcpy(features.data() + static_cast<std::size_t>(chunk.first_frame) * hidden, source,
                    static_cast<std::size_t>(chunk.frames) * hidden * sizeof(std::uint16_t));
    }
    return features;
}

const float* VibeVoiceAsrModel::forward(ITrtModule& module, const std::int32_t* ids,
                                        const std::uint16_t* embeds, const std::int32_t* mask,
                                        std::int32_t rows, std::int32_t position) {
    std::int32_t write_index = position;
    TensorMap inputs;
    inputs["input_ids"] = Tensor{const_cast<std::int32_t*>(ids), {rows}, DType::kInt32};
    inputs["audio_embeds"] =
        Tensor{const_cast<std::uint16_t*>(embeds), {rows, config_.hidden_size}, DType::kBFloat16};
    inputs["audio_mask"] = Tensor{const_cast<std::int32_t*>(mask), {rows}, DType::kInt32};
    inputs["cache_write_index"] = Tensor{&write_index, {1}, DType::kInt32};
    const auto outputs = module.forward(inputs);
    const auto& logits = outputs.at("logits");
    if (logits.dtype != DType::kFloat32 || logits.data == nullptr ||
        logits.numel() != static_cast<std::size_t>(config_.vocab_size))
        throw std::runtime_error("VibeVoice-ASR decoder returned unexpected logits");
    logits_.assign(static_cast<const float*>(logits.data),
                   static_cast<const float*>(logits.data) + config_.vocab_size);
    return logits_.data();
}

std::vector<std::int32_t> VibeVoiceAsrModel::generate(const Prompt& prompt,
                                                      const std::vector<std::uint16_t>& features,
                                                      std::int64_t max_new_tokens,
                                                      double& prefill_ms, double& decode_ms) {
    // Stale rows from an earlier request are masked out; zeros keep them inert.
    for (auto& tensor : cache_) {
        if (cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), prefill_->stream()) != cudaSuccess)
            throw std::runtime_error("VibeVoice-ASR could not clear its KV cache");
    }
    const auto start = Clock::now();
    const std::int32_t length = static_cast<std::int32_t>(prompt.ids.size());
    const std::size_t hidden = static_cast<std::size_t>(config_.hidden_size);
    std::vector<std::uint16_t> embeds;
    std::vector<std::int32_t> mask;
    const float* logits = nullptr;
    std::int32_t position = 0;
    while (position < length) {
        const std::int32_t rows = std::min(config_.max_prefill_rows, length - position);
        embeds.assign(static_cast<std::size_t>(rows) * hidden, 0);
        mask.assign(static_cast<std::size_t>(rows), 0);
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::int32_t speech = position + row - prompt.speech_offset;
            if (speech < 0 || speech >= prompt.speech_rows)
                continue;
            mask[static_cast<std::size_t>(row)] = 1;
            std::memcpy(embeds.data() + static_cast<std::size_t>(row) * hidden,
                        features.data() + static_cast<std::size_t>(speech) * hidden,
                        hidden * sizeof(std::uint16_t));
        }
        logits = forward(*prefill_, prompt.ids.data() + position, embeds.data(), mask.data(), rows,
                         position);
        position += rows;
    }
    prefill_ms += elapsed_ms(start);

    const auto decode_start = Clock::now();
    std::vector<std::int32_t> generated;
    std::int32_t next = argmax(logits, config_.vocab_size);
    const std::int32_t text_mask = 0;
    while (next != config_.token_ids.eos) {
        generated.push_back(next);
        if (static_cast<std::int64_t>(generated.size()) >= max_new_tokens ||
            position >= config_.max_sequence_length)
            break;
        logits = forward(*decode_, &next, zero_row_.data(), &text_mask, 1, position);
        ++position;
        next = argmax(logits, config_.vocab_size);
    }
    decode_ms += elapsed_ms(decode_start);
    return generated;
}

TextResult VibeVoiceAsrModel::run(const internal::SpeechTranscriptionRequest& request,
                                  internal::ConfigView options) {
    internal::validate_config(kFields, options);
    if (request.source_language.has_value())
        throw std::invalid_argument(
            "VibeVoice-ASR identifies the language itself and takes no source_language");
    const auto limit = internal::config_get<std::int64_t>(options, kFields, "max_output_tokens");
    if (limit.has_value() && *limit <= 0)
        throw std::invalid_argument("VibeVoice-ASR max_output_tokens must be positive");
    const auto context =
        internal::config_get<std::string_view>(options, kFields, "context_info").value();
    const bool raw = internal::config_get<bool>(options, kFields, "raw_output").value();

    std::lock_guard<std::mutex> lock(mutex_);
    TextResult result;
    const auto setup_start = Clock::now();
    const auto samples = prepare_audio(request.audio);
    const auto prompt = build_prompt(*tokenizer_, config_, samples.size(), context);
    if (static_cast<std::int64_t>(prompt.ids.size()) >= config_.max_sequence_length)
        throw std::invalid_argument("VibeVoice-ASR audio of " + std::to_string(samples.size()) +
                                    " samples needs a longer bundle max_sequence_length");
    result.setup_ms = elapsed_ms(setup_start);

    const auto encode_start = Clock::now();
    const auto features = encode_audio(samples);
    result.prefill_ms = elapsed_ms(encode_start);
    result.token_ids =
        generate(prompt, features, limit.value_or(std::numeric_limits<std::int64_t>::max()),
                 result.prefill_ms, result.decode_ms);

    const auto decoded = tokenizer_->decode(result.token_ids);
    auto transcript = parse_transcript(decoded);
    result.text = raw ? decoded : std::move(transcript.text);
    result.segments = std::move(transcript.segments);
    return result;
}

} // namespace trtmc::vibevoice_asr
