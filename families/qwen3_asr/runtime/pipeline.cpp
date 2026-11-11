/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_asr/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace trtmc::qwen3_asr {
namespace {

using Clock = std::chrono::steady_clock;

const internal::ConfigField kFields[] = {
    {"max_output_tokens", internal::ConfigKind::I64, std::nullopt,
     "Maximum generated tokens, the language tag included (default: the checkpoint's "
     "max_new_tokens)"},
};

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string cache_name(const char* kind, std::int32_t layer) {
    return std::string(kind) + "_" + std::to_string(layer);
}

} // namespace

Qwen3AsrPipeline::Qwen3AsrPipeline(std::unique_ptr<ITrtModule> encoder,
                                   std::unique_ptr<ITrtModule> prefill,
                                   std::unique_ptr<ITrtModule> decode, RuntimeConfig config,
                                   MelFilterbank filters, Vocabulary vocabulary)
    : encoder_(std::move(encoder)), prefill_(std::move(prefill)), decode_(std::move(decode)),
      config_(std::move(config)), filters_(std::move(filters)), vocabulary_(std::move(vocabulary)) {
    if (!encoder_ || !encoder_->ok() || !prefill_ || !prefill_->ok() || !decode_ || !decode_->ok())
        throw std::runtime_error("qwen3_asr: an engine failed to load");
    if (prefill_->stream() != decode_->stream())
        throw std::runtime_error("qwen3_asr: decoder profiles must share one stream");
    if (filters_.n_mel != config_.num_mel_bins)
        throw std::runtime_error("qwen3_asr: mel filterbank does not match runtime.json");
    if (vocabulary_.size() != static_cast<std::size_t>(config_.vocab_size))
        throw std::runtime_error("qwen3_asr: vocabulary table does not match runtime.json");
    if (decode_->tensor_dtype("logits") != DType::kFloat32 ||
        decode_->tensor_shape("logits") != std::vector<int64_t>{1, config_.vocab_size})
        throw std::runtime_error("qwen3_asr: decoder logits must be FP32 [1, vocab]");

    const std::vector<int64_t> cache_shape{1, config_.num_kv_heads, config_.max_cache_length,
                                           config_.head_dim};
    for (std::int32_t layer = 0; layer < config_.num_layers; ++layer) {
        for (const char* kind : {"cache_k", "cache_v"}) {
            const auto name = cache_name(kind, layer);
            if (decode_->tensor_shape(name) != cache_shape ||
                decode_->tensor_dtype(name) != config_.cache_dtype)
                throw std::runtime_error("qwen3_asr: decoder cache " + name +
                                         " does not match runtime.json");
        }
        // Zeroed once; the engine also masks every row at or after the valid length.
        cache_k_.push_back(
            DeviceTensor::zeros(cache_shape, config_.cache_dtype, prefill_->stream()));
        cache_v_.push_back(
            DeviceTensor::zeros(cache_shape, config_.cache_dtype, prefill_->stream()));
        if (!cache_k_.back().ok() || !cache_v_.back().ok())
            throw std::runtime_error("qwen3_asr: KV cache allocation failed");
    }
    bind_caches(*prefill_);
    bind_caches(*decode_);
    key_positions_.resize(static_cast<std::size_t>(config_.max_cache_length));
    std::iota(key_positions_.begin(), key_positions_.end(), 0);
    logits_.resize(static_cast<std::size_t>(config_.vocab_size));
}

std::vector<internal::TaskInstance> Qwen3AsrPipeline::task_bindings() {
    return {internal::bind<internal::ISpeechTranscription>(*this, kFields)};
}

void Qwen3AsrPipeline::bind_caches(ITrtModule& module) {
    for (std::int32_t layer = 0; layer < config_.num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        for (auto [kind, present, cache] : {std::tuple{"cache_k", "present_k", &cache_k_[index]},
                                            std::tuple{"cache_v", "present_v", &cache_v_[index]}}) {
            module.bind_external(cache_name(kind, layer), cache->data());
            if (module.device_ptr(cache_name(kind, layer)) != cache->data() ||
                module.device_ptr(cache_name(present, layer)) != cache->data())
                throw std::runtime_error("qwen3_asr: KV cache output does not alias its input");
        }
    }
}

std::vector<float> Qwen3AsrPipeline::mono_audio(const internal::AudioView& audio) const {
    if (audio.samples.data() == nullptr || audio.samples.empty())
        throw std::invalid_argument("qwen3_asr requires float32 PCM input");
    const auto rate =
        audio.sample_rate.value_or(static_cast<std::uint32_t>(config_.frontend.sample_rate));
    // The checkpoint's features are defined at one rate; this family ships no resampler.
    if (static_cast<std::int64_t>(rate) != config_.frontend.sample_rate)
        throw std::invalid_argument("qwen3_asr requires " +
                                    std::to_string(config_.frontend.sample_rate) +
                                    " Hz audio, not " + std::to_string(rate) + " Hz");
    const std::size_t channels = audio.channels == 0 ? 1 : audio.channels;
    if (audio.samples.size() % channels != 0)
        throw std::invalid_argument("qwen3_asr: sample count is not a multiple of the channels");
    std::vector<float> mono(audio.samples.size() / channels);
    for (std::size_t frame = 0; frame < mono.size(); ++frame) {
        float sum = 0.0F;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const float value = audio.samples[frame * channels + channel];
            if (!std::isfinite(value))
                throw std::invalid_argument("qwen3_asr: audio contains a non-finite sample");
            sum += value;
        }
        mono[frame] = sum / static_cast<float>(channels);
    }
    return mono;
}

std::vector<float> Qwen3AsrPipeline::encode(const EncoderBatch& batch) {
    const auto chunks = static_cast<std::int64_t>(config_.frontend.window_chunks);
    const auto mels = static_cast<std::int64_t>(config_.num_mel_bins);
    const auto frames = static_cast<std::int64_t>(config_.frontend.chunk_frames);
    const auto width = static_cast<std::size_t>(config_.audio_embed_size);
    const auto window_rows = static_cast<std::size_t>(chunks * config_.frontend.tokens_per_chunk);
    const auto window_values = static_cast<std::size_t>(chunks * mels * frames);
    std::vector<float> rows;
    rows.reserve(static_cast<std::size_t>(batch.windows) * window_rows * width);
    // Attention never crosses a window, so windows are encoded in groups of bounded size.
    for (std::int32_t first = 0; first < batch.windows;
         first += config_.encoder_windows_per_launch) {
        const auto count = std::min(config_.encoder_windows_per_launch, batch.windows - first);
        const auto offset = static_cast<std::size_t>(first);
        TensorMap inputs;
        inputs["input_features"] =
            Tensor{const_cast<float*>(batch.features.data() + offset * window_values),
                   {count, chunks, mels, frames},
                   DType::kFloat32};
        inputs["chunk_tokens"] =
            Tensor{const_cast<std::int32_t*>(batch.chunk_tokens.data() +
                                             offset * static_cast<std::size_t>(chunks)),
                   {count, chunks},
                   DType::kInt32};
        const auto outputs = encoder_->forward(inputs);
        const auto found = outputs.find("audio_embeds");
        const auto expected = static_cast<std::size_t>(count) * window_rows * width;
        if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
            found->second.data == nullptr || found->second.numel() != expected)
            throw std::runtime_error("qwen3_asr: the audio encoder returned unusable audio_embeds");
        const auto* values = static_cast<const float*>(found->second.data);
        rows.insert(rows.end(), values, values + expected);
    }
    // Only the last chunk can be partial, so the valid tokens are a prefix of the token grid.
    rows.resize(static_cast<std::size_t>(batch.audio_tokens) * width);
    return rows;
}

std::int32_t Qwen3AsrPipeline::launch(ITrtModule& module, std::vector<std::int32_t>& ids,
                                      std::vector<float>& audio_rows,
                                      std::vector<std::int32_t>& audio_mask,
                                      std::int32_t position) {
    const auto length = static_cast<std::int64_t>(ids.size());
    std::int32_t write_index = position;
    const std::int32_t rows = key_rows(config_, position + static_cast<std::int32_t>(length));
    TensorMap inputs;
    inputs["input_ids"] = Tensor{ids.data(), {1, length}, DType::kInt32};
    inputs["audio_embeds"] =
        Tensor{audio_rows.data(), {1, length, config_.audio_embed_size}, DType::kFloat32};
    inputs["audio_mask"] = Tensor{audio_mask.data(), {1, length}, DType::kInt32};
    inputs["cache_write_index"] = Tensor{&write_index, {1}, DType::kInt32};
    inputs["key_positions"] = Tensor{key_positions_.data(), {rows}, DType::kInt32};
    // Never forward(): it would copy every cache-sized output to the host.
    module.forward_async(inputs);
    module.sync();
    const void* logits = module.device_ptr("logits");
    if (logits == nullptr || cudaMemcpy(logits_.data(), logits, logits_.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("qwen3_asr: reading the decoder logits failed");
    // The first maximum, as torch.argmax returns it.
    return static_cast<std::int32_t>(std::max_element(logits_.begin(), logits_.end()) -
                                     logits_.begin());
}

TextResult Qwen3AsrPipeline::run(const internal::SpeechTranscriptionRequest& request,
                                 internal::ConfigView options) {
    internal::validate_config(kFields, options);
    const auto limit = internal::config_get<std::int64_t>(options, kFields, "max_output_tokens")
                           .value_or(config_.default_max_output_tokens);
    if (limit < 1 || limit > config_.max_cache_length)
        throw std::invalid_argument("qwen3_asr max_output_tokens must be in [1, " +
                                    std::to_string(config_.max_cache_length) + "]");
    const LanguagePrefill* language =
        request.source_language ? &find_language(config_, *request.source_language) : nullptr;
    auto mono = mono_audio(request.audio);
    // Refuse a request that cannot fit before any front-end work: the prompt length follows from
    // the clip length alone.
    const auto prompt_tokens =
        static_cast<std::int64_t>(config_.prompt_prefix_ids.size() +
                                  config_.prompt_suffix_ids.size() +
                                  (language ? language->ids.size() : 0)) +
        audio_token_count(static_cast<std::int64_t>(mono.size()), config_.frontend);
    const auto capacity = static_cast<std::int64_t>(config_.max_cache_length);
    if (prompt_tokens + limit > capacity) {
        throw std::invalid_argument(
            "qwen3_asr: " + std::to_string(prompt_tokens) +
            " prompt tokens plus max_output_tokens " + std::to_string(limit) +
            " exceed the bundle's " + std::to_string(capacity) +
            "-token cache; use less audio, fewer output tokens, or a larger max_sequence_length");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto setup_start = Clock::now();
    const auto batch = encoder_batch(log_mel(mono, config_.frontend, filters_), config_.frontend);
    auto prompt = prompt_ids(config_, batch.audio_tokens, language);
    if (static_cast<std::int64_t>(prompt.size()) != prompt_tokens)
        throw std::logic_error("qwen3_asr: the audio front end disagrees with its token count");
    const auto audio = encode(batch);
    const auto setup_end = Clock::now();

    const auto width = static_cast<std::size_t>(config_.audio_embed_size);
    std::vector<float> audio_rows(prompt.size() * width, 0.0F);
    std::vector<std::int32_t> audio_mask(prompt.size(), 0);
    const auto first = config_.prompt_prefix_ids.size();
    std::copy(audio.begin(), audio.end(),
              audio_rows.begin() + static_cast<std::ptrdiff_t>(first * width));
    std::fill_n(audio_mask.begin() + static_cast<std::ptrdiff_t>(first), batch.audio_tokens, 1);
    const auto prompt_length = static_cast<std::int32_t>(prompt.size());
    // A prompt longer than one prefill launch is appended to the cache in consecutive chunks;
    // only the last chunk's logits pick the first token.
    std::int32_t next = 0;
    for (std::int32_t start = 0; start < prompt_length; start += config_.prefill_chunk_tokens) {
        const auto count = std::min(config_.prefill_chunk_tokens, prompt_length - start);
        const auto begin = static_cast<std::ptrdiff_t>(start);
        const auto end = static_cast<std::ptrdiff_t>(start + count);
        const auto row = static_cast<std::ptrdiff_t>(width);
        std::vector<std::int32_t> ids(prompt.begin() + begin, prompt.begin() + end);
        std::vector<float> rows(audio_rows.begin() + begin * row, audio_rows.begin() + end * row);
        std::vector<std::int32_t> mask(audio_mask.begin() + begin, audio_mask.begin() + end);
        next = launch(*prefill_, ids, rows, mask, start);
    }
    const auto prefill_end = Clock::now();

    const auto& stops = config_.stop_token_ids;
    std::vector<std::int32_t> generated;
    std::vector<std::int32_t> token(1);
    std::vector<float> no_audio(width, 0.0F);
    std::vector<std::int32_t> no_mask(1, 0);
    while (std::find(stops.begin(), stops.end(), next) == stops.end()) {
        generated.push_back(next);
        if (static_cast<std::int64_t>(generated.size()) >= limit)
            break;
        token[0] = next;
        next = launch(*decode_, token, no_audio, no_mask,
                      prompt_length + static_cast<std::int32_t>(generated.size()) - 1);
    }
    const auto decode_end = Clock::now();

    TextResult result;
    result.text = transcription_from_output(vocabulary_.decode(generated), config_.asr_text_marker,
                                            config_.repetition_threshold);
    result.token_ids = std::move(generated);
    result.setup_ms = milliseconds(setup_start, setup_end);
    result.prefill_ms = milliseconds(setup_end, prefill_end);
    result.decode_ms = milliseconds(prefill_end, decode_end);
    return result;
}

} // namespace trtmc::qwen3_asr
