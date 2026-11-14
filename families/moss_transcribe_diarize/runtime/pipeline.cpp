/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/moss_transcribe_diarize/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::moss_transcribe_diarize {
namespace {

constexpr const char* kFamily = "moss_transcribe_diarize";
// Attended cache rows grow in steps of this many rows.
constexpr std::int32_t kKeyRowBucket = 32;

const internal::ConfigField kFields[] = {
    {"max_new_tokens", internal::ConfigKind::I64, std::nullopt,
     "Maximum generated tokens. Default: the checkpoint's generation limit, bounded by the "
     "sequence capacity left after the prompt."},
};

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

std::runtime_error engine_error(const std::string& what) {
    return std::runtime_error(std::string(kFamily) + " bundle: " + what);
}

} // namespace

TranscribeDiarizeModel::TranscribeDiarizeModel(Engines engines, RuntimeConfig config,
                                               MelFilterBank mel_filters, TokenDecoder decoder)
    : engines_(std::move(engines)), config_(std::move(config)),
      mel_(std::move(mel_filters), config_.n_fft, config_.hop_length, config_.chunk_samples,
           config_.mel_frames),
      decoder_(std::move(decoder)),
      work_dtype_(config_.precision == "bf16" ? DType::kBFloat16 : DType::kFloat32) {
    if (!engines_.audio || !engines_.prefill || !engines_.decode || !engines_.audio->ok() ||
        !engines_.prefill->ok() || !engines_.decode->ok())
        throw engine_error("an engine failed to load");
    if (mel_.mel_bins() != config_.num_mel_bins)
        throw engine_error("mel filter bank and runtime.json disagree on the mel bins");
    if (decoder_.size() != static_cast<std::size_t>(config_.vocab_size))
        throw engine_error("decode table and runtime.json disagree on the vocabulary");
    validate_engines();

    const cudaStream_t stream = engines_.prefill->stream();
    const std::vector<int64_t> cache_shape{1, config_.num_kv_heads, config_.max_sequence_length,
                                           config_.head_dim};
    // Zeroed once: rows past the valid prefix are masked and must stay finite.
    for (std::int32_t i = 0; i < 2 * config_.num_layers; ++i) {
        cache_.push_back(DeviceTensor::zeros(cache_shape, work_dtype_, stream));
        if (!cache_.back().ok())
            throw std::runtime_error("moss_transcribe_diarize KV-cache allocation failed");
    }
    // Every chunk but the last is full, so the valid rows of a request are a
    // prefix of the chunk-major engine outputs.
    const std::int64_t chunks = config_.max_sequence_length / config_.tokens_per_chunk + 2;
    audio_rows_ = DeviceTensor(
        {chunks * config_.tokens_per_chunk, static_cast<std::int64_t>(config_.hidden_size)},
        work_dtype_, stream);
    decode_audio_row_ = DeviceTensor::zeros({1, static_cast<std::int64_t>(config_.hidden_size)},
                                            work_dtype_, stream);
    if (!audio_rows_.ok() || !decode_audio_row_.ok())
        throw std::runtime_error("moss_transcribe_diarize audio buffer allocation failed");
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        throw std::runtime_error("moss_transcribe_diarize buffer initialisation failed");

    key_positions_.resize(static_cast<std::size_t>(config_.max_sequence_length));
    std::iota(key_positions_.begin(), key_positions_.end(), 0);
    bind_cache(*engines_.prefill);
    bind_cache(*engines_.decode);
    engines_.decode->bind_external("audio_embeds", decode_audio_row_.data(),
                                   {1, static_cast<std::int64_t>(config_.hidden_size)});
}

void TranscribeDiarizeModel::validate_engines() const {
    const auto& audio = *engines_.audio;
    const std::vector<int64_t> features{1, config_.num_mel_bins, config_.mel_frames};
    if (!audio.has_input("input_features") ||
        audio.tensor_dtype("input_features") != DType::kFloat32 ||
        audio.tensor_shape("input_features") != features)
        throw engine_error("audio engine input does not match runtime.json");
    const std::vector<int64_t> rows{config_.tokens_per_chunk, config_.hidden_size};
    if (!audio.has_output("audio_embeds") || audio.tensor_dtype("audio_embeds") != work_dtype_ ||
        audio.tensor_shape("audio_embeds") != rows)
        throw engine_error("audio engine output does not match runtime.json");

    const std::vector<int64_t> cache{1, config_.num_kv_heads, config_.max_sequence_length,
                                     config_.head_dim};
    for (const auto* module : {engines_.prefill.get(), engines_.decode.get()}) {
        for (const char* name :
             {"input_ids", "audio_index", "position_ids", "cache_write_index", "key_positions"}) {
            if (!module->has_input(name) || module->tensor_dtype(name) != DType::kInt32)
                throw engine_error(std::string("text engine input ") + name + " is missing");
        }
        if (module->tensor_dtype("audio_embeds") != work_dtype_)
            throw engine_error("text engine audio rows do not match the audio engine");
        const std::vector<int64_t> logits{1, config_.vocab_size};
        if (!module->has_output("logits") || module->tensor_dtype("logits") != DType::kFloat32 ||
            module->tensor_shape("logits") != logits)
            throw engine_error("text engine logits do not match runtime.json");
        for (std::int32_t layer = 0; layer < config_.num_layers; ++layer) {
            for (const char* stem : {"cache_k_", "cache_v_"}) {
                const auto name = stem + std::to_string(layer);
                if (!module->has_input(name) || module->tensor_dtype(name) != work_dtype_ ||
                    module->tensor_shape(name) != cache)
                    throw engine_error("text engine cache " + name + " does not match");
            }
        }
    }
}

void TranscribeDiarizeModel::bind_cache(ITrtModule& module) {
    for (std::int32_t layer = 0; layer < config_.num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(2 * layer);
        const auto suffix = std::to_string(layer);
        module.bind_external("cache_k_" + suffix, cache_[index].data());
        module.bind_external("cache_v_" + suffix, cache_[index + 1].data());
        if (module.device_ptr("present_k_" + suffix) != cache_[index].data() ||
            module.device_ptr("present_v_" + suffix) != cache_[index + 1].data())
            throw engine_error("present KV tensors do not alias the cache");
    }
}

std::vector<internal::TaskInstance> TranscribeDiarizeModel::task_bindings() {
    return {internal::bind<internal::ISpeechTranscription>(*this, kFields)};
}

std::vector<float> TranscribeDiarizeModel::mono_samples(const internal::AudioView& audio) const {
    if (audio.samples.data() == nullptr || audio.samples.empty())
        throw std::invalid_argument("moss_transcribe_diarize requires non-empty float32 PCM");
    const auto rate = audio.sample_rate.value_or(static_cast<std::uint32_t>(config_.sample_rate));
    if (rate != static_cast<std::uint32_t>(config_.sample_rate)) {
        // The checkpoint's processor consumes 16 kHz audio; this family ships no resampler.
        throw std::invalid_argument("moss_transcribe_diarize requires " +
                                    std::to_string(config_.sample_rate) + " Hz audio, not " +
                                    std::to_string(rate) + " Hz");
    }
    const std::size_t channels = audio.channels == 0 ? 1 : audio.channels;
    if (audio.samples.size() % channels != 0)
        throw std::invalid_argument("interleaved PCM must hold whole frames");
    const std::size_t frames = audio.samples.size() / channels;
    std::vector<float> mono(frames);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double sum = 0.0;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const float value = audio.samples[frame * channels + channel];
            if (!std::isfinite(value))
                throw std::invalid_argument("PCM samples must be finite");
            sum += value;
        }
        mono[frame] = static_cast<float>(sum / static_cast<double>(channels));
    }
    return mono;
}

std::int32_t TranscribeDiarizeModel::resolve_max_new_tokens(internal::ConfigView config,
                                                            std::size_t prompt_tokens) const {
    // The last generated token is never fed back, so the cache must hold
    // prompt_tokens + max_new_tokens - 1 rows.
    const auto room = static_cast<std::int64_t>(config_.max_sequence_length) -
                      static_cast<std::int64_t>(prompt_tokens) + 1;
    const auto requested = internal::config_get<std::int64_t>(config, kFields, "max_new_tokens");
    if (!requested)
        return static_cast<std::int32_t>(
            std::min<std::int64_t>(config_.default_max_new_tokens, room));
    if (*requested <= 0)
        throw internal::ConfigError("max_new_tokens must be positive");
    if (*requested > room) {
        throw internal::ConfigError("max_new_tokens=" + std::to_string(*requested) +
                                    " exceeds the " + std::to_string(room) +
                                    " tokens left after the " + std::to_string(prompt_tokens) +
                                    "-token prompt in this bundle's max_sequence_length=" +
                                    std::to_string(config_.max_sequence_length));
    }
    return static_cast<std::int32_t>(*requested);
}

void TranscribeDiarizeModel::encode_audio(const std::vector<float>& samples,
                                          const std::vector<AudioChunk>& chunks) {
    auto& engine = *engines_.audio;
    const std::size_t row_bytes =
        static_cast<std::size_t>(config_.hidden_size) * dtype_size(work_dtype_);
    for (std::size_t index = 0; index < chunks.size(); ++index) {
        const auto& chunk = chunks[index];
        auto features = mel_.chunk(samples.data() + chunk.first_sample, chunk.samples);
        auto* rows = static_cast<char*>(audio_rows_.data()) +
                     index * static_cast<std::size_t>(config_.tokens_per_chunk) * row_bytes;
        engine.bind_external("audio_embeds", rows);
        engine.forward({{"input_features", Tensor{features.data(),
                                                  {1, config_.num_mel_bins, config_.mel_frames},
                                                  DType::kFloat32}}});
    }
}

std::int32_t TranscribeDiarizeModel::key_rows(std::int32_t valid_rows) const {
    const auto rounded = (valid_rows + kKeyRowBucket - 1) / kKeyRowBucket * kKeyRowBucket;
    return std::min(rounded, config_.max_sequence_length);
}

std::int32_t TranscribeDiarizeModel::next_token(const TensorMap& outputs) const {
    const auto found = outputs.find("logits");
    if (found == outputs.end() || found->second.data == nullptr ||
        found->second.numel() != static_cast<std::size_t>(config_.vocab_size))
        throw engine_error("text engine returned no logits");
    const auto* logits = static_cast<const float*>(found->second.data);
    // Greedy decoding: the first maximal logit, as torch.argmax picks it.
    std::int32_t best = 0;
    for (std::int32_t i = 0; i < config_.vocab_size; ++i) {
        if (!std::isfinite(logits[i]))
            throw std::runtime_error("moss_transcribe_diarize produced non-finite logits");
        if (logits[i] > logits[best])
            best = i;
    }
    return best;
}

std::int32_t TranscribeDiarizeModel::prefill(const PromptPlan& plan) {
    auto& engine = *engines_.prefill;
    engine.bind_external("audio_embeds", audio_rows_.data(),
                         {plan.audio_tokens, static_cast<std::int64_t>(config_.hidden_size)});
    const auto total = static_cast<std::int32_t>(plan.input_ids.size());
    std::vector<std::int32_t> positions(static_cast<std::size_t>(config_.prefill_chunk_tokens));
    TensorMap outputs;
    for (std::int32_t start = 0; start < total; start += config_.prefill_chunk_tokens) {
        const std::int32_t count = std::min(config_.prefill_chunk_tokens, total - start);
        std::iota(positions.begin(), positions.begin() + count, start);
        std::int32_t write_index = start;
        auto* ids = const_cast<std::int32_t*>(plan.input_ids.data() + start);
        auto* rows = const_cast<std::int32_t*>(plan.audio_index.data() + start);
        outputs = engine.forward({
            {"input_ids", Tensor{ids, {count}, DType::kInt32}},
            {"audio_index", Tensor{rows, {count}, DType::kInt32}},
            {"position_ids", Tensor{positions.data(), {count}, DType::kInt32}},
            {"cache_write_index", Tensor{&write_index, {1}, DType::kInt32}},
            {"key_positions",
             Tensor{key_positions_.data(), {key_rows(start + count)}, DType::kInt32}},
        });
    }
    return next_token(outputs);
}

std::int32_t TranscribeDiarizeModel::decode_step(std::int32_t token, std::int32_t position) {
    std::int32_t audio_row = -1;
    std::int32_t write_index = position;
    return next_token(engines_.decode->forward({
        {"input_ids", Tensor{&token, {1}, DType::kInt32}},
        {"audio_index", Tensor{&audio_row, {1}, DType::kInt32}},
        {"position_ids", Tensor{&position, {1}, DType::kInt32}},
        {"cache_write_index", Tensor{&write_index, {1}, DType::kInt32}},
        {"key_positions", Tensor{key_positions_.data(), {key_rows(position + 1)}, DType::kInt32}},
    }));
}

bool TranscribeDiarizeModel::is_end_of_turn(std::int32_t token) const {
    return std::find(config_.eos_token_ids.begin(), config_.eos_token_ids.end(), token) !=
           config_.eos_token_ids.end();
}

TextResult TranscribeDiarizeModel::run(const internal::SpeechTranscriptionRequest& request,
                                       internal::ConfigView config) {
    internal::validate_config(kFields, config);
    if (request.source_language) {
        throw std::invalid_argument(
            "moss_transcribe_diarize identifies the spoken language itself and takes no "
            "source_language");
    }
    const auto samples = mono_samples(request.audio);
    const auto chunks = plan_chunks(static_cast<std::int64_t>(samples.size()),
                                    config_.chunk_samples, config_.samples_per_token);
    std::int32_t audio_tokens = 0;
    for (const auto& chunk : chunks)
        audio_tokens += chunk.audio_tokens;
    const auto plan = build_prompt(config_, audio_tokens);
    if (plan.input_ids.size() > static_cast<std::size_t>(config_.max_sequence_length)) {
        throw std::invalid_argument("moss_transcribe_diarize: this audio needs a " +
                                    std::to_string(plan.input_ids.size()) +
                                    "-token prompt, more than the bundle's max_sequence_length=" +
                                    std::to_string(config_.max_sequence_length) +
                                    "; rebuild the bundle with a larger max_sequence_length");
    }
    const auto limit = resolve_max_new_tokens(config, plan.input_ids.size());

    std::lock_guard<std::mutex> lock(mutex_);
    const auto begin = Clock::now();
    encode_audio(samples, chunks);
    const auto encoded = Clock::now();
    std::int32_t token = prefill(plan);
    const auto first = Clock::now();
    std::vector<std::int32_t> generated{token};
    auto position = static_cast<std::int32_t>(plan.input_ids.size());
    while (static_cast<std::int32_t>(generated.size()) < limit && !is_end_of_turn(token)) {
        token = decode_step(token, position++);
        generated.push_back(token);
    }
    const auto done = Clock::now();

    TextResult result;
    result.text = decoder_.decode(generated);
    result.token_ids = std::move(generated);
    for (auto& segment : parse_transcript(result.text)) {
        // The speaker label stays in the segment text, written as the model writes it.
        result.segments.push_back({segment.start_seconds,
                                   segment.end_seconds,
                                   "[" + segment.speaker + "]" + segment.text,
                                   {}});
    }
    result.setup_ms = elapsed_ms(begin, encoded);
    result.prefill_ms = elapsed_ms(encoded, first);
    result.decode_ms = elapsed_ms(first, done);
    return result;
}

} // namespace trtmc::moss_transcribe_diarize
