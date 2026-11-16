/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron3_diarization/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::nemotron3_diarization {
namespace {

using Clock = std::chrono::steady_clock;

// Key bias of a padding frame: it is excluded as a key and stays a query,
// because the speaker head's convolution reads its neighbours. Finite, and
// exp() of it is exactly zero.
constexpr float kMaskedKey = -1.0e30F;

const internal::ConfigField kFields[] = {
    {"threshold", internal::ConfigKind::F64, internal::ConfigValue{0.5},
     "Speaker activity probability above which a 10 ms frame counts as that speaker's speech."}};

double elapsed_ms(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

float threshold(internal::ConfigView supplied) {
    const double value = *internal::config_get<double>(supplied, kFields, "threshold");
    if (!std::isfinite(value) || value <= 0.0 || value >= 1.0)
        throw internal::ConfigError("threshold must lie strictly between 0 and 1");
    return static_cast<float>(value);
}

const Tensor& float_output(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || found->second.data == nullptr ||
        found->second.dtype != DType::kFloat32 || found->second.numel() != count)
        throw std::runtime_error(std::string("nemotron3_diarization engine returned an unusable ") +
                                 name);
    return found->second;
}

} // namespace

bool speaker_active(float logit, float threshold) {
    const float probability = 1.0F / (1.0F + std::exp(-logit));
    return probability > threshold;
}

std::vector<internal::TimedLabel> active_intervals(const std::vector<float>& logits,
                                                   std::int64_t frames, std::int64_t speakers,
                                                   float threshold, std::int32_t hop_length,
                                                   std::int32_t sample_rate) {
    struct Run {
        std::int64_t start, end, speaker;
    };
    std::vector<Run> runs;
    for (std::int64_t speaker = 0; speaker < speakers; ++speaker) {
        std::int64_t start = -1;
        for (std::int64_t frame = 0; frame <= frames; ++frame) {
            const bool active =
                frame < frames && speaker_active(logits[frame * speakers + speaker], threshold);
            if (active && start < 0)
                start = frame;
            if (!active && start >= 0) {
                runs.push_back({start, frame, speaker});
                start = -1;
            }
        }
    }
    std::sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) {
        return a.start != b.start ? a.start < b.start : a.speaker < b.speaker;
    });
    // Frame f starts at f * hop samples; the division is rounded once, so a
    // boundary lands on the nearest double to its decimal value (0.03, 15.43).
    const auto seconds = [&](std::int64_t frame) {
        return static_cast<double>(frame * hop_length) / static_cast<double>(sample_rate);
    };
    std::vector<internal::TimedLabel> intervals;
    intervals.reserve(runs.size());
    for (const auto& run : runs)
        intervals.push_back(
            {seconds(run.start), seconds(run.end), static_cast<std::uint32_t>(run.speaker)});
    return intervals;
}

DiarizationPipeline::DiarizationPipeline(std::unique_ptr<ITrtModule> embedder,
                                         std::unique_ptr<ITrtModule> encoder,
                                         DiarizationConfig config)
    : embedder_(std::move(embedder)), encoder_(std::move(encoder)), config_(std::move(config)),
      front_end_(config_.front_end) {
    if (!embedder_ || !encoder_)
        throw std::runtime_error("nemotron3_diarization pipeline is missing an engine");
    if (config_.labels.size() != static_cast<std::size_t>(config_.cache.num_speakers))
        throw std::runtime_error("nemotron3_diarization needs one label per speaker");
    if (config_.chunk_length <= 0 || config_.chunk_left_context < 0 ||
        config_.chunk_right_context < 0 ||
        config_.max_step_rows < config_.cache.cache_length + config_.cache.fifo_length +
                                    config_.chunk_left_context + config_.chunk_length +
                                    config_.chunk_right_context)
        throw std::runtime_error("nemotron3_diarization chunking does not fit one encoder step");
}

std::vector<internal::TaskInstance> DiarizationPipeline::task_bindings() {
    return {internal::bind<internal::IAudioToTimedLabels>(*this, kFields)};
}

std::vector<float> DiarizationPipeline::embed(const LogMel& features, std::int64_t& rows) {
    const auto factor = static_cast<std::int64_t>(config_.cache.subsampling_factor);
    const auto mels = static_cast<std::int64_t>(config_.front_end.num_mel_bins);
    const auto width = static_cast<std::int64_t>(config_.cache.hidden_size);
    // Groups of `factor` frames, the last one completed with zero frames.
    rows = (features.frames + factor - 1) / factor;
    std::vector<float> stacked(static_cast<std::size_t>(rows * factor * mels), 0.0F);
    std::copy(features.values.begin(), features.values.end(), stacked.begin());

    std::vector<float> embeddings(static_cast<std::size_t>(rows * width));
    const std::int64_t block = config_.max_step_rows;
    for (std::int64_t first = 0; first < rows; first += block) {
        const std::int64_t count = std::min(block, rows - first);
        Tensor input{
            stacked.data() + first * factor * mels, {count, factor * mels}, DType::kFloat32};
        const auto outputs = embedder_->forward({{"stacked_features", input}});
        const auto& out =
            float_output(outputs, "inputs_embeds", static_cast<std::size_t>(count * width));
        std::memcpy(embeddings.data() + first * width, out.data,
                    static_cast<std::size_t>(count * width) * sizeof(float));
    }
    return embeddings;
}

std::vector<float> DiarizationPipeline::diarize(const std::vector<float>& embeddings,
                                                std::int64_t rows,
                                                const std::vector<std::uint8_t>& valid_rows,
                                                std::int64_t frames) {
    const auto width = static_cast<std::int64_t>(config_.cache.hidden_size);
    const auto factor = static_cast<std::int64_t>(config_.cache.subsampling_factor);
    const auto speakers = static_cast<std::int64_t>(config_.cache.num_speakers);
    SpeakerCache cache(config_.cache, config_.silence_embedding);
    std::vector<float> logits;
    logits.reserve(static_cast<std::size_t>(rows * factor * speakers));
    std::vector<float> input, bias;
    std::vector<std::uint8_t> valid;
    for (std::int64_t start = 0; start < rows; start += config_.chunk_length) {
        const std::int64_t end = std::min<std::int64_t>(start + config_.chunk_length, rows);
        const std::int64_t first = std::max<std::int64_t>(start - config_.chunk_left_context, 0);
        const std::int64_t last = std::min<std::int64_t>(end + config_.chunk_right_context, rows);
        const std::int64_t cached = cache.cached_rows();
        const std::int64_t step = cached + (last - first);
        if (step > config_.max_step_rows)
            throw std::runtime_error("nemotron3_diarization step exceeds the engine profile");

        input.resize(static_cast<std::size_t>(step * width));
        cache.copy_cached_rows(input.data());
        std::memcpy(input.data() + cached * width, embeddings.data() + first * width,
                    static_cast<std::size_t>((last - first) * width) * sizeof(float));
        valid.assign(static_cast<std::size_t>(cached), 1U);
        valid.insert(valid.end(), valid_rows.begin() + first, valid_rows.begin() + last);
        bias.resize(valid.size());
        for (std::size_t i = 0; i < valid.size(); ++i)
            bias[i] = valid[i] ? 0.0F : kMaskedKey;

        const auto outputs =
            encoder_->forward({{"inputs_embeds", Tensor{input.data(), {step, width}}},
                               {"key_bias", Tensor{bias.data(), {step}}}});
        const auto& out =
            float_output(outputs, "logits", static_cast<std::size_t>(step * factor * speakers));
        const auto* step_logits = static_cast<const float*>(out.data);
        cache.update(input.data(), step, step_logits, valid, end - start, start - first);
        // Only the chunk's own frames are scored here; its look-ahead opens the next chunk.
        const std::int64_t begin = (cached + start - first) * factor * speakers;
        logits.insert(logits.end(), step_logits + begin,
                      step_logits + begin + (end - start) * factor * speakers);
    }
    // The last stacked group may end in padding frames past the spectrogram.
    logits.resize(static_cast<std::size_t>(frames * speakers));
    return logits;
}

internal::TimedLabelsResult
DiarizationPipeline::run(const internal::AudioToTimedLabelsRequest& request,
                         internal::ConfigView supplied) {
    const float activity_threshold = threshold(supplied);
    const auto& audio = request.audio;
    const auto rate =
        audio.sample_rate.value_or(static_cast<std::uint32_t>(config_.front_end.sample_rate));
    // Framing and mel band edges are fixed to the training rate; the family
    // does not resample.
    if (rate != static_cast<std::uint32_t>(config_.front_end.sample_rate))
        throw std::invalid_argument("nemotron3_diarization requires " +
                                    std::to_string(config_.front_end.sample_rate) +
                                    " Hz audio, not " + std::to_string(rate) + " Hz");
    const auto mono = mix_to_mono(audio.samples.data(), audio.samples.size(), audio.channels);
    if (mono.size() < static_cast<std::size_t>(config_.front_end.hop_length))
        throw std::invalid_argument(
            "nemotron3_diarization needs at least one 10 ms frame of audio");

    const auto setup = Clock::now();
    const auto features = front_end_.compute(mono.data(), static_cast<std::int64_t>(mono.size()));
    const auto factor = static_cast<std::int64_t>(config_.cache.subsampling_factor);
    std::int64_t rows = 0;
    const auto embeddings = embed(features, rows);
    // An encoder frame is valid when its first spectrogram frame is.
    std::vector<std::uint8_t> valid_rows(static_cast<std::size_t>(rows));
    for (std::int64_t row = 0; row < rows; ++row)
        valid_rows[static_cast<std::size_t>(row)] = row * factor < features.valid_frames;
    const double setup_ms = elapsed_ms(setup);

    const auto inference = Clock::now();
    auto logits = diarize(embeddings, rows, valid_rows, features.frames);
    const auto speakers = static_cast<std::int64_t>(config_.cache.num_speakers);
    // Frames past the audio are padding; the result covers the audio only.
    logits.resize(static_cast<std::size_t>(features.valid_frames * speakers));
    for (const float value : logits)
        if (!std::isfinite(value))
            throw std::runtime_error("nemotron3_diarization engine returned a non-finite logit");

    internal::TimedLabelsResult result;
    result.intervals =
        active_intervals(logits, features.valid_frames, speakers, activity_threshold,
                         config_.front_end.hop_length, config_.front_end.sample_rate);
    result.labels = config_.labels;
    result.frame_scores = std::move(logits);
    result.frame_count = static_cast<std::uint64_t>(features.valid_frames);
    result.frame_seconds = static_cast<double>(config_.front_end.hop_length) /
                           static_cast<double>(config_.front_end.sample_rate);
    result.score_kind = internal::ScoreKind::Logit;
    result.setup_ms = setup_ms;
    result.inference_ms = elapsed_ms(inference);
    return result;
}

} // namespace trtmc::nemotron3_diarization
