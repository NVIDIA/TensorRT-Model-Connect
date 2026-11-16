/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron3_diarization/runtime/speaker_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::nemotron3_diarization {
namespace {

// The reference's score constants, rounded once to float32 as torch does when
// it combines a Python float with a float32 tensor.
constexpr float kLogHalf = -0.6931471805599453F;
constexpr float kStrongBoost = 1.3862943611198906F; // -2 * log(0.5)
constexpr float kWeakBoost = 0.6931471805599453F;   // -log(0.5)
constexpr float kMinusInfinity = -std::numeric_limits<float>::infinity();
constexpr float kPlusInfinity = std::numeric_limits<float>::infinity();

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(std::string("nemotron3_diarization speaker cache: ") + message);
}

float sigmoid(float logit) {
    return 1.0F / (1.0F + std::exp(-logit));
}

void append_rows(std::vector<float>& destination, const float* rows, std::int64_t count,
                 std::int64_t width) {
    destination.insert(destination.end(), rows, rows + count * width);
}

} // namespace

std::vector<std::size_t> top_k_first_seen(const std::vector<float>& values, std::size_t k) {
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    // Larger first; NaN ranks above every number, as in torch.topk. The stable
    // sort keeps equal values in index order, so the k-th place is the first seen.
    const auto larger = [&](std::size_t a, std::size_t b) {
        const float x = values[a], y = values[b];
        return (std::isnan(x) && !std::isnan(y)) || x > y;
    };
    std::stable_sort(order.begin(), order.end(), larger);
    order.resize(std::min(k, order.size()));
    return order;
}

SpeakerCache::SpeakerCache(SpeakerCachePolicy policy, std::vector<float> silence_embedding)
    : policy_(policy), silence_(std::move(silence_embedding)) {
    const auto& p = policy_;
    require(p.num_speakers > 0 && p.subsampling_factor > 0 && p.hidden_size > 0,
            "geometry is unusable");
    require(p.silence_frames_per_speaker >= 0 &&
                p.cache_length >= (1 + p.silence_frames_per_speaker) * p.num_speakers,
            "the cache cannot hold every speaker");
    require(p.fifo_length >= 0 && p.update_period > 0, "FIFO sizes are unusable");
    require(p.min_positive_scores >= 0 && p.strong_boosted_frames >= 0 &&
                p.weak_boosted_frames >= 0,
            "frame budgets are unusable");
    require(silence_.size() == static_cast<std::size_t>(p.hidden_size),
            "silence embedding width differs from the encoder width");
}

void SpeakerCache::copy_cached_rows(float* destination) const {
    const auto width = static_cast<std::size_t>(policy_.hidden_size);
    std::memcpy(destination, cache_.data(),
                static_cast<std::size_t>(cache_rows_) * width * sizeof(float));
    std::memcpy(destination + static_cast<std::size_t>(cache_rows_) * width, fifo_.data(),
                static_cast<std::size_t>(fifo_rows_) * width * sizeof(float));
}

std::vector<float> SpeakerCache::pool_probabilities(const float* logits, std::int64_t rows,
                                                    const std::vector<std::uint8_t>& valid) const {
    const auto speakers = static_cast<std::int64_t>(policy_.num_speakers);
    const auto factor = static_cast<std::int64_t>(policy_.subsampling_factor);
    require(valid.size() == static_cast<std::size_t>(rows), "frame mask does not match the step");
    std::vector<float> probs(static_cast<std::size_t>(rows * speakers));
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t speaker = 0; speaker < speakers; ++speaker) {
            float total = 0.0F;
            for (std::int64_t sub = 0; sub < factor; ++sub)
                total += sigmoid(logits[(row * factor + sub) * speakers + speaker]);
            const float mask = valid[static_cast<std::size_t>(row)] ? 1.0F : 0.0F;
            probs[static_cast<std::size_t>(row * speakers + speaker)] =
                total / static_cast<float>(factor) * mask;
        }
    }
    return probs;
}

std::vector<float> SpeakerCache::frame_scores(const std::vector<float>& probs,
                                              std::int64_t rows) const {
    const auto speakers = static_cast<std::size_t>(policy_.num_speakers);
    const float floor = policy_.score_threshold;
    std::vector<float> scores(probs.size());
    std::vector<float> log_probs(speakers), log_complements(speakers);
    for (std::size_t row = 0; row < static_cast<std::size_t>(rows); ++row) {
        const float* p = probs.data() + row * speakers;
        float complement_total = 0.0F;
        for (std::size_t s = 0; s < speakers; ++s) {
            log_probs[s] = std::log(std::max(p[s], floor));
            log_complements[s] = std::log(std::max(1.0F - p[s], floor));
            complement_total += log_complements[s];
        }
        for (std::size_t s = 0; s < speakers; ++s) {
            const float difference = log_probs[s] - log_complements[s];
            const float joint = difference + complement_total;
            scores[row * speakers + s] = p[s] > 0.5F ? joint - kLogHalf : kMinusInfinity;
        }
    }
    // A speaker with enough positively scored frames loses its non-positive
    // ones: overlapped speech, which describes the speaker worst.
    std::vector<std::int64_t> positives(speakers, 0);
    for (std::size_t i = 0; i < scores.size(); ++i)
        positives[i % speakers] += scores[i] > 0.0F;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const bool speech = probs[i] > 0.5F;
        const bool enough = positives[i % speakers] >= policy_.min_positive_scores;
        if (!(scores[i] > 0.0F) && speech && enough)
            scores[i] = kMinusInfinity;
    }
    return scores;
}

namespace {

void boost_scores(std::vector<float>& scores, std::size_t rows, std::size_t speakers,
                  std::size_t count, float boost) {
    std::vector<float> column(rows);
    for (std::size_t speaker = 0; speaker < speakers; ++speaker) {
        for (std::size_t row = 0; row < rows; ++row)
            column[row] = scores[row * speakers + speaker];
        for (const auto row : top_k_first_seen(column, count))
            scores[row * speakers + speaker] += boost;
    }
}

} // namespace

void SpeakerCache::compress(std::vector<float>& embeddings, std::vector<float>& probs,
                            std::int64_t& rows) const {
    const auto frames = static_cast<std::size_t>(rows);
    const auto speakers = static_cast<std::size_t>(policy_.num_speakers);
    const auto width = static_cast<std::size_t>(policy_.hidden_size);
    const auto length = static_cast<std::size_t>(policy_.cache_length);
    auto scores = frame_scores(probs, rows);
    // Frames past the cache capacity are the ones just popped from the FIFO.
    for (std::size_t i = length * speakers; i < scores.size(); ++i)
        scores[i] += policy_.latest_frames_boost;
    boost_scores(scores, frames, speakers, static_cast<std::size_t>(policy_.strong_boosted_frames),
                 kStrongBoost);
    boost_scores(scores, frames, speakers, static_cast<std::size_t>(policy_.weak_boosted_frames),
                 kWeakBoost);

    // Speaker-major candidates, each speaker's frames followed by its reserved
    // silence slots, which always win.
    const std::size_t scored =
        frames + static_cast<std::size_t>(policy_.silence_frames_per_speaker);
    std::vector<float> flat(scored * speakers, kPlusInfinity);
    for (std::size_t speaker = 0; speaker < speakers; ++speaker)
        for (std::size_t frame = 0; frame < frames; ++frame)
            flat[speaker * scored + frame] = scores[frame * speakers + speaker];
    const std::size_t sentinel = flat.size();
    auto chosen = top_k_first_seen(flat, length);
    for (auto& index : chosen)
        if (flat[index] == kMinusInfinity)
            index = sentinel;
    std::sort(chosen.begin(), chosen.end());

    std::vector<float> kept_embeddings, kept_probs;
    kept_embeddings.reserve(length * width);
    kept_probs.reserve(length * speakers);
    const std::vector<float> no_speech(speakers, 0.0F);
    for (const auto index : chosen) {
        const std::size_t frame = index == sentinel ? frames : std::min(index % scored, frames);
        const float* embedding =
            frame < frames ? embeddings.data() + frame * width : silence_.data();
        const float* prob = frame < frames ? probs.data() + frame * speakers : no_speech.data();
        kept_embeddings.insert(kept_embeddings.end(), embedding, embedding + width);
        kept_probs.insert(kept_probs.end(), prob, prob + speakers);
    }
    embeddings = std::move(kept_embeddings);
    probs = std::move(kept_probs);
    rows = static_cast<std::int64_t>(chosen.size());
}

std::int64_t SpeakerCache::popped_frames(std::int64_t fifo_rows) const noexcept {
    if (fifo_rows <= policy_.fifo_length)
        return 0;
    const std::int64_t popped =
        std::max<std::int64_t>(policy_.update_period, fifo_rows - policy_.fifo_length);
    return std::min(popped, fifo_rows);
}

void SpeakerCache::update(const float* step_input, std::int64_t rows, const float* logits,
                          const std::vector<std::uint8_t>& valid, std::int64_t chunk_frames,
                          std::int64_t lookback_frames) {
    const auto width = static_cast<std::int64_t>(policy_.hidden_size);
    const auto speakers = static_cast<std::int64_t>(policy_.num_speakers);
    const std::int64_t chunk_start = cache_rows_ + fifo_rows_ + lookback_frames;
    require(chunk_frames > 0 && lookback_frames >= 0 && chunk_start + chunk_frames <= rows,
            "the chunk does not fit the step");
    const auto probs = pool_probabilities(logits, rows, valid);

    std::vector<float> queue = fifo_;
    append_rows(queue, step_input + chunk_start * width, chunk_frames, width);
    const std::int64_t queued = fifo_rows_ + chunk_frames;
    const std::int64_t popped = popped_frames(queued);
    if (popped > 0) {
        // An uncompressed cache still holds plain earlier frames, whose
        // probabilities this step re-estimates; a compressed one is out of
        // order, so the probabilities stored with its frames are the only ones.
        std::vector<float> merged_probs;
        if (compressed_)
            merged_probs = cache_probs_;
        else
            merged_probs.assign(probs.begin(), probs.begin() + cache_rows_ * speakers);
        std::vector<float> queue_probs(probs.begin() + cache_rows_ * speakers,
                                       probs.begin() + (cache_rows_ + fifo_rows_) * speakers);
        append_rows(queue_probs, probs.data() + chunk_start * speakers, chunk_frames, speakers);

        std::vector<float> merged = cache_;
        append_rows(merged, queue.data(), popped, width);
        append_rows(merged_probs, queue_probs.data(), popped, speakers);
        std::int64_t merged_rows = cache_rows_ + popped;
        queue.erase(queue.begin(), queue.begin() + popped * width);
        if (merged_rows > policy_.cache_length) {
            compress(merged, merged_probs, merged_rows);
            compressed_ = true;
        }
        cache_ = std::move(merged);
        cache_probs_ = std::move(merged_probs);
        cache_rows_ = merged_rows;
    }
    fifo_ = std::move(queue);
    fifo_rows_ = queued - popped;
}

} // namespace trtmc::nemotron3_diarization
