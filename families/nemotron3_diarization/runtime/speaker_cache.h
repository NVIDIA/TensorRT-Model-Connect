/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::nemotron3_diarization {

// The Arrival-Order Speaker Cache policy of the checkpoint's streaming config,
// with the FIFO sizes of offline mode. The frame budgets are integers derived
// at bundle time with the reference's own arithmetic.
struct SpeakerCachePolicy {
    std::int32_t num_speakers{8};
    std::int32_t subsampling_factor{8};
    std::int32_t hidden_size{512};
    std::int32_t cache_length{264};
    std::int32_t silence_frames_per_speaker{1};
    std::int32_t fifo_length{40};
    std::int32_t update_period{300};
    float score_threshold{0.25F};
    float latest_frames_boost{0.05F};
    std::int32_t min_positive_scores{16};
    std::int32_t strong_boosted_frames{24};
    std::int32_t weak_boosted_frames{48};
};

// Streaming state between encoder steps: the speaker cache and the FIFO queue
// of the most recent encoder *input* frames. Every step attends to these rows
// ahead of its chunk; after the step, the chunk joins the FIFO queue, and when
// the queue overflows its oldest frames move to the cache, which is compressed
// to the frames the step's own speaker scores rank highest, grouped by speaker
// in arrival order with learned silence frames between them.
class SpeakerCache {
  public:
    SpeakerCache(SpeakerCachePolicy policy, std::vector<float> silence_embedding);

    // Rows to put ahead of the next chunk: the cache frames, then the FIFO frames.
    std::int64_t cached_rows() const noexcept { return cache_rows_ + fifo_rows_; }
    void copy_cached_rows(float* destination) const;

    // step_input: [rows, hidden] that the encoder saw (cached rows, look-back
    // rows, the chunk, then its look-ahead). logits: [rows * factor, speakers].
    // valid: [rows], zero for padding frames, whose probabilities are zeroed.
    void update(const float* step_input, std::int64_t rows, const float* logits,
                const std::vector<std::uint8_t>& valid, std::int64_t chunk_frames,
                std::int64_t lookback_frames);

    bool compressed() const noexcept { return compressed_; }
    std::int64_t cache_rows() const noexcept { return cache_rows_; }
    std::int64_t fifo_rows() const noexcept { return fifo_rows_; }
    const std::vector<float>& cache_embeddings() const noexcept { return cache_; }
    const std::vector<float>& cache_probabilities() const noexcept { return cache_probs_; }

    // Exposed for the unit tests: speaker probabilities at the encoder frame
    // rate, the frame scores, and the compression of the policy.
    std::vector<float> pool_probabilities(const float* logits, std::int64_t rows,
                                          const std::vector<std::uint8_t>& valid) const;
    std::vector<float> frame_scores(const std::vector<float>& probs, std::int64_t rows) const;
    void compress(std::vector<float>& embeddings, std::vector<float>& probs,
                  std::int64_t& rows) const;

  private:
    std::int64_t popped_frames(std::int64_t fifo_rows) const noexcept;

    SpeakerCachePolicy policy_;
    std::vector<float> silence_;
    std::vector<float> cache_, cache_probs_, fifo_;
    std::int64_t cache_rows_{0}, fifo_rows_{0};
    bool compressed_{false};
};

// torch.topk(values, k, sorted=False) as CUDA resolves it: every value above
// the k-th largest, then values equal to it in index order (the first seen).
std::vector<std::size_t> top_k_first_seen(const std::vector<float>& values, std::size_t k);

} // namespace trtmc::nemotron3_diarization
