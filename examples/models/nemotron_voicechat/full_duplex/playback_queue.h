/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::examples::voicechat {

inline std::int16_t float_to_pcm16(float sample) noexcept {
    if (!std::isfinite(sample))
        return 0;
    if (sample <= -1.0F)
        return std::numeric_limits<std::int16_t>::min();
    if (sample >= 1.0F)
        return std::numeric_limits<std::int16_t>::max();
    return static_cast<std::int16_t>(std::lrint(sample * 32767.0F));
}

inline float pcm16_to_float(std::int16_t sample) noexcept {
    return static_cast<float>(sample) / 32768.0F;
}

enum class PlaybackQueueItemKind {
    kAudio,
    kFlush,
    kStopped,
};

struct PlaybackQueueItem {
    PlaybackQueueItemKind kind{PlaybackQueueItemKind::kStopped};
    std::uint64_t generation{0};
    std::vector<std::int16_t> samples;
};

// A bounded hand-off between the session event consumer and the one thread
// that owns the ALSA playback handle. A flush changes the generation so the
// playback thread can abandon a chunk that it has already popped.
class PlaybackQueue {
  public:
    explicit PlaybackQueue(std::size_t capacity_samples) : capacity_samples_(capacity_samples) {
        if (capacity_samples_ == 0)
            throw std::invalid_argument("playback queue capacity must be positive");
    }

    PlaybackQueue(const PlaybackQueue&) = delete;
    PlaybackQueue& operator=(const PlaybackQueue&) = delete;

    bool try_push(std::vector<std::int16_t> samples) {
        if (samples.empty())
            return true;
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_ || samples.size() > capacity_samples_ - queued_samples_)
            return false;
        queued_samples_ += samples.size();
        queue_.push_back(
            PlaybackQueueItem{PlaybackQueueItemKind::kAudio, generation_, std::move(samples)});
        cv_.notify_one();
        return true;
    }

    std::uint64_t request_flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_)
            return generation_;
        if (++generation_ == 0)
            ++generation_;
        queue_.clear();
        queued_samples_ = 0;
        flush_pending_ = true;
        cv_.notify_all();
        return generation_;
    }

    PlaybackQueueItem wait_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stopped_ || flush_pending_ || !queue_.empty(); });
        if (flush_pending_) {
            flush_pending_ = false;
            return {PlaybackQueueItemKind::kFlush, generation_, {}};
        }
        if (stopped_)
            return {PlaybackQueueItemKind::kStopped, generation_, {}};
        PlaybackQueueItem item = std::move(queue_.front());
        queue_.pop_front();
        queued_samples_ -= item.samples.size();
        return item;
    }

    bool generation_is_current(std::uint64_t generation) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !stopped_ && generation == generation_;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_)
            return;
        stopped_ = true;
        if (++generation_ == 0)
            ++generation_;
        queue_.clear();
        queued_samples_ = 0;
        flush_pending_ = false;
        cv_.notify_all();
    }

    std::size_t queued_samples() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queued_samples_;
    }

  private:
    const std::size_t capacity_samples_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<PlaybackQueueItem> queue_;
    std::size_t queued_samples_{0};
    std::uint64_t generation_{1};
    bool flush_pending_{false};
    bool stopped_{false};
};

} // namespace trtmc::examples::voicechat
