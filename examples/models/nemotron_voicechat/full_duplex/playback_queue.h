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
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::examples::voicechat {

inline float playback_gain_from_db(float gain_db) {
    if (!std::isfinite(gain_db))
        throw std::invalid_argument("playback gain must be finite");
    return std::pow(10.0F, gain_db / 20.0F);
}

inline std::int16_t float_to_pcm16(float sample, float linear_gain = 1.0F) noexcept {
    if (!std::isfinite(sample) || !std::isfinite(linear_gain))
        return 0;
    sample *= linear_gain;
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
    kTurnFinished,
    kFlush,
    kStopped,
};

struct PlaybackQueueItem {
    PlaybackQueueItemKind kind{PlaybackQueueItemKind::kStopped};
    std::uint64_t generation{0};
    std::uint64_t epoch{0};
    std::vector<std::int16_t> samples;
};

struct PlaybackRebufferNotice {
    std::uint64_t epoch{0};
    std::uint64_t underflow_count{0};
    std::size_t target_samples{0};
};

struct PlaybackLevelSummary {
    std::uint64_t epoch{0};
    std::size_t samples{0};
    float pre_gain_peak{0.0F};
    float pre_gain_rms{0.0F};
    std::size_t clipped_samples{0};
};

// Accumulates the exact levels seen by float_to_pcm16(). Keeping this separate
// from playback conditioning makes it possible to distinguish model-level
// changes from queue starvation and ALSA recovery without retaining audio.
class PlaybackLevelMeter {
  public:
    void observe(std::uint64_t epoch, const std::vector<float>& samples, float linear_gain) {
        if (epoch == 0 || samples.empty())
            return;
        if (epoch_ != epoch) {
            reset();
            epoch_ = epoch;
        }
        for (const float sample : samples) {
            if (!std::isfinite(sample) || !std::isfinite(linear_gain))
                continue;
            const float magnitude = std::abs(sample);
            peak_ = std::max(peak_, magnitude);
            sum_squares_ += static_cast<double>(sample) * sample;
            ++samples_;
            if (magnitude * std::abs(linear_gain) >= 1.0F)
                ++clipped_samples_;
        }
    }

    std::optional<PlaybackLevelSummary> finish(std::uint64_t epoch) {
        if (epoch == 0 || epoch != epoch_ || samples_ == 0)
            return std::nullopt;
        PlaybackLevelSummary summary;
        summary.epoch = epoch_;
        summary.samples = samples_;
        summary.pre_gain_peak = peak_;
        summary.pre_gain_rms =
            static_cast<float>(std::sqrt(sum_squares_ / static_cast<double>(samples_)));
        summary.clipped_samples = clipped_samples_;
        reset();
        return summary;
    }

    void reset() noexcept {
        epoch_ = 0;
        samples_ = 0;
        peak_ = 0.0F;
        sum_squares_ = 0.0;
        clipped_samples_ = 0;
    }

  private:
    std::uint64_t epoch_{0};
    std::size_t samples_{0};
    float peak_{0.0F};
    double sum_squares_{0.0};
    std::size_t clipped_samples_{0};
};

// A bounded hand-off between the session event consumer and the one thread
// that owns the ALSA playback handle. A flush changes the generation so the
// playback thread can abandon a chunk that it has already popped.
class PlaybackQueue {
  public:
    explicit PlaybackQueue(std::size_t capacity_samples, std::size_t prebuffer_samples = 0)
        : capacity_samples_(capacity_samples), prebuffer_samples_(prebuffer_samples) {
        if (capacity_samples_ == 0)
            throw std::invalid_argument("playback queue capacity must be positive");
        if (prebuffer_samples_ > capacity_samples_)
            throw std::invalid_argument("playback prebuffer exceeds queue capacity");
    }

    PlaybackQueue(const PlaybackQueue&) = delete;
    PlaybackQueue& operator=(const PlaybackQueue&) = delete;

    bool try_push(std::vector<std::int16_t> samples, std::uint64_t epoch = 1) {
        if (samples.empty())
            return true;
        if (epoch == 0)
            return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_ || samples.size() > capacity_samples_ - queued_samples_)
            return false;
        queued_samples_ += samples.size();
        queue_.push_back(PlaybackQueueItem{PlaybackQueueItemKind::kAudio, generation_, epoch,
                                           std::move(samples)});
        cv_.notify_one();
        return true;
    }

    bool finish_turn(std::uint64_t epoch) {
        if (epoch == 0)
            return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_)
            return false;
        queue_.push_back(
            PlaybackQueueItem{PlaybackQueueItemKind::kTurnFinished, generation_, epoch, {}});
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
        reset_playback_gate_locked();
        flush_pending_ = true;
        cv_.notify_all();
        return generation_;
    }

    PlaybackQueueItem wait_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            if (auto item = try_pop_locked())
                return std::move(*item);
            cv_.wait(lock);
        }
    }

    // Playback owns a continuously clocked ALSA stream, so an empty queue is
    // represented by digital silence rather than by blocking the device. Flush
    // and stop remain higher priority than audio, matching wait_pop().
    std::optional<PlaybackQueueItem> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        return try_pop_locked();
    }

    std::optional<PlaybackRebufferNotice> take_rebuffer_notice() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto notice = rebuffer_notice_;
        rebuffer_notice_.reset();
        return notice;
    }

    std::uint64_t underflow_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return underflow_count_;
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
        reset_playback_gate_locked();
        flush_pending_ = false;
        cv_.notify_all();
    }

    std::size_t queued_samples() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queued_samples_;
    }

  private:
    void reset_playback_gate_locked() {
        playback_epoch_ = 0;
        playback_streaming_ = false;
        rebuffer_notice_.reset();
    }

    void begin_epoch_locked(std::uint64_t epoch) {
        playback_epoch_ = epoch;
        playback_streaming_ = false;
    }

    void note_underflow_locked() {
        if (playback_epoch_ == 0 || !playback_streaming_)
            return;
        playback_streaming_ = false;
        ++underflow_count_;
        rebuffer_notice_ =
            PlaybackRebufferNotice{playback_epoch_, underflow_count_, prebuffer_samples_};
    }

    bool prebuffer_ready_locked() const {
        if (prebuffer_samples_ == 0)
            return true;
        std::size_t buffered = 0;
        for (const auto& item : queue_) {
            if (item.generation != generation_)
                continue;
            if (item.kind == PlaybackQueueItemKind::kTurnFinished) {
                if (item.epoch == playback_epoch_)
                    return true;
                continue;
            }
            if (item.kind != PlaybackQueueItemKind::kAudio || item.epoch != playback_epoch_)
                break;
            buffered += item.samples.size();
            if (buffered >= prebuffer_samples_)
                return true;
        }
        return false;
    }

    std::optional<PlaybackQueueItem> try_pop_locked() {
        if (flush_pending_) {
            flush_pending_ = false;
            return PlaybackQueueItem{PlaybackQueueItemKind::kFlush, generation_, 0, {}};
        }
        if (stopped_)
            return PlaybackQueueItem{PlaybackQueueItemKind::kStopped, generation_, 0, {}};

        while (!queue_.empty()) {
            const auto kind = queue_.front().kind;
            if (kind == PlaybackQueueItemKind::kTurnFinished) {
                const auto epoch = queue_.front().epoch;
                queue_.pop_front();
                if (epoch == playback_epoch_)
                    reset_playback_gate_locked();
                continue;
            }
            if (kind != PlaybackQueueItemKind::kAudio) {
                queue_.pop_front();
                continue;
            }

            const auto epoch = queue_.front().epoch;
            if (playback_epoch_ != epoch)
                begin_epoch_locked(epoch);
            if (!playback_streaming_) {
                if (!prebuffer_ready_locked())
                    return std::nullopt;
                playback_streaming_ = true;
            }

            PlaybackQueueItem item = std::move(queue_.front());
            queue_.pop_front();
            queued_samples_ -= item.samples.size();
            return item;
        }

        note_underflow_locked();
        return std::nullopt;
    }

    const std::size_t capacity_samples_;
    const std::size_t prebuffer_samples_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<PlaybackQueueItem> queue_;
    std::size_t queued_samples_{0};
    std::uint64_t generation_{1};
    std::uint64_t playback_epoch_{0};
    std::uint64_t underflow_count_{0};
    std::optional<PlaybackRebufferNotice> rebuffer_notice_;
    bool playback_streaming_{false};
    bool flush_pending_{false};
    bool stopped_{false};
};

} // namespace trtmc::examples::voicechat
