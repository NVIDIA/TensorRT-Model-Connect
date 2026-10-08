/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/stream.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace trtmc {

// Qwen's byte-level tokenizer may end a token inside a UTF-8 codepoint.
// Keep that suffix until another token arrives; replace malformed bytes only
// when their invalidity is known (or when generation has ended).
inline std::string qwen_utf8_text(std::string_view bytes, bool final) {
    std::string result;
    for (std::size_t i = 0; i < bytes.size();) {
        const auto lead = static_cast<unsigned char>(bytes[i]);
        const std::size_t length = lead < 0x80 ? 1 : lead >= 0xc2 && lead <= 0xdf ? 2
            : lead >= 0xe0 && lead <= 0xef ? 3 : lead >= 0xf0 && lead <= 0xf4 ? 4 : 0;
        bool valid = length != 0;
        const auto available = std::min(length, bytes.size() - i);
        for (std::size_t j = 1; j < available; ++j) {
            const auto byte = static_cast<unsigned char>(bytes[i + j]);
            valid = valid && byte >= 0x80 && byte <= 0xbf;
            if (j == 1)
                valid = valid && !(lead == 0xe0 && byte < 0xa0) &&
                    !(lead == 0xed && byte >= 0xa0) && !(lead == 0xf0 && byte < 0x90) &&
                    !(lead == 0xf4 && byte >= 0x90);
        }
        if (valid && available < length && !final)
            break;
        if (!valid || available < length) {
            result += "\xef\xbf\xbd";
            i += valid ? available : 1;
        } else {
            result.append(bytes.substr(i, length));
            i += length;
        }
    }
    return result;
}

// The producer owns Qwen generation. Eight queued events bound lookahead;
// cancellation wakes both readers and a producer blocked by a slow consumer.
class QwenTextStream final : public internal::ITextStream {
  public:
    using Emit = std::function<bool(internal::TextStreamEvent)>;
    using Produce = std::function<TextResult(const Emit&, const std::atomic<bool>&)>;
    explicit QwenTextStream(Produce produce) {
        producer_ = std::thread([this, produce = std::move(produce)] {
            try {
                auto result = produce([this](internal::TextStreamEvent event) {
                    return push(std::move(event));
                }, cancelled_);
                if (!cancelled_)
                    push({internal::StreamEventKind::Complete, {}, {}, std::move(result)});
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                failure_ = std::current_exception();
            }
            std::lock_guard<std::mutex> lock(mutex_);
            ended_ = true;
            cv_.notify_all();
        });
    }
    ~QwenTextStream() override {
        cancel();
        producer_.join();
    }
    std::optional<internal::TextStreamEvent> next(std::int64_t timeout_ms) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (terminal_)
            throw std::logic_error("Qwen stream terminal event already consumed");
        const auto available = [this] { return !events_.empty() || ended_; };
        if (timeout_ms < 0)
            cv_.wait(lock, available);
        else if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), available))
            return std::nullopt;
        if (cancelled_) {
            // Cancellation is confirmed only after the producer has stopped.
            if (!ended_)
                return std::nullopt;
            events_.clear();
            terminal_ = true;
            return internal::TextStreamEvent{internal::StreamEventKind::Cancelled, {}, {}, {}};
        }
        if (failure_)
            std::rethrow_exception(failure_);
        auto event = std::move(events_.front());
        events_.pop_front();
        terminal_ = event.kind != internal::StreamEventKind::Delta;
        cv_.notify_all();
        return event;
    }
    void cancel() noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        events_.clear();
        cv_.notify_all();
    }
  private:
    bool push(internal::TextStreamEvent event) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return cancelled_ || events_.size() < 8; });
        if (cancelled_)
            return false;
        events_.push_back(std::move(event));
        cv_.notify_all();
        return true;
    }
    std::atomic<bool> cancelled_{false};
    bool ended_{false}, terminal_{false};
    std::exception_ptr failure_;
    std::deque<internal::TextStreamEvent> events_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread producer_;
};
} // namespace trtmc
