/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "playback_queue.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

namespace {

using trtmc::examples::voicechat::float_to_pcm16;
using trtmc::examples::voicechat::pcm16_to_float;
using trtmc::examples::voicechat::PlaybackQueue;
using trtmc::examples::voicechat::PlaybackQueueItemKind;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void test_pcm_conversion() {
    check(float_to_pcm16(-2.0F) == std::numeric_limits<std::int16_t>::min(),
          "negative PCM conversion saturates");
    check(float_to_pcm16(2.0F) == std::numeric_limits<std::int16_t>::max(),
          "positive PCM conversion saturates");
    check(float_to_pcm16(std::numeric_limits<float>::quiet_NaN()) == 0,
          "non-finite PCM conversion is silent");
    check(pcm16_to_float(std::numeric_limits<std::int16_t>::min()) == -1.0F,
          "capture conversion preserves negative full scale");
}

void test_bound_and_fifo() {
    PlaybackQueue queue(4);
    check(queue.try_push({1, 2}), "first audio chunk is accepted");
    check(queue.try_push({3, 4}), "queue accepts samples up to its bound");
    check(!queue.try_push({5}), "queue rejects samples beyond its bound");
    check(queue.queued_samples() == 4, "queue accounts for pending samples");

    auto first = queue.wait_pop();
    auto second = queue.wait_pop();
    check(first.kind == PlaybackQueueItemKind::kAudio &&
              first.samples == std::vector<std::int16_t>({1, 2}),
          "queue preserves first audio chunk");
    check(second.kind == PlaybackQueueItemKind::kAudio &&
              second.samples == std::vector<std::int16_t>({3, 4}),
          "queue preserves FIFO order");
    check(queue.queued_samples() == 0, "pop releases queue capacity");
}

void test_flush_invalidates_popped_and_pending_audio() {
    PlaybackQueue queue(8);
    check(queue.try_push({1, 2}), "popped audio is accepted");
    auto popped = queue.wait_pop();
    check(queue.try_push({3, 4}), "pending stale audio is accepted");

    const auto next_generation = queue.request_flush();
    auto flush = queue.wait_pop();
    check(flush.kind == PlaybackQueueItemKind::kFlush && flush.generation == next_generation,
          "flush is observable by the playback owner");
    check(!queue.generation_is_current(popped.generation),
          "flush invalidates audio already owned by playback");
    check(queue.queued_samples() == 0, "flush discards pending audio");
    check(queue.try_push({9}), "replacement audio is accepted after flush");
    auto replacement = queue.wait_pop();
    check(replacement.kind == PlaybackQueueItemKind::kAudio &&
              replacement.generation == next_generation,
          "replacement audio uses the new generation");
}

void test_wait_and_stop() {
    PlaybackQueue queue(4);
    std::atomic<bool> waiting{false};
    PlaybackQueueItemKind observed = PlaybackQueueItemKind::kAudio;
    std::thread consumer([&] {
        waiting.store(true);
        observed = queue.wait_pop().kind;
    });
    while (!waiting.load())
        std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    queue.stop();
    consumer.join();
    check(observed == PlaybackQueueItemKind::kStopped, "stop wakes a blocked playback owner");
    check(!queue.try_push({1}), "stopped queue rejects new audio");
}

} // namespace

int main() {
    test_pcm_conversion();
    test_bound_and_fifo();
    test_flush_invalidates_popped_and_pending_audio();
    test_wait_and_stop();
    return failures == 0 ? 0 : 1;
}
