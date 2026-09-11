/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "playback_queue.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

namespace {

using trtmc::examples::voicechat::float_to_pcm16;
using trtmc::examples::voicechat::pcm16_to_float;
using trtmc::examples::voicechat::playback_gain_from_db;
using trtmc::examples::voicechat::PlaybackLevelMeter;
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

void test_playback_gain_conversion() {
    const float unity_gain = playback_gain_from_db(0.0F);
    check(std::abs(unity_gain - 1.0F) < 1.0e-6F, "zero dB maps to unity linear gain");
    check(float_to_pcm16(0.25F, unity_gain) == float_to_pcm16(0.25F),
          "explicit zero dB preserves default PCM conversion");

    const float boosted_gain = playback_gain_from_db(12.0F);
    check(std::abs(boosted_gain - 3.9810717F) < 1.0e-5F,
          "positive twelve dB maps to its linear amplitude gain");
    const auto expected_boosted =
        static_cast<std::int16_t>(std::lrint(0.125F * boosted_gain * 32767.0F));
    check(float_to_pcm16(0.125F, boosted_gain) == expected_boosted,
          "playback gain is applied before PCM quantization");
    check(float_to_pcm16(0.5F, boosted_gain) == std::numeric_limits<std::int16_t>::max(),
          "boosted positive playback saturates safely");
    check(float_to_pcm16(-0.5F, boosted_gain) == std::numeric_limits<std::int16_t>::min(),
          "boosted negative playback saturates safely");

    const float attenuated_gain = playback_gain_from_db(-24.0F);
    check(attenuated_gain > 0.0F && attenuated_gain < 0.064F,
          "negative twenty-four dB attenuates without changing polarity");
    check(float_to_pcm16(std::numeric_limits<float>::quiet_NaN(), boosted_gain) == 0,
          "gain-aware conversion keeps non-finite samples silent");
}

void test_bound_and_fifo() {
    PlaybackQueue queue(4);
    check(!queue.try_pop().has_value(), "nonblocking pop reports an empty queue immediately");
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

void test_initial_audio_waits_for_prebuffer() {
    PlaybackQueue queue(16, 4);
    check(queue.try_push({1, 2}, 7), "prebuffer accepts the first audio frame");
    check(!queue.try_pop().has_value(), "first audio frame waits below the prebuffer target");
    check(queue.underflow_count() == 0, "initial buffering is not an underflow");

    check(queue.try_push({3, 4}, 7), "prebuffer accepts the second audio frame");
    const auto first = queue.try_pop();
    const auto second = queue.try_pop();
    check(first.has_value() && first->samples == std::vector<std::int16_t>({1, 2}),
          "prebuffer releases the first frame at its target");
    check(second.has_value() && second->samples == std::vector<std::int16_t>({3, 4}),
          "prebuffer preserves the following frame");
}

void test_completed_short_turn_releases_tail() {
    PlaybackQueue queue(16, 4);
    check(queue.try_push({1, 2}, 8), "short turn audio is accepted");
    check(!queue.try_pop().has_value(), "short turn initially waits below target");
    check(queue.finish_turn(8), "short turn completion is accepted");

    const auto audio = queue.try_pop();
    check(audio.has_value() && audio->samples == std::vector<std::int16_t>({1, 2}),
          "completion releases a short turn without waiting forever");
    check(!queue.try_pop().has_value(), "completion marker is consumed internally");
    check(queue.underflow_count() == 0, "a completed short turn is not an underflow");
}

void test_mid_turn_starvation_rebuffers() {
    PlaybackQueue queue(16, 4);
    check(queue.try_push({1, 2}, 9), "first starvation fixture frame is accepted");
    check(queue.try_push({3, 4}, 9), "second starvation fixture frame is accepted");
    check(queue.try_pop().has_value(), "starvation fixture reaches its initial target");
    check(queue.try_pop().has_value(), "starvation fixture drains its initial buffer");
    check(!queue.try_pop().has_value(), "empty active turn enters rebuffering");

    const auto notice = queue.take_rebuffer_notice();
    check(notice.has_value() && notice->epoch == 9 && notice->underflow_count == 1 &&
              notice->target_samples == 4,
          "mid-turn starvation emits one bounded telemetry notice");
    check(queue.underflow_count() == 1, "mid-turn starvation increments its counter once");
    check(!queue.try_pop().has_value(), "repeated empty polls do not duplicate underflows");
    check(!queue.take_rebuffer_notice().has_value(), "underflow telemetry is edge-triggered");

    check(queue.try_push({5, 6}, 9), "starved turn accepts one replacement frame");
    check(!queue.try_pop().has_value(), "replacement audio waits below the rebuffer target");
    check(queue.try_push({7, 8}, 9), "starved turn accepts enough replacement audio");
    const auto resumed_first = queue.try_pop();
    const auto resumed_second = queue.try_pop();
    check(resumed_first.has_value() && resumed_first->samples == std::vector<std::int16_t>({5, 6}),
          "rebuffer resumes with the first held frame");
    check(resumed_second.has_value() &&
              resumed_second->samples == std::vector<std::int16_t>({7, 8}),
          "rebuffer resumes contiguously with the second held frame");
    check(queue.finish_turn(9), "resumed turn completion is accepted");
    check(!queue.try_pop().has_value(), "resumed turn consumes its completion marker");
    check(queue.underflow_count() == 1, "normal completion adds no underflow");
}

void test_flush_bypasses_prebuffer() {
    PlaybackQueue queue(16, 4);
    check(queue.try_push({1, 2}, 10), "flush fixture audio is accepted");
    check(!queue.try_pop().has_value(), "flush fixture waits in prebuffer");
    const auto generation = queue.request_flush();
    const auto flush = queue.try_pop();
    check(flush.has_value() && flush->kind == PlaybackQueueItemKind::kFlush &&
              flush->generation == generation,
          "flush remains immediate while audio is prebuffering");
    check(queue.queued_samples() == 0, "flush discards prebuffered audio");
}

void test_level_meter_reports_clipping_per_turn() {
    PlaybackLevelMeter meter;
    const float gain = playback_gain_from_db(6.0206F);
    meter.observe(11, {0.25F, -0.6F, std::numeric_limits<float>::quiet_NaN()}, gain);
    check(!meter.finish(12).has_value(), "level meter ignores another epoch's completion");
    const auto summary = meter.finish(11);
    check(summary.has_value() && summary->epoch == 11 && summary->samples == 2,
          "level meter reports the matching turn and finite sample count");
    check(summary.has_value() && std::abs(summary->pre_gain_peak - 0.6F) < 1.0e-6F,
          "level meter reports the pre-gain peak");
    const float expected_rms = std::sqrt((0.25F * 0.25F + 0.6F * 0.6F) / 2.0F);
    check(summary.has_value() && std::abs(summary->pre_gain_rms - expected_rms) < 1.0e-6F,
          "level meter reports the pre-gain RMS");
    check(summary.has_value() && summary->clipped_samples == 1,
          "level meter counts samples saturated by playback gain");
    check(!meter.finish(11).has_value(), "level meter clears a completed turn");
}

void test_nonblocking_pop_preserves_control_priority() {
    PlaybackQueue queue(8);
    check(queue.try_push({1, 2}), "nonblocking queue accepts audio");
    const auto next_generation = queue.request_flush();
    const auto flush = queue.try_pop();
    check(flush.has_value() && flush->kind == PlaybackQueueItemKind::kFlush &&
              flush->generation == next_generation,
          "nonblocking pop exposes flush before audio");
    check(!queue.try_pop().has_value(), "flush removes stale queued audio");

    check(queue.try_push({3, 4}), "queue accepts post-flush audio");
    const auto audio = queue.try_pop();
    check(audio.has_value() && audio->kind == PlaybackQueueItemKind::kAudio &&
              audio->samples == std::vector<std::int16_t>({3, 4}),
          "nonblocking pop returns replacement audio in FIFO order");
    queue.stop();
    const auto stopped = queue.try_pop();
    check(stopped.has_value() && stopped->kind == PlaybackQueueItemKind::kStopped,
          "nonblocking pop observes stop without waiting");
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
    test_playback_gain_conversion();
    test_bound_and_fifo();
    test_initial_audio_waits_for_prebuffer();
    test_completed_short_turn_releases_tail();
    test_mid_turn_starvation_rebuffers();
    test_flush_bypasses_prebuffer();
    test_level_meter_reports_clipping_per_turn();
    test_nonblocking_pop_preserves_control_priority();
    test_flush_invalidates_popped_and_pending_audio();
    test_wait_and_stop();
    return failures == 0 ? 0 : 1;
}
