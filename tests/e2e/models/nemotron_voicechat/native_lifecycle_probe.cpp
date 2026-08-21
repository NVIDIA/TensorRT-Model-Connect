/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Real-engine lifecycle probe for the public asynchronous speech-session API.
 * This is deliberately a standalone E2E executable: it links only trtmc_core,
 * loads the family-owned model plugin, and drives ISpeechSession without using
 * Python or any framework runtime.
 *
 * Example (from a configured source tree):
 *   c++ -std=c++17 -O2 -pthread -Iinclude -Isrc \
 *     tests/e2e/models/nemotron_voicechat/native_lifecycle_probe.cpp \
 *     -Lbuild-voicechat-pure -ltrtmc_core \
 *     -Wl,-rpath,$PWD/build-voicechat-pure \
 *     -o /tmp/voicechat_native_lifecycle_probe
 *   /tmp/voicechat_native_lifecycle_probe \
 *     MODEL.bundle sample_general.wav build-voicechat-pure \
 *     build-voicechat-pure/models/nemotron_voicechat \
 *     chunked.wav receipt.json
 */

#include "trtmc/pipeline.h"
#include "trtmc/speech_session.h"
#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using EventKind = trtmc::SpeechSessionEventKind;

constexpr int32_t kInputFrameSamples = 1280;
constexpr int32_t kOutputFrameSamples = 1764;
constexpr int32_t kExpectedOutputSamples = 345744;
constexpr int32_t kTailFrames = 3;
constexpr int32_t kPartialSamples = 317;
constexpr double kControlLatencyLimitMs = 500.0;
constexpr double kTailCompletionLimitMs = 15000.0;

constexpr const char* kExpectedAgentText =
    "Hi there! How can you? How can I help you today? The sky is blue. "
    "That blue color is because of something called Rayleigh scattering.";

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (ch < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(ch) << std::dec;
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    return out.str();
}

const char* json_bool(bool value) {
    return value ? "true" : "false";
}

std::string hex_u64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

std::string audio_fnv1a64(const std::vector<float>& samples) {
    constexpr std::uint64_t kOffset = UINT64_C(14695981039346656037);
    constexpr std::uint64_t kPrime = UINT64_C(1099511628211);
    std::uint64_t hash = kOffset;
    const auto* bytes = reinterpret_cast<const unsigned char*>(samples.data());
    const std::size_t count = samples.size() * sizeof(float);
    for (std::size_t index = 0; index < count; ++index) {
        hash ^= bytes[index];
        hash *= kPrime;
    }
    return hex_u64(hash);
}

bool bitwise_equal(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    return lhs.size() == rhs.size() &&
           (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0);
}

bool is_agent_payload(EventKind kind) {
    return kind == EventKind::kAgentAudio || kind == EventKind::kAgentText;
}

const char* event_kind_name(EventKind kind) {
    switch (kind) {
    case EventKind::kAgentAudio:
        return "agent_audio";
    case EventKind::kAgentText:
        return "agent_text";
    case EventKind::kUserTranscript:
        return "user_transcript";
    case EventKind::kTurnStarted:
        return "turn_started";
    case EventKind::kTurnFinished:
        return "turn_finished";
    case EventKind::kYielded:
        return "yielded";
    case EventKind::kCancelled:
        return "cancelled";
    case EventKind::kReset:
        return "reset";
    case EventKind::kError:
        return "error";
    case EventKind::kInputFinished:
        return "input_finished";
    }
    return "unknown";
}

struct ObservedEvent {
    EventKind kind{EventKind::kError};
    std::uint64_t epoch{0};
    std::uint64_t sequence{0};
    std::int64_t frame_index{-1};
    std::size_t audio_samples{0};
    bool is_final{false};
    std::string text;
};

struct TurnText {
    std::uint64_t epoch{0};
    std::string partial;
    std::string final;
    bool has_final{false};
};

struct Capture {
    std::vector<ObservedEvent> events;
    std::vector<float> audio;
    std::vector<TurnText> turns;

    void absorb(std::vector<trtmc::SpeechSessionEvent> ready) {
        for (auto& event : ready) {
            events.push_back({event.kind, event.epoch, event.sequence, event.frame_index,
                              event.audio_samples.size(), event.is_final, event.text});
            if (event.kind == EventKind::kError)
                throw std::runtime_error(event.text.empty() ? "native speech worker failed"
                                                            : event.text);
            if (event.kind == EventKind::kAgentAudio) {
                audio.insert(audio.end(), event.audio_samples.begin(), event.audio_samples.end());
            }
            if (event.kind == EventKind::kAgentText) {
                auto turn = std::find_if(turns.begin(), turns.end(), [&](const TurnText& value) {
                    return value.epoch == event.epoch;
                });
                if (turn == turns.end()) {
                    turns.push_back(TurnText{event.epoch});
                    turn = std::prev(turns.end());
                }
                if (event.is_final) {
                    turn->final = std::move(event.text);
                    turn->has_final = true;
                } else {
                    turn->partial += event.text;
                }
            }
        }
    }

    int count(EventKind kind) const {
        return static_cast<int>(std::count_if(
            events.begin(), events.end(), [&](const auto& event) { return event.kind == kind; }));
    }

    int count_with_text(EventKind kind, const std::string& text) const {
        return static_cast<int>(std::count_if(events.begin(), events.end(), [&](const auto& event) {
            return event.kind == kind && event.text == text;
        }));
    }

    bool has_agent_audio_for_epoch(std::uint64_t epoch) const {
        return std::any_of(events.begin(), events.end(), [&](const auto& event) {
            return event.kind == EventKind::kAgentAudio && event.epoch == epoch;
        });
    }

    bool has_partial_agent_text_for_epoch(std::uint64_t epoch) const {
        return std::any_of(events.begin(), events.end(), [&](const auto& event) {
            return event.kind == EventKind::kAgentText && event.epoch == epoch && !event.is_final &&
                   !event.text.empty();
        });
    }

    std::string agent_text() const {
        std::string text;
        for (const auto& turn : turns) {
            const auto& piece = turn.has_final ? turn.final : turn.partial;
            if (piece.empty())
                continue;
            if (!text.empty())
                text.push_back(' ');
            text += piece;
        }
        return text;
    }
};

void dump_event_trace(const char* label, const Capture& capture) {
    for (const auto& event : capture.events) {
        std::cerr << "[probe.event] phase=" << label << " kind=" << event_kind_name(event.kind)
                  << " epoch=" << event.epoch << " sequence=" << event.sequence
                  << " frame=" << event.frame_index << " audio_samples=" << event.audio_samples
                  << " final=" << json_bool(event.is_final) << " text=\"" << json_escape(event.text)
                  << "\"\n";
    }
}

void wait_until(trtmc::ISpeechSession& session, Capture& capture,
                const std::function<bool()>& predicate, int32_t timeout_ms,
                const char* description) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!predicate()) {
        const auto now = Clock::now();
        if (now >= deadline)
            throw std::runtime_error(std::string("timed out waiting for ") + description);
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const int32_t wait_ms =
            static_cast<int32_t>(std::max<std::int64_t>(1, std::min<std::int64_t>(500, remaining)));
        capture.absorb(session.wait_events(wait_ms));
    }
}

void finish_and_drain(trtmc::ISpeechSession& session, Capture& capture,
                      int32_t timeout_ms = 120000) {
    session.finish_input();
    wait_until(
        session, capture, [&] { return capture.count(EventKind::kInputFinished) != 0; }, timeout_ms,
        "kInputFinished");
    capture.absorb(session.take_events());
}

void pace_and_drain(trtmc::ISpeechSession& session, Capture& capture, int32_t pace_ms = 100) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(pace_ms);
    while (Clock::now() < deadline) {
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        capture.absorb(
            session.wait_events(static_cast<int32_t>(std::max<std::int64_t>(1, remaining))));
    }
    capture.absorb(session.take_events());
}

trtmc::SpeechSessionConfig base_config(int32_t input_sample_rate) {
    trtmc::SpeechSessionConfig config;
    config.input_sample_rate = input_sample_rate;
    config.output_sample_rate = 22050;
    config.emit_agent_audio = true;
    config.emit_agent_text = true;
    config.emit_user_transcript = true;
    config.enable_barge_in = false;
    config.seed = 0;
    config.finish_tail_frames = 0;
    return config;
}

std::uint64_t last_epoch(const Capture& capture, EventKind kind) {
    for (auto event = capture.events.rbegin(); event != capture.events.rend(); ++event) {
        if (event->kind == kind)
            return event->epoch;
    }
    return 0;
}

void write_failure_receipt(const std::string& path, const std::string& error) {
    std::ofstream out(path);
    if (out)
        out << "{\n  \"schema_version\": 2,\n  \"pass\": false,\n  \"error\": \""
            << json_escape(error) << "\"\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: native_lifecycle_probe BUNDLE WAV BACKEND_DIR "
                     "MODEL_PLUGIN_DIR OUTPUT_WAV RECEIPT_JSON\n";
        return 2;
    }

    const std::string bundle_path = argv[1];
    const std::string input_path = argv[2];
    const std::string backend_dir = argv[3];
    const std::string model_plugin_dir = argv[4];
    const std::string output_wav = argv[5];
    const std::string receipt_path = argv[6];

    try {
        trtmc::LoadOptions options;
        options.backend_search_paths.push_back(backend_dir);
        options.model_plugin_search_paths.push_back(model_plugin_dir);
        auto pipeline = trtmc::load(bundle_path, options);
        auto* session_provider = dynamic_cast<trtmc::ISpeechSessionProvider*>(pipeline.get());
        if (session_provider == nullptr)
            throw std::runtime_error("loaded pipeline does not expose ISpeechSession");

        const auto input = trtmc::io::read_wav(input_path);
        if (input.sample_rate != 16000 || input.num_samples != 249734)
            throw std::runtime_error("pinned model-card sample shape does not match contract");

        const auto config = base_config(input.sample_rate);

        Capture baseline;
        double baseline_append_ms = 0.0;
        {
            auto session = session_provider->create_speech_session(config);
            const auto append_start = Clock::now();
            session->append_audio(input.samples.data(), input.num_samples);
            baseline_append_ms = elapsed_ms(append_start, Clock::now());
            finish_and_drain(*session, baseline);
        }
        std::cerr << "[probe] baseline complete: audio_samples=" << baseline.audio.size()
                  << " audio_events=" << baseline.count(EventKind::kAgentAudio) << '\n';

        Capture irregular;
        int irregular_append_calls = 0;
        double irregular_max_append_ms = 0.0;
        int audio_events_before_finish = 0;
        {
            auto session = session_provider->create_speech_session(config);
            constexpr std::array<int32_t, 7> chunk_sizes = {257, 991, 64, 2048, 383, 1280, 17};
            std::size_t offset = 0;
            std::size_t chunk_index = 0;
            while (offset < input.samples.size()) {
                const int32_t count =
                    std::min<int32_t>(chunk_sizes[chunk_index++ % chunk_sizes.size()],
                                      static_cast<int32_t>(input.samples.size() - offset));
                const auto append_start = Clock::now();
                session->append_audio(input.samples.data() + offset, count);
                irregular_max_append_ms =
                    std::max(irregular_max_append_ms, elapsed_ms(append_start, Clock::now()));
                ++irregular_append_calls;
                offset += static_cast<std::size_t>(count);
                irregular.absorb(session->take_events());
            }
            if (irregular.count(EventKind::kAgentAudio) == 0) {
                wait_until(
                    *session, irregular,
                    [&] { return irregular.count(EventKind::kAgentAudio) != 0; }, 30000,
                    "pre-finish agent audio");
            }
            audio_events_before_finish = irregular.count(EventKind::kAgentAudio);
            finish_and_drain(*session, irregular);
        }
        std::cerr << "[probe] irregular chunking complete: audio_samples=" << irregular.audio.size()
                  << " pre_finish_audio_events=" << audio_events_before_finish << '\n';

        trtmc::AudioResult chunked_audio;
        chunked_audio.samples = irregular.audio;
        chunked_audio.num_samples = static_cast<int32_t>(chunked_audio.samples.size());
        chunked_audio.sample_rate = 22050;
        trtmc::io::write_wav(chunked_audio, output_wav);

        Capture barge_before;
        Capture barge_after;
        std::uint64_t interrupted_epoch = 0;
        std::uint64_t yielded_epoch = 0;
        std::uint64_t recovered_epoch = 0;
        int stale_agent_payloads = 0;
        int barge_frames_fed = 0;
        int interrupted_silence_frames = 0;
        int recovery_input_frames = 0;
        int recovery_silence_frames = 0;
        bool interrupted_audio_before_yield = false;
        bool interrupted_partial_text_before_yield = false;
        bool recovery_audio_before_finish = false;
        bool recovery_partial_text_before_finish = false;
        {
            auto barge_config = config;
            barge_config.enable_barge_in = true;
            auto session = session_provider->create_speech_session(barge_config);
            std::size_t offset = 0;
            while (offset + kInputFrameSamples <= input.samples.size() && interrupted_epoch == 0) {
                session->append_audio(input.samples.data() + offset, kInputFrameSamples);
                offset += kInputFrameSamples;
                ++barge_frames_fed;
                // Live sessions intentionally suppress idle/silence audio before
                // an agent turn. Pace at slightly slower than the model's 80 ms
                // frame clock and consume any event that becomes ready.
                pace_and_drain(*session, barge_before);
                interrupted_epoch = last_epoch(barge_before, EventKind::kTurnStarted);
            }
            if (interrupted_epoch == 0) {
                dump_event_trace("before_barge", barge_before);
                throw std::runtime_error("model did not start an agent turn for barge-in probe");
            }

            std::array<float, static_cast<std::size_t>(kInputFrameSamples)> silence{};
            while (interrupted_silence_frames < 64 &&
                   (!barge_before.has_agent_audio_for_epoch(interrupted_epoch) ||
                    !barge_before.has_partial_agent_text_for_epoch(interrupted_epoch))) {
                session->append_audio(silence.data(), static_cast<int32_t>(silence.size()));
                ++interrupted_silence_frames;
                pace_and_drain(*session, barge_before);
            }
            interrupted_audio_before_yield =
                barge_before.has_agent_audio_for_epoch(interrupted_epoch);
            interrupted_partial_text_before_yield =
                barge_before.has_partial_agent_text_for_epoch(interrupted_epoch);
            if (!interrupted_audio_before_yield || !interrupted_partial_text_before_yield) {
                dump_event_trace("before_barge", barge_before);
                throw std::runtime_error(
                    "agent did not publish audio and partial text before explicit barge-in");
            }

            std::array<float, static_cast<std::size_t>(kInputFrameSamples)> loud{};
            loud.fill(0.1F);
            session->append_audio(loud.data(), static_cast<int32_t>(loud.size()));
            barge_after.absorb(session->take_events());
            wait_until(
                *session, barge_after, [&] { return barge_after.count(EventKind::kYielded) != 0; },
                5000, "barge-in yield");
            yielded_epoch = last_epoch(barge_after, EventKind::kYielded);

            offset = 0;
            while (offset + kInputFrameSamples <= input.samples.size() && recovered_epoch == 0) {
                session->append_audio(input.samples.data() + offset, kInputFrameSamples);
                offset += kInputFrameSamples;
                ++recovery_input_frames;
                pace_and_drain(*session, barge_after);
                for (auto event = barge_after.events.rbegin(); event != barge_after.events.rend();
                     ++event) {
                    if (event->kind == EventKind::kTurnStarted && event->epoch > yielded_epoch) {
                        recovered_epoch = event->epoch;
                        break;
                    }
                }
            }
            if (recovered_epoch == 0) {
                dump_event_trace("after_barge", barge_after);
                throw std::runtime_error(
                    "same session did not start a recovered turn for the second utterance");
            }

            while (recovery_silence_frames < 64 &&
                   (!barge_after.has_agent_audio_for_epoch(recovered_epoch) ||
                    !barge_after.has_partial_agent_text_for_epoch(recovered_epoch))) {
                session->append_audio(silence.data(), static_cast<int32_t>(silence.size()));
                ++recovery_silence_frames;
                pace_and_drain(*session, barge_after);
            }
            recovery_audio_before_finish = barge_after.has_agent_audio_for_epoch(recovered_epoch);
            recovery_partial_text_before_finish =
                barge_after.has_partial_agent_text_for_epoch(recovered_epoch);
            if (!recovery_audio_before_finish || !recovery_partial_text_before_finish) {
                dump_event_trace("after_barge", barge_after);
                throw std::runtime_error(
                    "recovered turn did not publish audio and partial text before finish");
            }

            try {
                finish_and_drain(*session, barge_after, 30000);
            } catch (...) {
                dump_event_trace("after_barge", barge_after);
                throw;
            }

            stale_agent_payloads = static_cast<int>(std::count_if(
                barge_after.events.begin(), barge_after.events.end(), [&](const auto& event) {
                    return is_agent_payload(event.kind) && event.epoch == interrupted_epoch;
                }));
        }
        std::cerr << "[probe] barge-in complete: interrupted_epoch=" << interrupted_epoch
                  << " yielded_epoch=" << yielded_epoch << " recovered_epoch=" << recovered_epoch
                  << " stale_agent_payloads=" << stale_agent_payloads << '\n';
        dump_event_trace("barge_lifecycle", barge_after);

        Capture cancel_initial;
        Capture cancel_late;
        Capture reset_marker;
        Capture reset_prefix;
        double cancel_append_ms = 0.0;
        double cancel_call_ms = 0.0;
        bool append_after_cancel_rejected = false;
        {
            auto session = session_provider->create_speech_session(config);
            const auto append_start = Clock::now();
            session->append_audio(input.samples.data(), input.num_samples);
            cancel_append_ms = elapsed_ms(append_start, Clock::now());
            const auto cancel_start = Clock::now();
            session->cancel();
            cancel_call_ms = elapsed_ms(cancel_start, Clock::now());
            cancel_initial.absorb(session->take_events());

            try {
                const float zero = 0.0F;
                session->append_audio(&zero, 1);
            } catch (const std::logic_error&) {
                append_after_cancel_rejected = true;
            }

            std::this_thread::sleep_for(std::chrono::seconds(2));
            cancel_late.absorb(session->take_events());
            session->reset();
            reset_marker.absorb(session->take_events());

            const int32_t prefix_samples = 2 * kInputFrameSamples + kPartialSamples;
            session->append_audio(input.samples.data(), prefix_samples);
            finish_and_drain(*session, reset_prefix, 30000);
        }
        std::cerr << "[probe] cancel/reset complete: cancel_ms=" << cancel_call_ms
                  << " late_events=" << cancel_late.events.size() << '\n';

        Capture fresh_prefix;
        {
            auto session = session_provider->create_speech_session(config);
            const int32_t prefix_samples = 2 * kInputFrameSamples + kPartialSamples;
            session->append_audio(input.samples.data(), prefix_samples);
            finish_and_drain(*session, fresh_prefix, 30000);
        }
        std::cerr << "[probe] fresh deterministic prefix complete: audio_samples="
                  << fresh_prefix.audio.size() << '\n';

        Capture tail_before;
        Capture tail_after;
        std::uint64_t tail_turn_epoch = 0;
        int tail_frames_fed = 0;
        double tail_completion_ms = 0.0;
        {
            auto tail_config = config;
            tail_config.finish_tail_frames = kTailFrames;
            auto session = session_provider->create_speech_session(tail_config);
            std::size_t offset = 0;
            while (offset + kInputFrameSamples <= input.samples.size() && tail_turn_epoch == 0) {
                session->append_audio(input.samples.data() + offset, kInputFrameSamples);
                offset += kInputFrameSamples;
                ++tail_frames_fed;
                pace_and_drain(*session, tail_before);
                tail_turn_epoch = last_epoch(tail_before, EventKind::kTurnStarted);
            }
            if (tail_turn_epoch == 0)
                throw std::runtime_error("model did not start an agent turn for tail probe");
            wait_until(
                *session, tail_before,
                [&] { return tail_before.has_agent_audio_for_epoch(tail_turn_epoch); }, 30000,
                "agent audio before bounded tail");

            std::array<float, static_cast<std::size_t>(kPartialSamples)> partial{};
            session->append_audio(partial.data(), static_cast<int32_t>(partial.size()));
            const auto finish_start = Clock::now();
            finish_and_drain(*session, tail_after, 30000);
            tail_completion_ms = elapsed_ms(finish_start, Clock::now());
        }
        std::cerr << "[probe] partial finish/tail complete: post_finish_audio_events="
                  << tail_after.count(EventKind::kAgentAudio)
                  << " completion_ms=" << tail_completion_ms << '\n';

        const bool baseline_contract =
            static_cast<int32_t>(baseline.audio.size()) == kExpectedOutputSamples &&
            baseline.count(EventKind::kAgentAudio) ==
                kExpectedOutputSamples / kOutputFrameSamples &&
            baseline.agent_text() == kExpectedAgentText &&
            baseline.count(EventKind::kInputFinished) == 1;
        const bool irregular_parity = bitwise_equal(irregular.audio, baseline.audio) &&
                                      irregular.agent_text() == baseline.agent_text() &&
                                      audio_events_before_finish > 0 &&
                                      irregular.count(EventKind::kInputFinished) == 1 &&
                                      irregular_max_append_ms < kControlLatencyLimitMs;
        const bool barge_in_pass =
            interrupted_audio_before_yield && interrupted_partial_text_before_yield &&
            barge_after.count_with_text(EventKind::kYielded, "barge-in") == 1 &&
            yielded_epoch > interrupted_epoch && stale_agent_payloads == 0 &&
            recovered_epoch > yielded_epoch && recovery_audio_before_finish &&
            recovery_partial_text_before_finish &&
            barge_after.count(EventKind::kInputFinished) == 1;
        const bool cancel_pass = cancel_initial.count(EventKind::kCancelled) == 1 &&
                                 append_after_cancel_rejected && cancel_late.events.empty() &&
                                 cancel_append_ms < kControlLatencyLimitMs &&
                                 cancel_call_ms < kControlLatencyLimitMs;
        const bool reset_pass = reset_marker.count(EventKind::kReset) == 1 &&
                                bitwise_equal(reset_prefix.audio, fresh_prefix.audio) &&
                                reset_prefix.agent_text() == fresh_prefix.agent_text() &&
                                reset_prefix.audio.size() == 3U * kOutputFrameSamples &&
                                reset_prefix.count(EventKind::kInputFinished) == 1 &&
                                fresh_prefix.count(EventKind::kInputFinished) == 1;
        const int expected_tail_audio_events = 1 + kTailFrames;
        const bool tail_pass =
            tail_after.count(EventKind::kAgentAudio) == expected_tail_audio_events &&
            tail_after.audio.size() ==
                static_cast<std::size_t>(expected_tail_audio_events * kOutputFrameSamples) &&
            tail_after.count(EventKind::kInputFinished) == 1 &&
            tail_completion_ms < kTailCompletionLimitMs;
        const bool pass = baseline_contract && irregular_parity && barge_in_pass && cancel_pass &&
                          reset_pass && tail_pass;

        std::ofstream receipt(receipt_path);
        if (!receipt)
            throw std::runtime_error("cannot create receipt: " + receipt_path);
        receipt << std::fixed << std::setprecision(3);
        receipt << "{\n";
        receipt << "  \"schema_version\": 2,\n";
        receipt << "  \"pass\": " << json_bool(pass) << ",\n";
        receipt << "  \"runtime\": \"C++ ISpeechSession with TensorRT backend\",\n";
        receipt << "  \"baseline\": {\n";
        receipt << "    \"pass\": " << json_bool(baseline_contract) << ",\n";
        receipt << "    \"append_call_ms\": " << baseline_append_ms << ",\n";
        receipt << "    \"output_samples\": " << baseline.audio.size() << ",\n";
        receipt << "    \"audio_events\": " << baseline.count(EventKind::kAgentAudio) << ",\n";
        receipt << "    \"audio_fnv1a64\": \"" << audio_fnv1a64(baseline.audio) << "\",\n";
        receipt << "    \"agent_text\": \"" << json_escape(baseline.agent_text()) << "\"\n";
        receipt << "  },\n";
        receipt << "  \"irregular_chunking\": {\n";
        receipt << "    \"pass\": " << json_bool(irregular_parity) << ",\n";
        receipt << "    \"chunk_pattern\": [257, 991, 64, 2048, 383, 1280, 17],\n";
        receipt << "    \"append_calls\": " << irregular_append_calls << ",\n";
        receipt << "    \"max_append_call_ms\": " << irregular_max_append_ms << ",\n";
        receipt << "    \"audio_events_before_finish\": " << audio_events_before_finish << ",\n";
        receipt << "    \"output_samples\": " << irregular.audio.size() << ",\n";
        receipt << "    \"audio_fnv1a64\": \"" << audio_fnv1a64(irregular.audio) << "\",\n";
        receipt << "    \"bitwise_audio_equal_to_one_shot\": "
                << json_bool(bitwise_equal(irregular.audio, baseline.audio)) << ",\n";
        receipt << "    \"text_equal_to_one_shot\": "
                << json_bool(irregular.agent_text() == baseline.agent_text()) << "\n";
        receipt << "  },\n";
        receipt << "  \"barge_in\": {\n";
        receipt << "    \"pass\": " << json_bool(barge_in_pass) << ",\n";
        receipt << "    \"input_frames_before_turn\": " << barge_frames_fed << ",\n";
        receipt << "    \"interrupted_epoch\": " << interrupted_epoch << ",\n";
        receipt << "    \"interrupted_silence_frames\": " << interrupted_silence_frames << ",\n";
        receipt << "    \"interrupted_audio_before_yield\": "
                << json_bool(interrupted_audio_before_yield) << ",\n";
        receipt << "    \"interrupted_partial_text_before_yield\": "
                << json_bool(interrupted_partial_text_before_yield) << ",\n";
        receipt << "    \"yielded_epoch\": " << yielded_epoch << ",\n";
        receipt << "    \"yield_events_total\": " << barge_after.count(EventKind::kYielded)
                << ",\n";
        receipt << "    \"barge_in_yield_events\": "
                << barge_after.count_with_text(EventKind::kYielded, "barge-in") << ",\n";
        receipt << "    \"finish_bound_yield_events\": "
                << barge_after.count_with_text(EventKind::kYielded, "max-response-frames") << ",\n";
        receipt << "    \"stale_agent_payloads_after_yield\": " << stale_agent_payloads << ",\n";
        receipt << "    \"second_utterance_input_frames\": " << recovery_input_frames << ",\n";
        receipt << "    \"recovered_epoch\": " << recovered_epoch << ",\n";
        receipt << "    \"recovery_silence_frames\": " << recovery_silence_frames << ",\n";
        receipt << "    \"recovery_audio_before_finish\": "
                << json_bool(recovery_audio_before_finish) << ",\n";
        receipt << "    \"recovery_partial_text_before_finish\": "
                << json_bool(recovery_partial_text_before_finish) << ",\n";
        receipt << "    \"input_finished_events\": " << barge_after.count(EventKind::kInputFinished)
                << ",\n";
        receipt << "    \"event_trace\": [\n";
        for (std::size_t index = 0; index < barge_after.events.size(); ++index) {
            const auto& event = barge_after.events[index];
            receipt << "      {\"kind\": \"" << event_kind_name(event.kind)
                    << "\", \"epoch\": " << event.epoch << ", \"sequence\": " << event.sequence
                    << ", \"frame_index\": " << event.frame_index
                    << ", \"audio_samples\": " << event.audio_samples
                    << ", \"is_final\": " << json_bool(event.is_final) << ", \"text\": \""
                    << json_escape(event.text) << "\"}"
                    << (index + 1 == barge_after.events.size() ? "\n" : ",\n");
        }
        receipt << "    ]\n";
        receipt << "  },\n";
        receipt << "  \"cancel\": {\n";
        receipt << "    \"pass\": " << json_bool(cancel_pass) << ",\n";
        receipt << "    \"append_call_ms\": " << cancel_append_ms << ",\n";
        receipt << "    \"cancel_call_ms\": " << cancel_call_ms << ",\n";
        receipt << "    \"latency_limit_ms\": " << kControlLatencyLimitMs << ",\n";
        receipt << "    \"cancel_events\": " << cancel_initial.count(EventKind::kCancelled)
                << ",\n";
        receipt << "    \"append_after_cancel_rejected\": "
                << json_bool(append_after_cancel_rejected) << ",\n";
        receipt << "    \"observation_window_ms\": 2000,\n";
        receipt << "    \"late_events\": " << cancel_late.events.size() << "\n";
        receipt << "  },\n";
        receipt << "  \"reset_vs_fresh\": {\n";
        receipt << "    \"pass\": " << json_bool(reset_pass) << ",\n";
        receipt << "    \"input_prefix_samples\": " << (2 * kInputFrameSamples + kPartialSamples)
                << ",\n";
        receipt << "    \"reset_events\": " << reset_marker.count(EventKind::kReset) << ",\n";
        receipt << "    \"output_samples\": " << reset_prefix.audio.size() << ",\n";
        receipt << "    \"reset_audio_fnv1a64\": \"" << audio_fnv1a64(reset_prefix.audio)
                << "\",\n";
        receipt << "    \"fresh_audio_fnv1a64\": \"" << audio_fnv1a64(fresh_prefix.audio)
                << "\",\n";
        receipt << "    \"bitwise_audio_equal\": "
                << json_bool(bitwise_equal(reset_prefix.audio, fresh_prefix.audio)) << ",\n";
        receipt << "    \"text_equal\": "
                << json_bool(reset_prefix.agent_text() == fresh_prefix.agent_text()) << "\n";
        receipt << "  },\n";
        receipt << "  \"partial_finish_tail\": {\n";
        receipt << "    \"pass\": " << json_bool(tail_pass) << ",\n";
        receipt << "    \"input_frames_before_turn\": " << tail_frames_fed << ",\n";
        receipt << "    \"partial_input_samples\": " << kPartialSamples << ",\n";
        receipt << "    \"configured_tail_frames\": " << kTailFrames << ",\n";
        receipt << "    \"expected_audio_events_after_finish\": " << expected_tail_audio_events
                << ",\n";
        receipt << "    \"audio_events_after_finish\": " << tail_after.count(EventKind::kAgentAudio)
                << ",\n";
        receipt << "    \"output_samples_after_finish\": " << tail_after.audio.size() << ",\n";
        receipt << "    \"completion_ms\": " << tail_completion_ms << ",\n";
        receipt << "    \"completion_limit_ms\": " << kTailCompletionLimitMs << ",\n";
        receipt << "    \"input_finished_events\": " << tail_after.count(EventKind::kInputFinished)
                << "\n";
        receipt << "  }\n";
        receipt << "}\n";
        receipt.close();

        std::cout << "receipt=" << receipt_path << '\n';
        std::cout << "output_wav=" << output_wav << '\n';
        std::cout << "pass=" << json_bool(pass) << '\n';
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        write_failure_receipt(receipt_path, error.what());
        std::cerr << "native lifecycle probe failed: " << error.what() << '\n';
        return 1;
    }
}
