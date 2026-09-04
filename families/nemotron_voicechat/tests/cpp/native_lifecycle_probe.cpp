/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Real-engine lifecycle probe for the public asynchronous speech-session API.
 * This is deliberately a standalone family E2E executable. It loads the
 * bundle through the public Task API and emits primitive lifecycle evidence.
 */

#include "cli/io.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
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
constexpr int32_t kRealtimePaceMs = 70;
// The pinned function sample's single spoken request occupies this range.
constexpr int32_t kBargeSpeechStartFrame = 47;
constexpr int32_t kBargeSpeechEndFrame = 70;
constexpr int32_t kLiveInputCapacitySeconds = 30;
constexpr int32_t kConcurrencyDeadlineMs = 120000;
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

std::string json_string_field(const std::string& value, const std::string& field) {
    const std::string marker = "\"" + field + "\":\"";
    const auto start = value.find(marker);
    if (start == std::string::npos)
        return {};
    const auto content = start + marker.size();
    const auto end = value.find('"', content);
    return end == std::string::npos ? std::string{} : value.substr(content, end - content);
}

const char* json_bool(bool value) {
    return value ? "true" : "false";
}

bool bitwise_equal(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    return lhs.size() == rhs.size() &&
           (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0);
}

bool is_agent_payload(EventKind kind) {
    return kind == EventKind::kAgentAudio || kind == EventKind::kAgentText ||
           kind == EventKind::kFunctionCall || kind == EventKind::kFunctionCallStarted ||
           kind == EventKind::kFunctionResponseFinished;
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
    case EventKind::kUserSpeechStarted:
        return "user_speech_started";
    case EventKind::kUserSpeechStopped:
        return "user_speech_stopped";
    case EventKind::kFunctionCall:
        return "function_call";
    case EventKind::kFunctionCallStarted:
        return "function_call_started";
    case EventKind::kFunctionResponseFinished:
        return "function_response_finished";
    case EventKind::kInputCleared:
        return "input_cleared";
    }
    return "unknown";
}

struct ObservedEvent {
    EventKind kind{EventKind::kError};
    std::uint64_t epoch{0};
    std::uint64_t sequence{0};
    std::int64_t frame_index{-1};
    std::int64_t media_start_sample{-1};
    std::int64_t media_end_sample{-1};
    int32_t sample_rate{0};
    std::size_t audio_samples{0};
    double audio_square_sum{0.0};
    double audio_peak{0.0};
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
            double square_sum = 0.0;
            double peak = 0.0;
            for (const float sample : event.audio_samples) {
                square_sum += static_cast<double>(sample) * sample;
                peak = std::max(peak, std::abs(static_cast<double>(sample)));
            }
            events.push_back({event.kind, event.epoch, event.sequence, event.frame_index,
                              event.media_start_sample, event.media_end_sample, event.sample_rate,
                              event.audio_samples.size(), square_sum, peak, event.is_final,
                              event.text});
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
                    turns.push_back(TurnText{event.epoch, {}, {}, false});
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

    int count_final(EventKind kind) const {
        return static_cast<int>(std::count_if(events.begin(), events.end(), [&](const auto& event) {
            return event.kind == kind && event.is_final && !event.text.empty();
        }));
    }

    int count_nonfinal(EventKind kind) const {
        return static_cast<int>(std::count_if(events.begin(), events.end(), [&](const auto& event) {
            return event.kind == kind && !event.is_final && !event.text.empty();
        }));
    }

    int distinct_epochs(EventKind kind) const {
        std::vector<std::uint64_t> epochs;
        for (const auto& event : events) {
            if (event.kind == kind)
                epochs.push_back(event.epoch);
        }
        std::sort(epochs.begin(), epochs.end());
        return static_cast<int>(std::unique(epochs.begin(), epochs.end()) - epochs.begin());
    }

    bool has_event_for_epoch(EventKind kind, std::uint64_t epoch,
                             bool require_final = false) const {
        return std::any_of(events.begin(), events.end(), [&](const auto& event) {
            return event.kind == kind && event.epoch == epoch &&
                   (!require_final || (event.is_final && !event.text.empty()));
        });
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

    int count_for_epoch(EventKind kind, std::uint64_t epoch) const {
        return static_cast<int>(std::count_if(events.begin(), events.end(), [&](const auto& event) {
            return event.kind == kind && event.epoch == epoch;
        }));
    }

    int count_agent_payloads_for_epoch(std::uint64_t epoch) const {
        return static_cast<int>(std::count_if(events.begin(), events.end(), [&](const auto& event) {
            return event.epoch == epoch && is_agent_payload(event.kind);
        }));
    }

    std::size_t audio_samples_for_epoch(std::uint64_t epoch) const {
        std::size_t samples = 0;
        for (const auto& event : events) {
            if (event.kind == EventKind::kAgentAudio && event.epoch == epoch)
                samples += event.audio_samples;
        }
        return samples;
    }

    double audio_rms_for_epoch(std::uint64_t epoch) const {
        double square_sum = 0.0;
        std::size_t samples = 0;
        for (const auto& event : events) {
            if (event.kind != EventKind::kAgentAudio || event.epoch != epoch)
                continue;
            square_sum += event.audio_square_sum;
            samples += event.audio_samples;
        }
        return samples == 0 ? 0.0 : std::sqrt(square_sum / static_cast<double>(samples));
    }

    double audio_peak_for_epoch(std::uint64_t epoch) const {
        double peak = 0.0;
        for (const auto& event : events) {
            if (event.kind == EventKind::kAgentAudio && event.epoch == epoch)
                peak = std::max(peak, event.audio_peak);
        }
        return peak;
    }

    std::string final_text_for_epoch(EventKind kind, std::uint64_t epoch) const {
        for (auto event = events.rbegin(); event != events.rend(); ++event) {
            if (event->kind == kind && event->epoch == epoch && event->is_final)
                return event->text;
        }
        return {};
    }

    std::string final_text(EventKind kind) const {
        for (auto event = events.rbegin(); event != events.rend(); ++event) {
            if (event->kind == kind && event->is_final)
                return event->text;
        }
        return {};
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

struct EventContractEvidence {
    bool sequence_monotonic{true};
    int sequence_violations{0};
    int sessions_checked{0};
    int events_checked{0};
    bool media_contiguous{true};
    int media_violations{0};
    int audio_events_checked{0};
    int media_segments_checked{0};
};

struct MediaSegment {
    std::int64_t first_sample{0};
    std::vector<const Capture*> captures;
};

EventContractEvidence
validate_event_contracts(const std::vector<std::vector<const Capture*>>& sessions,
                         const std::vector<MediaSegment>& media_segments) {
    EventContractEvidence evidence;
    evidence.sessions_checked = static_cast<int>(sessions.size());
    for (const auto& session : sessions) {
        std::map<std::uint64_t, std::uint64_t> last_sequence;
        for (const auto* capture : session) {
            if (capture == nullptr)
                continue;
            for (const auto& event : capture->events) {
                ++evidence.events_checked;
                const auto previous = last_sequence.find(event.epoch);
                if (previous != last_sequence.end() && event.sequence <= previous->second) {
                    evidence.sequence_monotonic = false;
                    ++evidence.sequence_violations;
                }
                last_sequence[event.epoch] = event.sequence;
            }
        }
    }

    evidence.media_segments_checked = static_cast<int>(media_segments.size());
    for (const auto& segment : media_segments) {
        std::int64_t next_media_sample = segment.first_sample;
        for (const auto* capture : segment.captures) {
            if (capture == nullptr)
                continue;
            for (const auto& event : capture->events) {
                if (event.kind != EventKind::kAgentAudio)
                    continue;
                ++evidence.audio_events_checked;
                const bool valid_range = event.sample_rate == 22050 &&
                                         event.media_start_sample == next_media_sample &&
                                         event.media_end_sample >= event.media_start_sample &&
                                         event.media_end_sample - event.media_start_sample ==
                                             static_cast<std::int64_t>(event.audio_samples);
                if (!valid_range) {
                    evidence.media_contiguous = false;
                    ++evidence.media_violations;
                }
                if (event.media_end_sample >= 0)
                    next_media_sample = event.media_end_sample;
            }
        }
    }
    return evidence;
}

struct NormalMultiturnEvidence {
    bool same_session{false};
    bool every_turn_completed{false};
    int turn_started_events{0};
    int turn_finished_events{0};
    int distinct_turn_epochs{0};
    int final_agent_text_events{0};
    int final_user_transcript_events{0};
    int yield_events{0};
    int reset_events{0};
    int input_finished_events{0};

    bool pass() const {
        // VoiceChat starts with an autonomous assistant greeting.  The two user
        // utterances below must therefore complete three distinct assistant turns
        // while producing exactly two final user transcripts.
        return same_session && every_turn_completed && turn_started_events == 3 &&
               turn_finished_events == 3 && distinct_turn_epochs == 3 &&
               final_agent_text_events == 3 && final_user_transcript_events == 2 &&
               yield_events == 0 && reset_events == 0 && input_finished_events == 1;
    }
};

NormalMultiturnEvidence summarize_normal_multiturn(const Capture& capture, bool same_session) {
    NormalMultiturnEvidence evidence;
    evidence.same_session = same_session;
    evidence.turn_started_events = capture.count(EventKind::kTurnStarted);
    evidence.turn_finished_events = capture.count(EventKind::kTurnFinished);
    evidence.distinct_turn_epochs = capture.distinct_epochs(EventKind::kTurnStarted);
    evidence.final_agent_text_events = capture.count_final(EventKind::kAgentText);
    evidence.final_user_transcript_events = capture.count_final(EventKind::kUserTranscript);
    evidence.yield_events = capture.count(EventKind::kYielded);
    evidence.reset_events = capture.count(EventKind::kReset);
    evidence.input_finished_events = capture.count(EventKind::kInputFinished);
    evidence.every_turn_completed = true;
    for (const auto& event : capture.events) {
        if (event.kind != EventKind::kTurnStarted)
            continue;
        if (!capture.has_event_for_epoch(EventKind::kTurnFinished, event.epoch) ||
            !capture.has_event_for_epoch(EventKind::kAgentText, event.epoch, true)) {
            evidence.every_turn_completed = false;
        }
    }
    return evidence;
}

struct BackpressureConcurrencyEvidence {
    bool producer_thread_completed{false};
    bool consumer_thread_completed{false};
    bool events_observed_while_producing{false};
    bool bounded_queue{false};
    bool overflow_error_observed{false};
    bool no_deadlock{false};
    int producer_append_calls{0};
    int finish_input_calls{0};
    int input_finished_events{0};
    std::size_t live_capacity_samples{0};
    std::size_t overflow_attempt_samples{0};
    double max_append_call_ms{0.0};
    double overflow_call_ms{0.0};

    bool pass() const {
        return producer_thread_completed && consumer_thread_completed &&
               events_observed_while_producing && bounded_queue && overflow_error_observed &&
               no_deadlock && producer_append_calls > 1 && finish_input_calls == 1 &&
               input_finished_events == 1 && overflow_attempt_samples > live_capacity_samples &&
               max_append_call_ms < kControlLatencyLimitMs &&
               overflow_call_ms < kControlLatencyLimitMs;
    }
};

struct FunctionChannelEvidence {
    bool implemented{false};
    bool expected_tool_name_match{false};
    bool tool_response_submitted{false};
    bool stale_response_rejected{false};
    int sotc_events{0};
    int eotc_events{0};
    int eotr_events{0};
    int completed_calls{0};
    int tool_response_injections{0};
    int agent_resumed_audio_events{0};
    int agent_resumed_text_events{0};
    int stale_function_payloads{0};
    std::string call_id;

    bool pass() const {
        return implemented && expected_tool_name_match && tool_response_submitted &&
               stale_response_rejected && sotc_events == 1 && eotc_events == 1 &&
               eotr_events == 1 && completed_calls == 1 && tool_response_injections == 1 &&
               agent_resumed_audio_events > 0 && agent_resumed_text_events > 0 &&
               stale_function_payloads == 0 && !call_id.empty();
    }
};

struct ProcessedInputClearEvidence {
    bool implemented{false};
    bool clear_succeeded{false};
    int processed_append_calls{0};
    std::size_t processed_input_samples{0};
    int transcript_delta_events_before_clear{0};
    double clear_call_ms{0.0};
    int clear_completion_events{0};
    std::size_t cleared_output_samples{0};
    std::size_t fresh_output_samples{0};
    double cleared_audio_rms{0.0};
    double fresh_audio_rms{0.0};
    double cleared_audio_peak{0.0};
    double fresh_audio_peak{0.0};
    bool bitwise_audio_equal{false};
    bool agent_text_equal{false};
    bool user_transcript_equal{false};
    int cleared_turn_finished_events{0};
    int fresh_turn_finished_events{0};
    int cleared_input_finished_events{0};
    int fresh_input_finished_events{0};

    bool pass() const {
        return implemented && clear_succeeded && processed_append_calls > 1 &&
               processed_input_samples >= 2U * kInputFrameSamples &&
               transcript_delta_events_before_clear > 0 && clear_call_ms < kControlLatencyLimitMs &&
               clear_completion_events == 1 && cleared_output_samples > 0 &&
               fresh_output_samples > 0 && cleared_output_samples == fresh_output_samples &&
               cleared_audio_rms >= 0.001 && fresh_audio_rms >= 0.001 &&
               cleared_audio_peak >= 0.01 && fresh_audio_peak >= 0.01 && bitwise_audio_equal &&
               agent_text_equal && user_transcript_equal && cleared_turn_finished_events == 1 &&
               fresh_turn_finished_events == 1 && cleared_input_finished_events == 1 &&
               fresh_input_finished_events == 1;
    }
};

struct ResponseRecoveryEvidence {
    bool implemented{false};
    bool commit_without_response{false};
    std::uint64_t interrupted_epoch{0};
    std::uint64_t yielded_epoch{0};
    std::uint64_t replacement_epoch{0};
    int old_audio_events_before_control{0};
    int old_partial_text_events_before_control{0};
    int control_yield_events{0};
    double control_call_ms{0.0};
    std::int64_t played_output_samples{0};
    std::int64_t observed_output_span_samples{0};
    std::int64_t generated_output_samples{0};
    std::int64_t retained_output_samples{0};
    std::int64_t discarded_output_samples{0};
    int stale_agent_payloads_after_control{0};
    int replacement_audio_events{0};
    std::size_t replacement_audio_samples{0};
    double replacement_audio_rms{0.0};
    double replacement_audio_peak{0.0};
    int replacement_final_text_events{0};
    std::string replacement_final_text;
    int replacement_turn_finished_events{0};
    int input_finished_events{0};

    bool pass(bool truncate) const {
        const bool generated_boundary = generated_output_samples == observed_output_span_samples;
        const bool boundary =
            truncate
                ? (played_output_samples > 0 && played_output_samples < generated_output_samples &&
                   played_output_samples % kOutputFrameSamples == 0 &&
                   retained_output_samples == played_output_samples &&
                   discarded_output_samples == generated_output_samples - retained_output_samples &&
                   discarded_output_samples >= kOutputFrameSamples)
                : (played_output_samples == 0 &&
                   retained_output_samples == generated_output_samples &&
                   discarded_output_samples == 0);
        const int minimum_old_audio_events = truncate ? 2 : 1;
        return implemented && commit_without_response && interrupted_epoch > 0 &&
               yielded_epoch > interrupted_epoch && replacement_epoch > yielded_epoch &&
               old_audio_events_before_control >= minimum_old_audio_events &&
               old_partial_text_events_before_control > 0 && control_yield_events == 1 &&
               control_call_ms < kControlLatencyLimitMs && generated_boundary && boundary &&
               stale_agent_payloads_after_control == 0 && replacement_audio_events > 0 &&
               replacement_audio_samples > 0 && replacement_audio_rms >= 0.001 &&
               replacement_audio_peak >= 0.01 && replacement_final_text_events == 1 &&
               !replacement_final_text.empty() && replacement_turn_finished_events == 1 &&
               input_finished_events == 1;
    }
};

bool required_lifecycle_contract(bool baseline, bool irregular, bool barge_in, bool cancel,
                                 bool reset, bool tail, const EventContractEvidence& events,
                                 bool normal_multiturn, bool function_channel,
                                 bool backpressure_concurrency, bool processed_input_clear,
                                 bool response_cancel_recovery, bool response_truncate_recovery) {
    return baseline && irregular && barge_in && cancel && reset && tail &&
           events.sequence_monotonic && events.media_contiguous && normal_multiturn &&
           function_channel && backpressure_concurrency && processed_input_clear &&
           response_cancel_recovery && response_truncate_recovery;
}

int run_host_self_check() {
    const auto make_event = [](EventKind kind, std::uint64_t epoch, std::uint64_t sequence,
                               const char* text = nullptr, bool is_final = false) {
        trtmc::SpeechSessionEvent event;
        event.kind = kind;
        event.epoch = epoch;
        event.sequence = sequence;
        event.text = text == nullptr ? "" : text;
        event.is_final = is_final;
        return event;
    };
    const auto make_audio_event = [&](std::uint64_t epoch, std::uint64_t sequence,
                                      std::int64_t start, std::int64_t end) {
        auto event = make_event(EventKind::kAgentAudio, epoch, sequence);
        event.audio_samples.resize(static_cast<std::size_t>(end - start));
        event.sample_rate = 22050;
        event.media_start_sample = start;
        event.media_end_sample = end;
        return event;
    };

    Capture valid;
    valid.absorb({
        make_event(EventKind::kTurnStarted, 2, 0),
        make_audio_event(2, 1, 0, 2),
        make_event(EventKind::kAgentText, 2, 2, "ok"),
        make_audio_event(2, 3, 2, 3),
    });
    const auto valid_evidence = validate_event_contracts({{&valid}}, {{0, {&valid}}});
    if (!valid_evidence.sequence_monotonic || !valid_evidence.media_contiguous ||
        valid_evidence.events_checked != 4 || valid_evidence.audio_events_checked != 2 ||
        valid_evidence.media_segments_checked != 1)
        throw std::runtime_error("valid event trace did not satisfy the host contract");

    Capture invalid;
    invalid.absorb({
        make_audio_event(3, 0, 0, 1),
        make_audio_event(3, 0, 7, 8),
    });
    const auto invalid_evidence = validate_event_contracts({{&invalid}}, {{0, {&invalid}}});
    if (invalid_evidence.sequence_monotonic || invalid_evidence.media_contiguous ||
        invalid_evidence.sequence_violations != 1 || invalid_evidence.media_violations != 1)
        throw std::runtime_error("invalid event trace was not rejected by the host contract");

    Capture rewind_before;
    rewind_before.absorb({make_audio_event(4, 0, 0, 2)});
    Capture rewind_after;
    rewind_after.absorb({make_audio_event(6, 0, 1, 3)});
    const auto segmented_rewind = validate_event_contracts(
        {{&rewind_before, &rewind_after}}, {{0, {&rewind_before}}, {1, {&rewind_after}}});
    if (!segmented_rewind.sequence_monotonic || !segmented_rewind.media_contiguous ||
        segmented_rewind.media_segments_checked != 2)
        throw std::runtime_error("segmented response rewind did not satisfy the host contract");
    const auto unsegmented_rewind = validate_event_contracts(
        {{&rewind_before, &rewind_after}}, {{0, {&rewind_before, &rewind_after}}});
    if (unsegmented_rewind.media_contiguous)
        throw std::runtime_error("unsegmented response rewind did not fail closed");

    Capture multiturn;
    multiturn.absorb({
        make_event(EventKind::kTurnStarted, 1, 0),
        make_event(EventKind::kAgentText, 1, 1, "initial greeting", true),
        make_event(EventKind::kTurnFinished, 1, 2),
        make_event(EventKind::kUserTranscript, 2, 0, "first user", true),
        make_event(EventKind::kTurnStarted, 3, 0),
        make_event(EventKind::kAgentText, 3, 1, "first agent", true),
        make_event(EventKind::kTurnFinished, 3, 2),
        make_event(EventKind::kUserTranscript, 4, 0, "second user", true),
        make_event(EventKind::kTurnStarted, 5, 0),
        make_event(EventKind::kAgentText, 5, 1, "second agent", true),
        make_event(EventKind::kTurnFinished, 5, 2),
        make_event(EventKind::kInputFinished, 6, 0),
    });
    if (!summarize_normal_multiturn(multiturn, true).pass() ||
        summarize_normal_multiturn(multiturn, false).pass()) {
        throw std::runtime_error("normal multi-turn host contract did not fail closed");
    }

    BackpressureConcurrencyEvidence backpressure;
    backpressure.producer_thread_completed = backpressure.consumer_thread_completed = true;
    backpressure.events_observed_while_producing = backpressure.bounded_queue = true;
    backpressure.overflow_error_observed = backpressure.no_deadlock = true;
    backpressure.producer_append_calls = 3;
    backpressure.finish_input_calls = 1;
    backpressure.input_finished_events = 1;
    backpressure.live_capacity_samples = 480000;
    backpressure.overflow_attempt_samples = 480001;
    backpressure.max_append_call_ms = 1.0;
    backpressure.overflow_call_ms = 1.0;
    if (!backpressure.pass())
        throw std::runtime_error("complete backpressure host evidence did not pass");
    backpressure.overflow_error_observed = false;
    if (backpressure.pass())
        throw std::runtime_error("missing overflow evidence did not fail closed");

    ProcessedInputClearEvidence clear;
    clear.implemented = clear.clear_succeeded = true;
    clear.processed_append_calls = 2;
    clear.processed_input_samples = 2U * kInputFrameSamples;
    clear.transcript_delta_events_before_clear = 1;
    clear.clear_call_ms = 1.0;
    clear.clear_completion_events = 1;
    clear.cleared_output_samples = clear.fresh_output_samples = kOutputFrameSamples;
    clear.cleared_audio_rms = clear.fresh_audio_rms = 0.01;
    clear.cleared_audio_peak = clear.fresh_audio_peak = 0.1;
    clear.bitwise_audio_equal = clear.agent_text_equal = clear.user_transcript_equal = true;
    clear.cleared_turn_finished_events = clear.fresh_turn_finished_events = 1;
    clear.cleared_input_finished_events = clear.fresh_input_finished_events = 1;
    if (!clear.pass())
        throw std::runtime_error("complete processed-input clear evidence did not pass");
    clear.clear_completion_events = 0;
    if (clear.pass())
        throw std::runtime_error("missing input-clear completion did not fail closed");
    clear.clear_completion_events = 1;
    clear.cleared_audio_rms = 0.0;
    if (clear.pass())
        throw std::runtime_error("silent post-clear response did not fail closed");
    clear.cleared_audio_rms = 0.01;
    clear.fresh_turn_finished_events = 0;
    if (clear.pass())
        throw std::runtime_error("missing fresh terminal did not fail closed");
    clear.fresh_turn_finished_events = 1;
    clear.user_transcript_equal = false;
    if (clear.pass())
        throw std::runtime_error("contaminated post-clear transcript did not fail closed");

    ResponseRecoveryEvidence recovery;
    recovery.implemented = recovery.commit_without_response = true;
    recovery.interrupted_epoch = 2;
    recovery.yielded_epoch = 3;
    recovery.replacement_epoch = 4;
    recovery.old_audio_events_before_control = 1;
    recovery.old_partial_text_events_before_control = 1;
    recovery.control_yield_events = 1;
    recovery.control_call_ms = 1.0;
    recovery.observed_output_span_samples = kOutputFrameSamples;
    recovery.generated_output_samples = kOutputFrameSamples;
    recovery.retained_output_samples = kOutputFrameSamples;
    recovery.replacement_audio_events = 1;
    recovery.replacement_audio_samples = kOutputFrameSamples;
    recovery.replacement_audio_rms = 0.01;
    recovery.replacement_audio_peak = 0.1;
    recovery.replacement_final_text_events = 1;
    recovery.replacement_final_text = "complete";
    recovery.replacement_turn_finished_events = 1;
    recovery.input_finished_events = 1;
    if (!recovery.pass(false))
        throw std::runtime_error("complete cancel-recovery evidence did not pass");
    recovery.retained_output_samples = 0;
    if (recovery.pass(false))
        throw std::runtime_error("lost cancelled-response boundary did not fail closed");
    recovery.generated_output_samples = 2 * kOutputFrameSamples;
    recovery.observed_output_span_samples = recovery.generated_output_samples;
    recovery.old_audio_events_before_control = 2;
    recovery.played_output_samples = kOutputFrameSamples;
    recovery.retained_output_samples = recovery.played_output_samples;
    recovery.discarded_output_samples = kOutputFrameSamples;
    if (!recovery.pass(true))
        throw std::runtime_error("complete response-recovery evidence did not pass");
    recovery.played_output_samples = 0;
    recovery.stale_agent_payloads_after_control = 1;
    if (recovery.pass(false))
        throw std::runtime_error("stale cancelled-response payload did not fail closed");
    recovery.stale_agent_payloads_after_control = 0;
    recovery.played_output_samples = kOutputFrameSamples / 2;
    if (recovery.pass(true))
        throw std::runtime_error("wrong truncate playback boundary did not fail closed");
    recovery.played_output_samples = kOutputFrameSamples;
    recovery.discarded_output_samples = 0;
    if (recovery.pass(true))
        throw std::runtime_error("zero discarded response audio did not fail closed");
    recovery.discarded_output_samples = kOutputFrameSamples;
    recovery.replacement_turn_finished_events = 0;
    if (recovery.pass(true))
        throw std::runtime_error("missing truncated-response terminal did not fail closed");

    if (required_lifecycle_contract(true, true, true, true, true, true, valid_evidence, false, true,
                                    true, true, true, true))
        throw std::runtime_error("missing future lifecycle evidence did not fail closed");
    if (!required_lifecycle_contract(true, true, true, true, true, true, valid_evidence, true, true,
                                     true, true, true, true))
        throw std::runtime_error("complete lifecycle evidence did not pass the host contract");

    std::cout << "native lifecycle probe host self-check passed\n";
    return 0;
}

void dump_event_trace(const char* label, const Capture& capture) {
    for (const auto& event : capture.events) {
        std::cerr << "[probe.event] phase=" << label << " kind=" << event_kind_name(event.kind)
                  << " epoch=" << event.epoch << " sequence=" << event.sequence
                  << " frame=" << event.frame_index << " media_start=" << event.media_start_sample
                  << " media_end=" << event.media_end_sample << " sample_rate=" << event.sample_rate
                  << " audio_samples=" << event.audio_samples
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

bool wait_until_bounded(trtmc::ISpeechSession& session, Capture& capture,
                        const std::function<bool()>& predicate, int32_t timeout_ms) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!predicate()) {
        const auto now = Clock::now();
        if (now >= deadline)
            return false;
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const int32_t wait_ms =
            static_cast<int32_t>(std::max<std::int64_t>(1, std::min<std::int64_t>(500, remaining)));
        capture.absorb(session.wait_events(wait_ms));
    }
    return true;
}

void finish_and_drain(trtmc::ISpeechSession& session, Capture& capture,
                      int32_t timeout_ms = 120000) {
    session.finish_input();
    wait_until(
        session, capture, [&] { return capture.count(EventKind::kInputFinished) != 0; }, timeout_ms,
        "kInputFinished");
    capture.absorb(session.take_events());
}

void pace_and_drain(trtmc::ISpeechSession& session, Capture& capture, int32_t pace_ms = 40) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(pace_ms);
    while (Clock::now() < deadline) {
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        capture.absorb(
            session.wait_events(static_cast<int32_t>(std::max<std::int64_t>(1, remaining))));
    }
    capture.absorb(session.take_events());
}

void feed_realtime(trtmc::ISpeechSession& session, Capture& capture,
                   const std::vector<float>& samples) {
    std::size_t offset = 0;
    while (offset < samples.size()) {
        const int32_t count =
            std::min<int32_t>(kInputFrameSamples, static_cast<int32_t>(samples.size() - offset));
        session.append_audio(samples.data() + offset, count);
        offset += static_cast<std::size_t>(count);
        pace_and_drain(session, capture, kRealtimePaceMs);
    }
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

std::int64_t audio_span_for_epoch(const Capture& capture, std::uint64_t epoch) {
    std::int64_t start = -1;
    std::int64_t end = -1;
    for (const auto& event : capture.events) {
        if (event.kind != EventKind::kAgentAudio || event.epoch != epoch)
            continue;
        if (start < 0 || event.media_start_sample < start)
            start = event.media_start_sample;
        end = std::max(end, event.media_end_sample);
    }
    return start >= 0 && end > start ? end - start : 0;
}

struct ResponseRecoveryRun {
    ResponseRecoveryEvidence evidence;
    Capture before_control;
    Capture after_control;
    Capture replacement;
};

ResponseRecoveryRun run_response_recovery(trtmc::ISpeechSessionProvider& provider,
                                          const trtmc::SpeechSessionConfig& base,
                                          const std::vector<float>& speech, bool truncate) {
    ResponseRecoveryRun run;
    auto config = base;
    config.enable_barge_in = true;
    config.finish_tail_frames = 0;
    auto session = provider.create_speech_session(config);
    auto* control = dynamic_cast<trtmc::ISpeechRealtimeControl*>(session.get());
    if (control == nullptr)
        throw std::runtime_error("live response-recovery session lacks realtime controls");

    feed_realtime(*session, run.before_control, speech);
    control->commit_input_turn(false);
    run.before_control.absorb(session->take_events());
    run.evidence.commit_without_response = run.before_control.count(EventKind::kTurnStarted) == 0;
    control->create_response();
    run.before_control.absorb(session->take_events());
    wait_until(
        *session, run.before_control,
        [&] {
            const auto epoch = last_epoch(run.before_control, EventKind::kTurnStarted);
            const int minimum_audio_events = truncate ? 2 : 1;
            return epoch > 0 &&
                   run.before_control.count_for_epoch(EventKind::kAgentAudio, epoch) >=
                       minimum_audio_events &&
                   run.before_control.has_partial_agent_text_for_epoch(epoch);
        },
        120000, "response audio and partial text before local control");

    run.evidence.interrupted_epoch = last_epoch(run.before_control, EventKind::kTurnStarted);
    run.evidence.old_audio_events_before_control =
        run.before_control.count_for_epoch(EventKind::kAgentAudio, run.evidence.interrupted_epoch);
    run.evidence.old_partial_text_events_before_control = static_cast<int>(std::count_if(
        run.before_control.events.begin(), run.before_control.events.end(), [&](const auto& event) {
            return event.kind == EventKind::kAgentText &&
                   event.epoch == run.evidence.interrupted_epoch && !event.is_final &&
                   !event.text.empty();
        }));
    run.evidence.observed_output_span_samples =
        audio_span_for_epoch(run.before_control, run.evidence.interrupted_epoch);
    run.evidence.generated_output_samples = run.evidence.observed_output_span_samples;
    run.evidence.played_output_samples =
        truncate ? run.evidence.generated_output_samples - kOutputFrameSamples : 0;
    run.evidence.retained_output_samples =
        truncate ? run.evidence.played_output_samples : run.evidence.generated_output_samples;
    run.evidence.discarded_output_samples =
        run.evidence.generated_output_samples - run.evidence.retained_output_samples;

    const auto control_start = Clock::now();
    if (truncate) {
        control->truncate_response(run.evidence.interrupted_epoch,
                                   run.evidence.played_output_samples);
    } else {
        control->cancel_response();
    }
    run.evidence.control_call_ms = elapsed_ms(control_start, Clock::now());
    run.after_control.absorb(session->take_events());
    const char* reason = truncate ? "response-truncate" : "response-cancel";
    wait_until(
        *session, run.after_control,
        [&] { return run.after_control.count_with_text(EventKind::kYielded, reason) != 0; }, 120000,
        truncate ? "response-truncate completion" : "response-cancel completion");
    run.evidence.control_yield_events =
        run.after_control.count_with_text(EventKind::kYielded, reason);
    run.evidence.yielded_epoch = last_epoch(run.after_control, EventKind::kYielded);

    control->create_response();
    run.replacement.absorb(session->take_events());
    wait_until(
        *session, run.replacement,
        [&] {
            const auto epoch = last_epoch(run.replacement, EventKind::kTurnStarted);
            return epoch > run.evidence.yielded_epoch &&
                   run.replacement.has_agent_audio_for_epoch(epoch) &&
                   run.replacement.has_event_for_epoch(EventKind::kAgentText, epoch, true) &&
                   run.replacement.has_event_for_epoch(EventKind::kTurnFinished, epoch);
        },
        120000, "completed replacement response after local control");
    run.evidence.replacement_epoch = last_epoch(run.replacement, EventKind::kTurnStarted);
    finish_and_drain(*session, run.replacement, 30000);

    run.evidence.stale_agent_payloads_after_control =
        run.after_control.count_agent_payloads_for_epoch(run.evidence.interrupted_epoch) +
        run.replacement.count_agent_payloads_for_epoch(run.evidence.interrupted_epoch);
    run.evidence.replacement_audio_events =
        run.replacement.count_for_epoch(EventKind::kAgentAudio, run.evidence.replacement_epoch);
    run.evidence.replacement_audio_samples =
        run.replacement.audio_samples_for_epoch(run.evidence.replacement_epoch);
    run.evidence.replacement_audio_rms =
        run.replacement.audio_rms_for_epoch(run.evidence.replacement_epoch);
    run.evidence.replacement_audio_peak =
        run.replacement.audio_peak_for_epoch(run.evidence.replacement_epoch);
    run.evidence.replacement_final_text_events = static_cast<int>(std::count_if(
        run.replacement.events.begin(), run.replacement.events.end(), [&](const auto& event) {
            return event.kind == EventKind::kAgentText &&
                   event.epoch == run.evidence.replacement_epoch && event.is_final &&
                   !event.text.empty();
        }));
    run.evidence.replacement_final_text =
        run.replacement.final_text_for_epoch(EventKind::kAgentText, run.evidence.replacement_epoch);
    run.evidence.replacement_turn_finished_events =
        run.replacement.count_for_epoch(EventKind::kTurnFinished, run.evidence.replacement_epoch);
    run.evidence.input_finished_events = run.replacement.count(EventKind::kInputFinished);
    run.evidence.implemented = true;
    return run;
}

void write_response_recovery_receipt(std::ostream& out, const char* name,
                                     const ResponseRecoveryEvidence& evidence) {
    out << "  \"" << name << "\": {\n";
    out << "    \"implemented\": " << json_bool(evidence.implemented) << ",\n";
    out << "    \"commit_without_response\": " << json_bool(evidence.commit_without_response)
        << ",\n";
    out << "    \"interrupted_epoch\": " << evidence.interrupted_epoch << ",\n";
    out << "    \"yielded_epoch\": " << evidence.yielded_epoch << ",\n";
    out << "    \"replacement_epoch\": " << evidence.replacement_epoch << ",\n";
    out << "    \"old_audio_events_before_control\": " << evidence.old_audio_events_before_control
        << ",\n";
    out << "    \"old_partial_text_events_before_control\": "
        << evidence.old_partial_text_events_before_control << ",\n";
    out << "    \"control_yield_events\": " << evidence.control_yield_events << ",\n";
    out << "    \"control_call_ms\": " << evidence.control_call_ms << ",\n";
    out << "    \"played_output_samples\": " << evidence.played_output_samples << ",\n";
    out << "    \"observed_output_span_samples\": " << evidence.observed_output_span_samples
        << ",\n";
    out << "    \"generated_output_samples\": " << evidence.generated_output_samples << ",\n";
    out << "    \"retained_output_samples\": " << evidence.retained_output_samples << ",\n";
    out << "    \"discarded_output_samples\": " << evidence.discarded_output_samples << ",\n";
    out << "    \"stale_agent_payloads_after_control\": "
        << evidence.stale_agent_payloads_after_control << ",\n";
    out << "    \"replacement_audio_events\": " << evidence.replacement_audio_events << ",\n";
    out << "    \"replacement_audio_samples\": " << evidence.replacement_audio_samples << ",\n";
    out << "    \"replacement_audio_rms\": " << evidence.replacement_audio_rms << ",\n";
    out << "    \"replacement_audio_peak\": " << evidence.replacement_audio_peak << ",\n";
    out << "    \"replacement_final_text_events\": " << evidence.replacement_final_text_events
        << ",\n";
    out << "    \"replacement_final_text\": \"" << json_escape(evidence.replacement_final_text)
        << "\",\n";
    out << "    \"replacement_turn_finished_events\": " << evidence.replacement_turn_finished_events
        << ",\n";
    out << "    \"input_finished_events\": " << evidence.input_finished_events << "\n";
    out << "  },\n";
}

void write_failure_receipt(const std::string& path, const std::string& error) {
    std::ofstream out(path);
    if (out)
        out << "{\n  \"schema_version\": 3,\n  \"pass\": false,\n  \"error\": \""
            << json_escape(error) << "\"\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        try {
            return run_host_self_check();
        } catch (const std::exception& error) {
            std::cerr << "native lifecycle probe host self-check failed: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 6) {
        std::cerr << "usage: native_lifecycle_probe BUNDLE WAV RUNTIME_ROOT "
                     "OUTPUT_WAV RECEIPT_JSON\n";
        return 2;
    }

    const std::string bundle_path = argv[1];
    const std::string input_path = argv[2];
    const std::string runtime_root = argv[3];
    const std::string output_wav = argv[4];
    const std::string receipt_path = argv[5];

    try {
        auto pipeline = trtmc::load_task(bundle_path, runtime_root);
        auto* session_provider = dynamic_cast<trtmc::ISpeechSessionProvider*>(pipeline.get());
        if (session_provider == nullptr)
            throw std::runtime_error("loaded pipeline does not expose ISpeechSession");
        auto* batch_provider = dynamic_cast<trtmc::ISpeechBatchSessionProvider*>(pipeline.get());
        if (batch_provider == nullptr)
            throw std::runtime_error("loaded pipeline does not expose batch speech sessions");

        const auto input = trtmc::cli::io::read_wav(input_path);
        if (input.sample_rate != 16000 || input.num_samples != 249734)
            throw std::runtime_error("pinned model-card sample shape does not match contract");
        const auto slash = input_path.find_last_of('/');
        const std::string function_input_path =
            input_path.substr(0, slash == std::string::npos ? 0 : slash + 1U) + "sample_fc.wav";
        const auto function_input = trtmc::cli::io::read_wav(function_input_path);
        if (function_input.sample_rate != 16000 || function_input.num_samples != 190278)
            throw std::runtime_error("pinned function-call sample shape does not match contract");

        const auto config = base_config(input.sample_rate);

        Capture baseline;
        {
            auto session = batch_provider->create_batch_speech_session(config);
            session->append_audio(input.samples.data(), input.num_samples);
            finish_and_drain(*session, baseline);
        }
        std::cerr << "[probe] baseline complete: audio_samples=" << baseline.audio.size()
                  << " audio_events=" << baseline.count(EventKind::kAgentAudio) << '\n';

        Capture irregular;
        int irregular_append_calls = 0;
        double irregular_max_append_ms = 0.0;
        int audio_events_before_finish = 0;
        {
            auto session = batch_provider->create_batch_speech_session(config);
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
        trtmc::cli::io::write_wav(chunked_audio, output_wav);

        Capture barge_before;
        Capture barge_after;
        std::uint64_t interrupted_epoch = 0;
        std::uint64_t yielded_epoch = 0;
        std::uint64_t recovered_epoch = 0;
        int stale_agent_payloads = 0;
        int recovery_silence_frames = 0;
        bool interrupted_audio_before_yield = false;
        bool interrupted_partial_text_before_yield = false;
        bool recovery_audio_before_finish = false;
        bool recovery_partial_text_before_finish = false;
        {
            auto barge_config = config;
            barge_config.enable_barge_in = true;
            auto session = session_provider->create_speech_session(barge_config);
            auto* realtime_control = dynamic_cast<trtmc::ISpeechRealtimeControl*>(session.get());
            if (realtime_control == nullptr)
                throw std::runtime_error("barge session does not expose realtime controls");
            std::array<float, static_cast<std::size_t>(kInputFrameSamples)> silence{};
            std::size_t prompt_offset =
                static_cast<std::size_t>(kBargeSpeechStartFrame) * kInputFrameSamples;
            const std::size_t prompt_end =
                static_cast<std::size_t>(kBargeSpeechEndFrame) * kInputFrameSamples;
            while (prompt_offset + kInputFrameSamples <= prompt_end) {
                session->append_audio(function_input.samples.data() + prompt_offset,
                                      kInputFrameSamples);
                prompt_offset += kInputFrameSamples;
                pace_and_drain(*session, barge_before);
            }
            realtime_control->commit_input_turn(false);
            barge_before.absorb(session->take_events());

            std::vector<float> barge_audio;
            const std::size_t barge_source_start =
                static_cast<std::size_t>(kBargeSpeechStartFrame) * kInputFrameSamples;
            const std::size_t barge_source_samples = prompt_end - barge_source_start;
            const std::size_t barge_output_samples = barge_source_samples * 2U / 3U;
            barge_audio.reserve(barge_output_samples);
            for (std::size_t index = 0; index < barge_output_samples; ++index) {
                barge_audio.push_back(function_input.samples[barge_source_start + index * 3U / 2U]);
            }
            std::size_t offset = 0;
            for (int frame = 0; frame < 4 && offset + kInputFrameSamples <= barge_audio.size();
                 ++frame) {
                session->append_audio(barge_audio.data() + offset, kInputFrameSamples);
                offset += kInputFrameSamples;
                pace_and_drain(*session, barge_before);
            }
            if (barge_before.count(EventKind::kTurnStarted) != 0) {
                dump_event_trace("before_barge", barge_before);
                throw std::runtime_error("agent started before explicit response.create");
            }

            realtime_control->create_response();
            barge_after.absorb(session->take_events());
            interrupted_epoch = last_epoch(barge_after, EventKind::kTurnStarted);
            if (interrupted_epoch == 0)
                throw std::runtime_error("response.create did not start an agent turn");

            // Speak over the committed request's long answer. Time compression
            // retains real pinned speech while producing the consecutive
            // non-unknown RNNT frames required by the public BOU policy.
            while (offset + kInputFrameSamples <= barge_audio.size() &&
                   barge_after.count(EventKind::kYielded) == 0) {
                session->append_audio(barge_audio.data() + offset, kInputFrameSamples);
                offset += kInputFrameSamples;
                pace_and_drain(*session, barge_after);
            }
            if (barge_after.count(EventKind::kYielded) == 0) {
                dump_event_trace("recognized_barge_attempt", barge_after);
                throw std::runtime_error("recognized speech did not trigger barge-in yield");
            }
            yielded_epoch = last_epoch(barge_after, EventKind::kYielded);
            interrupted_audio_before_yield =
                barge_after.has_agent_audio_for_epoch(interrupted_epoch);
            interrupted_partial_text_before_yield =
                barge_after.has_partial_agent_text_for_epoch(interrupted_epoch);
            if (!interrupted_audio_before_yield || !interrupted_partial_text_before_yield) {
                dump_event_trace("recognized_barge_attempt", barge_after);
                throw std::runtime_error(
                    "agent did not publish audio and partial text before recognized barge-in");
            }

            while (recovered_epoch == 0 && recovery_silence_frames < 64) {
                if (offset + kInputFrameSamples <= barge_audio.size()) {
                    session->append_audio(barge_audio.data() + offset, kInputFrameSamples);
                    offset += kInputFrameSamples;
                } else {
                    session->append_audio(silence.data(), static_cast<int32_t>(silence.size()));
                    ++recovery_silence_frames;
                }
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

            bool yield_observed = false;
            for (const auto& event : barge_after.events) {
                if (event.kind == EventKind::kYielded && event.epoch == yielded_epoch)
                    yield_observed = true;
                else if (yield_observed && is_agent_payload(event.kind) &&
                         event.epoch == interrupted_epoch)
                    ++stale_agent_payloads;
            }
        }
        std::cerr << "[probe] barge-in complete: interrupted_epoch=" << interrupted_epoch
                  << " yielded_epoch=" << yielded_epoch << " recovered_epoch=" << recovered_epoch
                  << " stale_agent_payloads=" << stale_agent_payloads << '\n';
        if (stale_agent_payloads != 0)
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

        const std::size_t control_speech_start =
            static_cast<std::size_t>(kBargeSpeechStartFrame) * kInputFrameSamples;
        const std::size_t control_speech_end =
            static_cast<std::size_t>(kBargeSpeechEndFrame) * kInputFrameSamples;
        std::vector<float> control_speech(
            function_input.samples.begin() + static_cast<std::ptrdiff_t>(control_speech_start),
            function_input.samples.begin() + static_cast<std::ptrdiff_t>(control_speech_end));

        ProcessedInputClearEvidence processed_clear;
        Capture clear_processed;
        Capture clear_output;
        Capture clear_fresh;
        const auto run_clear_comparison_input = [&](trtmc::ISpeechSession& session,
                                                    trtmc::ISpeechRealtimeControl& control,
                                                    Capture& capture) {
            feed_realtime(session, capture, control_speech);
            control.commit_input_turn();
            capture.absorb(session.take_events());
            wait_until(
                session, capture,
                [&] {
                    return !capture.audio.empty() &&
                           capture.count_final(EventKind::kAgentText) > 0 &&
                           capture.count_final(EventKind::kUserTranscript) > 0 &&
                           capture.count(EventKind::kTurnFinished) > 0;
                },
                120000, "completed clear-comparison response");
            finish_and_drain(session, capture, 30000);
        };
        {
            auto clear_config = config;
            clear_config.enable_barge_in = true;
            auto session = session_provider->create_speech_session(clear_config);
            auto* realtime_control = dynamic_cast<trtmc::ISpeechRealtimeControl*>(session.get());
            if (realtime_control == nullptr)
                throw std::runtime_error("processed-input clear session lacks realtime controls");

            std::size_t offset = 0;
            while (offset < control_speech.size()) {
                const int32_t count = std::min<int32_t>(
                    kInputFrameSamples, static_cast<int32_t>(control_speech.size() - offset));
                session->append_audio(control_speech.data() + offset, count);
                offset += static_cast<std::size_t>(count);
                ++processed_clear.processed_append_calls;
                processed_clear.processed_input_samples += static_cast<std::size_t>(count);
                pace_and_drain(*session, clear_processed, kRealtimePaceMs);
            }
            wait_until(
                *session, clear_processed,
                [&] { return clear_processed.count_nonfinal(EventKind::kUserTranscript) > 0; },
                30000, "processed user transcript before input clear");
            processed_clear.transcript_delta_events_before_clear =
                clear_processed.count_nonfinal(EventKind::kUserTranscript);

            const auto clear_start = Clock::now();
            realtime_control->clear_pending_input();
            processed_clear.clear_call_ms = elapsed_ms(clear_start, Clock::now());
            processed_clear.clear_succeeded = true;
            clear_output.absorb(session->take_events());
            wait_until(
                *session, clear_output,
                [&] { return clear_output.count(EventKind::kInputCleared) != 0; }, 120000,
                "processed input clear completion");
            processed_clear.clear_completion_events = clear_output.count(EventKind::kInputCleared);

            run_clear_comparison_input(*session, *realtime_control, clear_output);
        }
        {
            auto clear_config = config;
            clear_config.enable_barge_in = true;
            auto session = session_provider->create_speech_session(clear_config);
            auto* realtime_control = dynamic_cast<trtmc::ISpeechRealtimeControl*>(session.get());
            if (realtime_control == nullptr)
                throw std::runtime_error("fresh control session lacks realtime controls");
            run_clear_comparison_input(*session, *realtime_control, clear_fresh);
        }
        processed_clear.cleared_output_samples = clear_output.audio.size();
        processed_clear.fresh_output_samples = clear_fresh.audio.size();
        const auto cleared_response_epoch = last_epoch(clear_output, EventKind::kTurnStarted);
        const auto fresh_response_epoch = last_epoch(clear_fresh, EventKind::kTurnStarted);
        processed_clear.cleared_audio_rms =
            clear_output.audio_rms_for_epoch(cleared_response_epoch);
        processed_clear.fresh_audio_rms = clear_fresh.audio_rms_for_epoch(fresh_response_epoch);
        processed_clear.cleared_audio_peak =
            clear_output.audio_peak_for_epoch(cleared_response_epoch);
        processed_clear.fresh_audio_peak = clear_fresh.audio_peak_for_epoch(fresh_response_epoch);
        processed_clear.bitwise_audio_equal = bitwise_equal(clear_output.audio, clear_fresh.audio);
        processed_clear.agent_text_equal = clear_output.final_text(EventKind::kAgentText) ==
                                           clear_fresh.final_text(EventKind::kAgentText);
        processed_clear.user_transcript_equal =
            clear_output.final_text(EventKind::kUserTranscript) ==
            clear_fresh.final_text(EventKind::kUserTranscript);
        processed_clear.cleared_turn_finished_events = clear_output.count(EventKind::kTurnFinished);
        processed_clear.fresh_turn_finished_events = clear_fresh.count(EventKind::kTurnFinished);
        processed_clear.cleared_input_finished_events =
            clear_output.count(EventKind::kInputFinished);
        processed_clear.fresh_input_finished_events = clear_fresh.count(EventKind::kInputFinished);
        processed_clear.implemented = true;
        std::cerr << "[probe] processed input clear complete: processed_appends="
                  << processed_clear.processed_append_calls
                  << " transcript_deltas=" << processed_clear.transcript_delta_events_before_clear
                  << " fresh_equal=" << json_bool(processed_clear.bitwise_audio_equal)
                  << " pass=" << json_bool(processed_clear.pass()) << '\n';

        auto response_cancel =
            run_response_recovery(*session_provider, config, control_speech, false);
        std::cerr << "[probe] local response cancel complete: interrupted_epoch="
                  << response_cancel.evidence.interrupted_epoch
                  << " replacement_epoch=" << response_cancel.evidence.replacement_epoch
                  << " stale_payloads="
                  << response_cancel.evidence.stale_agent_payloads_after_control
                  << " pass=" << json_bool(response_cancel.evidence.pass(false)) << '\n';

        auto response_truncate =
            run_response_recovery(*session_provider, config, control_speech, true);
        std::cerr << "[probe] local response truncate complete: interrupted_epoch="
                  << response_truncate.evidence.interrupted_epoch
                  << " played_samples=" << response_truncate.evidence.played_output_samples
                  << " replacement_epoch=" << response_truncate.evidence.replacement_epoch
                  << " stale_payloads="
                  << response_truncate.evidence.stale_agent_payloads_after_control
                  << " pass=" << json_bool(response_truncate.evidence.pass(true)) << '\n';

        Capture tail_before;
        Capture tail_after;
        std::uint64_t tail_turn_epoch = 0;
        int tail_partial_commit_audio_events = 0;
        double tail_completion_ms = 0.0;
        {
            auto tail_config = config;
            tail_config.finish_tail_frames = kTailFrames;
            auto session = session_provider->create_speech_session(tail_config);
            std::size_t offset = 0;
            while (offset + kInputFrameSamples <= input.samples.size() && tail_turn_epoch == 0) {
                session->append_audio(input.samples.data() + offset, kInputFrameSamples);
                offset += kInputFrameSamples;
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
            auto* realtime_control = dynamic_cast<trtmc::ISpeechRealtimeControl*>(session.get());
            if (realtime_control == nullptr)
                throw std::runtime_error("live session does not expose realtime controls");
            const int audio_events_before_commit = tail_before.count(EventKind::kAgentAudio);
            realtime_control->commit_input_turn(false);
            tail_before.absorb(session->take_events());
            tail_partial_commit_audio_events =
                tail_before.count(EventKind::kAgentAudio) - audio_events_before_commit;
            const auto finish_start = Clock::now();
            finish_and_drain(*session, tail_after, 30000);
            tail_completion_ms = elapsed_ms(finish_start, Clock::now());
        }
        std::cerr << "[probe] partial finish/tail complete: post_finish_audio_events="
                  << tail_after.count(EventKind::kAgentAudio)
                  << " completion_ms=" << tail_completion_ms << '\n';

        Capture normal_multiturn_capture;
        bool normal_same_session = false;
        {
            auto normal_config = config;
            normal_config.enable_barge_in = false;
            normal_config.finish_tail_frames = 0;
            auto session = session_provider->create_speech_session(normal_config);
            normal_same_session = true;

            std::array<float, static_cast<std::size_t>(kInputFrameSamples)> silence{};
            std::uint64_t greeting_epoch = 0;
            for (int frame = 0; frame < 128 && greeting_epoch == 0; ++frame) {
                session->append_audio(silence.data(), static_cast<int32_t>(silence.size()));
                pace_and_drain(*session, normal_multiturn_capture);
                greeting_epoch = last_epoch(normal_multiturn_capture, EventKind::kTurnStarted);
            }
            const bool greeting_complete =
                greeting_epoch != 0 && wait_until_bounded(
                                           *session, normal_multiturn_capture,
                                           [&] {
                                               return normal_multiturn_capture.has_event_for_epoch(
                                                   EventKind::kTurnFinished, greeting_epoch);
                                           },
                                           120000);

            feed_realtime(*session, normal_multiturn_capture, function_input.samples);
            const bool first_turn_complete = wait_until_bounded(
                *session, normal_multiturn_capture,
                [&] {
                    return normal_multiturn_capture.count(EventKind::kTurnFinished) >= 2 &&
                           normal_multiturn_capture.count_final(EventKind::kAgentText) >= 2 &&
                           normal_multiturn_capture.count_final(EventKind::kUserTranscript) >= 1;
                },
                120000);
            if (greeting_complete && first_turn_complete) {
                feed_realtime(*session, normal_multiturn_capture, function_input.samples);
                const bool second_turn_complete = wait_until_bounded(
                    *session, normal_multiturn_capture,
                    [&] {
                        return normal_multiturn_capture.count(EventKind::kTurnFinished) >= 3 &&
                               normal_multiturn_capture.count_final(EventKind::kAgentText) >= 3 &&
                               normal_multiturn_capture.count_final(EventKind::kUserTranscript) >=
                                   2;
                    },
                    120000);
                if (second_turn_complete) {
                    finish_and_drain(*session, normal_multiturn_capture, 30000);
                } else {
                    session->cancel();
                    normal_multiturn_capture.absorb(session->take_events());
                }
            } else {
                session->cancel();
                normal_multiturn_capture.absorb(session->take_events());
            }
        }
        const auto normal_multiturn =
            summarize_normal_multiturn(normal_multiturn_capture, normal_same_session);
        std::cerr << "[probe] normal multi-turn complete: started="
                  << normal_multiturn.turn_started_events
                  << " finished=" << normal_multiturn.turn_finished_events
                  << " distinct_epochs=" << normal_multiturn.distinct_turn_epochs
                  << " pass=" << json_bool(normal_multiturn.pass()) << '\n';

        Capture concurrency_capture;
        BackpressureConcurrencyEvidence backpressure;
        {
            auto live_config = config;
            live_config.enable_barge_in = true;
            live_config.finish_tail_frames = 0;
            auto session = session_provider->create_speech_session(live_config);
            std::atomic<bool> producer_finished{false};
            std::atomic<bool> consumer_finished{false};
            std::atomic<bool> stop_requested{false};
            std::atomic<bool> events_during_production{false};
            std::exception_ptr producer_error;
            std::exception_ptr consumer_error;
            int producer_append_calls = 0;
            int finish_input_calls = 0;
            double max_append_call_ms = 0.0;

            std::thread consumer([&] {
                try {
                    while (!stop_requested.load(std::memory_order_acquire) &&
                           concurrency_capture.count(EventKind::kInputFinished) == 0) {
                        auto events = session->wait_events(250);
                        if (!producer_finished.load(std::memory_order_acquire) && !events.empty())
                            events_during_production.store(true, std::memory_order_release);
                        concurrency_capture.absorb(std::move(events));
                    }
                } catch (...) {
                    consumer_error = std::current_exception();
                }
                consumer_finished.store(true, std::memory_order_release);
            });

            std::thread producer([&] {
                try {
                    auto next_tick = Clock::now();
                    std::size_t offset = 0;
                    while (offset < input.samples.size() &&
                           !stop_requested.load(std::memory_order_acquire)) {
                        const int32_t count =
                            std::min<int32_t>(kInputFrameSamples,
                                              static_cast<int32_t>(input.samples.size() - offset));
                        const auto append_start = Clock::now();
                        session->append_audio(input.samples.data() + offset, count);
                        max_append_call_ms =
                            std::max(max_append_call_ms, elapsed_ms(append_start, Clock::now()));
                        ++producer_append_calls;
                        offset += static_cast<std::size_t>(count);
                        next_tick += std::chrono::milliseconds(kRealtimePaceMs);
                        std::this_thread::sleep_until(next_tick);
                    }
                    if (!stop_requested.load(std::memory_order_acquire)) {
                        session->finish_input();
                        ++finish_input_calls;
                    }
                } catch (...) {
                    producer_error = std::current_exception();
                }
                producer_finished.store(true, std::memory_order_release);
            });

            const auto concurrency_deadline =
                Clock::now() + std::chrono::milliseconds(kConcurrencyDeadlineMs);
            while ((!producer_finished.load(std::memory_order_acquire) ||
                    !consumer_finished.load(std::memory_order_acquire)) &&
                   Clock::now() < concurrency_deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            const bool completed_before_deadline =
                producer_finished.load(std::memory_order_acquire) &&
                consumer_finished.load(std::memory_order_acquire);
            if (!completed_before_deadline) {
                stop_requested.store(true, std::memory_order_release);
                session->cancel();
            }
            producer.join();
            consumer.join();

            backpressure.producer_thread_completed =
                producer_finished.load(std::memory_order_acquire) && !producer_error;
            backpressure.consumer_thread_completed =
                consumer_finished.load(std::memory_order_acquire) && !consumer_error;
            backpressure.events_observed_while_producing =
                events_during_production.load(std::memory_order_acquire);
            backpressure.no_deadlock = completed_before_deadline;
            backpressure.producer_append_calls = producer_append_calls;
            backpressure.finish_input_calls = finish_input_calls;
            backpressure.input_finished_events =
                concurrency_capture.count(EventKind::kInputFinished);
            backpressure.max_append_call_ms = max_append_call_ms;
        }

        {
            auto overflow_config = config;
            overflow_config.enable_barge_in = true;
            overflow_config.finish_tail_frames = 0;
            overflow_config.emit_agent_audio = false;
            overflow_config.emit_agent_text = false;
            overflow_config.emit_user_transcript = false;
            auto session = session_provider->create_speech_session(overflow_config);
            backpressure.live_capacity_samples =
                static_cast<std::size_t>(input.sample_rate) * kLiveInputCapacitySeconds;
            backpressure.overflow_attempt_samples = backpressure.live_capacity_samples + 1U;
            std::vector<float> oversized(backpressure.overflow_attempt_samples, 0.0F);
            const auto overflow_start = Clock::now();
            try {
                session->append_audio(oversized.data(), static_cast<int32_t>(oversized.size()));
            } catch (const std::overflow_error&) {
                backpressure.overflow_error_observed = true;
            }
            backpressure.overflow_call_ms = elapsed_ms(overflow_start, Clock::now());
            backpressure.bounded_queue = backpressure.overflow_error_observed;
        }
        std::cerr << "[probe] backpressure concurrency complete: producer="
                  << json_bool(backpressure.producer_thread_completed)
                  << " consumer=" << json_bool(backpressure.consumer_thread_completed)
                  << " events_during_production="
                  << json_bool(backpressure.events_observed_while_producing)
                  << " overflow=" << json_bool(backpressure.overflow_error_observed)
                  << " pass=" << json_bool(backpressure.pass()) << '\n';

        Capture function_capture;
        FunctionChannelEvidence function_channel;
        auto* tool_provider = dynamic_cast<trtmc::ISpeechToolSessionProvider*>(pipeline.get());
        if (tool_provider != nullptr) {
            auto function_config = config;
            function_config.enable_barge_in = true;
            function_config.finish_tail_frames = 0;
            trtmc::SpeechToolSessionConfig tools;
            tools.tools_json = R"json([
              {
                "type": "function",
                "function": {
                  "name": "generate_random_number",
                  "description": "Generate a random integer between min and max (inclusive).",
                  "parameters": {
                    "type": "object",
                    "properties": {
                      "min": {"type": "integer", "description": "Minimum value (inclusive)"},
                      "max": {"type": "integer", "description": "Maximum value (inclusive)"}
                    },
                    "required": ["min", "max"]
                  }
                },
                "ack_messages": ["Sure, give me just a moment."]
              }
            ])json";
            tools.on_hold_messages_json =
                R"({"generate_random_number":"I am generating that number now."})";
            auto session = tool_provider->create_tool_speech_session(function_config, tools);
            auto* tool_session = dynamic_cast<trtmc::ISpeechToolSession*>(session.get());
            if (tool_session == nullptr)
                throw std::runtime_error("tool session factory did not expose response capability");

            feed_realtime(*session, function_capture, function_input.samples);
            wait_until(
                *session, function_capture,
                [&] { return function_capture.count(EventKind::kFunctionCall) == 1; }, 120000,
                "function call event");
            const auto call = std::find_if(
                function_capture.events.begin(), function_capture.events.end(),
                [](const ObservedEvent& event) { return event.kind == EventKind::kFunctionCall; });
            if (call == function_capture.events.end())
                throw std::runtime_error("function call event disappeared from capture");
            function_channel.call_id = json_string_field(call->text, "call_id");
            function_channel.expected_tool_name_match =
                json_string_field(call->text, "name") == "generate_random_number";
            const int audio_before_response = function_capture.count(EventKind::kAgentAudio);
            const int final_text_before_response =
                function_capture.count_final(EventKind::kAgentText);
            tool_session->submit_tool_response(call->epoch, function_channel.call_id,
                                               R"({"result":20})");
            function_channel.tool_response_submitted = true;
            function_channel.tool_response_injections = 1;

            wait_until(
                *session, function_capture,
                [&] {
                    return function_capture.count(EventKind::kFunctionResponseFinished) == 1 &&
                           function_capture.count(EventKind::kAgentAudio) > audio_before_response &&
                           function_capture.count_final(EventKind::kAgentText) >
                               final_text_before_response;
                },
                120000, "function response EOTR and resumed speech");
            function_channel.agent_resumed_audio_events =
                function_capture.count(EventKind::kAgentAudio) - audio_before_response;
            function_channel.agent_resumed_text_events =
                function_capture.count_final(EventKind::kAgentText) - final_text_before_response;
            try {
                tool_session->submit_tool_response(call->epoch, function_channel.call_id,
                                                   R"({"result":21})");
            } catch (const std::invalid_argument&) {
                function_channel.stale_response_rejected = true;
            }
            finish_and_drain(*session, function_capture, 30000);
            function_channel.implemented = true;
            function_channel.sotc_events = function_capture.count(EventKind::kFunctionCallStarted);
            function_channel.eotc_events = function_capture.count(EventKind::kFunctionCall);
            function_channel.eotr_events =
                function_capture.count(EventKind::kFunctionResponseFinished);
            function_channel.completed_calls = function_channel.eotc_events;
        }
        std::cerr << "[probe] function channel complete: calls=" << function_channel.completed_calls
                  << " eotr=" << function_channel.eotr_events
                  << " resumed_audio=" << function_channel.agent_resumed_audio_events
                  << " pass=" << json_bool(function_channel.pass()) << '\n';

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
        const int minimum_tail_audio_events = kTailFrames;
        const int maximum_tail_audio_events = kTailFrames + 1;
        const int tail_audio_events = tail_after.count(EventKind::kAgentAudio);
        const bool tail_pass =
            tail_partial_commit_audio_events >= 1 &&
            tail_audio_events >= minimum_tail_audio_events &&
            tail_audio_events <= maximum_tail_audio_events &&
            tail_after.audio.size() ==
                static_cast<std::size_t>(tail_audio_events * kOutputFrameSamples) &&
            tail_after.count(EventKind::kInputFinished) == 1 &&
            tail_completion_ms < kTailCompletionLimitMs;
        const auto event_contract = validate_event_contracts(
            {{&baseline},
             {&irregular},
             {&barge_before, &barge_after},
             {&cancel_initial, &cancel_late, &reset_marker, &reset_prefix},
             {&fresh_prefix},
             {&tail_before, &tail_after},
             {&clear_processed, &clear_output},
             {&clear_fresh},
             {&response_cancel.before_control, &response_cancel.after_control,
              &response_cancel.replacement},
             {&response_truncate.before_control, &response_truncate.after_control,
              &response_truncate.replacement},
             {&normal_multiturn_capture},
             {&concurrency_capture},
             {&function_capture}},
            {{0, {&baseline}},
             {0, {&irregular}},
             {0, {&barge_before, &barge_after}},
             {0, {&cancel_initial, &cancel_late}},
             {0, {&reset_marker, &reset_prefix}},
             {0, {&fresh_prefix}},
             {0, {&tail_before, &tail_after}},
             {0, {&clear_processed, &clear_output}},
             {0, {&clear_fresh}},
             {0, {&response_cancel.before_control}},
             {response_cancel.evidence.retained_output_samples,
              {&response_cancel.after_control, &response_cancel.replacement}},
             {0, {&response_truncate.before_control}},
             {response_truncate.evidence.retained_output_samples,
              {&response_truncate.after_control, &response_truncate.replacement}},
             {0, {&normal_multiturn_capture}},
             {0, {&concurrency_capture}},
             {0, {&function_capture}}});

        const bool normal_multiturn_pass = normal_multiturn.pass();
        const bool function_channel_pass = function_channel.pass();
        const bool backpressure_concurrency_pass = backpressure.pass();
        const bool processed_input_clear_pass = processed_clear.pass();
        const bool response_cancel_recovery_pass = response_cancel.evidence.pass(false);
        const bool response_truncate_recovery_pass = response_truncate.evidence.pass(true);
        const bool pass = required_lifecycle_contract(
            baseline_contract, irregular_parity, barge_in_pass, cancel_pass, reset_pass, tail_pass,
            event_contract, normal_multiturn_pass, function_channel_pass,
            backpressure_concurrency_pass, processed_input_clear_pass,
            response_cancel_recovery_pass, response_truncate_recovery_pass);

        std::ofstream receipt(receipt_path);
        if (!receipt)
            throw std::runtime_error("cannot create receipt: " + receipt_path);
        receipt << std::fixed << std::setprecision(3);
        receipt << "{\n";
        receipt << "  \"schema_version\": 3,\n";
        receipt << "  \"pass\": " << json_bool(pass) << ",\n";
        receipt << "  \"runtime\": \"C++ ISpeechSession with TensorRT backend\",\n";
        receipt << "  \"baseline\": {\n";
        receipt << "    \"output_samples\": " << baseline.audio.size() << ",\n";
        receipt << "    \"audio_events\": " << baseline.count(EventKind::kAgentAudio) << ",\n";
        receipt << "    \"agent_text\": \"" << json_escape(baseline.agent_text()) << "\",\n";
        receipt << "    \"input_finished_events\": " << baseline.count(EventKind::kInputFinished)
                << "\n";
        receipt << "  },\n";
        receipt << "  \"irregular_chunking\": {\n";
        receipt << "    \"append_calls\": " << irregular_append_calls << ",\n";
        receipt << "    \"max_append_call_ms\": " << irregular_max_append_ms << ",\n";
        receipt << "    \"audio_events_before_finish\": " << audio_events_before_finish << ",\n";
        receipt << "    \"output_samples\": " << irregular.audio.size() << ",\n";
        receipt << "    \"bitwise_audio_equal_to_one_shot\": "
                << json_bool(bitwise_equal(irregular.audio, baseline.audio)) << ",\n";
        receipt << "    \"text_equal_to_one_shot\": "
                << json_bool(irregular.agent_text() == baseline.agent_text()) << ",\n";
        receipt << "    \"input_finished_events\": " << irregular.count(EventKind::kInputFinished)
                << "\n";
        receipt << "  },\n";
        receipt << "  \"barge_in\": {\n";
        receipt << "    \"interrupted_epoch\": " << interrupted_epoch << ",\n";
        receipt << "    \"interrupted_audio_before_yield\": "
                << json_bool(interrupted_audio_before_yield) << ",\n";
        receipt << "    \"interrupted_partial_text_before_yield\": "
                << json_bool(interrupted_partial_text_before_yield) << ",\n";
        receipt << "    \"yielded_epoch\": " << yielded_epoch << ",\n";
        receipt << "    \"barge_in_yield_events\": "
                << barge_after.count_with_text(EventKind::kYielded, "barge-in") << ",\n";
        receipt << "    \"stale_agent_payloads_after_yield\": " << stale_agent_payloads << ",\n";
        receipt << "    \"recovered_epoch\": " << recovered_epoch << ",\n";
        receipt << "    \"recovery_audio_before_finish\": "
                << json_bool(recovery_audio_before_finish) << ",\n";
        receipt << "    \"recovery_partial_text_before_finish\": "
                << json_bool(recovery_partial_text_before_finish) << ",\n";
        receipt << "    \"input_finished_events\": " << barge_after.count(EventKind::kInputFinished)
                << "\n";
        receipt << "  },\n";
        receipt << "  \"cancel\": {\n";
        receipt << "    \"append_call_ms\": " << cancel_append_ms << ",\n";
        receipt << "    \"cancel_call_ms\": " << cancel_call_ms << ",\n";
        receipt << "    \"cancel_events\": " << cancel_initial.count(EventKind::kCancelled)
                << ",\n";
        receipt << "    \"append_after_cancel_rejected\": "
                << json_bool(append_after_cancel_rejected) << ",\n";
        receipt << "    \"late_events\": " << cancel_late.events.size() << "\n";
        receipt << "  },\n";
        receipt << "  \"reset_vs_fresh\": {\n";
        receipt << "    \"reset_events\": " << reset_marker.count(EventKind::kReset) << ",\n";
        receipt << "    \"output_samples\": " << reset_prefix.audio.size() << ",\n";
        receipt << "    \"bitwise_audio_equal\": "
                << json_bool(bitwise_equal(reset_prefix.audio, fresh_prefix.audio)) << ",\n";
        receipt << "    \"text_equal\": "
                << json_bool(reset_prefix.agent_text() == fresh_prefix.agent_text()) << ",\n";
        receipt << "    \"reset_input_finished_events\": "
                << reset_prefix.count(EventKind::kInputFinished) << ",\n";
        receipt << "    \"fresh_input_finished_events\": "
                << fresh_prefix.count(EventKind::kInputFinished) << "\n";
        receipt << "  },\n";
        receipt << "  \"processed_input_clear\": {\n";
        receipt << "    \"implemented\": " << json_bool(processed_clear.implemented) << ",\n";
        receipt << "    \"clear_succeeded\": " << json_bool(processed_clear.clear_succeeded)
                << ",\n";
        receipt << "    \"processed_append_calls\": " << processed_clear.processed_append_calls
                << ",\n";
        receipt << "    \"processed_input_samples\": " << processed_clear.processed_input_samples
                << ",\n";
        receipt << "    \"transcript_delta_events_before_clear\": "
                << processed_clear.transcript_delta_events_before_clear << ",\n";
        receipt << "    \"clear_call_ms\": " << processed_clear.clear_call_ms << ",\n";
        receipt << "    \"clear_completion_events\": " << processed_clear.clear_completion_events
                << ",\n";
        receipt << "    \"cleared_output_samples\": " << processed_clear.cleared_output_samples
                << ",\n";
        receipt << "    \"fresh_output_samples\": " << processed_clear.fresh_output_samples
                << ",\n";
        receipt << "    \"cleared_audio_rms\": " << processed_clear.cleared_audio_rms << ",\n";
        receipt << "    \"fresh_audio_rms\": " << processed_clear.fresh_audio_rms << ",\n";
        receipt << "    \"cleared_audio_peak\": " << processed_clear.cleared_audio_peak << ",\n";
        receipt << "    \"fresh_audio_peak\": " << processed_clear.fresh_audio_peak << ",\n";
        receipt << "    \"bitwise_audio_equal\": " << json_bool(processed_clear.bitwise_audio_equal)
                << ",\n";
        receipt << "    \"agent_text_equal\": " << json_bool(processed_clear.agent_text_equal)
                << ",\n";
        receipt << "    \"user_transcript_equal\": "
                << json_bool(processed_clear.user_transcript_equal) << ",\n";
        receipt << "    \"cleared_turn_finished_events\": "
                << processed_clear.cleared_turn_finished_events << ",\n";
        receipt << "    \"fresh_turn_finished_events\": "
                << processed_clear.fresh_turn_finished_events << ",\n";
        receipt << "    \"cleared_input_finished_events\": "
                << processed_clear.cleared_input_finished_events << ",\n";
        receipt << "    \"fresh_input_finished_events\": "
                << processed_clear.fresh_input_finished_events << "\n";
        receipt << "  },\n";
        write_response_recovery_receipt(receipt, "response_cancel_recovery",
                                        response_cancel.evidence);
        write_response_recovery_receipt(receipt, "response_truncate_recovery",
                                        response_truncate.evidence);
        receipt << "  \"partial_finish_tail\": {\n";
        receipt << "    \"partial_input_samples\": " << kPartialSamples << ",\n";
        receipt << "    \"pre_finish_committed_audio_events\": " << tail_partial_commit_audio_events
                << ",\n";
        receipt << "    \"configured_tail_frames\": " << kTailFrames << ",\n";
        receipt << "    \"minimum_audio_events_after_finish\": " << minimum_tail_audio_events
                << ",\n";
        receipt << "    \"maximum_audio_events_after_finish\": " << maximum_tail_audio_events
                << ",\n";
        receipt << "    \"audio_events_after_finish\": " << tail_after.count(EventKind::kAgentAudio)
                << ",\n";
        receipt << "    \"output_samples_after_finish\": " << tail_after.audio.size() << ",\n";
        receipt << "    \"completion_ms\": " << tail_completion_ms << ",\n";
        receipt << "    \"input_finished_events\": " << tail_after.count(EventKind::kInputFinished)
                << "\n";
        receipt << "  },\n";
        receipt << "  \"sequence_continuity\": {\n";
        receipt << "    \"pass\": " << json_bool(event_contract.sequence_monotonic) << ",\n";
        receipt << "    \"sessions_checked\": " << event_contract.sessions_checked << ",\n";
        receipt << "    \"events_checked\": " << event_contract.events_checked << ",\n";
        receipt << "    \"violations\": " << event_contract.sequence_violations << "\n";
        receipt << "  },\n";
        receipt << "  \"media_continuity\": {\n";
        receipt << "    \"pass\": " << json_bool(event_contract.media_contiguous) << ",\n";
        receipt << "    \"segments_checked\": " << event_contract.media_segments_checked << ",\n";
        receipt << "    \"audio_events_checked\": " << event_contract.audio_events_checked << ",\n";
        receipt << "    \"violations\": " << event_contract.media_violations << "\n";
        receipt << "  },\n";
        receipt << "  \"normal_multiturn\": {\n";
        receipt << "    \"implemented\": true,\n";
        receipt << "    \"same_session\": " << json_bool(normal_multiturn.same_session) << ",\n";
        receipt << "    \"every_turn_completed\": "
                << json_bool(normal_multiturn.every_turn_completed) << ",\n";
        receipt << "    \"turn_started_events\": " << normal_multiturn.turn_started_events << ",\n";
        receipt << "    \"turn_finished_events\": " << normal_multiturn.turn_finished_events
                << ",\n";
        receipt << "    \"distinct_turn_epochs\": " << normal_multiturn.distinct_turn_epochs
                << ",\n";
        receipt << "    \"final_agent_text_events\": " << normal_multiturn.final_agent_text_events
                << ",\n";
        receipt << "    \"final_user_transcript_events\": "
                << normal_multiturn.final_user_transcript_events << ",\n";
        receipt << "    \"yield_events\": " << normal_multiturn.yield_events << ",\n";
        receipt << "    \"reset_events\": " << normal_multiturn.reset_events << ",\n";
        receipt << "    \"input_finished_events\": " << normal_multiturn.input_finished_events
                << "\n";
        receipt << "  },\n";
        receipt << "  \"function_channel\": {\n";
        receipt << "    \"implemented\": " << json_bool(function_channel.implemented) << ",\n";
        receipt << "    \"sotc_events\": " << function_channel.sotc_events << ",\n";
        receipt << "    \"eotc_events\": " << function_channel.eotc_events << ",\n";
        receipt << "    \"eotr_events\": " << function_channel.eotr_events << ",\n";
        receipt << "    \"completed_calls\": " << function_channel.completed_calls << ",\n";
        receipt << "    \"tool_response_injections\": " << function_channel.tool_response_injections
                << ",\n";
        receipt << "    \"agent_resumed_audio_events\": "
                << function_channel.agent_resumed_audio_events << ",\n";
        receipt << "    \"agent_resumed_text_events\": "
                << function_channel.agent_resumed_text_events << ",\n";
        receipt << "    \"expected_tool_name_match\": "
                << json_bool(function_channel.expected_tool_name_match) << ",\n";
        receipt << "    \"tool_response_submitted\": "
                << json_bool(function_channel.tool_response_submitted) << ",\n";
        receipt << "    \"stale_response_rejected\": "
                << json_bool(function_channel.stale_response_rejected) << ",\n";
        receipt << "    \"stale_function_payloads\": " << function_channel.stale_function_payloads
                << "\n";
        receipt << "  },\n";
        receipt << "  \"backpressure_concurrency\": {\n";
        receipt << "    \"implemented\": true,\n";
        receipt << "    \"producer_thread_completed\": "
                << json_bool(backpressure.producer_thread_completed) << ",\n";
        receipt << "    \"consumer_thread_completed\": "
                << json_bool(backpressure.consumer_thread_completed) << ",\n";
        receipt << "    \"events_observed_while_producing\": "
                << json_bool(backpressure.events_observed_while_producing) << ",\n";
        receipt << "    \"bounded_queue\": " << json_bool(backpressure.bounded_queue) << ",\n";
        receipt << "    \"overflow_error_observed\": "
                << json_bool(backpressure.overflow_error_observed) << ",\n";
        receipt << "    \"no_deadlock\": " << json_bool(backpressure.no_deadlock) << ",\n";
        receipt << "    \"producer_append_calls\": " << backpressure.producer_append_calls << ",\n";
        receipt << "    \"finish_input_calls\": " << backpressure.finish_input_calls << ",\n";
        receipt << "    \"live_capacity_samples\": " << backpressure.live_capacity_samples << ",\n";
        receipt << "    \"overflow_attempt_samples\": " << backpressure.overflow_attempt_samples
                << ",\n";
        receipt << "    \"max_append_call_ms\": " << backpressure.max_append_call_ms << ",\n";
        receipt << "    \"overflow_call_ms\": " << backpressure.overflow_call_ms << ",\n";
        receipt << "    \"input_finished_events\": " << backpressure.input_finished_events << "\n";
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
