/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "serve/realtime_worker.h"
#include "trtmc/speech_session.h"

#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

struct SessionState {
    trtmc::SpeechSessionConfig config;
    trtmc::SpeechToolSessionConfig tool_config;
    std::deque<std::vector<trtmc::SpeechSessionEvent>> append_event_batches;
    std::vector<trtmc::SpeechSessionEvent> pending_events;
    std::vector<float> audio;
    std::uint64_t truncate_epoch{0};
    std::int64_t truncate_samples{-1};
    std::uint64_t tool_epoch{0};
    std::string tool_call_id;
    std::string tool_output;
    int normal_factory_calls{0};
    int tool_factory_calls{0};
    int append_calls{0};
    int commit_calls{0};
    int create_response_calls{0};
    int clear_calls{0};
    int response_cancel_calls{0};
    int truncate_calls{0};
    int tool_output_calls{0};
    int finish_calls{0};
    int session_cancel_calls{0};
    bool commit_create_response{true};
    bool throw_on_commit{false};
    bool throw_on_clear{false};
    bool throw_on_create_response{false};
    bool throw_on_response_cancel{false};
    bool throw_on_truncate{false};
};

trtmc::SpeechSessionEvent event(trtmc::SpeechSessionEventKind kind, std::uint64_t epoch,
                                std::uint64_t sequence, std::string text = {}) {
    trtmc::SpeechSessionEvent value;
    value.kind = kind;
    value.epoch = epoch;
    value.sequence = sequence;
    value.frame_index = static_cast<std::int64_t>(sequence);
    value.media_start_sample = static_cast<std::int64_t>(sequence * 10U);
    value.media_end_sample = value.media_start_sample + 10;
    value.text = std::move(text);
    return value;
}

class FakeSession final : public trtmc::ISpeechSession,
                          public trtmc::ISpeechRealtimeControl,
                          public trtmc::ISpeechToolSession {
  public:
    explicit FakeSession(std::shared_ptr<SessionState> state) : state_(std::move(state)) {}

    void append_audio(const float* samples, int32_t count) override {
        ++state_->append_calls;
        state_->audio.insert(state_->audio.end(), samples, samples + count);
        if (!state_->append_event_batches.empty()) {
            auto events = std::move(state_->append_event_batches.front());
            state_->append_event_batches.pop_front();
            state_->pending_events.insert(state_->pending_events.end(),
                                          std::make_move_iterator(events.begin()),
                                          std::make_move_iterator(events.end()));
        }
    }

    void finish_input() override {
        ++state_->finish_calls;
        state_->pending_events.push_back(
            event(trtmc::SpeechSessionEventKind::kInputFinished, 50, 1));
    }

    std::vector<trtmc::SpeechSessionEvent> take_events() override {
        auto result = std::move(state_->pending_events);
        state_->pending_events.clear();
        return result;
    }

    void cancel() override { ++state_->session_cancel_calls; }
    void reset() override {}
    trtmc::SpeechSessionConfig config() const override { return state_->config; }

    void commit_input_turn(bool create_response) override {
        ++state_->commit_calls;
        state_->commit_create_response = create_response;
        if (state_->throw_on_commit)
            throw std::runtime_error("sensitive commit diagnostic");
    }

    void create_response() override {
        ++state_->create_response_calls;
        if (state_->throw_on_create_response)
            throw std::runtime_error("sensitive create diagnostic");
    }

    void clear_pending_input() override {
        ++state_->clear_calls;
        if (state_->throw_on_clear)
            throw std::runtime_error("sensitive diagnostic must not escape");
    }

    void cancel_response() override {
        ++state_->response_cancel_calls;
        if (state_->throw_on_response_cancel)
            throw std::runtime_error("sensitive cancel diagnostic");
    }

    void truncate_response(std::uint64_t epoch, std::int64_t samples) override {
        ++state_->truncate_calls;
        state_->truncate_epoch = epoch;
        state_->truncate_samples = samples;
        if (state_->throw_on_truncate)
            throw std::runtime_error("sensitive truncate diagnostic");
    }

    void submit_tool_response(std::uint64_t epoch, const std::string& call_id,
                              const std::string& output) override {
        ++state_->tool_output_calls;
        state_->tool_epoch = epoch;
        state_->tool_call_id = call_id;
        state_->tool_output = output;
    }

  private:
    std::shared_ptr<SessionState> state_;
};

class FakePipeline final : public trtmc::IPipeline,
                           public trtmc::ISpeechSessionProvider,
                           public trtmc::ISpeechToolSessionProvider {
  public:
    explicit FakePipeline(std::shared_ptr<SessionState> state) : state_(std::move(state)) {}

    const char* model_id() const override { return "fake"; }
    const char* pipeline_type() const override { return "FakePipeline"; }

    std::unique_ptr<trtmc::ISpeechSession>
    create_speech_session(const trtmc::SpeechSessionConfig& config) override {
        ++state_->normal_factory_calls;
        state_->config = config;
        return std::make_unique<FakeSession>(state_);
    }

    std::unique_ptr<trtmc::ISpeechSession>
    create_tool_speech_session(const trtmc::SpeechSessionConfig& config,
                               const trtmc::SpeechToolSessionConfig& tool_config) override {
        ++state_->tool_factory_calls;
        state_->config = config;
        state_->tool_config = tool_config;
        return std::make_unique<FakeSession>(state_);
    }

  private:
    std::shared_ptr<SessionState> state_;
};

class BasicSession final : public trtmc::ISpeechSession {
  public:
    explicit BasicSession(std::shared_ptr<SessionState> state) : state_(std::move(state)) {}

    void append_audio(const float*, int32_t) override { ++state_->append_calls; }
    void finish_input() override { ++state_->finish_calls; }
    std::vector<trtmc::SpeechSessionEvent> take_events() override { return {}; }
    void cancel() override { ++state_->session_cancel_calls; }
    void reset() override {}
    trtmc::SpeechSessionConfig config() const override { return state_->config; }

  private:
    std::shared_ptr<SessionState> state_;
};

class BasicPipeline final : public trtmc::IPipeline, public trtmc::ISpeechSessionProvider {
  public:
    explicit BasicPipeline(std::shared_ptr<SessionState> state) : state_(std::move(state)) {}

    const char* model_id() const override { return "basic"; }
    const char* pipeline_type() const override { return "BasicPipeline"; }

    std::unique_ptr<trtmc::ISpeechSession>
    create_speech_session(const trtmc::SpeechSessionConfig& config) override {
        ++state_->normal_factory_calls;
        state_->config = config;
        return std::make_unique<BasicSession>(state_);
    }

  private:
    std::shared_ptr<SessionState> state_;
};

class NoSessionPipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "none"; }
    const char* pipeline_type() const override { return "NoSessionPipeline"; }
};

struct RunResult {
    int status{0};
    std::string raw;
    std::vector<json> messages;
};

std::string jsonl(std::initializer_list<json> messages, bool crlf = false) {
    std::string input;
    for (const auto& message : messages) {
        input += message.dump();
        input += crlf ? "\r\n" : "\n";
    }
    return input;
}

RunResult run_worker(trtmc::IPipeline& pipeline, const std::string& input,
                     const trtmc::serve::RealtimeWorkerLimits& limits = {}) {
    std::istringstream source(input);
    std::ostringstream output;
    RunResult result;
    result.status = trtmc::serve::run_realtime_worker(pipeline, source, output, limits);
    result.raw = output.str();
    std::istringstream lines(result.raw);
    for (std::string line; std::getline(lines, line);)
        result.messages.push_back(json::parse(line));
    return result;
}

std::vector<json> messages_of_type(const RunResult& result, const std::string& type) {
    std::vector<json> matches;
    for (const auto& message : result.messages) {
        if (message.value("type", "") == type)
            matches.push_back(message);
    }
    return matches;
}

std::vector<std::string> error_codes(const RunResult& result) {
    std::vector<std::string> codes;
    for (const auto& message : messages_of_type(result, "error"))
        codes.push_back(message.at("code").get<std::string>());
    return codes;
}

bool contains_code(const RunResult& result, const std::string& code) {
    const auto codes = error_codes(result);
    return std::find(codes.begin(), codes.end(), code) != codes.end();
}

json update_message(std::string event_id = "update-1") {
    return {{"type", "session.update"},          {"event_id", std::move(event_id)},
            {"input_sample_rate", 24000},        {"output_sample_rate", 24000},
            {"instructions", "Be concise."},     {"tools", json::array()},
            {"on_hold_messages", json::object()}};
}

void test_framing_audio_and_close_modes() {
    auto state = std::make_shared<SessionState>();
    auto audio = event(trtmc::SpeechSessionEventKind::kAgentAudio, 7, 3);
    audio.audio_samples = {-1.0F, 0.0F, 1.0F};
    audio.sample_rate = 24000;
    state->append_event_batches.push_back({audio});
    FakePipeline pipeline(state);

    const auto result = run_worker(
        pipeline, jsonl({update_message(),
                         {{"type", "input_audio_buffer.append"},
                          {"event_id", "audio-1"},
                          {"audio", "AIAAAP9/"}},
                         {{"type", "session.close"}, {"event_id", "close-1"}, {"mode", "finish"}}},
                        true));

    check(result.status == 0 && result.messages.front().at("type") == "session.ready" &&
              result.messages.back().at("type") == "session.end",
          "worker frames one JSON object per line from ready through end");
    check(state->normal_factory_calls == 1 && state->config.input_sample_rate == 24000 &&
              state->config.output_sample_rate == 24000 &&
              state->config.system_prompt == "Be concise." && state->config.enable_barge_in,
          "session.update creates a generic 24 kHz full-duplex session");
    check(state->audio.size() == 3 && state->audio[0] == -1.0F && state->audio[1] == 0.0F &&
              state->audio[2] > 0.999F,
          "PCM16LE base64 input is decoded to normalized float samples");
    const auto events = messages_of_type(result, "session.event");
    check(events.size() == 2 && events[0].at("audio") == "AIAAAP9/" && events[0].at("epoch") == 7 &&
              events[0].at("sequence") == 3 && events[1].at("kind") == "input_finished",
          "agent audio is PCM16LE base64 and native ordering coordinates are preserved");
    check(state->finish_calls == 1 && state->session_cancel_calls == 0 &&
              result.messages.back().at("reason") == "closed" &&
              result.messages.back().at("event_id") == "close-1",
          "finish close drains input completion without cancelling");

    auto cancel_state = std::make_shared<SessionState>();
    FakePipeline cancel_pipeline(cancel_state);
    const auto cancelled =
        run_worker(cancel_pipeline,
                   jsonl({update_message(), {{"type", "session.close"}, {"mode", "cancel"}}}));
    check(cancelled.status == 0 && cancel_state->finish_calls == 0 &&
              cancel_state->session_cancel_calls == 1 &&
              cancelled.messages.back().at("reason") == "cancelled",
          "cancel close aborts rather than pretending to finish");
}

void test_tools_controls_and_stale_epoch() {
    auto state = std::make_shared<SessionState>();
    state->append_event_batches.push_back(
        {event(trtmc::SpeechSessionEventKind::kTurnStarted, 9, 1)});
    state->append_event_batches.push_back(
        {event(trtmc::SpeechSessionEventKind::kFunctionResponseFinished, 9, 2)});
    FakePipeline pipeline(state);
    auto update = update_message();
    update["tools"] =
        json::array({{{"type", "function"},
                      {"function", {{"name", "weather"}, {"parameters", {{"type", "object"}}}}}}});
    update["on_hold_messages"] = {{"weather", "One moment."}};

    const auto result = run_worker(
        pipeline,
        jsonl(
            {update,
             {{"type", "input_audio_buffer.append"}, {"audio", "AIA="}},
             {{"type", "input_audio_buffer.commit"}, {"event_id", "commit-1"}},
             {{"type", "input_audio_buffer.clear"}, {"event_id", "clear-1"}},
             {{"type", "response.create"}, {"event_id", "create-1"}},
             {{"type", "response.cancel"}, {"event_id", "cancel-1"}},
             {{"type", "conversation.item.truncate"}, {"epoch", 8}, {"played_output_samples", 120}},
             {{"type", "conversation.item.truncate"},
              {"event_id", "truncate-1"},
              {"epoch", 9},
              {"played_output_samples", 240}},
             {{"type", "conversation.item.create"},
              {"epoch", 8},
              {"call_id", "call-1"},
              {"output", "stale"}},
             {{"type", "conversation.item.create"},
              {"epoch", 9},
              {"call_id", "call-1"},
              {"output", R"({"temperature":72})"}},
             {{"type", "response.create"}, {"event_id", "tool-response-create"}},
             {{"type", "input_audio_buffer.append"}, {"audio", "AAA="}},
             {{"type", "response.create"}, {"event_id", "after-tool-response"}},
             {{"type", "session.close"}, {"mode", "cancel"}}}));

    check(state->tool_factory_calls == 1 && state->normal_factory_calls == 0 &&
              json::parse(state->tool_config.tools_json) == update.at("tools") &&
              json::parse(state->tool_config.on_hold_messages_json) ==
                  update.at("on_hold_messages"),
          "tool session configuration is routed through the optional generic provider");
    check(state->commit_calls == 1 && !state->commit_create_response && state->clear_calls == 1 &&
              state->create_response_calls == 2 && state->response_cancel_calls == 1,
          "commit, response.create, clear, and cancel route to separate realtime controls");
    check(state->truncate_calls == 1 && state->truncate_epoch == 9 &&
              state->truncate_samples == 240 && state->tool_output_calls == 1 &&
              state->tool_epoch == 9 && state->tool_call_id == "call-1" &&
              state->tool_output == R"({"temperature":72})",
          "truncate and function output preserve current epoch and payload");
    const auto codes = error_codes(result);
    check(std::count(codes.begin(), codes.end(), "stale_epoch") == 2,
          "stale truncate and function output are rejected before reaching the session");
    check(std::count(codes.begin(), codes.end(), "invalid_state") == 1 &&
              state->create_response_calls == 2,
          "response.create is rejected during atomic tool resume and accepted after its finish");
    bool active_tool_error_correlated = false;
    for (const auto& message : messages_of_type(result, "error")) {
        active_tool_error_correlated |= message.value("code", "") == "invalid_state" &&
                                        message.value("event_id", "") == "tool-response-create";
    }
    check(active_tool_error_correlated,
          "response.create rejection preserves its native event_id correlation");
    const auto updated = messages_of_type(result, "session.updated");
    check(updated.size() == 1 && updated[0]["capabilities"]["realtime_controls"] == true &&
              updated[0]["capabilities"]["function_call_output"] == true,
          "session.updated reports capabilities from the created session");
    const auto commits = messages_of_type(result, "input_audio_buffer.committed");
    const auto clears = messages_of_type(result, "input_audio_buffer.cleared");
    const auto cancellations = messages_of_type(result, "response.cancelled");
    const auto truncations = messages_of_type(result, "conversation.item.truncated");
    check(commits.size() == 1 && commits[0].at("event_id") == "commit-1" && clears.size() == 1 &&
              clears[0].at("event_id") == "clear-1" && cancellations.size() == 1 &&
              cancellations[0].at("event_id") == "cancel-1" && truncations.size() == 1 &&
              truncations[0].at("event_id") == "truncate-1" && truncations[0].at("epoch") == 9 &&
              truncations[0].at("played_output_samples") == 240 &&
              messages_of_type(result, "response.created").empty(),
          "control ACKs follow native acceptance while response.create waits for turn_started");
}

void test_native_event_mapping_order_and_redaction() {
    auto state = std::make_shared<SessionState>();
    const std::vector<trtmc::SpeechSessionEventKind> kinds = {
        trtmc::SpeechSessionEventKind::kAgentAudio,
        trtmc::SpeechSessionEventKind::kAgentText,
        trtmc::SpeechSessionEventKind::kUserTranscript,
        trtmc::SpeechSessionEventKind::kTurnStarted,
        trtmc::SpeechSessionEventKind::kTurnFinished,
        trtmc::SpeechSessionEventKind::kYielded,
        trtmc::SpeechSessionEventKind::kCancelled,
        trtmc::SpeechSessionEventKind::kReset,
        trtmc::SpeechSessionEventKind::kError,
        trtmc::SpeechSessionEventKind::kInputFinished,
        trtmc::SpeechSessionEventKind::kUserSpeechStarted,
        trtmc::SpeechSessionEventKind::kUserSpeechStopped,
        trtmc::SpeechSessionEventKind::kFunctionCall,
        trtmc::SpeechSessionEventKind::kFunctionCallStarted,
        trtmc::SpeechSessionEventKind::kFunctionResponseFinished,
    };
    std::vector<trtmc::SpeechSessionEvent> batch;
    for (std::size_t index = 0; index < kinds.size(); ++index) {
        auto value = event(kinds[index], 4, index, "text-" + std::to_string(index));
        if (kinds[index] == trtmc::SpeechSessionEventKind::kAgentAudio) {
            value.audio_samples = {0.25F};
            value.sample_rate = 24000;
        }
        if (kinds[index] == trtmc::SpeechSessionEventKind::kError)
            value.text = "sensitive native diagnostic";
        batch.push_back(std::move(value));
    }
    state->append_event_batches.push_back(std::move(batch));
    FakePipeline pipeline(state);
    const auto result =
        run_worker(pipeline, jsonl({update_message(),
                                    {{"type", "input_audio_buffer.append"}, {"audio", "AAA="}},
                                    {{"type", "session.close"}, {"mode", "cancel"}}}));
    const auto events = messages_of_type(result, "session.event");
    const std::vector<std::string> expected = {
        "agent_audio",
        "agent_text",
        "user_transcript",
        "turn_started",
        "turn_finished",
        "yielded",
        "cancelled",
        "reset",
        "error",
        "input_finished",
        "user_speech_started",
        "user_speech_stopped",
        "function_call",
        "function_call_started",
        "function_response_finished",
    };
    bool ordered = events.size() == expected.size();
    for (std::size_t index = 0; ordered && index < events.size(); ++index)
        ordered =
            events[index].at("kind") == expected[index] && events[index].at("sequence") == index;
    check(ordered, "all native event kinds preserve take_events order and sequence");
    check(result.raw.find("sensitive native diagnostic") == std::string::npos &&
              events[8].at("text") == "speech session error",
          "native error details are redacted from the JSONL transport");
    check(events[0].contains("audio") && !events[1].contains("audio"),
          "only agent_audio events carry PCM audio payloads");
}

void test_stale_native_events_are_suppressed() {
    auto state = std::make_shared<SessionState>();
    state->append_event_batches.push_back(
        {event(trtmc::SpeechSessionEventKind::kAgentText, 5, 1, "current-1"),
         event(trtmc::SpeechSessionEventKind::kAgentText, 4, 99, "stale-epoch"),
         event(trtmc::SpeechSessionEventKind::kAgentText, 5, 2, "current-2"),
         event(trtmc::SpeechSessionEventKind::kAgentText, 5, 2, "duplicate")});
    FakePipeline pipeline(state);
    const auto result =
        run_worker(pipeline, jsonl({update_message(),
                                    {{"type", "input_audio_buffer.append"}, {"audio", "AAA="}},
                                    {{"type", "session.close"}, {"mode", "cancel"}}}));
    const auto events = messages_of_type(result, "session.event");
    check(events.size() == 2 && events[0].at("text") == "current-1" &&
              events[1].at("text") == "current-2" &&
              result.messages.back()["stats"]["stale_events"] == 2,
          "older epochs and duplicate sequences never reach the transport output");
}

void test_unsupported_capabilities_are_explicit() {
    auto state = std::make_shared<SessionState>();
    BasicPipeline pipeline(state);
    auto tools_update = update_message("tool-update");
    tools_update["tools"] = json::array({{{"name", "tool"}}});
    const auto result = run_worker(
        pipeline,
        jsonl({update_message(),
               {{"type", "input_audio_buffer.commit"}},
               {{"type", "input_audio_buffer.clear"}},
               {{"type", "response.create"}},
               {{"type", "response.cancel"}},
               {{"type", "conversation.item.truncate"}, {"epoch", 1}, {"played_output_samples", 0}},
               {{"type", "conversation.item.create"},
                {"epoch", 1},
                {"call_id", "call"},
                {"output", "ok"}},
               tools_update,
               {{"type", "session.close"}, {"mode", "cancel"}}}));
    const auto codes = error_codes(result);
    check(std::count(codes.begin(), codes.end(), "unsupported") == 7 &&
              state->session_cancel_calls == 1,
          "missing realtime and tool capabilities return unsupported without fake success");

    NoSessionPipeline no_session;
    const auto absent = run_worker(
        no_session, jsonl({update_message(), {{"type", "session.close"}, {"mode", "cancel"}}}));
    check(contains_code(absent, "unsupported"),
          "a pipeline without persistent sessions is reported as unsupported");
}

void test_malformed_input_and_all_bounds() {
    auto state = std::make_shared<SessionState>();
    FakePipeline pipeline(state);
    trtmc::serve::RealtimeWorkerLimits limits;
    limits.max_line_bytes = 256;
    limits.max_queued_commands = 1;
    limits.max_queued_bytes = 256;
    limits.max_audio_bytes = 4;
    limits.max_session_audio_samples = 2;
    limits.max_session_config_bytes = 64;
    std::string input = "\n{\n[]\n";
    input += json({{"type", "unknown"}}).dump() + "\n";
    input += json({{"type", "input_audio_buffer.append"}, {"audio", "AAA="}}).dump() + "\n";
    auto wrong_rate = update_message();
    wrong_rate["input_sample_rate"] = 16000;
    input += wrong_rate.dump() + "\n";
    auto extra_field = update_message();
    extra_field["implementation_detail"] = "sensitive value";
    input += extra_field.dump() + "\n";
    input += update_message().dump() + "\n";
    input += json({{"type", "input_audio_buffer.commit"}, {"create_response", true}}).dump() + "\n";
    input += json({{"type", "input_audio_buffer.append"}, {"audio", "AA=="}}).dump() + "\n";
    input += json({{"type", "input_audio_buffer.append"}, {"audio", "@@@@"}}).dump() + "\n";
    input += json({{"type", "input_audio_buffer.append"}, {"audio", "AAAAAAAA"}}).dump() + "\n";
    input += json({{"type", "input_audio_buffer.append"}, {"audio", "AAAAAA=="}}).dump() + "\n";
    input += json({{"type", "input_audio_buffer.append"}, {"audio", "AAA="}}).dump() + "\n";
    input += update_message("late-update").dump() + "\n";
    input += std::string(300, 'x') + "\n";
    input += json({{"type", "session.close"}, {"mode", "cancel"}}).dump() + "\n";
    const auto result = run_worker(pipeline, input, limits);

    check(contains_code(result, "invalid_json") && contains_code(result, "invalid_message") &&
              contains_code(result, "unknown_event") && contains_code(result, "invalid_state") &&
              contains_code(result, "unsupported") && contains_code(result, "invalid_audio") &&
              contains_code(result, "audio_too_large") &&
              contains_code(result, "session_too_large") && contains_code(result, "line_too_large"),
          "malformed framing, schema, audio, session, and line bounds are explicit");
    check(result.raw.find("sensitive value") == std::string::npos && state->append_calls == 1 &&
              state->audio.size() == 2 && state->commit_calls == 0,
          "invalid payloads are neither echoed nor delivered to the native session");
    for (std::istringstream lines(result.raw); lines.good();) {
        std::string line;
        std::getline(lines, line);
        if (!line.empty())
            check(line.size() <= limits.max_line_bytes, "every output JSONL line is bounded");
    }
}

void test_native_event_and_config_bounds() {
    auto state = std::make_shared<SessionState>();
    state->append_event_batches.push_back(
        {event(trtmc::SpeechSessionEventKind::kAgentText, 1, 1, "one"),
         event(trtmc::SpeechSessionEventKind::kAgentText, 1, 2, "two")});
    FakePipeline pipeline(state);
    trtmc::serve::RealtimeWorkerLimits limits;
    limits.max_native_events_per_drain = 1;
    const auto queue_limited =
        run_worker(pipeline,
                   jsonl({update_message(),
                          {{"type", "input_audio_buffer.append"}, {"audio", "AAA="}},
                          {{"type", "session.close"}, {"mode", "cancel"}}}),
                   limits);
    check(contains_code(queue_limited, "event_queue_limit") &&
              messages_of_type(queue_limited, "session.event").empty(),
          "oversized native event batches are rejected as one bounded unit");

    auto text_state = std::make_shared<SessionState>();
    text_state->append_event_batches.push_back(
        {event(trtmc::SpeechSessionEventKind::kAgentText, 1, 1, "too-long")});
    FakePipeline text_pipeline(text_state);
    limits.max_native_events_per_drain = 8;
    limits.max_event_text_bytes = 3;
    const auto event_limited =
        run_worker(text_pipeline,
                   jsonl({update_message(),
                          {{"type", "input_audio_buffer.append"}, {"audio", "AAA="}},
                          {{"type", "session.close"}, {"mode", "cancel"}}}),
                   limits);
    check(contains_code(event_limited, "event_too_large"),
          "oversized native event content is replaced with a bounded error");

    auto audio_state = std::make_shared<SessionState>();
    audio_state->append_event_batches.push_back(
        {event(trtmc::SpeechSessionEventKind::kAgentAudio, 1, 1)});
    FakePipeline audio_pipeline(audio_state);
    limits.max_event_text_bytes = 256U * 1024U;
    const auto invalid_audio_event =
        run_worker(audio_pipeline,
                   jsonl({update_message(),
                          {{"type", "input_audio_buffer.append"}, {"audio", "AAA="}},
                          {{"type", "session.close"}, {"mode", "cancel"}}}),
                   limits);
    check(contains_code(invalid_audio_event, "invalid_event"),
          "empty or non-24 kHz native audio never reaches the JSONL host");

    auto config_state = std::make_shared<SessionState>();
    FakePipeline config_pipeline(config_state);
    limits.max_event_text_bytes = 256U * 1024U;
    limits.max_session_config_bytes = 12;
    auto large_update = update_message();
    large_update["instructions"] = "long configuration";
    const auto config_limited =
        run_worker(config_pipeline,
                   jsonl({large_update, {{"type", "session.close"}, {"mode", "cancel"}}}), limits);
    check(contains_code(config_limited, "session_too_large") &&
              config_state->normal_factory_calls == 0,
          "oversized session configuration is rejected before provider creation");
}

void test_errors_are_sanitized_and_factory_is_owned() {
    auto state = std::make_shared<SessionState>();
    state->throw_on_commit = true;
    state->throw_on_clear = true;
    state->throw_on_create_response = true;
    state->throw_on_response_cancel = true;
    state->throw_on_truncate = true;
    FakePipeline pipeline(state);
    const auto sanitized = run_worker(
        pipeline, jsonl({update_message(),
                         {{"type", "input_audio_buffer.commit"}, {"event_id", "commit-fail"}},
                         {{"type", "input_audio_buffer.clear"}, {"event_id", "clear-fail"}},
                         {{"type", "response.create"}, {"event_id", "create-fail"}},
                         {{"type", "response.cancel"}, {"event_id", "cancel-fail"}},
                         {{"type", "conversation.item.truncate"},
                          {"event_id", "truncate-fail"},
                          {"epoch", 1},
                          {"played_output_samples", 10}},
                         {{"type", "session.close"}, {"mode", "cancel"}}}));
    const auto sanitized_codes = error_codes(sanitized);
    check(std::count(sanitized_codes.begin(), sanitized_codes.end(), "session_error") == 5 &&
              sanitized.raw.find("sensitive") == std::string::npos &&
              messages_of_type(sanitized, "input_audio_buffer.committed").empty() &&
              messages_of_type(sanitized, "input_audio_buffer.cleared").empty() &&
              messages_of_type(sanitized, "response.cancelled").empty() &&
              messages_of_type(sanitized, "conversation.item.truncated").empty(),
          "control exceptions are sanitized and never receive a success ACK");

    int factory_calls = 0;
    std::istringstream source(jsonl({{{"type", "session.close"}, {"mode", "cancel"}}}));
    std::ostringstream output;
    const auto status = trtmc::serve::run_realtime_worker(
        [&]() -> std::unique_ptr<trtmc::IPipeline> {
            ++factory_calls;
            return std::make_unique<NoSessionPipeline>();
        },
        source, output);
    check(status == 0 && factory_calls == 1 &&
              output.str().find("session.ready") != std::string::npos,
          "factory overload creates and owns exactly one pipeline");

    std::istringstream null_source;
    std::ostringstream null_output;
    const auto null_status = trtmc::serve::run_realtime_worker(
        []() -> std::unique_ptr<trtmc::IPipeline> { return nullptr; }, null_source, null_output);
    check(null_status != 0 && null_output.str().find("pipeline_unavailable") != std::string::npos &&
              null_output.str().find("session.end") != std::string::npos,
          "factory startup failure returns a bounded sanitized terminal response");
}

void test_malformed_close_terminates_at_bounded_eof() {
    NoSessionPipeline pipeline;
    const auto result =
        run_worker(pipeline, jsonl({{{"type", "session.close"}, {"mode", "not-a-close-mode"}}}));
    check(result.status == 0 && contains_code(result, "invalid_message") &&
              result.messages.back().at("type") == "session.end" &&
              result.messages.back().at("reason") == "eof",
          "a malformed terminal frame cannot leave the worker waiting after its reader exits");
}

} // namespace

int main() {
    test_framing_audio_and_close_modes();
    test_tools_controls_and_stale_epoch();
    test_native_event_mapping_order_and_redaction();
    test_stale_native_events_are_suppressed();
    test_unsupported_capabilities_are_explicit();
    test_malformed_input_and_all_bounds();
    test_native_event_and_config_bounds();
    test_errors_are_sanitized_and_factory_is_owned();
    test_malformed_close_terminates_at_bounded_eof();
    return failures;
}
