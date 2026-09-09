/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/speech.hpp"

#include <chrono>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <set>
#include <thread>

namespace {
int failures;
void check(bool passed, const char* label) {
    if (!passed) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}
template <class F>
bool fails(trtmc_status code, F function) {
    try {
        function();
    } catch (const trtmc::Error& error) {
        return error.code() == code;
    }
    return false;
}
void write_bundle(const std::filesystem::path& path, const std::string& mode) {
    const char magic[]{'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header = "{\"format\":1,\"family\":\"speech_fixture\",\"task\":\"" + mode +
                               "\",\"backend\":\"fake\",\"sections\":{}}";
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.write(magic, sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
}
std::set<trtmc::SpeechEventKind> seen;
void record(const trtmc::SpeechReadResult& read) {
    if (read.events)
        for (std::size_t i = 0; i < read.events->size(); ++i)
            seen.insert(read.events->at(i).kind);
}
trtmc::SpeechDialogueRequest dialogue() {
    return {{std::nullopt, 2}, std::string("role prompt")};
}

void test_asr(const trtmc::Model& model) {
    auto provider = model.task<trtmc::StreamingSpeechTranscription>();
    auto stream = provider.create(
        {{std::nullopt, 2}, std::string{"fr"}},
        {{"text", "kept:"}, {"strings", std::vector<std::string>{"copied", "list"}}});
    const float values[]{0.1F, 0.2F, 0.3F, 0.4F};
    auto first = stream.accept_audio(values);
    auto second = stream.accept_audio(values, true);
    check(first.transcript().text == "kept:frames:2" && second.transcript().text == "kept:frames:4",
          "ASR chunks return cumulative snapshots");
    check(first.chunk_index() == 1 && second.chunk_index() == 2 && second.accepted_samples() == 4 &&
              second.is_final(),
          "ASR chunk/finality/per-channel sample counters preserved");
    check(stream.finish().transcript().text == second.transcript().text, "finish is idempotent");
    check(fails(TRTMC_INVALID_ARGUMENT, [&] { (void)stream.accept_audio(values); }),
          "finished ASR requires reset");
    check(fails(TRTMC_BUSY, [&] { (void)model.task<trtmc::TextContinuation>().run({"busy"}); }),
          "finished resettable ASR retains its model execution owner");
    auto info = stream.info();
    auto config = stream.config();
    check(info.input.sample_rate == 16000 && info.input.channels == 2 &&
              info.source_language == "fr",
          "ASR exposes resolved input info");
    stream.reset();
    check(stream.accept_audio(values).accepted_samples() == 2,
          "reset reuses ASR handle and clears state");
    stream.close();
    check(first.transcript().text == "kept:frames:2" && first.transcript().segments.size() == 1,
          "ASR results outlive stream release");
    const auto& entries = config.entries();
    check(entries.size() == 7 && entries[0].value.get<std::int64_t>() == 0 &&
              entries[1].value.get<double>() == 0 && !entries[2].value.get<bool>() &&
              entries[3].value.get<std::string>() == "kept:",
          "effective config preserves zero/false/text values");
    check(entries[4].value.get<std::vector<std::int64_t>>()[1] == 9007199254740993LL &&
              entries[5].value.get<std::vector<double>>().size() == 2 &&
              entries[6].value.get<std::vector<std::string>>() ==
                  std::vector<std::string>({"copied", "list"}),
          "effective config deep-owns all homogeneous list kinds");
    check(model.task<trtmc::TextContinuation>().run({"ok"}).text() == "sync",
          "ASR release makes model idle");
}

void test_tts(const trtmc::Model& model) {
    auto tts = model.task<trtmc::StreamingTextToSpeech>();
    const auto thread = std::this_thread::get_id();
    std::size_t chunks = 0;
    std::vector<float> copied;
    auto summary = tts.run({"Hello"}, [&](const trtmc::AudioView& audio) {
        check(std::this_thread::get_id() == thread, "TTS callbacks use the calling thread");
        check(audio.sample_rate == 24000 && audio.channels == 2,
              "TTS callback has actual PCM format");
        copied.insert(copied.end(), audio.samples.begin(), audio.samples.end());
        ++chunks;
        check(model.info().family == "speech_fixture",
              "TTS callback can query metadata without deadlock");
        check(fails(TRTMC_BUSY, [&] { (void)model.task<trtmc::TextContinuation>().run({"busy"}); }),
              "TTS reentrant same-model execution returns BUSY");
    });
    check(chunks == 3 && copied.size() == 12 && copied[8] == 2 &&
              summary.emitted_sample_count == 12 && summary.emitted_frame_count == 6 &&
              summary.outcome == trtmc::AudioDeliveryOutcome::Complete,
          "TTS final summary preserves delivered chunks and counts");
    auto stopped = tts.run({"Hello"}, [](const trtmc::AudioView&) { return false; });
    check(stopped.outcome == trtmc::AudioDeliveryOutcome::Stopped &&
              stopped.emitted_sample_count == 4,
          "callback stop returns a normal stopped summary");
    bool exception = false;
    try {
        (void)tts.run({"Hello"},
                      [](const trtmc::AudioView&) { throw std::logic_error("caller callback"); });
    } catch (const std::logic_error& error) {
        exception = std::string(error.what()) == "caller callback";
    }
    check(exception, "callback exception is rethrown after safe C unwind");
    check(model.task<trtmc::TextContinuation>().run({"ok"}).text() == "sync",
          "callback error releases execution owner");
}

void test_dialogue(const trtmc::Model& model) {
    const float values[]{0, 0, 0.25F, 0.5F};
    auto offline = model.task<trtmc::OfflineSpeechDialogue>().create(dialogue());
    check(fails(TRTMC_UNSUPPORTED, [&] { (void)offline.realtime(); }),
          "offline session disables realtime despite C++ inheritance");
    check(offline.append_audio(values), "offline session accepts one waveform");
    offline.finish_input();
    auto events = offline.wait_events();
    record(events);
    check(events.events && events.events->state() == trtmc::SpeechReadState::EpochEnded,
          "offline completion ends an epoch, not a request batch");
    bool waveform = false;
    for (std::size_t i = 0; i < events.events->size(); ++i) {
        const auto event = events.events->at(i);
        if (event.kind == trtmc::SpeechEventKind::AgentAudio) {
            waveform = event.audio.samples.size() == 4 && event.audio.samples[0] == 0 &&
                       event.audio.channels == 2 && event.media_end_sample == 2;
        }
    }
    check(waveform, "offline dialogue retains silence and frame-locked PCM");
    check(offline.take_events().status == trtmc::SpeechPollStatus::EpochEnd,
          "drained epoch is distinct from timeout");
    check(fails(TRTMC_BUSY, [&] { (void)model.task<trtmc::TextContinuation>().run({"busy"}); }),
          "ended dialogue keeps resettable model state reserved");
    offline.reset();
    record(offline.take_events());
    check(offline.append_audio(values), "same offline handle works after reset");
    offline.cancel();
    record(offline.take_events());
    offline.reset();
    record(offline.take_events());
    offline.close();

    auto live = model.task<trtmc::DuplexSpeechDialogue>().create(dialogue());
    check(live.take_events().status == trtmc::SpeechPollStatus::Timeout,
          "empty active dialogue is a timeout");
    check(fails(TRTMC_UNSUPPORTED, [&] { (void)live.tools(); }),
          "ordinary dialogue has no tool control");
    auto control = live.realtime();
    check(fails(TRTMC_INVALID_ARGUMENT, [&] { control.create_response(); }),
          "invalid turn control remains recoverable");
    std::vector<float> too_large(20);
    check(!live.append_audio({too_large.data(), too_large.size()}),
          "backpressure accepts no oversized input");
    check(live.append_audio(values), "smaller append works after backpressure");
    record(live.take_events());
    control.commit_input_turn(false);
    control.create_response();
    auto response = live.take_events();
    record(response);
    std::uint64_t epoch = 0;
    for (std::size_t i = 0; i < response.events->size(); ++i)
        epoch = response.events->at(i).epoch;
    check(fails(TRTMC_INVALID_ARGUMENT, [&] { control.truncate_response(epoch + 1, 0); }),
          "stale response cursor is rejected without closing session");
    control.truncate_response(epoch, 1);
    record(live.take_events());
    control.create_response();
    record(live.take_events());
    control.cancel_response();
    record(live.take_events());
    control.clear_pending_input();
    record(live.take_events());
    live.cancel();
    record(live.take_events());
    live.reset();
    record(live.take_events());
    live.close();
    check(fails(TRTMC_INVALID_ARGUMENT, [&] { control.create_response(); }),
          "control proxy after explicit close cannot use a freed handle");
}

void test_tools(const trtmc::Model& model) {
    trtmc::ToolSpeechDialogueRequest input{
        dialogue(),
        {{"lookup", "Lookup", "{}"}, {"other", "Other", "{}"}},
        {{"lookup", {"first ack", "chosen ack"}}},
        std::vector<std::string>{"first default", "chosen default"}};
    auto session = model.task<trtmc::ToolSpeechDialogue>().create(input);
    input.tools.clear();
    input.acknowledgements.clear();
    input.default_acknowledgements.reset();
    const float samples[]{0.1F, 0.2F};
    check(session.append_audio(samples), "tool session accepts input");
    session.realtime().commit_input_turn();
    auto read = session.take_events();
    record(read);
    std::vector<std::pair<std::uint64_t, trtmc::ToolCall>> calls;
    bool ack = false, fallback = false;
    for (std::size_t i = 0; i < read.events->size(); ++i) {
        auto event = read.events->at(i);
        if (event.tool_call)
            calls.push_back({event.epoch, *event.tool_call});
        ack |= event.text == "chosen ack";
        fallback |= event.text == "chosen default";
    }
    check(ack && fallback && calls.size() == 2,
          "all acknowledgement list entries reach family selection and survive input destruction");
    for (const auto& call : calls)
        check(call.second.state == trtmc::ToolCallState::Unknown,
              "speech preserves family Unknown tool state despite complete-looking fields");
    auto tools = session.tools();
    check(fails(TRTMC_INVALID_ARGUMENT,
                [&] {
                    tools.submit_tool_result(calls[0].first + 1,
                                             {calls[0].second.call_id, "stale", false});
                }),
          "tool result requires matching epoch");
    tools.submit_tool_result(calls[0].first, {calls[0].second.call_id, "recoverable error", true});
    auto error_event = session.take_events();
    record(error_event);
    check(error_event.events->state() == trtmc::SpeechReadState::Active,
          "error/is_final event does not imply fatal session end");
    check(fails(TRTMC_INVALID_ARGUMENT,
                [&] {
                    tools.submit_tool_result(calls[0].first,
                                             {calls[0].second.call_id, "duplicate", false});
                }),
          "duplicate tool result is recoverable user error");
    tools.submit_tool_result(calls[1].first, {calls[1].second.call_id, "result", false});
    record(session.take_events());
    session.cancel();
    record(session.take_events());
    session.reset();
    record(session.take_events());
    session.close();
}

void test_concurrency(const std::string& ordinary, const std::string& fatal_path,
                      const trtmc::LoadOptions& options, int (*waiting)()) {
    for (bool fatal : {false, true}) {
        auto model = trtmc::Model::load(fatal ? fatal_path : ordinary, options);
        auto session = model.task<trtmc::DuplexSpeechDialogue>().create(dialogue());
        auto future = std::async(std::launch::async, [&] { return session.wait_events(); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!waiting() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        check(waiting() > 0, "reader blocks in family before concurrent operation");
        check(fails(TRTMC_BUSY, [&] { (void)session.take_events(); }),
              "second reader returns BUSY without blocking");
        if (fatal) {
            const float values[]{0, 0};
            check(fails(TRTMC_INTERNAL_ERROR, [&] { (void)session.append_audio(values); }),
                  "unexpected logic error is fatal, not a masked user error");
        } else
            session.cancel();
        if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
            std::abort();
        auto result = future.get();
        record(result);
        if (fatal) {
            check(model.task<trtmc::TextContinuation>().run({"ok"}).text() == "sync",
                  "fatal mutator stops blocked read and releases model after cleanup");
            check(fails(TRTMC_INVALID_ARGUMENT, [&] { session.reset(); }),
                  "fatal session cannot be reset as if intact");
        } else {
            session.reset();
            record(session.take_events());
            check(fails(TRTMC_BUSY,
                        [&] { (void)model.task<trtmc::TextContinuation>().run({"busy"}); }),
                  "normal cancel/reset retains execution owner");
        }
        std::thread release([owned = std::move(session)]() mutable { owned.close(); });
        release.join();
        check(model.task<trtmc::TextContinuation>().run({"ok"}).text() == "sync",
              "release on another thread makes model idle");
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const std::filesystem::path root(argv[1]);
    const auto path = root / "speech-cpp.bundle", fatal = root / "speech-fatal.bundle",
               malformed = root / "speech-malformed.bundle";
    try {
        write_bundle(path, "ordinary");
        write_bundle(fatal, "fatal_append");
        write_bundle(malformed, "bad_event");
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        auto model = trtmc::Model::load(path.string(), options);
        test_asr(model);
        test_tts(model);
        test_dialogue(model);
        test_tools(model);
        void* library =
            dlopen((root / "libtrtmc_model_speech_fixture.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library)
            throw std::runtime_error("cannot inspect speech fixture");
        auto waiting =
            reinterpret_cast<int (*)()>(dlsym(library, "trtmc_test_speech_waiting_readers"));
        if (!waiting)
            throw std::runtime_error("missing speech fixture probe");
        test_concurrency(path.string(), fatal.string(), options, waiting);
        dlclose(library);
        auto bad_model = trtmc::Model::load(malformed.string(), options);
        auto bad_session = bad_model.task<trtmc::DuplexSpeechDialogue>().create(dialogue());
        check(fails(TRTMC_INTERNAL_ERROR, [&] { (void)bad_session.take_events(); }),
              "malformed consumed event closes failed session");
        check(bad_model.task<trtmc::TextContinuation>().run({"ok"}).text() == "sync",
              "packing failure releases execution owner");
        for (unsigned kind = 1; kind <= 16; ++kind)
            check(seen.count(static_cast<trtmc::SpeechEventKind>(kind)) != 0,
                  "every legacy speech event kind represented");
    } catch (const std::exception& error) {
        std::cerr << "Unexpected: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove(path);
    std::filesystem::remove(fatal);
    std::filesystem::remove(malformed);
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
