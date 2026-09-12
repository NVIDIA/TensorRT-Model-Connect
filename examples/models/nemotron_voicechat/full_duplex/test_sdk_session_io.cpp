/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "session_io.h"
#include "trtmc/trtmc.hpp"

#include <filesystem>
#include <fstream>
#include <future>
#include <sstream>

namespace {
using namespace trtmc;
using namespace trtmc::examples::voicechat;
void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
Model load(const std::filesystem::path& root, const std::string& mode) {
    const auto path = root / (mode + "_voicechat_example.bundle");
    const std::string header =
        "{\"format\":1,\"family\":\"speech_fixture\",\"task\":\"" + mode +
        "\",\"backend\":\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}";
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write("BUNDLE\1\0", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((header.size() >> shift) & 255U));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("PLAN", 4);
    output.close();
    LoadOptions options;
    options.runtime_root = root.string();
    return Model::load(path.string(), options);
}
template <class Function>
void rejects(Function function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, message);
}
void exercise(const std::filesystem::path& root) {
    auto model = load(root, "example_voicechat");
    auto session = create_sdk_session(model, 16000, 48000, std::nullopt, 7);
    check(!session.info().system_prompt, "omitted prompt must remain distinct from empty prompt");
    auto effective_config = session.config();
    auto config = effective_config.c_entries();
    check(config.view().count == 6, "every requested voice option reaches the family");
    PlaybackQueue queue(48000 * 4);
    RunState state;
    TranscriptPrinter printer;
    volatile std::sig_atomic_t signal = 0;
    check(poll_sdk_session(session, 48000, queue, printer, state, 0) && !state.stopping(),
          "timeout must not finish the conversation");
    const float audio[]{0.25F, -0.5F};
    append_captured_audio(session, audio, state, signal);
    check(poll_sdk_session(session, 48000, queue, printer, state, 0),
          "active events continue session");
    // Event ownership is gone after poll. Playback must own its PCM copy.
    auto pcm = queue.wait_pop();
    check(pcm.kind == PlaybackQueueItemKind::kAudio && pcm.samples.size() == 2 &&
              pcm.samples[0] == float_to_pcm16(0.25F) && pcm.samples[1] == float_to_pcm16(-0.5F),
          "backpressured input is retried intact and playback owns converted output");
    check(queue.try_push({1, 2}), "queue accepts stale output before reset");
    session.reset();
    check(poll_sdk_session(session, 48000, queue, printer, state, 0),
          "reset preserves active session");
    check(queue.wait_pop().kind == PlaybackQueueItemKind::kFlush && queue.queued_samples() == 0,
          "reset flushes queued audio using the existing generation guard");
    auto pending = std::async(std::launch::async, [&] {
        return poll_sdk_session(session, 48000, queue, printer, state, -1);
    });
    session.cancel();
    check(pending.wait_for(std::chrono::seconds(1)) == std::future_status::ready && !pending.get(),
          "cancel unblocks the event reader and ends the session epoch");
    check(state.stopping() && queue.wait_pop().kind == PlaybackQueueItemKind::kFlush,
          "cancel flushes output and requests shutdown");
    session.close();
    check(model.task<TextContinuation>().run({"after"}).text() == "sync",
          "closing joined session releases model execution ownership");
    {
        auto empty = create_sdk_session(model, 16000, 48000, std::string{}, 0);
        check(empty.info().system_prompt && empty.info().system_prompt->empty(),
              "explicit empty system prompt must not become an omitted default");
        PlaybackQueue end_queue(64);
        RunState end_state;
        empty.finish_input();
        check(!poll_sdk_session(empty, 48000, end_queue, printer, end_state, 0),
              "input completion is not a timeout or a fabricated new turn");
    }
    {
        auto yielded =
            create_sdk_session(load(root, "example_voicechat_yield"), 16000, 48000, {}, 0);
        PlaybackQueue yielded_queue(64);
        RunState yielded_state;
        append_captured_audio(yielded, audio, yielded_state, signal);
        check(poll_sdk_session(yielded, 48000, yielded_queue, printer, yielded_state, 0),
              "barge-in yield flushes playback without ending the live session");
        check(yielded_queue.wait_pop().kind == PlaybackQueueItemKind::kFlush &&
                  yielded_queue.queued_samples() == 0,
              "yield discards stale queued audio");
    }
    rejects(
        [&] {
            (void)create_sdk_session(load(root, "example_voicechat_bad_format"), 16000, 48000, {},
                                     0);
        },
        "negotiated format mismatch must fail before opening session threads");
    {
        auto failed =
            create_sdk_session(load(root, "example_voicechat_failed"), 16000, 48000, {}, 0);
        PlaybackQueue failed_queue(64);
        RunState failed_state;
        rejects([&] { poll_sdk_session(failed, 48000, failed_queue, printer, failed_state, 0); },
                "Failed read state without an Error event must not be called success");
    }
    {
        auto bad =
            create_sdk_session(load(root, "example_voicechat_bad_audio_rate"), 16000, 48000, {}, 0);
        PlaybackQueue bad_queue(64);
        RunState bad_state;
        append_captured_audio(bad, audio, bad_state, signal);
        rejects([&] { poll_sdk_session(bad, 48000, bad_queue, printer, bad_state, 0); },
                "PCM output rate cannot silently change after endpoint negotiation");
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    std::ostringstream transcript;
    auto* original = std::cout.rdbuf(transcript.rdbuf());
    try {
        exercise(argv[1]);
        std::cout.rdbuf(original);
        check(transcript.str().find("heard 2 samples after 2 attempts") != std::string::npos,
              "transcript proves backpressure retry without input loss");
        check(transcript.str().find("agent> reply\n") != std::string::npos &&
                  transcript.str().find("replyreply") == std::string::npos,
              "delta and final transcript handling does not duplicate the response");
        std::cout << "ALL PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout.rdbuf(original);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
