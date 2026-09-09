/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/control.hpp"
#include "trtmc/stream.hpp"

#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <new>
#include <thread>

namespace {
thread_local bool fail_next_allocation = false;
void arm_allocation_failure() {
    fail_next_allocation = true;
}
} // namespace

// Test-only interposition: the family arms this immediately before returning
// an event, so the fault exercises C result packing rather than family work.
void* operator new(std::size_t size) {
    if (std::exchange(fail_next_allocation, false))
        throw std::bad_alloc();
    if (void* memory = std::malloc(size ? size : 1))
        return memory;
    throw std::bad_alloc();
}
// Keep the test-only replacement boundary intact under Release optimization.
[[gnu::noinline]] void operator delete(void* memory) noexcept {
    std::free(memory);
}
[[gnu::noinline]] void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

namespace {
int failures = 0;
void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}
template <class Function>
bool fails(trtmc_status code, Function&& function) {
    try {
        function();
    } catch (const trtmc::Error& e) {
        return e.code() == code;
    }
    return false;
}
void write_bundle(const std::filesystem::path& path, const std::string& mode) {
    const char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header = "{\"format\":1,\"family\":\"stream_fixture\",\"task\":\"" + mode +
                               "\",\"backend\":\"fake\",\"sections\":{}}";
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(magic, sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
}
trtmc::TextConversationRequest conversation_input() {
    return {{{trtmc::MessageRole::System, {trtmc::TextPart{"rules"}}},
             {trtmc::MessageRole::Assistant,
              {trtmc::ReasoningPart{"prior"},
               trtmc::ToolCall{"old", "lookup", "{}", trtmc::ToolCallState::Unknown}}},
             {trtmc::MessageRole::Tool, {trtmc::ToolResult{"old", "done", false}}},
             {trtmc::MessageRole::User, {trtmc::TextPart{"question"}}}},
            {{"lookup", "first", R"({"type":"object"})"},
             {"search", "second", R"({"type":"object"})"}}};
}
std::vector<trtmc::ConversationStreamEvent> drain(trtmc::ConversationStream& stream) {
    std::vector<trtmc::ConversationStreamEvent> result;
    while (auto event = stream.next())
        result.push_back(std::move(*event));
    return result;
}
template <class Request>
Request media_history(const trtmc::TextConversationRequest& input) {
    Request output;
    output.tools = input.tools;
    for (const auto& message : input.messages) {
        auto& copy = output.messages.emplace_back();
        copy.role = message.role;
        for (const auto& part : message.parts)
            std::visit([&](const auto& value) { copy.parts.emplace_back(value); }, part);
    }
    return output;
}
void media_conversation_tests(trtmc::Model& model) {
    float a[3]{0.25F, 0, 0}, b[3]{0.5F, 0, 0};
    const trtmc::ImageInput first{trtmc::Span<const float>{a}, 1, 1},
        second{trtmc::Span<const float>{b}, 1, 1};
    trtmc::VideoInput video{{first, second}, {2.0, 4.5}};
    auto plain_video = trtmc::VideoTextToTextRequest::from_parts(
        {trtmc::TextPart{"before"}, video, trtmc::TextPart{"after"}});
    auto video_stream = model.task<trtmc::StreamingVideoTextToText>().start(plain_video);
    a[0] = 0.75F;
    plain_video.messages.clear();
    auto before = video_stream.next();
    auto tail = video_stream.next();
    auto final = video_stream.next();
    check(before->text_delta() == "before" &&
              tail->text_delta() ==
                  "|video:0.250000:V[0.250000@2.000000;0.500000@4.500000;]|after" &&
              final->final_result()->text ==
                  "before|video:0.250000:V[0.250000@2.000000;0.500000@4.500000;]|after" &&
              !video_stream.next(),
          "plain video stream preserves temporal input and owns retained frames");
    video_stream.close();
    a[0] = 0.25F;
    auto mixed = trtmc::ImageVideoTextToTextRequest::from_parts(
        {trtmc::TextPart{"before"}, second, video, trtmc::TextPart{"after"}});
    auto mixed_stream = model.task<trtmc::StreamingImageVideoTextToText>().start(mixed);
    mixed.messages.clear();
    b[0] = 0.75F;
    check(mixed_stream.next()->text_delta() == "before" &&
              mixed_stream.next()->text_delta() ==
                  "|image_video:0.500000:V[0.250000@2.000000;0.500000@4.500000;]|after",
          "genuine mixed-image/video stream preserves distinct image role and temporal clip");
    (void)mixed_stream.next();
    check(!mixed_stream.next(), "mixed plain stream terminates once");
    mixed_stream.close();
    b[0] = 0.5F;
    auto images = media_history<trtmc::ImagesTextToTextRequest>(conversation_input());
    images.messages.back().parts.emplace_back(first);
    images.messages.back().parts.emplace_back(second);
    const auto once_images = model.task<trtmc::ImagesTextConversation>().run(images);
    auto images_stream = model.task<trtmc::StreamingImagesTextConversation>().start(images);
    images.messages.clear();
    images.tools.clear();
    a[0] = 0.75F;
    auto image_events = drain(images_stream);
    const auto image_final = image_events.back().final_result();
    const auto image_message = image_final->message();
    check(std::get<trtmc::TextPart>(image_message.parts[1]).text ==
                  "question -> answer!:I[0.250000]:I[0.500000]" &&
              std::get<trtmc::TextPart>(image_message.parts[1]).text ==
                  std::get<trtmc::TextPart>(once_images.message().parts[1]).text &&
              std::get<trtmc::ReasoningPart>(image_message.parts[0]).text == "prior -> thinking",
          "structured images one-shot/stream share complete reasoning/tool/media DTO without "
          "flattening");
    images_stream.close();
    a[0] = 0.25F;
    auto videos = media_history<trtmc::VideoTextToTextRequest>(conversation_input());
    videos.messages.back().parts.emplace_back(video);
    const auto once_video = model.task<trtmc::VideoTextConversation>().run(videos);
    auto structured_video = model.task<trtmc::StreamingVideoTextConversation>().start(videos);
    videos.messages.clear();
    videos.tools.clear();
    a[0] = 0.75F;
    auto video_events = drain(structured_video);
    const auto video_final = video_events.back().final_result();
    check(std::get<trtmc::TextPart>(video_final->message().parts[1]).text ==
                  "question -> answer!:V[0.250000@2.000000;0.500000@4.500000;]" &&
              std::get<trtmc::TextPart>(video_final->message().parts[1]).text ==
                  std::get<trtmc::TextPart>(once_video.message().parts[1]).text &&
              std::get<trtmc::ToolCall>(video_final->message().parts[2]).state ==
                  trtmc::ToolCallState::Complete,
          "structured video retains exact frame-time association, history and tool states");
    structured_video.close();
    auto missing = media_history<trtmc::ImagesTextToTextRequest>(conversation_input());
    check(fails(TRTMC_INVALID_ARGUMENT,
                [&] { (void)model.task<trtmc::StreamingImagesTextConversation>().start(missing); }),
          "structured images still require actual image operands");
    auto bad_role = media_history<trtmc::VideoTextToTextRequest>(conversation_input());
    bad_role.messages.back().parts.emplace_back(video);
    bad_role.messages.back().parts.emplace_back(trtmc::ReasoningPart{"not an assistant"});
    check(fails(TRTMC_INVALID_ARGUMENT,
                [&] { (void)model.task<trtmc::StreamingVideoTextConversation>().start(bad_role); }),
          "expanded media DTO rejects user-role reasoning injection");
    check(model.task<trtmc::TextContinuation>().run({"free"}).text() == "sync",
          "typed media input failure releases model reservation");
}
void conversation_tests(trtmc::Model& model, const std::filesystem::path& root) {
    auto task = model.task<trtmc::StreamingTextConversation>();
    auto request = conversation_input();
    const auto one_shot = model.task<trtmc::TextConversation>().run(request);
    auto stream = task.start(request);
    request.messages.clear();
    request.tools.clear();
    check(stream.poll(0).status == trtmc::StreamPollStatus::Timeout,
          "structured stream timeout is not completion");
    auto events = drain(stream);
    check(events.size() == 11 &&
              events.back().kind() == trtmc::ConversationStreamEventKind::Complete,
          "typed conversation emits independent events and exactly one terminal");
    std::array<std::string, 4> names, arguments;
    std::array<std::optional<std::string>, 4> ids;
    std::string reasoning, answer;
    std::vector<std::int32_t> tokens;
    bool delayed_id = false;
    int completed_calls = 0;
    for (const auto& event : events) {
        if (const auto delta = event.reasoning_delta())
            reasoning += delta->text;
        if (const auto delta = event.text_delta())
            answer += delta->text;
        if (const auto delta = event.tool_delta()) {
            names[delta->part_index] += delta->name_delta;
            arguments[delta->part_index] += delta->arguments_delta;
            if (delta->call_id)
                ids[delta->part_index] = std::string(*delta->call_id);
            else if (delta->part_index == 2 && !ids[2])
                delayed_id = true;
        }
        if (const auto ended = event.tool_end())
            completed_calls += ended->state == trtmc::ToolCallState::Complete;
        const auto delta_tokens = event.token_ids();
        if (!delta_tokens.empty())
            tokens.insert(tokens.end(), delta_tokens.begin(), delta_tokens.end());
    }
    check(reasoning == "prior -> thinking" && answer == "question -> answer!" && delayed_id &&
              completed_calls == 2,
          "structured roles/history survive start; reasoning is not answer text and delayed IDs "
          "remain explicit");
    const auto final = events.back().final_result();
    const auto message = final->message();
    const auto expected = one_shot.message();
    check(names[2] == "lookup" && arguments[2] == R"({"a":"x\"y"})" && ids[2] == "call-a" &&
              names[3] == "search" && arguments[3] == R"({"b":2})" && ids[3] == "call-b",
          "interleaved tool fragments retain partial escapes, names and correlated IDs");
    check(std::get<trtmc::ReasoningPart>(message.parts[0]).text ==
                  std::get<trtmc::ReasoningPart>(expected.parts[0]).text &&
              std::get<trtmc::ToolCall>(message.parts[2]).arguments_json == arguments[2] &&
              std::get<trtmc::ToolCall>(message.parts[2]).state == trtmc::ToolCallState::Complete,
          "stream and one-shot share the same structured DTO and final result");
    check(tokens == std::vector<std::int32_t>({31, 32}) &&
              final->finish_reason() == trtmc::FinishReason::ToolCalls &&
              final->usage().input_tokens == 11 && final->usage().output_tokens == 7 &&
              final->usage().total_tokens == 18,
          "family-reported usage/finish survive without guessing from two token IDs");
    stream.close();
    check(events[0].reasoning_delta()->text == "prior -> thinking",
          "owned structured events outlive stream close");
    for (const std::string fault :
         {"unknown", "zero_usage", "finish_other", "partial", "malformed_tool"}) {
        auto value = task.start(conversation_input(), {{"failure", fault}});
        auto output = drain(value);
        auto terminal = output.back().final_result();
        auto result = terminal->message();
        const auto& call = std::get<trtmc::ToolCall>(result.parts[2]);
        if (fault == "unknown") {
            check(terminal->finish_reason() == trtmc::FinishReason::Unknown &&
                      !terminal->usage().input_tokens && !terminal->usage().output_tokens &&
                      !terminal->usage().total_tokens &&
                      call.state == trtmc::ToolCallState::Unknown,
                  "Unknown tool/finish and absent usage stay unknown and absent");
        } else if (fault == "zero_usage") {
            check(terminal->usage().input_tokens == 0 && terminal->usage().output_tokens == 0 &&
                      terminal->usage().total_tokens == 0,
                  "explicit zero usage remains distinct from absence");
        } else if (fault == "finish_other") {
            check(terminal->finish_reason() == trtmc::FinishReason::Other &&
                      terminal->other_finish_reason() == "provider_exit",
                  "other finish reason preserves provider value");
        } else if (fault == "partial") {
            check(
                call.state == trtmc::ToolCallState::Incomplete &&
                    call.arguments_json == R"({"a":)" &&
                    terminal->finish_reason() == trtmc::FinishReason::Length,
                "length stop preserves incomplete tool arguments without claiming a complete call");
        } else
            check(call.state == trtmc::ToolCallState::Malformed && call.arguments_json == "{]",
                  "family-reported malformed arguments remain inspectable");
    }
    const auto close_library = [](void* value) noexcept {
        if (value)
            dlclose(value);
    };
    std::unique_ptr<void, decltype(close_library)> library(
        dlopen((root / "libtrtmc_model_stream_fixture.so").c_str(), RTLD_NOW | RTLD_LOCAL),
        close_library);
    if (!library)
        throw std::runtime_error("cannot inspect conversation fixture");
    using Probe = int (*)();
    using Hook = void (*)(void (*)());
    const auto waiting =
        reinterpret_cast<Probe>(dlsym(library.get(), "trtmc_test_stream_waiting_readers"));
    const auto hook =
        reinterpret_cast<Hook>(dlsym(library.get(), "trtmc_test_stream_allocation_hook"));
    if (!waiting || !hook)
        throw std::runtime_error("missing conversation fixture probes");
    auto cancellable = task.start(conversation_input(), {{"wait_for_cancel", true}});
    auto pending = std::async(std::launch::async, [&] { return cancellable.next(); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!waiting() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    check(waiting() != 0, "structured reader is genuinely blocked");
    check(fails(TRTMC_BUSY, [&] { (void)cancellable.poll(0); }),
          "structured competing reader returns BUSY");
    cancellable.cancel();
    if (pending.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
        std::abort();
    check(pending.get()->kind() == trtmc::ConversationStreamEventKind::Cancelled &&
              !cancellable.next(),
          "concurrent cancel wakes structured next and produces one terminal");
    for (const std::string fault : {"bad_index", "changed_id", "duplicate_id", "after_end",
                                    "bad_final", "throw", "blocking_timeout", "packing_oom"}) {
        if (fault == "packing_oom")
            hook(arm_allocation_failure);
        auto broken = task.start(conversation_input(), {{"failure", fault}});
        check(fails(fault == "packing_oom" ? TRTMC_OUT_OF_MEMORY : TRTMC_INTERNAL_ERROR,
                    [&] { (void)drain(broken); }),
              "malformed structured lifecycle or fatal packing error is reported");
        check(!broken.next(), "fatal structured error ends execution, not replay");
        check(model.task<trtmc::TextContinuation>().run({"free"}).text() == "sync",
              "fatal structured error releases model reservation");
    }
    hook(nullptr);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const auto root = std::filesystem::path(argv[1]);
    const auto path = root / "cpp-stream.bundle";
    const auto disabled_path = root / "cpp-stream-disabled.bundle";
    try {
        write_bundle(path, "enabled");
        write_bundle(disabled_path, "disabled");
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        auto model = trtmc::Model::load(path.string(), options);
        auto disabled = trtmc::Model::load(disabled_path.string(), options);
        check(!disabled.supports<trtmc::StreamingTextContinuation>(),
              "same class can disable streaming per bundle");
        auto streaming = model.task<trtmc::StreamingTextContinuation>();
        auto text = model.task<trtmc::TextContinuation>();
        check(fails(TRTMC_INVALID_CONFIG,
                    [&] { (void)streaming.start({"Hello"}, {{"unknown", true}}); }),
              "invalid config fails before session is retained");
        check(text.run({"ok"}).text() == "sync", "failed start releases the execution owner");
        auto stream = [&] {
            trtmc::TextContinuationRequest input{"Hello"};
            trtmc::Config config{{"suffix", " copied"}};
            auto value = streaming.start(input, config);
            input.prefix = std::string("overwritten");
            config = {};
            return value;
        }();
        check(stream.poll().status == trtmc::StreamPollStatus::Timeout,
              "timeout is not end of stream");
        check(fails(TRTMC_BUSY, [&] { (void)text.run({"blocked"}); }),
              "sync call cannot race a live stream");
        check(fails(TRTMC_BUSY, [&] { (void)streaming.start({"blocked"}); }),
              "second stream cannot race same model");
        check(fails(TRTMC_BUSY, [&] { model.lora_adapters().load("id", "path"); }),
              "LoRA mutation cannot race stream execution");
        auto delta = stream.next();
        check(delta && delta->kind() == trtmc::StreamEventKind::Delta &&
                  delta->text_delta() == "Hello copied" && delta->token_ids().size() == 1,
              "start owns input/config and delta is readable");
        auto final = stream.next();
        check(final && final->kind() == trtmc::StreamEventKind::Complete && final->final_result() &&
                  final->final_result()->text == "Hello copied" &&
                  final->final_result()->decode_ms == 0.5,
              "one complete event retains final text and timings");
        check(!stream.next(), "end follows terminal event");
        check(text.run({"ok"}).text() == "sync", "terminal cleanup makes model available again");
        stream.close();
        check(delta->text_delta() == "Hello copied" && final->final_result()->token_ids[0] == 42,
              "owned events outlive stream destruction");

        auto cancelled = streaming.start({"waiting"}, {{"wait_for_cancel", true}});
        void* library =
            dlopen((root / "libtrtmc_model_stream_fixture.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library)
            throw std::runtime_error("cannot inspect stream fixture");
        using Probe = int (*)();
        auto waiting = reinterpret_cast<Probe>(dlsym(library, "trtmc_test_stream_waiting_readers"));
        using SetAllocationHook = void (*)(void (*)());
        auto set_allocation_hook = reinterpret_cast<SetAllocationHook>(
            dlsym(library, "trtmc_test_stream_allocation_hook"));
        if (!waiting)
            throw std::runtime_error("missing stream probe");
        auto pending = std::async(std::launch::async, [&] { return cancelled.next(); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!waiting() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        check(waiting() != 0, "reader is genuinely blocked in family next");
        auto second_reader = std::async(std::launch::async, [&] {
            return fails(TRTMC_BUSY, [&] { (void)cancelled.poll(0); });
        });
        const bool reader_ready =
            second_reader.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        check(reader_ready, "poll(0) does not wait behind an active reader");
        cancelled.cancel();
        if (pending.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
            std::cerr << "FAIL: cancellation did not wake next\n";
            std::abort();
        }
        auto event = pending.get();
        check(second_reader.get(), "competing reader returns BUSY without terminating stream");
        check(event && event->kind() == trtmc::StreamEventKind::Cancelled,
              "concurrent cancel wakes reader with terminal event");
        check(text.run({"ok"}).text() == "sync", "execution owner can finish on another thread");
        cancelled.close();

        for (const std::string failure :
             {"throw", "malformed", "blocking_timeout", "packing_oom"}) {
            if (failure == "packing_oom") {
                if (!set_allocation_hook)
                    throw std::runtime_error("missing allocation hook");
                set_allocation_hook(arm_allocation_failure);
            }
            auto broken = streaming.start({"fault"}, {{"failure", failure}});
            const auto code = failure == "packing_oom" ? TRTMC_OUT_OF_MEMORY : TRTMC_INTERNAL_ERROR;
            check(fails(code, [&] { (void)broken.next(); }),
                  "fatal read/packing failure is reported");
            check(!broken.next(), "fatal failure consumes stream, subsequent read is END");
            check(text.run({"ok"}).text() == "sync",
                  "fatal read failure releases model execution state");
            broken.close();
        }
        set_allocation_hook(nullptr);
        dlclose(library);

        auto early = streaming.start({"unfinished"}, {{"wait_for_cancel", true}});
        std::thread closer([value = std::move(early)]() mutable { value.close(); });
        closer.join();
        check(text.run({"ok"}).text() == "sync",
              "early release on another thread cancels and restores model");
        auto detached = [&] {
            auto temporary = trtmc::Model::load(path.string(), options);
            return temporary.task<trtmc::StreamingTextContinuation>().start({"detached"});
        }();
        check(detached.next()->text_delta() == "detached!", "stream retains originating model");
        detached.close();
        float pixels[3]{0.25F, 0, 0};
        auto visual_request = trtmc::ImagesTextToTextRequest::from_parts(
            {trtmc::TextPart{"before"}, trtmc::ImageInput{trtmc::Span<const float>{pixels}, 1, 1},
             trtmc::TextPart{"after"}});
        auto visual = model.task<trtmc::StreamingImagesTextToText>().start(visual_request);
        pixels[0] = 0.75F;
        visual_request.messages.clear();
        auto visual_first = visual.next();
        auto visual_second = visual.next();
        auto visual_final = visual.next();
        check(visual_first->text_delta() == "before" &&
                  visual_second->text_delta() == "|image:0.250000|after",
              "typed image stream retains original image and ordered message input before start "
              "returns");
        check(visual_final->final_result()->text == "before|image:0.250000|after" && !visual.next(),
              "image stream reuses plain text events and one owned final result");
        visual.close();
        check(visual_first->text_delta() == "before", "image event outlives stream close");
        conversation_tests(model, root);
        media_conversation_tests(model);
    } catch (const std::exception& e) {
        std::cerr << "Unexpected: " << e.what() << '\n';
        ++failures;
    }
    std::filesystem::remove(path);
    std::filesystem::remove(disabled_path);
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
