/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/stream.h"

#include "api_internal.h"
#include "language_internal.h"
#include "trtmc/stream.h"

#include <memory>
#include <mutex>
#include <set>
#include <type_traits>

namespace trtmc::api {
namespace {

template <class Implementation>
struct StreamRun {
    using Interface = Implementation;
    StreamRun(std::unique_ptr<ModelSession> owner,
              std::unique_ptr<Implementation> implementation) noexcept
        : model(std::move(owner)), stream(std::move(implementation)) {}
    ~StreamRun() { stream->cancel(); }
    // Reverse destruction stops the family stream before making the model idle.
    std::unique_ptr<ModelSession> model;
    std::unique_ptr<Implementation> stream;
};
using TextStreamRun = StreamRun<internal::ITextStream>;
using ConversationStreamRun = StreamRun<internal::IConversationStream>;
template <class Run>
struct StreamHandle {
    using RunType = Run;
    explicit StreamHandle(std::shared_ptr<Run> value) : run(std::move(value)) {}
    std::mutex next_mutex, state_mutex;
    std::shared_ptr<Run> run;
};
struct ConversationValidation {
    struct Part {
        std::uint32_t kind;
        std::optional<std::string> id;
        bool has_name{false}, ended{false};
        internal::ToolCallState state{internal::ToolCallState::Unknown};
    };
    // Only transport identity/lifecycle metadata. Never accumulate text or arguments.
    std::vector<Part> parts;
    std::set<std::string> ids;
    static void check(bool value, const char* message) {
        if (!value)
            throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
    }
    Part& part(std::uint64_t index, std::uint32_t kind, bool may_create = true) {
        check(index < parts.size() || (may_create && index == parts.size()),
              "conversation part index is not in first-appearance order");
        if (index == parts.size())
            parts.push_back({kind, std::nullopt, false, false, internal::ToolCallState::Unknown});
        auto& value = parts[static_cast<std::size_t>(index)];
        check(value.kind == kind && !value.ended,
              "conversation part kind changed or received data after end");
        return value;
    }
    void accept(const internal::ConversationStreamEvent& event) {
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, internal::ConversationTextDelta>)
                    (void)part(value.part_index, TRTMC_CONVERSATION_TEXT);
                else if constexpr (std::is_same_v<T, internal::ConversationReasoningDelta>)
                    (void)part(value.part_index, TRTMC_CONVERSATION_REASONING);
                else if constexpr (std::is_same_v<T, internal::ConversationToolDelta>) {
                    auto& state = part(value.part_index, TRTMC_CONVERSATION_TOOL_CALL);
                    state.has_name = state.has_name || !value.name_delta.empty();
                    if (value.call_id) {
                        check(!value.call_id->empty(), "present tool-call ID is empty");
                        if (state.id)
                            check(*state.id == *value.call_id,
                                  "tool-call ID changed during stream");
                        else {
                            check(ids.insert(*value.call_id).second,
                                  "tool-call ID reused by another part");
                            state.id = value.call_id;
                        }
                    }
                } else if constexpr (std::is_same_v<T, internal::ConversationToolEnd>) {
                    auto& state = part(value.part_index, TRTMC_CONVERSATION_TOOL_CALL, false);
                    check(static_cast<std::uint32_t>(value.state) <= TRTMC_TOOL_CALL_MALFORMED,
                          "unknown tool-call completion state");
                    check(value.state != internal::ToolCallState::Complete ||
                              (state.id && state.has_name),
                          "completed tool call lacks emitted identity/name");
                    state.ended = true;
                    state.state = value.state;
                } else if constexpr (std::is_same_v<T, internal::ConversationComplete>) {
                    check(value.result.parts.size() == parts.size(),
                          "final conversation part count differs from observed parts");
                    for (std::size_t i = 0; i < parts.size(); ++i) {
                        const auto& final = value.result.parts[i];
                        const auto& state = parts[i];
                        const auto kind =
                            std::holds_alternative<internal::TextPart>(final)
                                ? TRTMC_CONVERSATION_TEXT
                                : (std::holds_alternative<internal::ReasoningPart>(final)
                                       ? TRTMC_CONVERSATION_REASONING
                                       : TRTMC_CONVERSATION_TOOL_CALL);
                        check(static_cast<std::uint32_t>(kind) == state.kind,
                              "final conversation part kind differs from stream");
                        if (const auto* call = std::get_if<internal::ToolCall>(&final)) {
                            check(state.ended && state.state == call->state,
                                  "final tool-call state differs from its end event");
                            check(state.id ? *state.id == call->call_id : call->call_id.empty(),
                                  "final tool-call ID differs from stream");
                        }
                    }
                }
            },
            event);
    }
};

struct TextEventStorage final : ResultStorage {
    explicit TextEventStorage(internal::TextStreamEvent event) : event(std::move(event)) {
        const auto kind = this->event.kind;
        if (kind == internal::StreamEventKind::Complete) {
            if (!this->event.final_result || !this->event.text_delta.empty() ||
                !this->event.token_ids.empty())
                throw ApiFailure{TRTMC_INTERNAL_ERROR, "malformed complete text event"};
            final = std::make_unique<TextResultStorage>(std::move(*this->event.final_result));
        } else if (kind == internal::StreamEventKind::Delta ||
                   kind == internal::StreamEventKind::Cancelled) {
            if (this->event.final_result ||
                (kind == internal::StreamEventKind::Cancelled &&
                 (!this->event.text_delta.empty() || !this->event.token_ids.empty())))
                throw ApiFailure{TRTMC_INTERNAL_ERROR, "malformed text stream event"};
        } else {
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "unknown text stream event kind"};
        }
    }
    internal::TextStreamEvent event;
    std::unique_ptr<TextResultStorage> final;
};
struct ConversationEventStorage final : ResultStorage {
    explicit ConversationEventStorage(internal::ConversationStreamEvent value)
        : event(std::move(value)) {
        std::visit(
            [&](auto& input) {
                using T = std::decay_t<decltype(input)>;
                if constexpr (std::is_same_v<T, internal::ConversationTextDelta>) {
                    view.kind = TRTMC_CONVERSATION_STREAM_TEXT;
                    view.content.text = {input.part_index, borrowed_string(input.text)};
                } else if constexpr (std::is_same_v<T, internal::ConversationReasoningDelta>) {
                    view.kind = TRTMC_CONVERSATION_STREAM_REASONING;
                    view.content.reasoning = {input.part_index, borrowed_string(input.text)};
                } else if constexpr (std::is_same_v<T, internal::ConversationToolDelta>) {
                    view.kind = TRTMC_CONVERSATION_STREAM_TOOL_DELTA;
                    view.content.tool_delta = {
                        input.part_index, input.call_id ? 1U : 0U,
                        input.call_id ? borrowed_string(*input.call_id) : trtmc_string_view{},
                        borrowed_string(input.name_delta), borrowed_string(input.arguments_delta)};
                } else if constexpr (std::is_same_v<T, internal::ConversationToolEnd>) {
                    view.kind = TRTMC_CONVERSATION_STREAM_TOOL_END;
                    view.content.tool_end = {input.part_index,
                                             static_cast<std::uint32_t>(input.state)};
                } else if constexpr (std::is_same_v<T, internal::ConversationTokenDelta>) {
                    view.kind = TRTMC_CONVERSATION_STREAM_TOKENS;
                    view.content.token_ids = {input.token_ids.data(), input.token_ids.size()};
                } else if constexpr (std::is_same_v<T, internal::ConversationComplete>) {
                    final = std::make_unique<language_detail::ConversationStorage>(
                        std::move(input.result));
                    view.kind = TRTMC_CONVERSATION_STREAM_COMPLETE;
                    view.content.complete = final->view;
                } else {
                    view.kind = TRTMC_CONVERSATION_STREAM_CANCELLED;
                }
            },
            event);
    }
    internal::ConversationStreamEvent event;
    std::unique_ptr<language_detail::ConversationStorage> final;
    trtmc_conversation_stream_event_view_v1 view{};
};

} // namespace
} // namespace trtmc::api

struct trtmc_text_stream : trtmc::api::StreamHandle<trtmc::api::TextStreamRun> {
    using StreamHandle::StreamHandle;
};
struct trtmc_conversation_stream : trtmc::api::StreamHandle<trtmc::api::ConversationStreamRun> {
    using StreamHandle::StreamHandle;
    trtmc::api::ConversationValidation validation;
};

namespace trtmc::api {
namespace {

template <class Handle>
std::shared_ptr<typename Handle::RunType> current_run(Handle* stream) {
    const std::lock_guard<std::mutex> lock(stream->state_mutex);
    return stream->run;
}

template <class Interface, class Handle, class Request, class Invoke>
trtmc_status start_stream(trtmc_model* model, const Request* request,
                          const trtmc_config_view_v1* config, Handle** out, trtmc_error** error,
                          Invoke invoke) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request && out, "stream request or output is null");
        std::unique_ptr<ModelSession> owner;
        std::unique_ptr<typename Handle::RunType::Interface> implementation;
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<Interface>(model, Interface::kTask);
            const ConvertedConfig options(config);
            validate_task_config(model_owner(model), internal::contract_key<Interface>(),
                                 options.view());
            owner = std::make_unique<ModelSession>(model);
            implementation = invoke(family, *request, options.view());
        }
        if (!implementation)
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned a null text stream"};
        auto run =
            std::make_shared<typename Handle::RunType>(std::move(owner), std::move(implementation));
        *out = new Handle(std::move(run));
    });
}
trtmc_status TRTMC_CALL start_text(trtmc_model* model,
                                   const trtmc_text_continuation_request_v1* request,
                                   const trtmc_config_view_v1* config, trtmc_text_stream** out,
                                   trtmc_error** error) noexcept {
    return start_stream<internal::IStreamingTextContinuation>(
        model, request, config, out, error, [](auto& family, const auto& input, auto options) {
            return family.start({text_source(input.prefix)}, options);
        });
}
trtmc_status TRTMC_CALL start_images_text(trtmc_model* model,
                                          const trtmc_images_text_to_text_request_v1* request,
                                          const trtmc_config_view_v1* config,
                                          trtmc_text_stream** out, trtmc_error** error) noexcept {
    return start_stream<internal::IStreamingImagesTextToText>(
        model, request, config, out, error, [](auto& family, const auto& input, auto options) {
            language_detail::LanguageInputs converted;
            return family.start_images_text(converted.convert(input), options);
        });
}
trtmc_status TRTMC_CALL start_text_conversation(trtmc_model* model,
                                                const trtmc_text_conversation_request_v1* request,
                                                const trtmc_config_view_v1* config,
                                                trtmc_conversation_stream** out,
                                                trtmc_error** error) noexcept {
    return start_stream<internal::IStreamingTextConversation>(
        model, request, config, out, error, [](auto& family, const auto& input, auto options) {
            language_detail::LanguageInputs converted;
            return family.start_text_conversation(converted.convert(input), options);
        });
}
#define MEDIA_STREAM_START(Function, Interface, Request, Handle, Method)                           \
    trtmc_status TRTMC_CALL Function(trtmc_model* model, const Request* request,                   \
                                     const trtmc_config_view_v1* config, Handle** out,             \
                                     trtmc_error** error) noexcept {                               \
        return start_stream<internal::Interface>(                                                  \
            model, request, config, out, error,                                                    \
            [](auto& family, const auto& input, auto options) {                                    \
                language_detail::LanguageInputs converted;                                         \
                return family.Method(converted.convert(input), options);                           \
            });                                                                                    \
    }
MEDIA_STREAM_START(start_video_text, IStreamingVideoTextToText, trtmc_video_text_to_text_request_v1,
                   trtmc_text_stream, start_video_text)
MEDIA_STREAM_START(start_image_video_text, IStreamingImageVideoTextToText,
                   trtmc_image_video_text_to_text_request_v1, trtmc_text_stream,
                   start_image_video_text)
MEDIA_STREAM_START(start_images_conversation, IStreamingImagesTextConversation,
                   trtmc_images_text_to_text_request_v1, trtmc_conversation_stream,
                   start_images_conversation)
MEDIA_STREAM_START(start_video_conversation, IStreamingVideoTextConversation,
                   trtmc_video_text_to_text_request_v1, trtmc_conversation_stream,
                   start_video_conversation)
#undef MEDIA_STREAM_START
bool terminal_event(const internal::TextStreamEvent& event) {
    return event.kind != internal::StreamEventKind::Delta;
}
bool terminal_event(const internal::ConversationStreamEvent& event) {
    return std::holds_alternative<internal::ConversationComplete>(event) ||
           std::holds_alternative<internal::ConversationCancelled>(event);
}

template <class Handle, class Pack>
trtmc_status next_stream(Handle* stream, std::int64_t timeout_ms, trtmc_result** out,
                         trtmc_error** error, Pack pack) noexcept {
    if (out)
        *out = nullptr;
    trtmc_status disposition = TRTMC_OK;
    std::unique_lock<std::mutex> next_lock;
    std::shared_ptr<typename Handle::RunType> run;
    bool entered = false, terminal = false;
    const auto status = guarded(error, [&] {
        require(stream && out, "stream or event output is null");
        require(timeout_ms >= -1, "stream timeout must be -1 or nonnegative");
        next_lock = std::unique_lock<std::mutex>(stream->next_mutex, std::try_to_lock);
        if (!next_lock.owns_lock())
            throw ApiFailure{TRTMC_BUSY, "another stream reader is active"};
        run = current_run(stream);
        if (!run) {
            disposition = TRTMC_END;
            return;
        }
        entered = true;
        auto event = run->stream->next(timeout_ms);
        if (!event) {
            if (timeout_ms == -1)
                throw ApiFailure{TRTMC_INTERNAL_ERROR, "blocking next returned no event"};
            disposition = TRTMC_AGAIN;
            return;
        }
        terminal = terminal_event(*event);
        auto result = std::unique_ptr<trtmc_result>(pack(std::move(*event)));
        *out = result.release();
    });
    // Capture the exception's message before destroying the family object.
    // Once next has been entered, an execution/packing error is fatal: the
    // consumed event cannot safely be replayed. Pre-call argument errors and
    // competing readers do not alter the active stream.
    if (run && (terminal || (entered && status != TRTMC_OK))) {
        const std::lock_guard<std::mutex> lock(stream->state_mutex);
        stream->run.reset();
    }
    // Keep next_lock until detach/cleanup completes. cancel's independent run
    // reference, if any, retains the family/model until that call completes.
    run.reset();
    return status == TRTMC_OK ? disposition : status;
}
trtmc_status TRTMC_CALL next_text(trtmc_text_stream* stream, std::int64_t timeout,
                                  trtmc_result** out, trtmc_error** error) noexcept {
    return next_stream(stream, timeout, out, error,
                       [](auto event) { return make_result<TextEventStorage>(std::move(event)); });
}
trtmc_status TRTMC_CALL next_conversation(trtmc_conversation_stream* stream, std::int64_t timeout,
                                          trtmc_result** out, trtmc_error** error) noexcept {
    return next_stream(stream, timeout, out, error, [stream](auto event) {
        stream->validation.accept(event);
        return make_result<ConversationEventStorage>(std::move(event));
    });
}

trtmc_status TRTMC_CALL text_event_view(const trtmc_result* result,
                                        trtmc_text_stream_event_view_v1* out,
                                        trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "stream event view output is null");
        const auto& storage = require_result<TextEventStorage>(result);
        switch (storage.event.kind) {
        case internal::StreamEventKind::Delta:
            out->kind = TRTMC_STREAM_DELTA;
            break;
        case internal::StreamEventKind::Complete:
            out->kind = TRTMC_STREAM_COMPLETE;
            break;
        case internal::StreamEventKind::Cancelled:
            out->kind = TRTMC_STREAM_CANCELLED;
            break;
        }
        out->text_delta = borrowed_string(storage.event.text_delta);
        out->token_ids = {storage.event.token_ids.data(), storage.event.token_ids.size()};
        if (storage.final)
            fill_text_result_view(*storage.final, &out->final_result);
    });
}

template <class Handle>
trtmc_status TRTMC_CALL cancel_stream(Handle* stream, trtmc_error** error) noexcept {
    return guarded(error, [&] {
        require(stream != nullptr, "stream is null");
        auto run = current_run(stream);
        if (run)
            run->stream->cancel();
    });
}

template <class Handle>
void TRTMC_CALL release_stream(Handle* stream) noexcept {
    delete stream;
}
trtmc_status TRTMC_CALL conversation_event_view(const trtmc_result* result,
                                                trtmc_conversation_stream_event_view_v1* out,
                                                trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "conversation event output is null");
        *out = require_result<ConversationEventStorage>(result).view;
    });
}

const trtmc_text_stream_api_v1 text_stream_api{{1, 0, sizeof(text_stream_api)},
                                               next_text,
                                               text_event_view,
                                               cancel_stream<trtmc_text_stream>,
                                               release_stream<trtmc_text_stream>};
const trtmc_conversation_stream_api_v1 conversation_stream_api{
    {1, 0, sizeof(conversation_stream_api)},
    next_conversation,
    conversation_event_view,
    cancel_stream<trtmc_conversation_stream>,
    release_stream<trtmc_conversation_stream>};
const trtmc_streaming_text_continuation_api_v1 text_api{
    {1, 0, sizeof(text_api)}, start_text, &text_stream_api};
const trtmc_streaming_images_text_to_text_api_v1 images_api{
    {1, 0, sizeof(images_api)}, start_images_text, &text_stream_api};
const trtmc_streaming_text_conversation_api_v1 conversation_api{
    {1, 0, sizeof(conversation_api)}, start_text_conversation, &conversation_stream_api};
const trtmc_streaming_video_text_to_text_api_v1 video_text_api{
    {1, 0, sizeof(video_text_api)}, start_video_text, &text_stream_api};
const trtmc_streaming_image_video_text_to_text_api_v1 image_video_text_api{
    {1, 0, sizeof(image_video_text_api)}, start_image_video_text, &text_stream_api};
const trtmc_streaming_images_text_conversation_api_v1 images_conversation_api{
    {1, 0, sizeof(images_conversation_api)}, start_images_conversation, &conversation_stream_api};
const trtmc_streaming_video_text_conversation_api_v1 video_conversation_api{
    {1, 0, sizeof(video_conversation_api)}, start_video_conversation, &conversation_stream_api};
const TaskBinding bindings[] = {
    {internal::IStreamingTextContinuation::kTask, 1, 0, &text_api.header},
    {internal::IStreamingImagesTextToText::kTask, 1, 0, &images_api.header},
    {internal::IStreamingTextConversation::kTask, 1, 0, &conversation_api.header},
    {internal::IStreamingVideoTextToText::kTask, 1, 0, &video_text_api.header},
    {internal::IStreamingImageVideoTextToText::kTask, 1, 0, &image_video_text_api.header},
    {internal::IStreamingImagesTextConversation::kTask, 1, 0, &images_conversation_api.header},
    {internal::IStreamingVideoTextConversation::kTask, 1, 0, &video_conversation_api.header},
};

} // namespace

Span<const TaskBinding> stream_task_bindings() noexcept {
    return bindings;
}

} // namespace trtmc::api
