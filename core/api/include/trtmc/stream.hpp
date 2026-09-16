/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/core.hpp"
#include "trtmc/language.hpp"
#include "trtmc/stream.h"

namespace trtmc {

namespace detail {
template <class Traits>
class MediaStreamTask;
inline void validate_text_stream_api(const trtmc_text_stream_api_v1* api) {
    if (!api || api->header.major != 1 || api->header.minor != 0 ||
        api->header.byte_size < sizeof(*api))
        throw Error(TRTMC_VERSION_MISMATCH, "incompatible text stream reader API");
}
inline void validate_conversation_stream_api(const trtmc_conversation_stream_api_v1* api) {
    if (!api || api->header.major != 1 || api->header.minor != 0 ||
        api->header.byte_size < sizeof(*api))
        throw Error(TRTMC_VERSION_MISMATCH, "incompatible conversation stream reader");
}
} // namespace detail

enum class StreamEventKind : std::uint32_t {
    Delta = TRTMC_STREAM_DELTA,
    Complete = TRTMC_STREAM_COMPLETE,
    Cancelled = TRTMC_STREAM_CANCELLED
};
enum class StreamPollStatus { Event, Timeout, End };

class TextStreamEvent {
  public:
    TextStreamEvent(const TextStreamEvent&) = delete;
    TextStreamEvent& operator=(const TextStreamEvent&) = delete;
    TextStreamEvent(TextStreamEvent&& other) noexcept
        : owner_(std::move(other.owner_)), view_(std::exchange(other.view_, {})) {}
    TextStreamEvent& operator=(TextStreamEvent&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }
    StreamEventKind kind() const noexcept { return static_cast<StreamEventKind>(view_.kind); }
    std::string_view text_delta() const { return detail::string_view(view_.text_delta); }
    Span<const std::int32_t> token_ids() const noexcept {
        return {view_.token_ids.data, static_cast<std::size_t>(view_.token_ids.size)};
    }
    std::optional<TextResultView> final_result() const {
        if (kind() != StreamEventKind::Complete)
            return std::nullopt;
        return detail::text_result_view(view_.final_result);
    }

  private:
    friend class TextStream;
    TextStreamEvent(const trtmc_core_api_v1& api, trtmc_result* result) noexcept
        : owner_(api, result) {}
    detail::ResultOwner owner_;
    trtmc_text_stream_event_view_v1 view_{};
};

struct TextStreamPoll {
    StreamPollStatus status;
    std::optional<TextStreamEvent> event;
};

class TextStream {
  public:
    ~TextStream() { close(); }
    TextStream(const TextStream&) = delete;
    TextStream& operator=(const TextStream&) = delete;
    TextStream(TextStream&& other) noexcept
        : state_(std::move(other.state_)), api_(other.api_),
          handle_(std::exchange(other.handle_, nullptr)) {}
    TextStream& operator=(TextStream&& other) noexcept {
        if (this != &other) {
            close();
            state_ = std::move(other.state_);
            api_ = other.api_;
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    TextStreamPoll poll(std::int64_t timeout_ms = 0) {
        if (!handle_)
            return {StreamPollStatus::End, std::nullopt};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->next(handle_, timeout_ms, &raw, &error);
        TextStreamEvent event(state_->api, raw);
        if (status == TRTMC_AGAIN || status == TRTMC_END) {
            state_->api.error_release(error);
            if (raw)
                throw Error(TRTMC_INTERNAL_ERROR,
                            "stream returned an event with a non-event status");
            return {status == TRTMC_AGAIN ? StreamPollStatus::Timeout : StreamPollStatus::End,
                    std::nullopt};
        }
        detail::check(state_->api, status, error);
        error = nullptr;
        status = api_->event_view(raw, &event.view_, &error);
        detail::check(state_->api, status, error);
        return {StreamPollStatus::Event, std::move(event)};
    }
    std::optional<TextStreamEvent> next() {
        auto result = poll(-1);
        return std::move(result.event);
    }
    // May run concurrently with poll/next, but close/destruction must be
    // synchronized by the caller with every operation on this wrapper.
    void cancel() {
        if (!handle_)
            return;
        trtmc_error* error = nullptr;
        const auto status = api_->cancel(handle_, &error);
        detail::check(state_->api, status, error);
    }
    void close() noexcept {
        if (handle_)
            api_->release(std::exchange(handle_, nullptr));
        state_.reset();
    }

  private:
    friend class StreamingTextContinuation;
    friend class StreamingImagesTextToText;
    template <class Traits>
    friend class detail::MediaStreamTask;
    TextStream(std::shared_ptr<detail::ModelState> state, const trtmc_text_stream_api_v1* api,
               trtmc_text_stream* handle) noexcept
        : state_(std::move(state)), api_(api), handle_(handle) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_text_stream_api_v1* api_;
    trtmc_text_stream* handle_;
};

class StreamingTextContinuation {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_STREAMING_TEXT_CONTINUATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_streaming_text_continuation_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible text stream API table");
        detail::validate_text_stream_api(
            reinterpret_cast<const trtmc_streaming_text_continuation_api_v1*>(table)->stream_api);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    TextStream start(const TextContinuationRequest& request, const Config& config = {}) const {
        trtmc_text_continuation_request_v1 input{};
        if (const auto* text = std::get_if<std::string>(&request.prefix)) {
            input.prefix.kind = TRTMC_TEXT_UTF8;
            input.prefix.as.text = detail::c_string(*text);
        } else {
            const auto& ids = std::get<std::vector<std::int32_t>>(request.prefix);
            input.prefix.kind = TRTMC_TEXT_TOKEN_IDS;
            input.prefix.as.token_ids = {ids.data(), ids.size()};
        }
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_text_stream* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->start(state_->handle, &input, &options, &raw, &error);
        TextStream stream(state_, api_->stream_api, raw);
        detail::check(state_->api, status, error);
        return stream;
    }

  private:
    friend class Model;
    StreamingTextContinuation(std::shared_ptr<detail::ModelState> state,
                              const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_streaming_text_continuation_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_streaming_text_continuation_api_v1* api_;
};

class StreamingImagesTextToText {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_STREAMING_IMAGES_TEXT_TO_TEXT;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_streaming_images_text_to_text_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible images text stream table");
        detail::validate_text_stream_api(
            reinterpret_cast<const trtmc_streaming_images_text_to_text_api_v1*>(table)->stream_api);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    TextStream start(const ImagesTextToTextRequest& request, const Config& config = {}) const {
        detail::LanguageWireInputs storage;
        const auto input = storage.convert(request);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_text_stream* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->start(state_->handle, &input, &options, &raw, &error);
        TextStream stream(state_, api_->stream_api, raw);
        detail::check(state_->api, status, error);
        return stream;
    }

  private:
    friend class Model;
    StreamingImagesTextToText(std::shared_ptr<detail::ModelState> state,
                              const trtmc_api_header* table)
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_streaming_images_text_to_text_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_streaming_images_text_to_text_api_v1* api_;
};

enum class ConversationStreamEventKind : std::uint32_t {
    Text = TRTMC_CONVERSATION_STREAM_TEXT,
    Reasoning = TRTMC_CONVERSATION_STREAM_REASONING,
    ToolDelta = TRTMC_CONVERSATION_STREAM_TOOL_DELTA,
    ToolEnd = TRTMC_CONVERSATION_STREAM_TOOL_END,
    Tokens = TRTMC_CONVERSATION_STREAM_TOKENS,
    Complete = TRTMC_CONVERSATION_STREAM_COMPLETE,
    Cancelled = TRTMC_CONVERSATION_STREAM_CANCELLED
};
struct ConversationPartDeltaView {
    std::uint64_t part_index;
    std::string_view text;
};
struct ConversationToolDeltaView {
    std::uint64_t part_index;
    std::optional<std::string_view> call_id;
    std::string_view name_delta, arguments_delta;
};
struct ConversationToolEndView {
    std::uint64_t part_index;
    ToolCallState state;
};
class ConversationStreamEvent : public detail::ViewResult<trtmc_conversation_stream_event_view_v1> {
  public:
    using ViewResult::ViewResult;
    ConversationStreamEventKind kind() const {
        return static_cast<ConversationStreamEventKind>(view().kind);
    }
    std::optional<ConversationPartDeltaView> text_delta() const {
        const auto value = view();
        if (value.kind != TRTMC_CONVERSATION_STREAM_TEXT)
            return std::nullopt;
        return ConversationPartDeltaView{value.content.text.part_index,
                                         detail::string_view(value.content.text.text_delta)};
    }
    std::optional<ConversationPartDeltaView> reasoning_delta() const {
        const auto value = view();
        if (value.kind != TRTMC_CONVERSATION_STREAM_REASONING)
            return std::nullopt;
        return ConversationPartDeltaView{value.content.reasoning.part_index,
                                         detail::string_view(value.content.reasoning.text_delta)};
    }
    std::optional<ConversationToolDeltaView> tool_delta() const {
        const auto value = view();
        if (value.kind != TRTMC_CONVERSATION_STREAM_TOOL_DELTA)
            return std::nullopt;
        const auto& input = value.content.tool_delta;
        return ConversationToolDeltaView{
            input.part_index,
            input.has_call_id ? std::optional<std::string_view>{detail::string_view(input.call_id)}
                              : std::nullopt,
            detail::string_view(input.name_delta), detail::string_view(input.arguments_delta)};
    }
    std::optional<ConversationToolEndView> tool_end() const {
        const auto value = view();
        if (value.kind != TRTMC_CONVERSATION_STREAM_TOOL_END)
            return std::nullopt;
        return ConversationToolEndView{value.content.tool_end.part_index,
                                       static_cast<ToolCallState>(value.content.tool_end.state)};
    }
    Span<const std::int32_t> token_ids() const {
        const auto value = view();
        if (value.kind != TRTMC_CONVERSATION_STREAM_TOKENS)
            return {};
        return {value.content.token_ids.data,
                static_cast<std::size_t>(value.content.token_ids.size)};
    }
    // Borrowed view: the event owns the terminal snapshot.
    std::optional<ConversationResultView> final_result() const {
        const auto value = view();
        if (value.kind != TRTMC_CONVERSATION_STREAM_COMPLETE)
            return std::nullopt;
        return ConversationResultView{value.content.complete};
    }
};
struct ConversationStreamPoll {
    StreamPollStatus status;
    std::optional<ConversationStreamEvent> event;
};
class ConversationStream {
  public:
    ~ConversationStream() { close(); }
    ConversationStream(const ConversationStream&) = delete;
    ConversationStream& operator=(const ConversationStream&) = delete;
    ConversationStream(ConversationStream&& other) noexcept
        : state_(std::move(other.state_)), api_(other.api_),
          handle_(std::exchange(other.handle_, nullptr)) {}
    ConversationStream& operator=(ConversationStream&& other) noexcept {
        if (this != &other) {
            close();
            state_ = std::move(other.state_);
            api_ = other.api_;
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    ConversationStreamPoll poll(std::int64_t timeout_ms = 0) {
        if (!handle_)
            return {StreamPollStatus::End, std::nullopt};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->next(handle_, timeout_ms, &raw, &error);
        detail::ResultOwner owner(state_->api, raw);
        if (status == TRTMC_AGAIN || status == TRTMC_END) {
            state_->api.error_release(error);
            if (raw)
                throw Error(TRTMC_INTERNAL_ERROR,
                            "conversation stream returned event with non-event status");
            return {status == TRTMC_AGAIN ? StreamPollStatus::Timeout : StreamPollStatus::End,
                    std::nullopt};
        }
        detail::check(state_->api, status, error);
        return {StreamPollStatus::Event,
                ConversationStreamEvent(std::move(owner), api_->event_view)};
    }
    std::optional<ConversationStreamEvent> next() { return std::move(poll(-1).event); }
    void cancel() {
        if (!handle_)
            return;
        trtmc_error* error = nullptr;
        const auto status = api_->cancel(handle_, &error);
        detail::check(state_->api, status, error);
    }
    void close() noexcept {
        if (handle_)
            api_->release(std::exchange(handle_, nullptr));
        state_.reset();
    }

  private:
    friend class StreamingTextConversation;
    template <class Traits>
    friend class detail::MediaStreamTask;
    ConversationStream(std::shared_ptr<detail::ModelState> state,
                       const trtmc_conversation_stream_api_v1* api,
                       trtmc_conversation_stream* handle) noexcept
        : state_(std::move(state)), api_(api), handle_(handle) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_conversation_stream_api_v1* api_;
    trtmc_conversation_stream* handle_;
};
class StreamingTextConversation {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_STREAMING_TEXT_CONVERSATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_streaming_text_conversation_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible conversation stream factory");
        const auto* reader =
            reinterpret_cast<const trtmc_streaming_text_conversation_api_v1*>(table)->stream_api;
        detail::validate_conversation_stream_api(reader);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    ConversationStream start(const TextConversationRequest& request,
                             const Config& config = {}) const {
        detail::LanguageWireInputs storage;
        const auto input = storage.convert(request);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_conversation_stream* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->start(state_->handle, &input, &options, &raw, &error);
        ConversationStream stream(state_, api_->stream_api, raw);
        detail::check(state_->api, status, error);
        return stream;
    }

  private:
    friend class Model;
    StreamingTextConversation(std::shared_ptr<detail::ModelState> state,
                              const trtmc_api_header* table)
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_streaming_text_conversation_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_streaming_text_conversation_api_v1* api_;
};
namespace detail {
template <class Traits>
class MediaStreamTask {
  public:
    static constexpr std::string_view kTask = Traits::id;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    using Request = typename Traits::Request;
    using Stream = typename Traits::Stream;
    using Table = typename Traits::Table;
    using Handle = typename Traits::Handle;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 || table->byte_size < sizeof(Table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible typed media stream table");
        const auto* reader = reinterpret_cast<const Table*>(table)->stream_api;
        if constexpr (std::is_same_v<Stream, TextStream>)
            validate_text_stream_api(reader);
        else
            validate_conversation_stream_api(reader);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    Stream start(const Request& request, const Config& config = {}) const {
        LanguageWireInputs storage;
        const auto input = storage.convert(request);
        auto entries = config.c_entries();
        auto options = entries.view();
        Handle* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->start(state_->handle, &input, &options, &raw, &error);
        Stream stream(state_, api_->stream_api, raw);
        detail::check(state_->api, status, error);
        return stream;
    }

  private:
    friend class trtmc::Model;
    MediaStreamTask(std::shared_ptr<ModelState> state, const trtmc_api_header* table)
        : state_(std::move(state)), api_(reinterpret_cast<const Table*>(table)) {}
    std::shared_ptr<ModelState> state_;
    const Table* api_;
};
struct StreamingVideoTextTraits {
    static constexpr std::string_view id = TRTMC_TASK_STREAMING_VIDEO_TEXT_TO_TEXT;
    using Request = VideoTextToTextRequest;
    using Stream = TextStream;
    using Table = trtmc_streaming_video_text_to_text_api_v1;
    using Handle = trtmc_text_stream;
};
struct StreamingImageVideoTextTraits {
    static constexpr std::string_view id = TRTMC_TASK_STREAMING_IMAGE_VIDEO_TEXT_TO_TEXT;
    using Request = ImageVideoTextToTextRequest;
    using Stream = TextStream;
    using Table = trtmc_streaming_image_video_text_to_text_api_v1;
    using Handle = trtmc_text_stream;
};
struct StreamingImagesConversationTraits {
    static constexpr std::string_view id = TRTMC_TASK_STREAMING_IMAGES_TEXT_CONVERSATION;
    using Request = ImagesTextToTextRequest;
    using Stream = ConversationStream;
    using Table = trtmc_streaming_images_text_conversation_api_v1;
    using Handle = trtmc_conversation_stream;
};
struct StreamingVideoConversationTraits {
    static constexpr std::string_view id = TRTMC_TASK_STREAMING_VIDEO_TEXT_CONVERSATION;
    using Request = VideoTextToTextRequest;
    using Stream = ConversationStream;
    using Table = trtmc_streaming_video_text_conversation_api_v1;
    using Handle = trtmc_conversation_stream;
};
} // namespace detail
using StreamingVideoTextToText = detail::MediaStreamTask<detail::StreamingVideoTextTraits>;
using StreamingImageVideoTextToText =
    detail::MediaStreamTask<detail::StreamingImageVideoTextTraits>;
using StreamingImagesTextConversation =
    detail::MediaStreamTask<detail::StreamingImagesConversationTraits>;
using StreamingVideoTextConversation =
    detail::MediaStreamTask<detail::StreamingVideoConversationTraits>;
} // namespace trtmc
