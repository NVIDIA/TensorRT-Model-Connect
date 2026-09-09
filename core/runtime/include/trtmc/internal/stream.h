/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/language.h"
#include "trtmc/internal/text.h"

#include <memory>
#include <optional>

namespace trtmc::internal {

enum class StreamEventKind : std::uint32_t { Delta, Complete, Cancelled };

struct TextStreamEvent {
    StreamEventKind kind{StreamEventKind::Delta};
    std::string text_delta;
    std::vector<std::int32_t> token_ids;
    // Present only for Complete. The final result does not repeat a delta.
    std::optional<TextResult> final_result;
};

class ITextStream {
  public:
    virtual ~ITextStream() = default;
    // timeout_ms: -1 waits for an event, 0 polls, positive values bound waiting.
    // nullopt means timeout, never completion. Emit exactly one Complete or
    // Cancelled event. The destructor stops/joins any family-owned work.
    virtual std::optional<TextStreamEvent> next(std::int64_t timeout_ms) = 0;
    // Idempotent, thread-safe, and must wake a blocked next().
    virtual void cancel() noexcept = 0;
};

class IStreamingTextContinuation {
  public:
    static constexpr std::string_view kTask = "streaming_text_continuation";
    virtual ~IStreamingTextContinuation() = default;
    // Copy/parse every retained input/config before returning. The family
    // owns the session implementation and its execution, not the C wrapper.
    virtual std::unique_ptr<ITextStream> start(const TextContinuationRequest&, ConfigView) = 0;
};
class IStreamingImagesTextToText {
  public:
    static constexpr std::string_view kTask = "streaming_images_text_to_text";
    virtual ~IStreamingImagesTextToText() = default;
    virtual std::unique_ptr<ITextStream> start_images_text(const ImagesTextToTextRequest&,
                                                           ConfigView) = 0;
};

struct ConversationTextDelta {
    std::uint64_t part_index;
    std::string text;
};
struct ConversationReasoningDelta {
    std::uint64_t part_index;
    std::string text;
};
struct ConversationToolDelta {
    std::uint64_t part_index;
    std::optional<std::string> call_id;
    std::string name_delta, arguments_delta;
};
struct ConversationToolEnd {
    std::uint64_t part_index;
    ToolCallState state{ToolCallState::Unknown};
};
struct ConversationTokenDelta {
    std::vector<std::int32_t> token_ids;
};
struct ConversationComplete {
    ConversationResult result;
};
struct ConversationCancelled {};
using ConversationStreamEvent =
    std::variant<ConversationTextDelta, ConversationReasoningDelta, ConversationToolDelta,
                 ConversationToolEnd, ConversationTokenDelta, ConversationComplete,
                 ConversationCancelled>;
class IConversationStream {
  public:
    virtual ~IConversationStream() = default;
    virtual std::optional<ConversationStreamEvent> next(std::int64_t timeout_ms) = 0;
    virtual void cancel() noexcept = 0;
};
class IStreamingTextConversation {
  public:
    static constexpr std::string_view kTask = "streaming_text_conversation";
    virtual ~IStreamingTextConversation() = default;
    virtual std::unique_ptr<IConversationStream>
    start_text_conversation(const TextConversationRequest&, ConfigView) = 0;
};

class IStreamingVideoTextToText {
  public:
    static constexpr std::string_view kTask = "streaming_video_text_to_text";
    virtual ~IStreamingVideoTextToText() = default;
    virtual std::unique_ptr<ITextStream> start_video_text(const VideoTextToTextRequest&,
                                                          ConfigView) = 0;
};
class IStreamingImageVideoTextToText {
  public:
    static constexpr std::string_view kTask = "streaming_image_video_text_to_text";
    virtual ~IStreamingImageVideoTextToText() = default;
    virtual std::unique_ptr<ITextStream> start_image_video_text(const ImageVideoTextToTextRequest&,
                                                                ConfigView) = 0;
};
class IStreamingImagesTextConversation {
  public:
    static constexpr std::string_view kTask = "streaming_images_text_conversation";
    virtual ~IStreamingImagesTextConversation() = default;
    virtual std::unique_ptr<IConversationStream>
    start_images_conversation(const ImagesTextToTextRequest&, ConfigView) = 0;
};
class IStreamingVideoTextConversation {
  public:
    static constexpr std::string_view kTask = "streaming_video_text_conversation";
    virtual ~IStreamingVideoTextConversation() = default;
    virtual std::unique_ptr<IConversationStream>
    start_video_conversation(const VideoTextToTextRequest&, ConfigView) = 0;
};
} // namespace trtmc::internal
