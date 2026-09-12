/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/core.hpp"
#include "trtmc/language.h"
#include "trtmc/text.hpp"
#include "trtmc/tools.hpp"
#include "trtmc/video.hpp"

namespace trtmc {

// Own text, history and descriptor arrays; media buffers remain borrowed until
// run returns. Separate closed part variants keep each Task's inputs explicit.
enum class MessageRole : uint32_t { System = 1, Developer = 2, User = 3, Assistant = 4, Tool = 5 };
enum class FinishReason : std::uint32_t {
    Unknown = TRTMC_FINISH_UNKNOWN,
    Stop = TRTMC_FINISH_STOP,
    Length = TRTMC_FINISH_LENGTH,
    ToolCalls = TRTMC_FINISH_TOOL_CALLS,
    ContentFilter = TRTMC_FINISH_CONTENT_FILTER,
    Other = TRTMC_FINISH_OTHER
};
struct TokenUsage {
    std::optional<std::uint64_t> input_tokens, output_tokens, total_tokens;
};
struct TextPart {
    std::string text;
};
struct ReasoningPart {
    std::string text;
};
struct AlignedAudioVideoInput {
    VideoInput video;
    AudioView audio;
    std::optional<double> audio_start_seconds{};
};
template <class Part>
struct MediaMessage {
    MessageRole role{MessageRole::User};
    std::vector<Part> parts;
};
using ImagesTextPart = std::variant<TextPart, ImageInput, ReasoningPart, ToolCall, ToolResult>;
using ImagesTextMessage = MediaMessage<ImagesTextPart>;
using VideoTextPart = std::variant<TextPart, VideoInput, ReasoningPart, ToolCall, ToolResult>;
using VideoTextMessage = MediaMessage<VideoTextPart>;
using ImageVideoTextPart = std::variant<TextPart, ImageInput, VideoInput>;
using ImageVideoTextMessage = MediaMessage<ImageVideoTextPart>;
using AudioTextPart = std::variant<TextPart, AudioView>;
using AudioTextMessage = MediaMessage<AudioTextPart>;
using ImageAudioTextPart = std::variant<TextPart, ImageInput, AudioView>;
using ImageAudioTextMessage = MediaMessage<ImageAudioTextPart>;
using AudioVideoTextPart = std::variant<TextPart, AlignedAudioVideoInput>;
using AudioVideoTextMessage = MediaMessage<AudioVideoTextPart>;
struct ImagesTextToTextRequest {
    std::vector<ImagesTextMessage> messages;
    std::vector<ToolDefinition> tools{};
    static ImagesTextToTextRequest from_parts(std::vector<ImagesTextPart> parts) {
        return {{{MessageRole::User, std::move(parts)}}};
    }
};
struct VideoTextToTextRequest {
    std::vector<VideoTextMessage> messages;
    std::vector<ToolDefinition> tools{};
    static VideoTextToTextRequest from_parts(std::vector<VideoTextPart> parts) {
        return {{{MessageRole::User, std::move(parts)}}};
    }
};
struct ImageVideoTextToTextRequest {
    std::vector<ImageVideoTextMessage> messages;
    static ImageVideoTextToTextRequest from_parts(std::vector<ImageVideoTextPart> parts) {
        return {{{MessageRole::User, std::move(parts)}}};
    }
};
struct AudioTextToTextRequest {
    std::vector<AudioTextMessage> messages;
    static AudioTextToTextRequest from_parts(std::vector<AudioTextPart> parts) {
        return {{{MessageRole::User, std::move(parts)}}};
    }
};
struct ImageAudioToTextRequest {
    std::vector<ImageAudioTextMessage> messages;
    static ImageAudioToTextRequest from_parts(std::vector<ImageAudioTextPart> parts) {
        return {{{MessageRole::User, std::move(parts)}}};
    }
};
struct AudioVideoTextToTextRequest {
    std::vector<AudioVideoTextMessage> messages;
    static AudioVideoTextToTextRequest from_parts(std::vector<AudioVideoTextPart> parts) {
        return {{{MessageRole::User, std::move(parts)}}};
    }
};
struct ImageAudioTextToTextRequest {
    std::vector<ImageAudioTextMessage> messages;
    static ImageAudioTextToTextRequest from_parts(std::vector<ImageAudioTextPart> parts) {
        return {{{MessageRole::User, std::move(parts)}}};
    }
};
struct ImageAudioTextToTextSpeechResponseRequest {
    std::vector<ImageAudioTextMessage> messages;
    static ImageAudioTextToTextSpeechResponseRequest
    from_parts(std::vector<ImageAudioTextPart> parts) {
        return {{{MessageRole::User, std::move(parts)}}};
    }
};

using ConversationPart = std::variant<TextPart, ReasoningPart, ToolCall, ToolResult>;
struct ConversationMessage {
    MessageRole role{MessageRole::User};
    std::vector<ConversationPart> parts;
};
struct TextConversationRequest {
    std::vector<ConversationMessage> messages;
    std::vector<ToolDefinition> tools{};
};
struct TextLabelClassificationRequest {
    std::string text;
};
struct TextPairLabelClassificationRequest {
    std::string first, second;
};
struct LanguageTokenSequence {
    Span<const int32_t> token_ids;
    Span<const uint8_t> attention_mask{}; // Empty means all valid.
};
struct TextEncoderDecoderHiddenStatesRequest {
    LanguageTokenSequence source, decoder;
};

using AudioTextConversationPart =
    std::variant<TextPart, AudioView, ReasoningPart, ToolCall, ToolResult>;
struct AudioTextConversationRequest {
    std::vector<MediaMessage<AudioTextConversationPart>> messages;
    std::vector<ToolDefinition> tools{};
    static AudioTextConversationRequest from_parts(std::vector<AudioTextConversationPart> parts) {
        return {{{MessageRole::User, std::move(parts)}}};
    }
};
using ImageAudioTextConversationPart =
    std::variant<TextPart, ImageInput, AudioView, ReasoningPart, ToolCall, ToolResult>;
struct ImageAudioTextConversationRequest {
    std::vector<MediaMessage<ImageAudioTextConversationPart>> messages;
    std::vector<ToolDefinition> tools{};
    static ImageAudioTextConversationRequest
    from_parts(std::vector<ImageAudioTextConversationPart> parts) {
        return {{{MessageRole::User, std::move(parts)}}};
    }
};
using TextImagesVideoConversationRequest =
    std::variant<TextConversationRequest, ImagesTextToTextRequest, VideoTextToTextRequest>;
using TextImagesAudioConversationRequest =
    std::variant<TextConversationRequest, ImagesTextToTextRequest, AudioTextConversationRequest,
                 ImageAudioTextConversationRequest>;
class TextSpeechResult : public detail::ViewResult<trtmc_text_speech_result_view_v1> {
  public:
    using ViewResult::ViewResult;
    std::string_view text() const { return detail::string_view(view().text.text); }
    AudioView speech() const {
        const auto audio = view().speech.audio;
        return {{audio.samples, static_cast<size_t>(audio.sample_count)},
                audio.sample_rate,
                audio.channels};
    }
};
class ConversationResultView {
  public:
    explicit ConversationResultView(trtmc_conversation_result_view_v1 value) : value_(value) {}
    // Copy the structured assistant response into caller-owned history. No tool
    // is executed, and reasoning is never flattened into the final text.
    ConversationMessage message() const {
        ConversationMessage output{MessageRole::Assistant, {}};
        const auto& data = value_;
        for (uint64_t i = 0; i < data.part_count; ++i) {
            const auto& part = data.parts[i];
            switch (part.kind) {
            case TRTMC_CONVERSATION_TEXT:
                output.parts.emplace_back(
                    TextPart{std::string(detail::string_view(part.content.text))});
                break;
            case TRTMC_CONVERSATION_REASONING:
                output.parts.emplace_back(
                    ReasoningPart{std::string(detail::string_view(part.content.text))});
                break;
            case TRTMC_CONVERSATION_TOOL_CALL:
                output.parts.emplace_back(detail::copy_tool_call(part.content.tool_call));
                break;
            default:
                throw Error(TRTMC_INTERNAL_ERROR, "invalid assistant result part");
            }
        }
        return output;
    }
    FinishReason finish_reason() const noexcept {
        return static_cast<FinishReason>(value_.finish_reason);
    }
    std::string_view other_finish_reason() const {
        return detail::string_view(value_.other_finish_reason);
    }
    TokenUsage usage() const {
        const auto& input = value_.usage;
        return {input.has_input_tokens ? std::optional<std::uint64_t>{input.input_tokens}
                                       : std::nullopt,
                input.has_output_tokens ? std::optional<std::uint64_t>{input.output_tokens}
                                        : std::nullopt,
                input.has_total_tokens ? std::optional<std::uint64_t>{input.total_tokens}
                                       : std::nullopt};
    }
    Span<const std::int32_t> token_ids() const noexcept {
        return {value_.token_ids.data, static_cast<std::size_t>(value_.token_ids.size)};
    }
    const trtmc_conversation_result_view_v1& c_view() const noexcept { return value_; }

  private:
    trtmc_conversation_result_view_v1 value_;
};
class ConversationResult : public detail::ViewResult<trtmc_conversation_result_view_v1> {
  public:
    using ViewResult::ViewResult;
    ConversationResultView result() const { return ConversationResultView{view()}; }
    ConversationMessage message() const { return result().message(); }
    FinishReason finish_reason() const { return result().finish_reason(); }
    TokenUsage usage() const { return result().usage(); }
};
class GeneratedLabelResult : public detail::ViewResult<trtmc_generated_label_result_view_v1> {
  public:
    using ViewResult::ViewResult;
    std::string_view raw_label() const { return detail::string_view(view().raw_label); }
    int64_t label_index() const { return view().label_index; }
    std::vector<std::string_view> vocabulary() const {
        const auto names = view().vocabulary;
        std::vector<std::string_view> out;
        for (uint64_t i = 0; i < names.size; ++i)
            out.push_back(detail::string_view(names.data[i]));
        return out;
    }
};
using EncoderDecoderStatesResult = detail::ViewResult<trtmc_encoder_decoder_states_result_view_v1>;

namespace detail {
class LanguageWireInputs {
  public:
    trtmc_images_text_part_v1 part(const ImagesTextPart& input) {
        trtmc_images_text_part_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextPart>) {
                    out.kind = TRTMC_MEDIA_TEXT;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, ImageInput>) {
                    out.kind = TRTMC_MEDIA_IMAGE;
                    out.content.image = value.wire;
                } else if constexpr (std::is_same_v<T, ReasoningPart>) {
                    out.kind = TRTMC_MEDIA_REASONING;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, ToolCall>) {
                    out.kind = TRTMC_MEDIA_TOOL_CALL;
                    out.content.tool_call = c_tool_call(value);
                } else if constexpr (std::is_same_v<T, ToolResult>) {
                    out.kind = TRTMC_MEDIA_TOOL_RESULT;
                    out.content.tool_result = c_tool_result(value);
                }
            },
            input);
        return out;
    }
    trtmc_video_text_part_v1 part(const VideoTextPart& input) {
        trtmc_video_text_part_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextPart>) {
                    out.kind = TRTMC_MEDIA_TEXT;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, VideoInput>) {
                    out.kind = TRTMC_MEDIA_VIDEO;
                    out.content.video = videos_.video(value);
                } else if constexpr (std::is_same_v<T, ReasoningPart>) {
                    out.kind = TRTMC_MEDIA_REASONING;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, ToolCall>) {
                    out.kind = TRTMC_MEDIA_TOOL_CALL;
                    out.content.tool_call = c_tool_call(value);
                } else if constexpr (std::is_same_v<T, ToolResult>) {
                    out.kind = TRTMC_MEDIA_TOOL_RESULT;
                    out.content.tool_result = c_tool_result(value);
                }
            },
            input);
        return out;
    }
    trtmc_image_video_text_part_v1 part(const ImageVideoTextPart& input) {
        trtmc_image_video_text_part_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextPart>) {
                    out.kind = TRTMC_MEDIA_TEXT;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, ImageInput>) {
                    out.kind = TRTMC_MEDIA_IMAGE;
                    out.content.image = value.wire;
                } else if constexpr (std::is_same_v<T, VideoInput>) {
                    out.kind = TRTMC_MEDIA_VIDEO;
                    out.content.video = videos_.video(value);
                }
            },
            input);
        return out;
    }
    trtmc_audio_text_part_v1 part(const AudioTextPart& input) {
        trtmc_audio_text_part_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextPart>) {
                    out.kind = TRTMC_MEDIA_TEXT;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, AudioView>) {
                    out.kind = TRTMC_MEDIA_AUDIO;
                    out.content.audio = c_audio(value);
                }
            },
            input);
        return out;
    }
    trtmc_image_audio_text_part_v1 part(const ImageAudioTextPart& input) {
        trtmc_image_audio_text_part_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextPart>) {
                    out.kind = TRTMC_MEDIA_TEXT;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, ImageInput>) {
                    out.kind = TRTMC_MEDIA_IMAGE;
                    out.content.image = value.wire;
                } else if constexpr (std::is_same_v<T, AudioView>) {
                    out.kind = TRTMC_MEDIA_AUDIO;
                    out.content.audio = c_audio(value);
                }
            },
            input);
        return out;
    }
    trtmc_audio_video_text_part_v1 part(const AudioVideoTextPart& input) {
        trtmc_audio_video_text_part_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextPart>) {
                    out.kind = TRTMC_MEDIA_TEXT;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, AlignedAudioVideoInput>) {
                    out.kind = TRTMC_MEDIA_ALIGNED_AUDIO_VIDEO;
                    out.content.aligned = trtmc_aligned_audio_video_input_v1{
                        videos_.video(value.video), c_audio(value.audio),
                        value.audio_start_seconds ? 1U : 0U, value.audio_start_seconds.value_or(0)};
                }
            },
            input);
        return out;
    }
    template <class Message, class WirePart, class WireMessage>
    void messages(const std::vector<Message>& input, std::vector<std::vector<WirePart>>& parts,
                  std::vector<WireMessage>& output) {
        parts.reserve(input.size());
        output.reserve(input.size());
        for (const auto& message : input) {
            auto& body = parts.emplace_back();
            body.reserve(message.parts.size());
            for (const auto& item : message.parts)
                body.push_back(part(item));
            output.push_back({static_cast<uint32_t>(message.role), body.data(), body.size()});
        }
    }
    trtmc_images_text_to_text_request_v1 convert(const ImagesTextToTextRequest& input) {
        messages(input.messages, images_text_parts_, images_text_messages_);
        tool_declarations(input.tools);
        return {images_text_messages_.data(), images_text_messages_.size(), tools_.data(),
                tools_.size()};
    }
    trtmc_video_text_to_text_request_v1 convert(const VideoTextToTextRequest& input) {
        messages(input.messages, video_text_parts_, video_text_messages_);
        tool_declarations(input.tools);
        return {video_text_messages_.data(), video_text_messages_.size(), tools_.data(),
                tools_.size()};
    }
    trtmc_image_video_text_to_text_request_v1 convert(const ImageVideoTextToTextRequest& input) {
        messages(input.messages, image_video_text_parts_, image_video_text_messages_);
        return {image_video_text_messages_.data(), image_video_text_messages_.size()};
    }
    trtmc_audio_text_to_text_request_v1 convert(const AudioTextToTextRequest& input) {
        messages(input.messages, audio_text_parts_, audio_text_messages_);
        return {audio_text_messages_.data(), audio_text_messages_.size()};
    }
    trtmc_image_audio_to_text_request_v1 convert(const ImageAudioToTextRequest& input) {
        messages(input.messages, image_audio_text_parts_, image_audio_text_messages_);
        return {image_audio_text_messages_.data(), image_audio_text_messages_.size()};
    }
    trtmc_audio_video_text_to_text_request_v1 convert(const AudioVideoTextToTextRequest& input) {
        messages(input.messages, audio_video_text_parts_, audio_video_text_messages_);
        return {audio_video_text_messages_.data(), audio_video_text_messages_.size()};
    }
    trtmc_image_audio_text_to_text_request_v1 convert(const ImageAudioTextToTextRequest& input) {
        messages(input.messages, image_audio_text_parts_, image_audio_text_messages_);
        return {image_audio_text_messages_.data(), image_audio_text_messages_.size()};
    }
    trtmc_image_audio_text_to_text_speech_response_request_v1
    convert(const ImageAudioTextToTextSpeechResponseRequest& input) {
        messages(input.messages, image_audio_text_parts_, image_audio_text_messages_);
        return {image_audio_text_messages_.data(), image_audio_text_messages_.size()};
    }

    trtmc_conversation_part_v1 part(const ConversationPart& input) {
        trtmc_conversation_part_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextPart>) {
                    out.kind = TRTMC_CONVERSATION_TEXT;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, ReasoningPart>) {
                    out.kind = TRTMC_CONVERSATION_REASONING;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, ToolCall>) {
                    out.kind = TRTMC_CONVERSATION_TOOL_CALL;
                    out.content.tool_call = c_tool_call(value);
                } else {
                    out.kind = TRTMC_CONVERSATION_TOOL_RESULT;
                    out.content.tool_result = c_tool_result(value);
                }
            },
            input);
        return out;
    }
    trtmc_text_conversation_request_v1 convert(const TextConversationRequest& input) {
        messages(input.messages, conversation_parts_, conversation_messages_);
        tool_declarations(input.tools);
        return {conversation_messages_.data(), conversation_messages_.size(), tools_.data(),
                tools_.size()};
    }
    trtmc_audio_text_conversation_part_v1 part(const AudioTextConversationPart& input) {
        trtmc_audio_text_conversation_part_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextPart>) {
                    out.kind = TRTMC_MEDIA_TEXT;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, AudioView>) {
                    out.kind = TRTMC_MEDIA_AUDIO;
                    out.content.audio = c_audio(value);
                } else if constexpr (std::is_same_v<T, ReasoningPart>) {
                    out.kind = TRTMC_MEDIA_REASONING;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, ToolCall>) {
                    out.kind = TRTMC_MEDIA_TOOL_CALL;
                    out.content.tool_call = c_tool_call(value);
                } else {
                    out.kind = TRTMC_MEDIA_TOOL_RESULT;
                    out.content.tool_result = c_tool_result(value);
                }
            },
            input);
        return out;
    }
    trtmc_audio_text_conversation_request_v1 convert(const AudioTextConversationRequest& input) {
        messages(input.messages, audio_text_conversation_parts_, audio_text_conversation_messages_);
        tool_declarations(input.tools);
        return {audio_text_conversation_messages_.data(), audio_text_conversation_messages_.size(),
                tools_.data(), tools_.size()};
    }
    trtmc_image_audio_text_conversation_part_v1 part(const ImageAudioTextConversationPart& input) {
        trtmc_image_audio_text_conversation_part_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextPart>) {
                    out.kind = TRTMC_MEDIA_TEXT;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, ImageInput>) {
                    out.kind = TRTMC_MEDIA_IMAGE;
                    out.content.image = value.wire;
                } else if constexpr (std::is_same_v<T, AudioView>) {
                    out.kind = TRTMC_MEDIA_AUDIO;
                    out.content.audio = c_audio(value);
                } else if constexpr (std::is_same_v<T, ReasoningPart>) {
                    out.kind = TRTMC_MEDIA_REASONING;
                    out.content.text = c_string(value.text);
                } else if constexpr (std::is_same_v<T, ToolCall>) {
                    out.kind = TRTMC_MEDIA_TOOL_CALL;
                    out.content.tool_call = c_tool_call(value);
                } else {
                    out.kind = TRTMC_MEDIA_TOOL_RESULT;
                    out.content.tool_result = c_tool_result(value);
                }
            },
            input);
        return out;
    }
    trtmc_image_audio_text_conversation_request_v1
    convert(const ImageAudioTextConversationRequest& input) {
        messages(input.messages, image_audio_text_conversation_parts_,
                 image_audio_text_conversation_messages_);
        tool_declarations(input.tools);
        return {image_audio_text_conversation_messages_.data(),
                image_audio_text_conversation_messages_.size(), tools_.data(), tools_.size()};
    }
    trtmc_text_images_video_conversation_request_v1
    convert(const TextImagesVideoConversationRequest& input) {
        trtmc_text_images_video_conversation_request_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextConversationRequest>) {
                    out.kind = TRTMC_BATCH_CONVERSATION_TEXT;
                    out.input.text = convert(value);
                } else if constexpr (std::is_same_v<T, ImagesTextToTextRequest>) {
                    out.kind = TRTMC_BATCH_CONVERSATION_IMAGES_TEXT;
                    out.input.images_text = convert(value);
                } else {
                    out.kind = TRTMC_BATCH_CONVERSATION_VIDEO_TEXT;
                    out.input.video_text = convert(value);
                }
            },
            input);
        return out;
    }
    trtmc_text_images_audio_conversation_request_v1
    convert(const TextImagesAudioConversationRequest& input) {
        trtmc_text_images_audio_conversation_request_v1 out{};
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, TextConversationRequest>) {
                    out.kind = TRTMC_BATCH_CONVERSATION_TEXT;
                    out.input.text = convert(value);
                } else if constexpr (std::is_same_v<T, ImagesTextToTextRequest>) {
                    out.kind = TRTMC_BATCH_CONVERSATION_IMAGES_TEXT;
                    out.input.images_text = convert(value);
                } else if constexpr (std::is_same_v<T, AudioTextConversationRequest>) {
                    out.kind = TRTMC_BATCH_CONVERSATION_AUDIO_TEXT;
                    out.input.audio_text = convert(value);
                } else {
                    out.kind = TRTMC_BATCH_CONVERSATION_IMAGE_AUDIO_TEXT;
                    out.input.image_audio_text = convert(value);
                }
            },
            input);
        return out;
    }
    trtmc_text_label_classification_request_v1
    convert(const TextLabelClassificationRequest& input) {
        return {c_string(input.text)};
    }
    trtmc_text_pair_label_classification_request_v1
    convert(const TextPairLabelClassificationRequest& input) {
        return {c_string(input.first), c_string(input.second)};
    }
    static trtmc_language_token_sequence_v1 sequence(const LanguageTokenSequence& input) {
        return {{input.token_ids.data(), input.token_ids.size()},
                {input.attention_mask.data(), input.attention_mask.size()}};
    }
    trtmc_text_encoder_decoder_hidden_states_request_v1
    convert(const TextEncoderDecoderHiddenStatesRequest& input) {
        return {sequence(input.source), sequence(input.decoder)};
    }

  private:
    void tool_declarations(const std::vector<ToolDefinition>& input) {
        tools_.reserve(input.size());
        for (const auto& tool : input)
            tools_.push_back(c_tool_definition(tool));
    }
    VideoWireInputs videos_;
    std::vector<std::vector<trtmc_images_text_part_v1>> images_text_parts_;
    std::vector<trtmc_images_text_message_v1> images_text_messages_;
    std::vector<std::vector<trtmc_video_text_part_v1>> video_text_parts_;
    std::vector<trtmc_video_text_message_v1> video_text_messages_;
    std::vector<std::vector<trtmc_image_video_text_part_v1>> image_video_text_parts_;
    std::vector<trtmc_image_video_text_message_v1> image_video_text_messages_;
    std::vector<std::vector<trtmc_audio_text_part_v1>> audio_text_parts_;
    std::vector<trtmc_audio_text_message_v1> audio_text_messages_;
    std::vector<std::vector<trtmc_image_audio_text_part_v1>> image_audio_text_parts_;
    std::vector<trtmc_image_audio_text_message_v1> image_audio_text_messages_;
    std::vector<std::vector<trtmc_audio_video_text_part_v1>> audio_video_text_parts_;
    std::vector<trtmc_audio_video_text_message_v1> audio_video_text_messages_;
    std::vector<std::vector<trtmc_conversation_part_v1>> conversation_parts_;
    std::vector<trtmc_conversation_message_v1> conversation_messages_;
    std::vector<std::vector<trtmc_audio_text_conversation_part_v1>> audio_text_conversation_parts_;
    std::vector<trtmc_audio_text_conversation_message_v1> audio_text_conversation_messages_;
    std::vector<std::vector<trtmc_image_audio_text_conversation_part_v1>>
        image_audio_text_conversation_parts_;
    std::vector<trtmc_image_audio_text_conversation_message_v1>
        image_audio_text_conversation_messages_;
    std::vector<trtmc_tool_definition_v1> tools_;
};

template <class Traits>
class LanguageTask {
  public:
    static constexpr std::string_view kTask = Traits::id;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    using Request = typename Traits::Request;
    using Result = typename Traits::Result;
    Result run(const Request& input, const Config& config = {}) const {
        LanguageWireInputs storage;
        const auto request = storage.convert(input);
        const auto entries = config.c_entries();
        const auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = table_->run(state_->handle, &request, &options, &raw, &error);
        if constexpr (std::is_same_v<Result, TextContinuationResult>) {
            return finish_text_call(state_, status, raw, error, table_->result_view);
        } else {
            ResultOwner owner(state_, raw);
            check(state_->api, status, error);
            return Result(std::move(owner), table_->result_view);
        }
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(typename Traits::Table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible language Task API table");
        const auto* typed = reinterpret_cast<const typename Traits::Table*>(table);
        if (!typed->run || !typed->result_view)
            throw Error(TRTMC_VERSION_MISMATCH, "incomplete language Task API table");
    }

  private:
    friend class ::trtmc::Model;
    LanguageTask(std::shared_ptr<ModelState> state, const trtmc_api_header* table)
        : state_(std::move(state)), table_(reinterpret_cast<const typename Traits::Table*>(table)) {
    }
    std::shared_ptr<ModelState> state_;
    const typename Traits::Table* table_;
};
struct ImagesTextToTextTraits {
    static constexpr std::string_view id = TRTMC_TASK_IMAGES_TEXT_TO_TEXT;
    using Request = ImagesTextToTextRequest;
    using Result = TextContinuationResult;
    using Table = trtmc_images_text_to_text_api_v1;
};
struct VideoTextToTextTraits {
    static constexpr std::string_view id = TRTMC_TASK_VIDEO_TEXT_TO_TEXT;
    using Request = VideoTextToTextRequest;
    using Result = TextContinuationResult;
    using Table = trtmc_video_text_to_text_api_v1;
};
struct ImageVideoTextToTextTraits {
    static constexpr std::string_view id = TRTMC_TASK_IMAGE_VIDEO_TEXT_TO_TEXT;
    using Request = ImageVideoTextToTextRequest;
    using Result = TextContinuationResult;
    using Table = trtmc_image_video_text_to_text_api_v1;
};
struct AudioTextToTextTraits {
    static constexpr std::string_view id = TRTMC_TASK_AUDIO_TEXT_TO_TEXT;
    using Request = AudioTextToTextRequest;
    using Result = TextContinuationResult;
    using Table = trtmc_audio_text_to_text_api_v1;
};
struct ImageAudioToTextTraits {
    static constexpr std::string_view id = TRTMC_TASK_IMAGE_AUDIO_TO_TEXT;
    using Request = ImageAudioToTextRequest;
    using Result = TextContinuationResult;
    using Table = trtmc_image_audio_to_text_api_v1;
};
struct AudioVideoTextToTextTraits {
    static constexpr std::string_view id = TRTMC_TASK_AUDIO_VIDEO_TEXT_TO_TEXT;
    using Request = AudioVideoTextToTextRequest;
    using Result = TextContinuationResult;
    using Table = trtmc_audio_video_text_to_text_api_v1;
};
struct ImageAudioTextToTextTraits {
    static constexpr std::string_view id = TRTMC_TASK_IMAGE_AUDIO_TEXT_TO_TEXT;
    using Request = ImageAudioTextToTextRequest;
    using Result = TextContinuationResult;
    using Table = trtmc_image_audio_text_to_text_api_v1;
};
struct ImageAudioTextToTextSpeechResponseTraits {
    static constexpr std::string_view id = TRTMC_TASK_IMAGE_AUDIO_TEXT_TO_TEXT_SPEECH_RESPONSE;
    using Request = ImageAudioTextToTextSpeechResponseRequest;
    using Result = TextSpeechResult;
    using Table = trtmc_image_audio_text_to_text_speech_response_api_v1;
};
struct TextConversationTraits {
    static constexpr std::string_view id = TRTMC_TASK_TEXT_CONVERSATION;
    using Request = TextConversationRequest;
    using Result = ConversationResult;
    using Table = trtmc_text_conversation_api_v1;
};
struct ImagesTextConversationTraits {
    static constexpr std::string_view id = TRTMC_TASK_IMAGES_TEXT_CONVERSATION;
    using Request = ImagesTextToTextRequest;
    using Result = ConversationResult;
    using Table = trtmc_images_text_conversation_api_v1;
};
struct VideoTextConversationTraits {
    static constexpr std::string_view id = TRTMC_TASK_VIDEO_TEXT_CONVERSATION;
    using Request = VideoTextToTextRequest;
    using Result = ConversationResult;
    using Table = trtmc_video_text_conversation_api_v1;
};
struct TextLabelClassificationTraits {
    static constexpr std::string_view id = TRTMC_TASK_TEXT_LABEL_CLASSIFICATION;
    using Request = TextLabelClassificationRequest;
    using Result = GeneratedLabelResult;
    using Table = trtmc_text_label_classification_api_v1;
};
struct TextPairLabelClassificationTraits {
    static constexpr std::string_view id = TRTMC_TASK_TEXT_PAIR_LABEL_CLASSIFICATION;
    using Request = TextPairLabelClassificationRequest;
    using Result = GeneratedLabelResult;
    using Table = trtmc_text_pair_label_classification_api_v1;
};
struct TextEncoderDecoderHiddenStatesTraits {
    static constexpr std::string_view id = TRTMC_TASK_TEXT_ENCODER_DECODER_HIDDEN_STATES;
    using Request = TextEncoderDecoderHiddenStatesRequest;
    using Result = EncoderDecoderStatesResult;
    using Table = trtmc_text_encoder_decoder_hidden_states_api_v1;
};
} // namespace detail
using ImagesTextToText = detail::LanguageTask<detail::ImagesTextToTextTraits>;
using VideoTextToText = detail::LanguageTask<detail::VideoTextToTextTraits>;
using ImageVideoTextToText = detail::LanguageTask<detail::ImageVideoTextToTextTraits>;
using AudioTextToText = detail::LanguageTask<detail::AudioTextToTextTraits>;
using ImageAudioToText = detail::LanguageTask<detail::ImageAudioToTextTraits>;
using AudioVideoTextToText = detail::LanguageTask<detail::AudioVideoTextToTextTraits>;
using ImageAudioTextToText = detail::LanguageTask<detail::ImageAudioTextToTextTraits>;
using ImageAudioTextToTextSpeechResponse =
    detail::LanguageTask<detail::ImageAudioTextToTextSpeechResponseTraits>;
using TextConversation = detail::LanguageTask<detail::TextConversationTraits>;
using ImagesTextConversation = detail::LanguageTask<detail::ImagesTextConversationTraits>;
using VideoTextConversation = detail::LanguageTask<detail::VideoTextConversationTraits>;
using TextLabelClassification = detail::LanguageTask<detail::TextLabelClassificationTraits>;
using TextPairLabelClassification = detail::LanguageTask<detail::TextPairLabelClassificationTraits>;
using TextEncoderDecoderHiddenStates =
    detail::LanguageTask<detail::TextEncoderDecoderHiddenStatesTraits>;

struct BatchTextConversationItem {
    TextConversationRequest input;
    Config config{};
};
struct BatchTextConversationRequest {
    std::vector<BatchTextConversationItem> items;
};

struct BatchVideoTextConversationItem {
    VideoTextToTextRequest input;
    Config config{};
};
struct BatchVideoTextConversationRequest {
    std::vector<BatchVideoTextConversationItem> items;
};

struct BatchAudioTextConversationItem {
    AudioTextConversationRequest input;
    Config config{};
};
struct BatchAudioTextConversationRequest {
    std::vector<BatchAudioTextConversationItem> items;
};

struct BatchImageAudioTextConversationItem {
    ImageAudioTextConversationRequest input;
    Config config{};
};
struct BatchImageAudioTextConversationRequest {
    std::vector<BatchImageAudioTextConversationItem> items;
};

struct BatchTextImagesVideoConversationsItem {
    TextImagesVideoConversationRequest input;
    Config config{};
};
struct BatchTextImagesVideoConversationsRequest {
    std::vector<BatchTextImagesVideoConversationsItem> items;
};

struct BatchTextImagesAudioConversationsItem {
    TextImagesAudioConversationRequest input;
    Config config{};
};
struct BatchTextImagesAudioConversationsRequest {
    std::vector<BatchTextImagesAudioConversationsItem> items;
};
struct BatchImagesTextConversationItem {
    ImagesTextToTextRequest input;
    Config config{};
};
struct BatchImagesTextConversationRequest {
    std::vector<BatchImagesTextConversationItem> items;
};
class ConversationBatchResult {
  public:
    using Count = trtmc_status(TRTMC_CALL*)(const trtmc_result*, uint64_t*, trtmc_error**);
    using ItemView = trtmc_status(TRTMC_CALL*)(const trtmc_result*, uint64_t,
                                               trtmc_conversation_result_view_v1*, trtmc_error**);
    ConversationBatchResult(detail::ResultOwner owner, Count count, ItemView item)
        : owner_(std::move(owner)), item_(item) {
        trtmc_error* error = nullptr;
        const auto status = count(owner_.get(), &count_, &error);
        detail::check(owner_.api(), status, error);
    }
    ConversationBatchResult(const ConversationBatchResult&) = delete;
    ConversationBatchResult& operator=(const ConversationBatchResult&) = delete;
    ConversationBatchResult(ConversationBatchResult&& other) noexcept
        : owner_(std::move(other.owner_)), item_(other.item_),
          count_(std::exchange(other.count_, 0)) {}
    ConversationBatchResult& operator=(ConversationBatchResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            item_ = other.item_;
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    uint64_t size() const noexcept { return count_; }
    // The returned view borrows this result, not the model or request.
    ConversationResultView item(uint64_t index) const {
        trtmc_conversation_result_view_v1 view{};
        trtmc_error* error = nullptr;
        const auto status = item_(owner_.get(), index, &view, &error);
        detail::check(owner_.api(), status, error);
        return ConversationResultView{view};
    }
    ConversationResultView operator[](uint64_t index) const { return item(index); }

  private:
    detail::ResultOwner owner_;
    ItemView item_;
    uint64_t count_{0};
};

namespace detail {
template <class Traits>
class ConversationBatchTask {
  public:
    static constexpr std::string_view kTask = Traits::id;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    using Request = typename Traits::Request;
    using Result = ConversationBatchResult;
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    Result run(const Request& request) const {
        std::vector<LanguageWireInputs> storage;
        std::vector<Config::CEntries> configs;
        std::vector<typename Traits::WireItem> items;
        storage.reserve(request.items.size());
        configs.reserve(request.items.size());
        items.reserve(request.items.size());
        for (const auto& item : request.items) {
            storage.emplace_back();
            configs.push_back(item.config.c_entries());
            items.push_back({storage.back().convert(item.input), configs.back().view()});
        }
        const typename Traits::WireRequest input{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = table_->run(state_->handle, &input, &raw, &error);
        ResultOwner owner(state_, raw);
        check(state_->api, status, error);
        return Result{std::move(owner), table_->result_count, table_->result_item_view};
    }
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(typename Traits::Table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible conversation batch Task table");
        const auto* typed = reinterpret_cast<const typename Traits::Table*>(table);
        if (!typed->run || !typed->result_count || !typed->result_item_view)
            throw Error(TRTMC_VERSION_MISMATCH, "incomplete conversation batch Task table");
    }

  private:
    friend class ::trtmc::Model;
    ConversationBatchTask(std::shared_ptr<ModelState> state, const trtmc_api_header* table)
        : state_(std::move(state)), table_(reinterpret_cast<const typename Traits::Table*>(table)) {
    }
    std::shared_ptr<ModelState> state_;
    const typename Traits::Table* table_;
};

struct BatchTextConversationTraits {
    static constexpr std::string_view id = TRTMC_TASK_BATCH_TEXT_CONVERSATION;
    using Request = BatchTextConversationRequest;
    using WireItem = trtmc_batch_text_conversation_item_v1;
    using WireRequest = trtmc_batch_text_conversation_request_v1;
    using Table = trtmc_batch_text_conversation_api_v1;
};

struct BatchVideoTextConversationTraits {
    static constexpr std::string_view id = TRTMC_TASK_BATCH_VIDEO_TEXT_CONVERSATION;
    using Request = BatchVideoTextConversationRequest;
    using WireItem = trtmc_batch_video_text_conversation_item_v1;
    using WireRequest = trtmc_batch_video_text_conversation_request_v1;
    using Table = trtmc_batch_video_text_conversation_api_v1;
};

struct BatchAudioTextConversationTraits {
    static constexpr std::string_view id = TRTMC_TASK_BATCH_AUDIO_TEXT_CONVERSATION;
    using Request = BatchAudioTextConversationRequest;
    using WireItem = trtmc_batch_audio_text_conversation_item_v1;
    using WireRequest = trtmc_batch_audio_text_conversation_request_v1;
    using Table = trtmc_batch_audio_text_conversation_api_v1;
};

struct BatchImageAudioTextConversationTraits {
    static constexpr std::string_view id = TRTMC_TASK_BATCH_IMAGE_AUDIO_TEXT_CONVERSATION;
    using Request = BatchImageAudioTextConversationRequest;
    using WireItem = trtmc_batch_image_audio_text_conversation_item_v1;
    using WireRequest = trtmc_batch_image_audio_text_conversation_request_v1;
    using Table = trtmc_batch_image_audio_text_conversation_api_v1;
};

struct BatchTextImagesVideoConversationsTraits {
    static constexpr std::string_view id = TRTMC_TASK_BATCH_TEXT_IMAGES_VIDEO_CONVERSATIONS;
    using Request = BatchTextImagesVideoConversationsRequest;
    using WireItem = trtmc_batch_text_images_video_conversations_item_v1;
    using WireRequest = trtmc_batch_text_images_video_conversations_request_v1;
    using Table = trtmc_batch_text_images_video_conversations_api_v1;
};

struct BatchTextImagesAudioConversationsTraits {
    static constexpr std::string_view id = TRTMC_TASK_BATCH_TEXT_IMAGES_AUDIO_CONVERSATIONS;
    using Request = BatchTextImagesAudioConversationsRequest;
    using WireItem = trtmc_batch_text_images_audio_conversations_item_v1;
    using WireRequest = trtmc_batch_text_images_audio_conversations_request_v1;
    using Table = trtmc_batch_text_images_audio_conversations_api_v1;
};
struct BatchImagesTextConversationTraits {
    static constexpr std::string_view id = TRTMC_TASK_BATCH_IMAGES_TEXT_CONVERSATION;
    using Request = BatchImagesTextConversationRequest;
    using WireItem = trtmc_batch_images_text_conversation_item_v1;
    using WireRequest = trtmc_batch_images_text_conversation_request_v1;
    using Table = trtmc_batch_images_text_conversation_api_v1;
};
} // namespace detail
using BatchImagesTextConversation =
    detail::ConversationBatchTask<detail::BatchImagesTextConversationTraits>;
using BatchTextConversation = detail::ConversationBatchTask<detail::BatchTextConversationTraits>;
using BatchVideoTextConversation =
    detail::ConversationBatchTask<detail::BatchVideoTextConversationTraits>;
using BatchAudioTextConversation =
    detail::ConversationBatchTask<detail::BatchAudioTextConversationTraits>;
using BatchImageAudioTextConversation =
    detail::ConversationBatchTask<detail::BatchImageAudioTextConversationTraits>;
using BatchTextImagesVideoConversations =
    detail::ConversationBatchTask<detail::BatchTextImagesVideoConversationsTraits>;
using BatchTextImagesAudioConversations =
    detail::ConversationBatchTask<detail::BatchTextImagesAudioConversationsTraits>;
} // namespace trtmc
