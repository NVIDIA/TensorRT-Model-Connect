/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "api_internal.h"
#include "trtmc/internal/language.h"
#include "trtmc/language.h"

#include <cmath>
#include <set>
#include <type_traits>

namespace trtmc::api::language_detail {
inline void output_check(bool condition, const char* message) {
    if (!condition)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
}
inline internal::MessageRole role(uint32_t value, bool tools) {
    require(value >= TRTMC_ROLE_SYSTEM && value <= (tools ? TRTMC_ROLE_TOOL : TRTMC_ROLE_ASSISTANT),
            "unknown or unsupported message role");
    return static_cast<internal::MessageRole>(value);
}
inline internal::ToolDefinitionView tool_definition(const trtmc_tool_definition_v1& value) {
    auto name = string_view(value.name);
    require(!name.empty(), "tool definition name must not be empty");
    return {name, string_view(value.description), string_view(value.parameters_schema_json)};
}
inline internal::ToolCallView tool_call(const trtmc_tool_call_v1& value) {
    auto id = string_view(value.call_id);
    auto name = string_view(value.name);
    require(value.state <= TRTMC_TOOL_CALL_MALFORMED, "unknown tool-call state");
    require(value.state != TRTMC_TOOL_CALL_COMPLETE || (!id.empty() && !name.empty()),
            "complete tool call requires an ID and name");
    return {id, name, string_view(value.arguments_json),
            static_cast<internal::ToolCallState>(value.state)};
}
inline internal::ToolResultView tool_result(const trtmc_tool_result_v1& value) {
    auto id = string_view(value.call_id);
    require(!id.empty() && value.is_error <= 1, "tool result requires an ID and a valid bool");
    return {id, string_view(value.content_text), value.is_error != 0};
}
template <class Part>
void validate_part_role(internal::MessageRole role, const Part& part) {
    std::visit(
        [role](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, internal::ToolResultView>)
                require(role == internal::MessageRole::Tool,
                        "tool results belong to tool messages");
            else {
                require(role != internal::MessageRole::Tool,
                        "tool messages require typed tool results");
                if constexpr (std::is_same_v<T, internal::ToolCallView> ||
                              std::is_same_v<T, internal::ReasoningPartView>)
                    require(role == internal::MessageRole::Assistant,
                            "reasoning and tool calls belong to assistant messages");
            }
        },
        part);
}
inline internal::LanguageTokenSequence sequence(const trtmc_language_token_sequence_v1& value) {
    auto ids = checked_span(value.token_ids.data, value.token_ids.size);
    auto mask = checked_span(value.attention_mask.data, value.attention_mask.count);
    require(!ids.empty(), "encoder/decoder token sequences must be explicit and nonempty");
    require(mask.empty() || mask.size() == ids.size(), "token attention-mask length mismatch");
    for (const auto bit : mask)
        require(bit <= 1, "token mask must contain zero or one");
    return {ids, mask};
}
class LanguageInputs {
  public:
    internal::AlignedAudioVideoView aligned(const trtmc_aligned_audio_video_input_v1& value) {
        require(value.has_audio_start <= 1, "audio origin presence must be zero or one");
        require(!value.has_audio_start || std::isfinite(value.audio_start_seconds),
                "audio origin must be finite");
        return {video_input(value.video, video_frames_.emplace_back()), audio_view(value.audio),
                value.has_audio_start ? std::optional<double>{value.audio_start_seconds}
                                      : std::nullopt};
    }
    internal::ImagesTextPart part(const trtmc_images_text_part_v1& input) {
        switch (input.kind) {
        case TRTMC_MEDIA_TEXT:
            return internal::TextPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_IMAGE:
            return image_input(input.content.image);
        case TRTMC_MEDIA_REASONING:
            return internal::ReasoningPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_TOOL_CALL:
            return tool_call(input.content.tool_call);
        case TRTMC_MEDIA_TOOL_RESULT:
            return tool_result(input.content.tool_result);
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "part kind is not accepted by this Task"};
        }
    }
    internal::VideoTextPart part(const trtmc_video_text_part_v1& input) {
        switch (input.kind) {
        case TRTMC_MEDIA_TEXT:
            return internal::TextPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_VIDEO:
            return video_input(input.content.video, video_frames_.emplace_back());
        case TRTMC_MEDIA_REASONING:
            return internal::ReasoningPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_TOOL_CALL:
            return tool_call(input.content.tool_call);
        case TRTMC_MEDIA_TOOL_RESULT:
            return tool_result(input.content.tool_result);
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "part kind is not accepted by this Task"};
        }
    }
    internal::ImageVideoTextPart part(const trtmc_image_video_text_part_v1& input) {
        switch (input.kind) {
        case TRTMC_MEDIA_TEXT:
            return internal::TextPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_IMAGE:
            return image_input(input.content.image);
        case TRTMC_MEDIA_VIDEO:
            return video_input(input.content.video, video_frames_.emplace_back());
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "part kind is not accepted by this Task"};
        }
    }
    internal::AudioTextPart part(const trtmc_audio_text_part_v1& input) {
        switch (input.kind) {
        case TRTMC_MEDIA_TEXT:
            return internal::TextPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_AUDIO:
            return audio_view(input.content.audio);
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "part kind is not accepted by this Task"};
        }
    }
    internal::ImageAudioTextPart part(const trtmc_image_audio_text_part_v1& input) {
        switch (input.kind) {
        case TRTMC_MEDIA_TEXT:
            return internal::TextPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_IMAGE:
            return image_input(input.content.image);
        case TRTMC_MEDIA_AUDIO:
            return audio_view(input.content.audio);
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "part kind is not accepted by this Task"};
        }
    }
    internal::AudioVideoTextPart part(const trtmc_audio_video_text_part_v1& input) {
        switch (input.kind) {
        case TRTMC_MEDIA_TEXT:
            return internal::TextPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_ALIGNED_AUDIO_VIDEO:
            return aligned(input.content.aligned);
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "part kind is not accepted by this Task"};
        }
    }

    template <class Part, class WireMessage>
    Span<const internal::MediaMessage<Part>>
    messages(const WireMessage* data, uint64_t count, uint32_t required,
             std::vector<std::vector<Part>>& parts,
             std::vector<internal::MediaMessage<Part>>& output, bool accepts_tools = false) {
        const auto input = checked_span(data, count);
        require(!input.empty(), "multimodal request requires messages");
        parts.reserve(input.size());
        output.reserve(input.size());
        uint32_t present = 0;
        for (const auto& message : input) {
            const auto message_role = role(message.role, accepts_tools);
            const auto source = checked_span(message.parts, message.part_count);
            auto& body = parts.emplace_back();
            body.reserve(source.size());
            for (const auto& item : source) {
                body.push_back(part(item));
                if (accepts_tools)
                    validate_part_role(message_role, body.back());
                present |= 1U << item.kind; // Only validated closed-union kinds reach here.
            }
            output.push_back({message_role, {body.data(), body.size()}});
        }
        require((present & required) == required, "request omitted a required Task modality");
        return {output.data(), output.size()};
    }
    Span<const internal::ToolDefinitionView>
    tool_declarations(const trtmc_tool_definition_v1* input, std::uint64_t count) {
        const auto declarations = checked_span(input, count);
        tools_.reserve(declarations.size());
        for (const auto& item : declarations)
            tools_.push_back(tool_definition(item));
        return {tools_.data(), tools_.size()};
    }
    internal::ImagesTextToTextRequest convert(const trtmc_images_text_to_text_request_v1& input) {
        return {messages(input.messages, input.message_count,
                         (1U << TRTMC_MEDIA_TEXT) | (1U << TRTMC_MEDIA_IMAGE), images_text_parts_,
                         images_text_messages_, true),
                tool_declarations(input.tools, input.tool_count)};
    }
    internal::VideoTextToTextRequest convert(const trtmc_video_text_to_text_request_v1& input) {
        return {messages(input.messages, input.message_count,
                         (1U << TRTMC_MEDIA_TEXT) | (1U << TRTMC_MEDIA_VIDEO), video_text_parts_,
                         video_text_messages_, true),
                tool_declarations(input.tools, input.tool_count)};
    }
    internal::ImageVideoTextToTextRequest
    convert(const trtmc_image_video_text_to_text_request_v1& input) {
        return {messages(input.messages, input.message_count,
                         (1U << TRTMC_MEDIA_TEXT) | (1U << TRTMC_MEDIA_IMAGE) |
                             (1U << TRTMC_MEDIA_VIDEO),
                         image_video_text_parts_, image_video_text_messages_)};
    }
    internal::AudioTextToTextRequest convert(const trtmc_audio_text_to_text_request_v1& input) {
        return {messages(input.messages, input.message_count,
                         (1U << TRTMC_MEDIA_TEXT) | (1U << TRTMC_MEDIA_AUDIO), audio_text_parts_,
                         audio_text_messages_)};
    }
    internal::ImageAudioToTextRequest convert(const trtmc_image_audio_to_text_request_v1& input) {
        return {messages(input.messages, input.message_count,
                         (1U << TRTMC_MEDIA_IMAGE) | (1U << TRTMC_MEDIA_AUDIO),
                         image_audio_text_parts_, image_audio_text_messages_)};
    }
    internal::AudioVideoTextToTextRequest
    convert(const trtmc_audio_video_text_to_text_request_v1& input) {
        return {messages(input.messages, input.message_count,
                         (1U << TRTMC_MEDIA_ALIGNED_AUDIO_VIDEO), audio_video_text_parts_,
                         audio_video_text_messages_)};
    }
    internal::ImageAudioTextToTextRequest
    convert(const trtmc_image_audio_text_to_text_request_v1& input) {
        return {messages(input.messages, input.message_count,
                         (1U << TRTMC_MEDIA_TEXT) | (1U << TRTMC_MEDIA_IMAGE) |
                             (1U << TRTMC_MEDIA_AUDIO),
                         image_audio_text_parts_, image_audio_text_messages_)};
    }
    internal::ImageAudioTextToTextSpeechResponseRequest
    convert(const trtmc_image_audio_text_to_text_speech_response_request_v1& input) {
        return {messages(input.messages, input.message_count,
                         (1U << TRTMC_MEDIA_TEXT) | (1U << TRTMC_MEDIA_IMAGE) |
                             (1U << TRTMC_MEDIA_AUDIO),
                         image_audio_text_parts_, image_audio_text_messages_)};
    }

    internal::ConversationPartView conversation_part(const trtmc_conversation_part_v1& input) {
        switch (input.kind) {
        case TRTMC_CONVERSATION_TEXT:
            return internal::TextPartView{string_view(input.content.text)};
        case TRTMC_CONVERSATION_REASONING:
            return internal::ReasoningPartView{string_view(input.content.text)};
        case TRTMC_CONVERSATION_TOOL_CALL:
            return tool_call(input.content.tool_call);
        case TRTMC_CONVERSATION_TOOL_RESULT:
            return tool_result(input.content.tool_result);
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "unknown conversation part kind"};
        }
    }
    internal::TextConversationRequest convert(const trtmc_text_conversation_request_v1& input) {
        const auto source = checked_span(input.messages, input.message_count);
        require(!source.empty(), "conversation requires at least one message");
        conversation_parts_.reserve(source.size());
        conversation_.reserve(source.size());
        for (const auto& message : source) {
            const auto message_role = role(message.role, true);
            const auto source_parts = checked_span(message.parts, message.part_count);
            auto& parts = conversation_parts_.emplace_back();
            parts.reserve(source_parts.size());
            for (const auto& item : source_parts) {
                auto value = conversation_part(item);
                validate_part_role(message_role, value);
                parts.push_back(std::move(value));
            }
            conversation_.push_back({message_role, {parts.data(), parts.size()}});
        }
        return {{conversation_.data(), conversation_.size()},
                tool_declarations(input.tools, input.tool_count)};
    }
    internal::AudioTextConversationPart part(const trtmc_audio_text_conversation_part_v1& input) {
        switch (input.kind) {
        case TRTMC_MEDIA_TEXT:
            return internal::TextPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_AUDIO:
            return audio_view(input.content.audio);
        case TRTMC_MEDIA_REASONING:
            return internal::ReasoningPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_TOOL_CALL:
            return tool_call(input.content.tool_call);
        case TRTMC_MEDIA_TOOL_RESULT:
            return tool_result(input.content.tool_result);
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "part kind is not accepted by this Task"};
        }
    }
    internal::AudioTextConversationRequest
    convert(const trtmc_audio_text_conversation_request_v1& input) {
        return {messages(input.messages, input.message_count,
                         (1U << TRTMC_MEDIA_TEXT) | (1U << TRTMC_MEDIA_AUDIO),
                         audio_text_conversation_parts_, audio_text_conversation_messages_, true),
                tool_declarations(input.tools, input.tool_count)};
    }
    internal::ImageAudioTextConversationPart
    part(const trtmc_image_audio_text_conversation_part_v1& input) {
        switch (input.kind) {
        case TRTMC_MEDIA_TEXT:
            return internal::TextPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_IMAGE:
            return image_input(input.content.image);
        case TRTMC_MEDIA_AUDIO:
            return audio_view(input.content.audio);
        case TRTMC_MEDIA_REASONING:
            return internal::ReasoningPartView{string_view(input.content.text)};
        case TRTMC_MEDIA_TOOL_CALL:
            return tool_call(input.content.tool_call);
        case TRTMC_MEDIA_TOOL_RESULT:
            return tool_result(input.content.tool_result);
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "part kind is not accepted by this Task"};
        }
    }
    internal::ImageAudioTextConversationRequest
    convert(const trtmc_image_audio_text_conversation_request_v1& input) {
        return {messages(input.messages, input.message_count,
                         (1U << TRTMC_MEDIA_TEXT) | (1U << TRTMC_MEDIA_AUDIO) |
                             (1U << TRTMC_MEDIA_IMAGE),
                         image_audio_text_conversation_parts_,
                         image_audio_text_conversation_messages_, true),
                tool_declarations(input.tools, input.tool_count)};
    }
    internal::TextImagesVideoConversationRequest
    convert(const trtmc_text_images_video_conversation_request_v1& input) {
        switch (input.kind) {
        case TRTMC_BATCH_CONVERSATION_TEXT:
            return convert(input.input.text);
        case TRTMC_BATCH_CONVERSATION_IMAGES_TEXT:
            return convert(input.input.images_text);
        case TRTMC_BATCH_CONVERSATION_VIDEO_TEXT:
            return convert(input.input.video_text);
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "unsupported visual batch request kind"};
        }
    }
    internal::TextImagesAudioConversationRequest
    convert(const trtmc_text_images_audio_conversation_request_v1& input) {
        switch (input.kind) {
        case TRTMC_BATCH_CONVERSATION_TEXT:
            return convert(input.input.text);
        case TRTMC_BATCH_CONVERSATION_IMAGES_TEXT:
            return convert(input.input.images_text);
        case TRTMC_BATCH_CONVERSATION_AUDIO_TEXT:
            return convert(input.input.audio_text);
        case TRTMC_BATCH_CONVERSATION_IMAGE_AUDIO_TEXT:
            return convert(input.input.image_audio_text);
        default:
            throw ApiFailure{TRTMC_INVALID_ARGUMENT, "unsupported audio batch request kind"};
        }
    }
    internal::TextLabelClassificationRequest
    convert(const trtmc_text_label_classification_request_v1& input) {
        return {string_view(input.text)};
    }
    internal::TextPairLabelClassificationRequest
    convert(const trtmc_text_pair_label_classification_request_v1& input) {
        return {string_view(input.first), string_view(input.second)};
    }
    internal::TextEncoderDecoderHiddenStatesRequest
    convert(const trtmc_text_encoder_decoder_hidden_states_request_v1& input) {
        return {sequence(input.source), sequence(input.decoder)};
    }

  private:
    std::vector<std::vector<internal::ImageView>> video_frames_;
    std::vector<std::vector<internal::ImagesTextPart>> images_text_parts_;
    std::vector<internal::ImagesTextMessage> images_text_messages_;
    std::vector<std::vector<internal::VideoTextPart>> video_text_parts_;
    std::vector<internal::VideoTextMessage> video_text_messages_;
    std::vector<std::vector<internal::ImageVideoTextPart>> image_video_text_parts_;
    std::vector<internal::ImageVideoTextMessage> image_video_text_messages_;
    std::vector<std::vector<internal::AudioTextPart>> audio_text_parts_;
    std::vector<internal::AudioTextMessage> audio_text_messages_;
    std::vector<std::vector<internal::ImageAudioTextPart>> image_audio_text_parts_;
    std::vector<internal::ImageAudioTextMessage> image_audio_text_messages_;
    std::vector<std::vector<internal::AudioVideoTextPart>> audio_video_text_parts_;
    std::vector<internal::AudioVideoTextMessage> audio_video_text_messages_;
    std::vector<std::vector<internal::ConversationPartView>> conversation_parts_;
    std::vector<internal::ConversationMessageView> conversation_;
    std::vector<std::vector<internal::AudioTextConversationPart>> audio_text_conversation_parts_;
    std::vector<internal::MediaMessage<internal::AudioTextConversationPart>>
        audio_text_conversation_messages_;
    std::vector<std::vector<internal::ImageAudioTextConversationPart>>
        image_audio_text_conversation_parts_;
    std::vector<internal::MediaMessage<internal::ImageAudioTextConversationPart>>
        image_audio_text_conversation_messages_;
    std::vector<internal::ToolDefinitionView> tools_;
};

struct ConversationStorage final : ResultStorage {
    explicit ConversationStorage(internal::ConversationResult value) : result(std::move(value)) {
        std::set<std::string_view> call_ids;
        parts.reserve(result.parts.size());
        for (const auto& part : result.parts) {
            trtmc_conversation_part_v1 out{};
            if (const auto* text = std::get_if<internal::TextPart>(&part)) {
                out.kind = TRTMC_CONVERSATION_TEXT;
                out.content.text = borrowed_string(text->text);
            } else if (const auto* reasoning = std::get_if<internal::ReasoningPart>(&part)) {
                out.kind = TRTMC_CONVERSATION_REASONING;
                out.content.text = borrowed_string(reasoning->text);
            } else {
                const auto& call = std::get<internal::ToolCall>(part);
                const auto state = static_cast<std::uint32_t>(call.state);
                output_check(state <= TRTMC_TOOL_CALL_MALFORMED,
                             "unknown assistant tool-call state");
                output_check(state != TRTMC_TOOL_CALL_COMPLETE ||
                                 (!call.call_id.empty() && !call.name.empty()),
                             "complete assistant tool call requires an ID and name");
                output_check(call.call_id.empty() || call_ids.insert(call.call_id).second,
                             "assistant tool calls require unique known IDs");
                out.kind = TRTMC_CONVERSATION_TOOL_CALL;
                out.content.tool_call = {borrowed_string(call.call_id), borrowed_string(call.name),
                                         borrowed_string(call.arguments_json), state};
            }
            parts.push_back(out);
        }
        const auto reason = static_cast<std::uint32_t>(result.finish_reason);
        output_check(reason <= TRTMC_FINISH_OTHER, "unknown finish reason");
        output_check((reason == TRTMC_FINISH_OTHER) == !result.other_finish_reason.empty(),
                     "Other finish reason requires its reported value");
        view = {parts.data(),
                parts.size(),
                {result.token_ids.data(), result.token_ids.size()},
                result.setup_ms,
                result.prefill_ms,
                result.decode_ms,
                reason,
                borrowed_string(result.other_finish_reason),
                {result.usage.input_tokens.has_value() ? 1U : 0U,
                 result.usage.output_tokens.has_value() ? 1U : 0U,
                 result.usage.total_tokens.has_value() ? 1U : 0U,
                 result.usage.input_tokens.value_or(0), result.usage.output_tokens.value_or(0),
                 result.usage.total_tokens.value_or(0)}};
    }
    internal::ConversationResult result;
    std::vector<trtmc_conversation_part_v1> parts;
    trtmc_conversation_result_view_v1 view{};
};

} // namespace trtmc::api::language_detail
