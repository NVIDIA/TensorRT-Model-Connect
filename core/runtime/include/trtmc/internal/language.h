/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/audio.h"
#include "trtmc/internal/matrix.h"
#include "trtmc/internal/tools.h"
#include "trtmc/internal/video.h"

namespace trtmc::internal {

enum class MessageRole : uint32_t { System = 1, Developer = 2, User = 3, Assistant = 4, Tool = 5 };
struct TextPartView {
    std::string_view text;
};
struct ReasoningPartView {
    std::string_view text;
};
struct TextPart {
    std::string text;
};
struct ReasoningPart {
    std::string text;
};
struct AlignedAudioVideoView {
    VideoView video;
    AudioView audio;
    // Same clock as video timestamps. Missing timing/rate may only be resolved
    // by an explicitly declared family default; otherwise reject the input.
    std::optional<double> audio_start_seconds{};
};
template <class Part>
struct MediaMessage {
    MessageRole role{MessageRole::User};
    Span<const Part> parts;
};

using ImagesTextPart =
    std::variant<TextPartView, ImageView, ReasoningPartView, ToolCallView, ToolResultView>;
using ImagesTextMessage = MediaMessage<ImagesTextPart>;
using VideoTextPart =
    std::variant<TextPartView, VideoView, ReasoningPartView, ToolCallView, ToolResultView>;
using VideoTextMessage = MediaMessage<VideoTextPart>;
using ImageVideoTextPart = std::variant<TextPartView, ImageView, VideoView>;
using ImageVideoTextMessage = MediaMessage<ImageVideoTextPart>;
using AudioTextPart = std::variant<TextPartView, AudioView>;
using AudioTextMessage = MediaMessage<AudioTextPart>;
using ImageAudioTextPart = std::variant<TextPartView, ImageView, AudioView>;
using ImageAudioTextMessage = MediaMessage<ImageAudioTextPart>;
using AudioVideoTextPart = std::variant<TextPartView, AlignedAudioVideoView>;
using AudioVideoTextMessage = MediaMessage<AudioVideoTextPart>;

struct ImagesTextToTextRequest {
    Span<const ImagesTextMessage> messages;
    Span<const ToolDefinitionView> tools{};
};
struct VideoTextToTextRequest {
    Span<const VideoTextMessage> messages;
    Span<const ToolDefinitionView> tools{};
};
struct ImageVideoTextToTextRequest {
    Span<const ImageVideoTextMessage> messages;
};
struct AudioTextToTextRequest {
    Span<const AudioTextMessage> messages;
};
struct ImageAudioToTextRequest {
    Span<const ImageAudioTextMessage> messages;
};
struct AudioVideoTextToTextRequest {
    Span<const AudioVideoTextMessage> messages;
};
struct ImageAudioTextToTextRequest {
    Span<const ImageAudioTextMessage> messages;
};
struct ImageAudioTextToTextSpeechResponseRequest {
    Span<const ImageAudioTextMessage> messages;
};

using ConversationPartView =
    std::variant<TextPartView, ReasoningPartView, ToolCallView, ToolResultView>;
struct ConversationMessageView {
    MessageRole role{MessageRole::User};
    Span<const ConversationPartView> parts;
};
struct TextConversationRequest {
    Span<const ConversationMessageView> messages;
    Span<const ToolDefinitionView> tools;
};
using AssistantPart = std::variant<TextPart, ReasoningPart, ToolCall>;
enum class FinishReason : std::uint32_t {
    Unknown = 0,
    Stop = 1,
    Length = 2,
    ToolCalls = 3,
    ContentFilter = 4,
    Other = 5
};
struct TokenUsage {
    std::optional<std::uint64_t> input_tokens, output_tokens, total_tokens;
};
struct ConversationResult {
    std::vector<AssistantPart> parts; // Assistant response; no automatic tool execution.
    std::vector<int32_t> token_ids;
    double setup_ms{0}, prefill_ms{0}, decode_ms{0};
    FinishReason finish_reason{FinishReason::Unknown};
    std::string other_finish_reason{};
    TokenUsage usage{};
};
struct TextSpeechResult {
    TextResult text;
    AudioResult speech; // A paired response, not promised token/word alignment.
};
struct TextLabelClassificationRequest {
    std::string_view text;
};
struct TextPairLabelClassificationRequest {
    std::string_view first;
    std::string_view second;
};
struct GeneratedLabelResult {
    std::string raw_label;
    int64_t label_index{-1}; // -1 preserves an invalid/unmapped generated label.
    std::vector<std::string> vocabulary;
    std::vector<int32_t> token_ids;
};
struct LanguageTokenSequence {
    Span<const int32_t> token_ids; // Exact caller-supplied sequence, including special/pad tokens.
    Span<const uint8_t> attention_mask; // Empty means all valid; no automatic token insertion.
};
struct TextEncoderDecoderHiddenStatesRequest {
    LanguageTokenSequence source;
    LanguageTokenSequence decoder;
};
struct EncoderDecoderStatesResult {
    FloatMatrix encoder_last_hidden_state; // [source token, encoder hidden], including pad rows.
    FloatMatrix decoder_last_hidden_state; // [decoder token, decoder hidden], separate axis.
};

class IImagesTextToText {
  public:
    static constexpr std::string_view kTask = "images_text_to_text";
    virtual ~IImagesTextToText() = default;
    virtual TextResult run(const ImagesTextToTextRequest&, ConfigView) = 0;
};

class IVideoTextToText {
  public:
    static constexpr std::string_view kTask = "video_text_to_text";
    virtual ~IVideoTextToText() = default;
    virtual TextResult run(const VideoTextToTextRequest&, ConfigView) = 0;
};

class IImageVideoTextToText {
  public:
    static constexpr std::string_view kTask = "image_video_text_to_text";
    virtual ~IImageVideoTextToText() = default;
    virtual TextResult run(const ImageVideoTextToTextRequest&, ConfigView) = 0;
};

class IAudioTextToText {
  public:
    static constexpr std::string_view kTask = "audio_text_to_text";
    virtual ~IAudioTextToText() = default;
    virtual TextResult run(const AudioTextToTextRequest&, ConfigView) = 0;
};

class IImageAudioToText {
  public:
    static constexpr std::string_view kTask = "image_audio_to_text";
    virtual ~IImageAudioToText() = default;
    virtual TextResult run(const ImageAudioToTextRequest&, ConfigView) = 0;
};

class IAudioVideoTextToText {
  public:
    static constexpr std::string_view kTask = "audio_video_text_to_text";
    virtual ~IAudioVideoTextToText() = default;
    virtual TextResult run(const AudioVideoTextToTextRequest&, ConfigView) = 0;
};

class IImageAudioTextToText {
  public:
    static constexpr std::string_view kTask = "image_audio_text_to_text";
    virtual ~IImageAudioTextToText() = default;
    virtual TextResult run(const ImageAudioTextToTextRequest&, ConfigView) = 0;
};

class IImageAudioTextToTextSpeechResponse {
  public:
    static constexpr std::string_view kTask = "image_audio_text_to_text_speech_response";
    virtual ~IImageAudioTextToTextSpeechResponse() = default;
    virtual TextSpeechResult run(const ImageAudioTextToTextSpeechResponseRequest&, ConfigView) = 0;
};

class ITextConversation {
  public:
    static constexpr std::string_view kTask = "text_conversation";
    virtual ~ITextConversation() = default;
    virtual ConversationResult run(const TextConversationRequest&, ConfigView) = 0;
};
class IImagesTextConversation {
  public:
    static constexpr std::string_view kTask = "images_text_conversation";
    virtual ~IImagesTextConversation() = default;
    virtual ConversationResult run_conversation(const ImagesTextToTextRequest&, ConfigView) = 0;
};
class IVideoTextConversation {
  public:
    static constexpr std::string_view kTask = "video_text_conversation";
    virtual ~IVideoTextConversation() = default;
    virtual ConversationResult run_conversation(const VideoTextToTextRequest&, ConfigView) = 0;
};

// Complete audio conversations are separate from older plain-text and paired-speech Tasks.
using AudioTextConversationPart =
    std::variant<TextPartView, AudioView, ReasoningPartView, ToolCallView, ToolResultView>;
struct AudioTextConversationRequest {
    Span<const MediaMessage<AudioTextConversationPart>> messages;
    Span<const ToolDefinitionView> tools;
};
using ImageAudioTextConversationPart =
    std::variant<TextPartView, ImageView, AudioView, ReasoningPartView, ToolCallView,
                 ToolResultView>;
struct ImageAudioTextConversationRequest {
    Span<const MediaMessage<ImageAudioTextConversationPart>> messages;
    Span<const ToolDefinitionView> tools;
};
// Closed alternatives describe independent request domains, not a universal media input.
using TextImagesVideoConversationRequest =
    std::variant<TextConversationRequest, ImagesTextToTextRequest, VideoTextToTextRequest>;
using TextImagesAudioConversationRequest =
    std::variant<TextConversationRequest, ImagesTextToTextRequest, AudioTextConversationRequest,
                 ImageAudioTextConversationRequest>;

struct BatchTextConversationItem {
    TextConversationRequest input;
    ConfigView config;
};
struct BatchTextConversationRequest {
    using Item = BatchTextConversationItem;
    Span<const Item> items;
};
class IBatchTextConversation {
  public:
    static constexpr std::string_view kTask = "batch_text_conversation";
    using Request = BatchTextConversationRequest;
    virtual ~IBatchTextConversation() = default;
    virtual std::vector<ConversationResult> run_batch(const Request&) = 0;
};

struct BatchVideoTextConversationItem {
    VideoTextToTextRequest input;
    ConfigView config;
};
struct BatchVideoTextConversationRequest {
    using Item = BatchVideoTextConversationItem;
    Span<const Item> items;
};
class IBatchVideoTextConversation {
  public:
    static constexpr std::string_view kTask = "batch_video_text_conversation";
    using Request = BatchVideoTextConversationRequest;
    virtual ~IBatchVideoTextConversation() = default;
    virtual std::vector<ConversationResult> run_batch(const Request&) = 0;
};

struct BatchAudioTextConversationItem {
    AudioTextConversationRequest input;
    ConfigView config;
};
struct BatchAudioTextConversationRequest {
    using Item = BatchAudioTextConversationItem;
    Span<const Item> items;
};
class IBatchAudioTextConversation {
  public:
    static constexpr std::string_view kTask = "batch_audio_text_conversation";
    using Request = BatchAudioTextConversationRequest;
    virtual ~IBatchAudioTextConversation() = default;
    virtual std::vector<ConversationResult> run_batch(const Request&) = 0;
};

struct BatchImageAudioTextConversationItem {
    ImageAudioTextConversationRequest input;
    ConfigView config;
};
struct BatchImageAudioTextConversationRequest {
    using Item = BatchImageAudioTextConversationItem;
    Span<const Item> items;
};
class IBatchImageAudioTextConversation {
  public:
    static constexpr std::string_view kTask = "batch_image_audio_text_conversation";
    using Request = BatchImageAudioTextConversationRequest;
    virtual ~IBatchImageAudioTextConversation() = default;
    virtual std::vector<ConversationResult> run_batch(const Request&) = 0;
};

struct BatchTextImagesVideoConversationsItem {
    TextImagesVideoConversationRequest input;
    ConfigView config;
};
struct BatchTextImagesVideoConversationsRequest {
    using Item = BatchTextImagesVideoConversationsItem;
    Span<const Item> items;
};
class IBatchTextImagesVideoConversations {
  public:
    static constexpr std::string_view kTask = "batch_text_images_video_conversations";
    using Request = BatchTextImagesVideoConversationsRequest;
    virtual ~IBatchTextImagesVideoConversations() = default;
    virtual std::vector<ConversationResult> run_batch(const Request&) = 0;
};

struct BatchTextImagesAudioConversationsItem {
    TextImagesAudioConversationRequest input;
    ConfigView config;
};
struct BatchTextImagesAudioConversationsRequest {
    using Item = BatchTextImagesAudioConversationsItem;
    Span<const Item> items;
};
class IBatchTextImagesAudioConversations {
  public:
    static constexpr std::string_view kTask = "batch_text_images_audio_conversations";
    using Request = BatchTextImagesAudioConversationsRequest;
    virtual ~IBatchTextImagesAudioConversations() = default;
    virtual std::vector<ConversationResult> run_batch(const Request&) = 0;
};
// Independent conversations, not multiple images inside a single request.
// Family owns native batching and validates all item options before execution.
struct BatchImagesTextConversationItem {
    ImagesTextToTextRequest input;
    ConfigView config;
};
struct BatchImagesTextConversationRequest {
    using Item = BatchImagesTextConversationItem;
    Span<const Item> items;
};
class IBatchImagesTextConversation {
  public:
    static constexpr std::string_view kTask = "batch_images_text_conversation";
    using Request = BatchImagesTextConversationRequest;
    virtual ~IBatchImagesTextConversation() = default;
    virtual std::vector<ConversationResult> run_batch(const Request&) = 0;
};

class ITextLabelClassification {
  public:
    static constexpr std::string_view kTask = "text_label_classification";
    virtual ~ITextLabelClassification() = default;
    virtual GeneratedLabelResult run(const TextLabelClassificationRequest&, ConfigView) = 0;
};

class ITextPairLabelClassification {
  public:
    static constexpr std::string_view kTask = "text_pair_label_classification";
    virtual ~ITextPairLabelClassification() = default;
    virtual GeneratedLabelResult run(const TextPairLabelClassificationRequest&, ConfigView) = 0;
};

class ITextEncoderDecoderHiddenStates {
  public:
    static constexpr std::string_view kTask = "text_encoder_decoder_hidden_states";
    virtual ~ITextEncoderDecoderHiddenStates() = default;
    virtual EncoderDecoderStatesResult run(const TextEncoderDecoderHiddenStatesRequest&,
                                           ConfigView) = 0;
};

} // namespace trtmc::internal
