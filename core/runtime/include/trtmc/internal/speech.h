/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/audio.h"
#include "trtmc/internal/tools.h"

#include <functional>
#include <memory>

namespace trtmc::internal {

struct SpeechInputFormat {
    std::optional<std::uint32_t> sample_rate{};
    std::uint32_t channels{0};
};
struct SpeechAudioFormat {
    std::uint32_t sample_rate{0}, channels{0};
};
struct SpeechSessionInfo {
    SpeechAudioFormat input;
    std::optional<SpeechAudioFormat> output{};
    std::optional<std::string> source_language{};
    std::optional<std::string> system_prompt{};
};
struct StreamingSpeechTranscriptionRequest {
    SpeechInputFormat input;
    std::optional<std::string_view> source_language{};
};
struct SpeechTranscriptUpdate {
    TextResult transcript;
    bool is_final{false};
    std::uint64_t chunk_index{0};
    // Per-channel sample frames, not interleaved scalar values.
    std::uint64_t accepted_samples{0};
    SpeechAudioFormat input;
};

class ISpeechTranscriptionStream {
  public:
    virtual ~ISpeechTranscriptionStream() = default;
    // Reject recoverable input/lifecycle errors before state changes with
    // invalid_argument, ConfigError or UnsupportedTask. Unexpected execution
    // errors and failed update packaging permanently close the C handle.
    // Cumulative snapshots. finish is idempotent; reset reuses this stream.
    virtual SpeechTranscriptUpdate accept_audio(Span<const float>, bool is_final) = 0;
    virtual SpeechTranscriptUpdate finish() = 0;
    virtual void reset() = 0;
    virtual SpeechSessionInfo info() const = 0;
    // Values borrow immutable session-owned configuration; the C layer copies.
    virtual ConfigView effective_config() const = 0;
};
class IStreamingSpeechTranscription {
  public:
    using TaskInterface = IStreamingSpeechTranscription;
    static constexpr std::string_view kTask = "streaming_speech_transcription";
    virtual ~IStreamingSpeechTranscription() = default;
    virtual std::unique_ptr<ISpeechTranscriptionStream>
    create(const StreamingSpeechTranscriptionRequest&, ConfigView) = 0;
};

enum class AudioDeliveryOutcome : std::uint32_t { Complete = 1, Stopped = 2 };
struct StreamingAudioSummary {
    std::uint64_t emitted_sample_count{0}, emitted_frame_count{0};
    SpeechAudioFormat output;
    AudioDeliveryOutcome outcome{AudioDeliveryOutcome::Complete};
    double setup_ms{0}, inference_ms{0};
};
using AudioChunkSink = std::function<bool(const AudioView&)>;
class IStreamingTextToSpeech {
  public:
    using TaskInterface = IStreamingTextToSpeech;
    static constexpr std::string_view kTask = "streaming_text_to_speech";
    virtual ~IStreamingTextToSpeech() = default;
    // Invoke serially on the calling thread, only during run. Chunks borrow
    // family storage for the callback; false stops at that delivery boundary.
    virtual StreamingAudioSummary run(const TextToSpeechRequest&, ConfigView,
                                      const AudioChunkSink&) = 0;
};

struct SpeechDialogueRequest {
    SpeechInputFormat input;
    std::optional<std::string_view> system_prompt{};
};
struct ToolAcknowledgementView {
    std::string_view tool_name;
    Span<const std::string_view> messages;
};
struct ToolSpeechDialogueRequest {
    SpeechDialogueRequest dialogue;
    Span<const ToolDefinitionView> tools;
    Span<const ToolAcknowledgementView> acknowledgements;
    std::optional<Span<const std::string_view>> default_acknowledgements{};
};

enum class SpeechEventKind : std::uint32_t {
    AgentAudio = 1,
    AgentText,
    UserTranscript,
    TurnStarted,
    TurnFinished,
    Yielded,
    Cancelled,
    Reset,
    Error,
    InputFinished,
    UserSpeechStarted,
    UserSpeechStopped,
    FunctionCall,
    FunctionCallStarted,
    FunctionResponseFinished,
    InputCleared
};
struct SpeechDialogueEvent {
    SpeechEventKind kind{SpeechEventKind::Error};
    std::uint64_t epoch{0}, sequence{0};
    AudioResult audio;
    // -1 means unavailable. Media bounds are per-channel sample positions at
    // audio.sample_rate. frame_index is the family's logical model frame.
    std::int64_t media_start_sample{-1}, media_end_sample{-1}, frame_index{-1};
    std::string text;
    bool is_final{false};
    std::optional<ToolCall> tool_call{};
};
enum class SpeechReadState : std::uint32_t { Active = 1, EpochEnded = 2, Failed = 3 };
struct SpeechEventBatch {
    std::vector<SpeechDialogueEvent> events;
    SpeechReadState state{SpeechReadState::Active};
};

// Reuse the existing five-method internal C++ control contract unchanged.
using ISpeechRealtimeControl = trtmc::ISpeechRealtimeControl;
class ISpeechToolControl {
  public:
    virtual ~ISpeechToolControl() = default;
    virtual void submit_tool_result(std::uint64_t epoch, const ToolResultView&) = 0;
};
class ISpeechDialogueSession {
  public:
    // Destruction stops/joins family-owned work; it must not invoke user code.
    virtual ~ISpeechDialogueSession() = default;
    // Permanent, idempotent shutdown: wake readers/producers without allocation.
    // Distinct from resettable epoch cancel; destruction joins stopped work.
    virtual void stop() noexcept = 0;
    // Recoverable precondition errors use invalid_argument/ConfigError/
    // UnsupportedTask before changing state. Unexpected execution errors are
    // fatal. Any read_events exception is fatal because events may be consumed.
    // Thread-safe with event reading and controls. false accepts no samples and
    // reports bounded-queue backpressure; neither core nor caller drops input.
    virtual bool append_audio(Span<const float>) = 0;
    virtual void finish_input() = 0;
    virtual SpeechEventBatch read_events(std::int64_t timeout_ms) = 0;
    virtual void cancel() = 0;
    virtual void reset() = 0;
    virtual SpeechSessionInfo info() const = 0;
    virtual ConfigView effective_config() const = 0;
    virtual ISpeechRealtimeControl* realtime_control() noexcept { return nullptr; }
    virtual ISpeechToolControl* tool_control() noexcept { return nullptr; }
};

class IDuplexSpeechDialogue {
  public:
    using TaskInterface = IDuplexSpeechDialogue;
    static constexpr std::string_view kTask = "duplex_speech_dialogue";
    virtual ~IDuplexSpeechDialogue() = default;
    virtual std::unique_ptr<ISpeechDialogueSession>
    create_duplex_session(const SpeechDialogueRequest&, ConfigView) = 0;
};
class IOfflineSpeechDialogue {
  public:
    using TaskInterface = IOfflineSpeechDialogue;
    static constexpr std::string_view kTask = "offline_speech_dialogue";
    virtual ~IOfflineSpeechDialogue() = default;
    virtual std::unique_ptr<ISpeechDialogueSession>
    create_offline_session(const SpeechDialogueRequest&, ConfigView) = 0;
};
class IToolSpeechDialogue {
  public:
    using TaskInterface = IToolSpeechDialogue;
    static constexpr std::string_view kTask = "tool_speech_dialogue";
    virtual ~IToolSpeechDialogue() = default;
    virtual std::unique_ptr<ISpeechDialogueSession>
    create_tool_session(const ToolSpeechDialogueRequest&, ConfigView) = 0;
};

} // namespace trtmc::internal
