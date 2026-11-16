/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/config.h"
#include "trtmc/internal/scores.h"
#include "trtmc/internal/text.h"

namespace trtmc::internal {

struct AudioView {
    // Interleaved host PCM float32. The span counts scalar values; dividing by
    // channels gives audio frames. An omitted rate uses only the loaded family's
    // declared input rate; a family without that default rejects omission.
    Span<const float> samples;
    std::optional<std::uint32_t> sample_rate{};
    std::uint32_t channels{0};
};

struct AudioResult {
    std::vector<float> samples;
    std::uint32_t sample_rate{0};
    std::uint32_t channels{0};
    double setup_ms{0};
    double inference_ms{0};
};

struct TextToAudioRequest {
    std::string_view prompt;
};

// Integer token history, not PCM or floating speaker embeddings. The two
// acoustic matrices are codebook-major and have independent frame lengths.
struct AudioCodebookTokensView {
    Span<const std::int32_t> tokens;
    std::uint64_t codebooks{0}, frames{0};
};
struct SemanticAcousticHistoryView {
    Span<const std::int32_t> semantic_tokens;
    AudioCodebookTokensView coarse_tokens, fine_tokens;
};
struct TextAudioTokenHistoryToAudioRequest {
    std::string_view prompt;
    SemanticAcousticHistoryView history;
};
struct TextToSpeechRequest {
    std::string_view text;
    std::optional<std::string_view> language{};
};
struct SpeechTranscriptionRequest {
    AudioView audio;
    std::optional<std::string_view> source_language{};
};
struct SpeechTranslationRequest {
    AudioView audio;
    std::optional<std::string_view> target_language{};
    std::optional<std::string_view> source_language{};
};
struct AudioLanguageIdentificationRequest {
    AudioView audio;
};
struct AudioToClassScoresRequest {
    AudioView audio;
};
struct AudioToTimedLabelsRequest {
    AudioView audio;
};

// One active interval of one label, [start_seconds, end_seconds) seconds from
// the first input frame. label indexes TimedLabelsResult::labels.
struct TimedLabel {
    double start_seconds{0};
    double end_seconds{0};
    std::uint32_t label{0};
};

struct TimedLabelsResult {
    // Ordered by start, then label; one label's intervals never overlap.
    std::vector<TimedLabel> intervals;
    // One nonempty name per label ordinal. An empty vocabulary_id means the
    // names have no identity beyond this result (anonymous speakers numbered
    // by first appearance in each input).
    std::vector<std::string> labels;
    std::string vocabulary_id;
    // Optional raw frame scores, row-major [frame_count, labels.size()]; frame
    // f covers [f * frame_seconds, (f + 1) * frame_seconds). Empty with
    // frame_count and frame_seconds zero when the family has none.
    std::vector<float> frame_scores;
    std::uint64_t frame_count{0};
    double frame_seconds{0};
    ScoreKind score_kind{ScoreKind::Logit};
    double setup_ms{0};
    double inference_ms{0};
};
struct SpeechToSpeechResponseRequest {
    AudioView audio;
};

class ITextToAudio {
  public:
    using TaskInterface = ITextToAudio;
    static constexpr std::string_view kTask = "text_to_audio";
    virtual ~ITextToAudio() = default;
    virtual AudioResult run(const TextToAudioRequest&, ConfigView) = 0;
};
class ITextAudioTokenHistoryToAudio {
  public:
    using TaskInterface = ITextAudioTokenHistoryToAudio;
    static constexpr std::string_view kTask = "text_audio_token_history_to_audio";
    virtual ~ITextAudioTokenHistoryToAudio() = default;
    // Required history borrows caller storage through synchronous return.
    // Family owns vocabulary/count/length/alignment rules and never mutates it.
    virtual AudioResult run(const TextAudioTokenHistoryToAudioRequest&, ConfigView) = 0;
};
class ITextToSpeech {
  public:
    using TaskInterface = ITextToSpeech;
    static constexpr std::string_view kTask = "text_to_speech";
    virtual ~ITextToSpeech() = default;
    // Speech realizes the supplied transcript; speaker/normalization controls
    // belong to the family-declared config, not a different output Task.
    virtual AudioResult run(const TextToSpeechRequest&, ConfigView) = 0;
};
class ISpeechTranscription {
  public:
    using TaskInterface = ISpeechTranscription;
    static constexpr std::string_view kTask = "speech_transcription";
    virtual ~ISpeechTranscription() = default;
    // Same-language transcript. Segment timestamps are not implied word alignment.
    virtual TextResult run(const SpeechTranscriptionRequest&, ConfigView) = 0;
};
class ISpeechTranslation {
  public:
    using TaskInterface = ISpeechTranslation;
    static constexpr std::string_view kTask = "speech_translation";
    virtual ~ISpeechTranslation() = default;
    // Absent language values use only the loaded family's declared defaults;
    // if no applicable default exists, the family rejects the request.
    virtual TextResult run(const SpeechTranslationRequest&, ConfigView) = 0;
};
class IAudioLanguageIdentification {
  public:
    using TaskInterface = IAudioLanguageIdentification;
    static constexpr std::string_view kTask = "audio_language_identification";
    virtual ~IAudioLanguageIdentification() = default;
    // Every score needs its actual language label; ordinal-only output is not
    // sufficient for language identification. The kind states score semantics.
    virtual LabelScoresResult run(const AudioLanguageIdentificationRequest&, ConfigView) = 0;
};
class IAudioToClassScores {
  public:
    using TaskInterface = IAudioToClassScores;
    static constexpr std::string_view kTask = "audio_to_class_scores";
    virtual ~IAudioToClassScores() = default;
    // Nonempty scores over the family's own sound-event vocabulary, in class-ordinal
    // order. This is general audio tagging, not language identification: the labels
    // name sound events. Empty labels and vocabulary_id mean model-local ordinal
    // order with unknown identity, not cross-model compatibility. Supplied labels
    // have one entry per score; without a vocabulary_id every label must be
    // nonempty. ScoreKind declares interpretation without shared normalization,
    // reordering or invented class metadata. Multi-label families return per-class
    // scores that do not sum to one; the application never renormalizes them.
    virtual LabelScoresResult run(const AudioToClassScoresRequest&, ConfigView) = 0;
};
class IAudioToTimedLabels {
  public:
    using TaskInterface = IAudioToTimedLabels;
    static constexpr std::string_view kTask = "audio_to_timed_labels";
    virtual ~IAudioToTimedLabels() = default;
    // Who or what is active when over the whole input: speaker diarization,
    // voice activity or sound-event detection. Intervals are on the input clock,
    // in seconds, ordered by start then label, and never overlap within one
    // label. Every label ordinal is named. Frame scores, when the family has
    // them, are the raw per-frame per-label scores the intervals were derived
    // from, in the declared ScoreKind; nothing renormalizes across labels.
    virtual TimedLabelsResult run(const AudioToTimedLabelsRequest&, ConfigView) = 0;
};
class ISpeechToSpeechResponse {
  public:
    using TaskInterface = ISpeechToSpeechResponse;
    static constexpr std::string_view kTask = "speech_to_speech_response";
    virtual ~ISpeechToSpeechResponse() = default;
    // A response to the complete user speech, not arbitrary audio transformation.
    virtual AudioResult run(const SpeechToSpeechResponseRequest&, ConfigView) = 0;
};

struct BatchSpeechTranscriptionItem {
    SpeechTranscriptionRequest input;
    ConfigView config;
};
struct BatchSpeechTranscriptionRequest {
    Span<const BatchSpeechTranscriptionItem> items;
};
struct BatchSpeechTranslationItem {
    SpeechTranslationRequest input;
    ConfigView config;
};
struct BatchSpeechTranslationRequest {
    Span<const BatchSpeechTranslationItem> items;
};

class IBatchSpeechTranscription {
  public:
    using TaskInterface = IBatchSpeechTranscription;
    static constexpr std::string_view kTask = "batch_speech_transcription";
    virtual ~IBatchSpeechTranscription() = default;
    virtual BatchTextResult run_batch(const BatchSpeechTranscriptionRequest&) = 0;
};
class IBatchSpeechTranslation {
  public:
    using TaskInterface = IBatchSpeechTranslation;
    static constexpr std::string_view kTask = "batch_speech_translation";
    virtual ~IBatchSpeechTranslation() = default;
    virtual BatchTextResult run_batch(const BatchSpeechTranslationRequest&) = 0;
};

using MixedSpeechTextRequest = std::variant<SpeechTranscriptionRequest, SpeechTranslationRequest>;
struct MixedBatchSpeechToTextItem {
    MixedSpeechTextRequest input;
    ConfigView config;
};
struct MixedBatchSpeechToTextRequest {
    Span<const MixedBatchSpeechToTextItem> items;
};

// A native batch of independently typed transcription/translation requests.
// Implementing a homogeneous batch does not imply this mixed contract.
class IMixedBatchSpeechToText {
  public:
    using TaskInterface = IMixedBatchSpeechToText;
    static constexpr std::string_view kTask = "mixed_batch_speech_to_text";
    virtual ~IMixedBatchSpeechToText() = default;
    virtual BatchTextResult run_batch(const MixedBatchSpeechToTextRequest&) = 0;
};

struct BatchTextToAudioItem {
    TextToAudioRequest input;
    ConfigView config;
};
struct BatchTextToSpeechItem {
    TextToSpeechRequest input;
    ConfigView config;
};
struct BatchTextToAudioRequest {
    Span<const BatchTextToAudioItem> items;
};
struct BatchTextAudioTokenHistoryToAudioRequest {
    SemanticAcousticHistoryView history; // Exactly one shared speaker/history for this batch.
    Span<const BatchTextToAudioItem> items;
};
struct BatchTextToSpeechRequest {
    Span<const BatchTextToSpeechItem> items;
};
using BatchAudioResult = std::vector<AudioResult>;

// Nonempty native request batches. Resolve/validate all per-item options and
// cross-item restrictions before generation. Success preserves order/count;
// failure has no partial result. Do not synthesize these with single.run loops.
// Each result owns only its actual PCM, excluding batch padding, with its
// actual sample rate/channels. Voice compatibility and RNG policy are family-owned.
class IBatchTextToAudio {
  public:
    using TaskInterface = IBatchTextToAudio;
    static constexpr std::string_view kTask = "batch_text_to_audio";
    virtual ~IBatchTextToAudio() = default;
    virtual BatchAudioResult run_batch(const BatchTextToAudioRequest&) = 0;
};
class IBatchTextAudioTokenHistoryToAudio {
  public:
    using TaskInterface = IBatchTextAudioTokenHistoryToAudio;
    static constexpr std::string_view kTask = "batch_text_audio_token_history_to_audio";
    virtual ~IBatchTextAudioTokenHistoryToAudio() = default;
    // One immutable typed history conditions every independent prompt. This
    // does not imply support for different histories per item.
    virtual BatchAudioResult run_batch(const BatchTextAudioTokenHistoryToAudioRequest&) = 0;
};
class IBatchTextToSpeech {
  public:
    using TaskInterface = IBatchTextToSpeech;
    static constexpr std::string_view kTask = "batch_text_to_speech";
    virtual ~IBatchTextToSpeech() = default;
    virtual BatchAudioResult run_batch(const BatchTextToSpeechRequest&) = 0;
};

} // namespace trtmc::internal
