/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/audio.h"
#include "trtmc/scores.hpp"
#include "trtmc/text.hpp"

namespace trtmc {

struct AudioView {
    // Borrowed interleaved host PCM; samples.size() counts scalar values.
    Span<const float> samples;
    std::optional<std::uint32_t> sample_rate{};
    std::uint32_t channels{0};
};
struct TextToAudioRequest {
    std::string prompt;
};
struct AudioCodebookTokensView {
    Span<const std::int32_t> tokens;
    std::uint64_t codebooks{0}, frames{0};
    trtmc_audio_codebook_tokens_view_v1 c_view() const noexcept {
        return {{tokens.data(), tokens.size()}, codebooks, frames};
    }
};
struct SemanticAcousticHistoryView {
    Span<const std::int32_t> semantic_tokens;
    AudioCodebookTokensView coarse_tokens, fine_tokens;
    trtmc_semantic_acoustic_history_view_v1 c_view() const noexcept {
        return {{semantic_tokens.data(), semantic_tokens.size()},
                coarse_tokens.c_view(),
                fine_tokens.c_view()};
    }
};
struct TextAudioTokenHistoryToAudioRequest {
    std::string prompt;
    SemanticAcousticHistoryView history; // Tokens borrow caller storage through run.
};
struct TextToSpeechRequest {
    std::string text;
    std::optional<std::string> language{};
};
struct SpeechTranscriptionRequest {
    AudioView audio;
    std::optional<std::string> source_language{};
};
struct SpeechTranslationRequest {
    AudioView audio;
    std::optional<std::string> target_language{};
    std::optional<std::string> source_language{};
};
struct AudioLanguageIdentificationRequest {
    AudioView audio;
};
struct SpeechToSpeechResponseRequest {
    AudioView audio;
};

class AudioGenerationResult {
  public:
    AudioGenerationResult(std::shared_ptr<detail::ModelState> state, trtmc_result* result) noexcept
        : owner_(std::move(state), result) {}
    AudioGenerationResult(const AudioGenerationResult&) = delete;
    AudioGenerationResult& operator=(const AudioGenerationResult&) = delete;
    AudioGenerationResult(AudioGenerationResult&& other) noexcept
        : owner_(std::move(other.owner_)), view_(std::exchange(other.view_, {})) {}
    AudioGenerationResult& operator=(AudioGenerationResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }
    // Samples borrow this owner and retain their actual rate/channel layout.
    Span<const float> samples() const noexcept {
        return {view_.audio.samples, static_cast<std::size_t>(view_.audio.sample_count)};
    }
    std::uint32_t sample_rate() const noexcept { return view_.audio.sample_rate; }
    std::uint32_t channels() const noexcept { return view_.audio.channels; }
    std::size_t frame_count() const noexcept {
        return channels() ? samples().size() / channels() : 0;
    }
    double setup_ms() const noexcept { return view_.setup_ms; }
    double inference_ms() const noexcept { return view_.inference_ms; }
    trtmc_audio_result_view_v1& wire_view() noexcept { return view_; }

  private:
    detail::ResultOwner owner_;
    trtmc_audio_result_view_v1 view_{};
};

using TextToAudioResult = AudioGenerationResult;
using TextToSpeechResult = AudioGenerationResult;
using SpeechToSpeechResponseResult = AudioGenerationResult;
using SpeechTranscriptionResult = TextContinuationResult;
using SpeechTranslationResult = TextContinuationResult;
using AudioLanguageIdentificationResult = LabelScoresResult;
using BatchSpeechTranscriptionResult = BatchTextContinuationResult;
using BatchSpeechTranslationResult = BatchTextContinuationResult;
using MixedBatchSpeechToTextResult = BatchTextContinuationResult;

namespace detail {

inline trtmc_audio_view_v1 c_audio(const AudioView& input) noexcept {
    return {input.samples.data(), input.samples.size(), input.sample_rate ? 1U : 0U,
            input.sample_rate.value_or(0), input.channels};
}
inline trtmc_speech_transcription_request_v1
c_transcription(const SpeechTranscriptionRequest& input) noexcept {
    return {c_audio(input.audio), input.source_language ? 1U : 0U,
            input.source_language ? c_string(*input.source_language)
                                  : trtmc_string_view{nullptr, 0}};
}
inline trtmc_speech_translation_request_v1
c_speech_translation(const SpeechTranslationRequest& input) noexcept {
    return {
        c_audio(input.audio), input.target_language ? 1U : 0U,
        input.target_language ? c_string(*input.target_language) : trtmc_string_view{nullptr, 0},
        input.source_language ? 1U : 0U,
        input.source_language ? c_string(*input.source_language) : trtmc_string_view{nullptr, 0}};
}
template <class Table>
void validate_audio_table(const trtmc_api_header* table) {
    if (table == nullptr || table->major != 1 || table->minor != 0 ||
        table->byte_size < sizeof(Table))
        throw Error(TRTMC_VERSION_MISMATCH, "incompatible audio Task API table");
}
template <class Table, class Request>
AudioGenerationResult run_audio(const std::shared_ptr<ModelState>& state, const Table* table,
                                const Request& input, const Config& config) {
    auto entries = config.c_entries();
    auto options = entries.view();
    trtmc_result* raw = nullptr;
    trtmc_error* error = nullptr;
    auto status = table->run(state->handle, &input, &options, &raw, &error);
    AudioGenerationResult result(state, raw);
    check(state->api, status, error);
    error = nullptr;
    status = table->result_view(raw, &result.wire_view(), &error);
    check(state->api, status, error);
    return result;
}

} // namespace detail

class TextAudioTokenHistoryToAudio {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_AUDIO_TOKEN_HISTORY_TO_AUDIO;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    AudioGenerationResult run(const TextAudioTokenHistoryToAudioRequest& input,
                              const Config& config = {}) const {
        const trtmc_text_audio_token_history_to_audio_request_v1 request{
            detail::c_string(input.prompt), input.history.c_view()};
        return detail::run_audio(state_, api_, request, config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_text_audio_token_history_to_audio_api_v1>(table);
    }

  private:
    friend class Model;
    TextAudioTokenHistoryToAudio(std::shared_ptr<detail::ModelState> state,
                                 const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_text_audio_token_history_to_audio_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_text_audio_token_history_to_audio_api_v1* api_;
};

class TextToAudio {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_TO_AUDIO;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    TextToAudioResult run(const TextToAudioRequest& input, const Config& config = {}) const {
        const trtmc_text_to_audio_request_v1 request{detail::c_string(input.prompt)};
        return detail::run_audio(state_, api_, request, config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_text_to_audio_api_v1>(table);
    }

  private:
    friend class Model;
    TextToAudio(std::shared_ptr<detail::ModelState> state, const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_text_to_audio_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_text_to_audio_api_v1* api_;
};

class TextToSpeech {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_TO_SPEECH;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    TextToSpeechResult run(const TextToSpeechRequest& input, const Config& config = {}) const {
        const trtmc_text_to_speech_request_v1 request{
            detail::c_string(input.text), input.language ? 1U : 0U,
            input.language ? detail::c_string(*input.language) : trtmc_string_view{nullptr, 0}};
        return detail::run_audio(state_, api_, request, config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_text_to_speech_api_v1>(table);
    }

  private:
    friend class Model;
    TextToSpeech(std::shared_ptr<detail::ModelState> state, const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_text_to_speech_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_text_to_speech_api_v1* api_;
};

class SpeechTranscription {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_SPEECH_TRANSCRIPTION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    SpeechTranscriptionResult run(const SpeechTranscriptionRequest& input,
                                  const Config& config = {}) const {
        return detail::run_text(state_, api_, detail::c_transcription(input), config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_speech_transcription_api_v1>(table);
    }

  private:
    friend class Model;
    SpeechTranscription(std::shared_ptr<detail::ModelState> state,
                        const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_speech_transcription_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_speech_transcription_api_v1* api_;
};

class SpeechTranslation {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_SPEECH_TRANSLATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    SpeechTranslationResult run(const SpeechTranslationRequest& input,
                                const Config& config = {}) const {
        return detail::run_text(state_, api_, detail::c_speech_translation(input), config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_speech_translation_api_v1>(table);
    }

  private:
    friend class Model;
    SpeechTranslation(std::shared_ptr<detail::ModelState> state,
                      const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_speech_translation_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_speech_translation_api_v1* api_;
};

class AudioLanguageIdentification {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_AUDIO_LANGUAGE_IDENTIFICATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    AudioLanguageIdentificationResult run(const AudioLanguageIdentificationRequest& input,
                                          const Config& config = {}) const {
        const trtmc_audio_language_identification_request_v1 request{detail::c_audio(input.audio)};
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run(state_->handle, &request, &options, &raw, &error);
        LabelScoresResult result(state_, raw);
        detail::check(state_->api, status, error);
        error = nullptr;
        status = api_->result_view(raw, &result.wire_view(), &error);
        detail::check(state_->api, status, error);
        return result;
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_audio_language_identification_api_v1>(table);
    }

  private:
    friend class Model;
    AudioLanguageIdentification(std::shared_ptr<detail::ModelState> state,
                                const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_audio_language_identification_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_audio_language_identification_api_v1* api_;
};

class SpeechToSpeechResponse {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_SPEECH_TO_SPEECH_RESPONSE;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    SpeechToSpeechResponseResult run(const SpeechToSpeechResponseRequest& input,
                                     const Config& config = {}) const {
        const trtmc_speech_to_speech_response_request_v1 request{detail::c_audio(input.audio)};
        return detail::run_audio(state_, api_, request, config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_speech_to_speech_response_api_v1>(table);
    }

  private:
    friend class Model;
    SpeechToSpeechResponse(std::shared_ptr<detail::ModelState> state,
                           const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_speech_to_speech_response_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_speech_to_speech_response_api_v1* api_;
};

struct BatchSpeechTranscriptionItem {
    SpeechTranscriptionRequest input;
    Config config;
};
struct BatchSpeechTranscriptionRequest {
    std::vector<BatchSpeechTranscriptionItem> items;
};
struct BatchSpeechTranslationItem {
    SpeechTranslationRequest input;
    Config config;
};
struct BatchSpeechTranslationRequest {
    std::vector<BatchSpeechTranslationItem> items;
};

class BatchSpeechTranscription {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_SPEECH_TRANSCRIPTION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    BatchSpeechTranscriptionResult run(const BatchSpeechTranscriptionRequest& input) const {
        std::vector<Config::CEntries> configs;
        std::vector<trtmc_batch_speech_transcription_item_v1> items;
        configs.reserve(input.items.size());
        items.reserve(input.items.size());
        for (const auto& item : input.items) {
            configs.push_back(item.config.c_entries());
            items.push_back({detail::c_transcription(item.input), configs.back().view()});
        }
        const trtmc_batch_speech_transcription_request_v1 request{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run_batch(state_->handle, &request, &raw, &error);
        return detail::BatchTextResultAccess::finish(state_, status, raw, error, api_->result_count,
                                                     api_->result_item_view);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_batch_speech_transcription_api_v1>(table);
    }

  private:
    friend class Model;
    BatchSpeechTranscription(std::shared_ptr<detail::ModelState> state,
                             const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_batch_speech_transcription_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_batch_speech_transcription_api_v1* api_;
};

class BatchSpeechTranslation {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_SPEECH_TRANSLATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    BatchSpeechTranslationResult run(const BatchSpeechTranslationRequest& input) const {
        std::vector<Config::CEntries> configs;
        std::vector<trtmc_batch_speech_translation_item_v1> items;
        configs.reserve(input.items.size());
        items.reserve(input.items.size());
        for (const auto& item : input.items) {
            configs.push_back(item.config.c_entries());
            items.push_back({detail::c_speech_translation(item.input), configs.back().view()});
        }
        const trtmc_batch_speech_translation_request_v1 request{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run_batch(state_->handle, &request, &raw, &error);
        return detail::BatchTextResultAccess::finish(state_, status, raw, error, api_->result_count,
                                                     api_->result_item_view);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_batch_speech_translation_api_v1>(table);
    }

  private:
    friend class Model;
    BatchSpeechTranslation(std::shared_ptr<detail::ModelState> state,
                           const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_batch_speech_translation_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_batch_speech_translation_api_v1* api_;
};

using MixedSpeechTextRequest = std::variant<SpeechTranscriptionRequest, SpeechTranslationRequest>;
struct MixedBatchSpeechToTextItem {
    MixedSpeechTextRequest input;
    Config config;
};
struct MixedBatchSpeechToTextRequest {
    std::vector<MixedBatchSpeechToTextItem> items;
};

class MixedBatchSpeechToText {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_MIXED_BATCH_SPEECH_TO_TEXT;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    MixedBatchSpeechToTextResult run(const MixedBatchSpeechToTextRequest& input) const {
        std::vector<Config::CEntries> configs;
        std::vector<trtmc_mixed_batch_speech_to_text_item_v1> items;
        configs.reserve(input.items.size());
        items.reserve(input.items.size());
        for (const auto& item : input.items) {
            configs.push_back(item.config.c_entries());
            trtmc_mixed_batch_speech_to_text_item_v1 wire{};
            wire.config = configs.back().view();
            if (const auto* transcription = std::get_if<SpeechTranscriptionRequest>(&item.input)) {
                wire.kind = TRTMC_SPEECH_TEXT_TRANSCRIPTION;
                wire.input.transcription = detail::c_transcription(*transcription);
            } else {
                wire.kind = TRTMC_SPEECH_TEXT_TRANSLATION;
                wire.input.translation =
                    detail::c_speech_translation(std::get<SpeechTranslationRequest>(item.input));
            }
            items.push_back(wire);
        }
        const trtmc_mixed_batch_speech_to_text_request_v1 request{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run_batch(state_->handle, &request, &raw, &error);
        return detail::BatchTextResultAccess::finish(state_, status, raw, error, api_->result_count,
                                                     api_->result_item_view);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_mixed_batch_speech_to_text_api_v1>(table);
    }

  private:
    friend class Model;
    MixedBatchSpeechToText(std::shared_ptr<detail::ModelState> state,
                           const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_mixed_batch_speech_to_text_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_mixed_batch_speech_to_text_api_v1* api_;
};

struct BatchTextToAudioItem {
    TextToAudioRequest input;
    Config config;
};
struct BatchTextToSpeechItem {
    TextToSpeechRequest input;
    Config config;
};
struct BatchTextToAudioRequest {
    std::vector<BatchTextToAudioItem> items;
};
struct BatchTextAudioTokenHistoryToAudioRequest {
    SemanticAcousticHistoryView history;
    std::vector<BatchTextToAudioItem> items;
};
struct BatchTextToSpeechRequest {
    std::vector<BatchTextToSpeechItem> items;
};
struct AudioResultView {
    Span<const float> samples;
    std::uint32_t sample_rate;
    std::uint32_t channels;
    double setup_ms;
    double inference_ms;
    std::size_t frame_count() const noexcept { return channels ? samples.size() / channels : 0; }
};

class AudioBatchResult {
  public:
    AudioBatchResult(const AudioBatchResult&) = delete;
    AudioBatchResult& operator=(const AudioBatchResult&) = delete;
    AudioBatchResult(AudioBatchResult&& other) noexcept
        : owner_(std::move(other.owner_)), read_(other.read_),
          count_(std::exchange(other.count_, 0)) {}
    AudioBatchResult& operator=(AudioBatchResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            read_ = other.read_;
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    std::uint64_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }
    // Views borrow this batch result. They contain actual, unpadded PCM and do
    // not create independently owned result/model handles.
    AudioResultView at(std::uint64_t index) const {
        if (index >= count_)
            throw Error(TRTMC_INVALID_ARGUMENT, "audio item index is out of range");
        trtmc_audio_result_view_v1 view{};
        trtmc_error* error = nullptr;
        const auto status = read_(owner_.get(), index, &view, &error);
        detail::check(owner_.api(), status, error);
        return {{view.audio.samples, static_cast<std::size_t>(view.audio.sample_count)},
                view.audio.sample_rate,
                view.audio.channels,
                view.setup_ms,
                view.inference_ms};
    }
    AudioResultView operator[](std::uint64_t index) const { return at(index); }

  private:
    friend class BatchTextToAudio;
    friend class BatchTextAudioTokenHistoryToAudio;
    friend class BatchTextToSpeech;
    using Count = trtmc_status(TRTMC_CALL*)(const trtmc_result*, std::uint64_t*, trtmc_error**);
    using Read = trtmc_status(TRTMC_CALL*)(const trtmc_result*, std::uint64_t,
                                           trtmc_audio_result_view_v1*, trtmc_error**);
    AudioBatchResult(std::shared_ptr<detail::ModelState> state, trtmc_result* result,
                     Read read) noexcept
        : owner_(std::move(state), result), read_(read) {}
    static AudioBatchResult finish(const std::shared_ptr<detail::ModelState>& state,
                                   trtmc_status status, trtmc_result* raw, trtmc_error* error,
                                   Count count, Read read) {
        AudioBatchResult result(state, raw, read); // Own before checking either call.
        detail::check(state->api, status, error);
        error = nullptr;
        status = count(raw, &result.count_, &error);
        detail::check(state->api, status, error);
        return result;
    }
    detail::ResultOwner owner_;
    Read read_;
    std::uint64_t count_{0};
};
using BatchTextToAudioResult = AudioBatchResult;
using BatchTextToSpeechResult = AudioBatchResult;

class BatchTextAudioTokenHistoryToAudio {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_TEXT_AUDIO_TOKEN_HISTORY_TO_AUDIO;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    AudioBatchResult run(const BatchTextAudioTokenHistoryToAudioRequest& input) const {
        std::vector<Config::CEntries> configs;
        std::vector<trtmc_batch_text_to_audio_item_v1> items;
        configs.reserve(input.items.size());
        items.reserve(input.items.size());
        for (const auto& item : input.items) {
            configs.push_back(item.config.c_entries());
            items.push_back({{detail::c_string(item.input.prompt)}, configs.back().view()});
        }
        const trtmc_batch_text_audio_token_history_to_audio_request_v1 request{
            input.history.c_view(), items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run_batch(state_->handle, &request, &raw, &error);
        return AudioBatchResult::finish(state_, status, raw, error, api_->result_count,
                                        api_->result_item_view);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_batch_text_audio_token_history_to_audio_api_v1>(table);
    }

  private:
    friend class Model;
    BatchTextAudioTokenHistoryToAudio(std::shared_ptr<detail::ModelState> state,
                                      const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_batch_text_audio_token_history_to_audio_api_v1*>(
              table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_batch_text_audio_token_history_to_audio_api_v1* api_;
};

class BatchTextToAudio {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_TEXT_TO_AUDIO;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    BatchTextToAudioResult run(const BatchTextToAudioRequest& input) const {
        std::vector<Config::CEntries> configs;
        std::vector<trtmc_batch_text_to_audio_item_v1> items;
        configs.reserve(input.items.size());
        items.reserve(input.items.size());
        for (const auto& item : input.items) {
            configs.push_back(item.config.c_entries());
            items.push_back({{detail::c_string(item.input.prompt)}, configs.back().view()});
        }
        const trtmc_batch_text_to_audio_request_v1 request{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run_batch(state_->handle, &request, &raw, &error);
        return AudioBatchResult::finish(state_, status, raw, error, api_->result_count,
                                        api_->result_item_view);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_batch_text_to_audio_api_v1>(table);
    }

  private:
    friend class Model;
    BatchTextToAudio(std::shared_ptr<detail::ModelState> state,
                     const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_batch_text_to_audio_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_batch_text_to_audio_api_v1* api_;
};

class BatchTextToSpeech {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_TEXT_TO_SPEECH;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    BatchTextToSpeechResult run(const BatchTextToSpeechRequest& input) const {
        std::vector<Config::CEntries> configs;
        std::vector<trtmc_batch_text_to_speech_item_v1> items;
        configs.reserve(input.items.size());
        items.reserve(input.items.size());
        for (const auto& item : input.items) {
            configs.push_back(item.config.c_entries());
            items.push_back({{detail::c_string(item.input.text), item.input.language ? 1U : 0U,
                              item.input.language ? detail::c_string(*item.input.language)
                                                  : trtmc_string_view{}},
                             configs.back().view()});
        }
        const trtmc_batch_text_to_speech_request_v1 request{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run_batch(state_->handle, &request, &raw, &error);
        return AudioBatchResult::finish(state_, status, raw, error, api_->result_count,
                                        api_->result_item_view);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_audio_table<trtmc_batch_text_to_speech_api_v1>(table);
    }

  private:
    friend class Model;
    BatchTextToSpeech(std::shared_ptr<detail::ModelState> state,
                      const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_batch_text_to_speech_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_batch_text_to_speech_api_v1* api_;
};

} // namespace trtmc
