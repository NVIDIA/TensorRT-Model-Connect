/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "api_internal.h"

namespace trtmc::api {

internal::AudioView audio_view(const trtmc_audio_view_v1& input) {
    require(input.channels > 0, "audio channel count must be positive");
    require(input.has_sample_rate <= 1, "sample-rate presence must be zero or one");
    std::optional<std::uint32_t> sample_rate;
    if (input.has_sample_rate) {
        require(input.sample_rate > 0, "present audio sample rate must be positive");
        sample_rate = input.sample_rate;
    }
    const auto samples = checked_span(input.samples, input.sample_count);
    require(samples.size() % input.channels == 0, "interleaved audio has an incomplete frame");
    return {samples, sample_rate, input.channels};
}

AudioResultStorage::AudioResultStorage(internal::AudioResult value) : result(std::move(value)) {
    if (result.sample_rate == 0 || result.channels == 0 ||
        result.samples.size() % result.channels != 0)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned invalid PCM audio metadata"};
}

void fill_audio_result_view(const AudioResultStorage& storage,
                            trtmc_audio_result_view_v1* output) noexcept {
    const auto& result = storage.result;
    *output = {
        {result.samples.data(), result.samples.size(), 1, result.sample_rate, result.channels},
        result.setup_ms,
        result.inference_ms};
}

trtmc_status TRTMC_CALL audio_result_view(const trtmc_result* result,
                                          trtmc_audio_result_view_v1* output,
                                          trtmc_error** error) noexcept {
    if (output)
        *output = {};
    return guarded(error, [&] {
        require(output != nullptr, "audio result view output is required");
        fill_audio_result_view(require_result<AudioResultStorage>(result), output);
    });
}

namespace {

internal::AudioCodebookTokensView
codebook_history(const trtmc_audio_codebook_tokens_view_v1& input) {
    const auto tokens = checked_span(input.tokens.data, input.tokens.size);
    require(input.frames == 0 || input.codebooks <= UINT64_MAX / input.frames,
            "audio history shape product overflows");
    require(input.tokens.size == input.codebooks * input.frames,
            "audio history token count differs from codebook/frame shape");
    return {tokens, input.codebooks, input.frames};
}
internal::SemanticAcousticHistoryView
token_history(const trtmc_semantic_acoustic_history_view_v1& input) {
    return {checked_span(input.semantic_tokens.data, input.semantic_tokens.size),
            codebook_history(input.coarse_tokens), codebook_history(input.fine_tokens)};
}
trtmc_status TRTMC_CALL history_audio_run(
    trtmc_model* model, const trtmc_text_audio_token_history_to_audio_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "audio history request and output are required");
        const internal::TextAudioTokenHistoryToAudioRequest request{string_view(input->prompt),
                                                                    token_history(input->history)};
        const ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::ITextAudioTokenHistoryToAudio>(
            model, internal::ITextAudioTokenHistoryToAudio::kTask);
        *output = make_result<AudioResultStorage>(family.run(request, options.view()));
    });
}

std::optional<std::string_view> language(std::uint32_t present, trtmc_string_view value) {
    require(present <= 1, "language presence must be zero or one");
    if (!present)
        return std::nullopt;
    const auto text = string_view(value);
    require(!text.empty(), "present language must not be empty");
    return text;
}

internal::SpeechTranscriptionRequest
transcription_request(const trtmc_speech_transcription_request_v1& input) {
    return {audio_view(input.audio), language(input.has_source_language, input.source_language)};
}

internal::SpeechTranslationRequest
translation_request(const trtmc_speech_translation_request_v1& input) {
    return {audio_view(input.audio), language(input.has_target_language, input.target_language),
            language(input.has_source_language, input.source_language)};
}

extern "C" trtmc_status TRTMC_CALL text_to_audio_run(trtmc_model* model,
                                                     const trtmc_text_to_audio_request_v1* input,
                                                     const trtmc_config_view_v1* config,
                                                     trtmc_result** output,
                                                     trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "audio request and result output are required");
        const internal::TextToAudioRequest request{string_view(input->prompt)};
        const ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task =
            require_interface<internal::ITextToAudio>(model, internal::ITextToAudio::kTask);
        *output = make_result<AudioResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL text_to_speech_run(trtmc_model* model,
                                                      const trtmc_text_to_speech_request_v1* input,
                                                      const trtmc_config_view_v1* config,
                                                      trtmc_result** output,
                                                      trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "speech request and result output are required");
        const internal::TextToSpeechRequest request{string_view(input->text),
                                                    language(input->has_language, input->language)};
        const ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task =
            require_interface<internal::ITextToSpeech>(model, internal::ITextToSpeech::kTask);
        *output = make_result<AudioResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL speech_transcription_run(
    trtmc_model* model, const trtmc_speech_transcription_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "transcription request and result output are required");
        const auto request = transcription_request(*input);
        const ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::ISpeechTranscription>(
            model, internal::ISpeechTranscription::kTask);
        *output = make_result<TextResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL speech_translation_run(
    trtmc_model* model, const trtmc_speech_translation_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "speech translation request and result output are required");
        const auto request = translation_request(*input);
        const ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::ISpeechTranslation>(
            model, internal::ISpeechTranslation::kTask);
        *output = make_result<TextResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL language_identification_run(
    trtmc_model* model, const trtmc_audio_language_identification_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "language identification request and result output are required");
        const internal::AudioLanguageIdentificationRequest request{audio_view(input->audio)};
        const ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::IAudioLanguageIdentification>(
            model, internal::IAudioLanguageIdentification::kTask);
        auto result = task.run(request, options.view());
        if (result.labels.size() != result.scores.size())
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "language identification needs one language label per score"};
        for (const auto& label : result.labels)
            if (label.empty())
                throw ApiFailure{TRTMC_INTERNAL_ERROR,
                                 "language identification returned an empty language label"};
        *output = make_result<LabelScoresStorage>(std::move(result));
    });
}

extern "C" trtmc_status TRTMC_CALL speech_response_run(
    trtmc_model* model, const trtmc_speech_to_speech_response_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "speech response request and result output are required");
        const internal::SpeechToSpeechResponseRequest request{audio_view(input->audio)};
        const ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::ISpeechToSpeechResponse>(
            model, internal::ISpeechToSpeechResponse::kTask);
        *output = make_result<AudioResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL batch_transcription_run(
    trtmc_model* model, const trtmc_batch_speech_transcription_request_v1* input,
    trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "transcription batch and result output are required");
        const auto source = checked_span(input->items, input->count);
        std::vector<ConvertedConfig> configs;
        std::vector<internal::BatchSpeechTranscriptionItem> items;
        configs.reserve(source.size());
        items.reserve(source.size());
        for (const auto& item : source) {
            configs.emplace_back(&item.config);
            items.push_back({transcription_request(item.input), configs.back().view()});
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::IBatchSpeechTranscription>(
            model, internal::IBatchSpeechTranscription::kTask);
        auto result = task.run_batch({{items.data(), items.size()}});
        if (result.size() != items.size())
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "family ASR batch result count differs from input count"};
        *output = make_result<BatchTextResultStorage>(std::move(result));
    });
}

extern "C" trtmc_status TRTMC_CALL
batch_translation_run(trtmc_model* model, const trtmc_batch_speech_translation_request_v1* input,
                      trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "translation batch and result output are required");
        const auto source = checked_span(input->items, input->count);
        std::vector<ConvertedConfig> configs;
        std::vector<internal::BatchSpeechTranslationItem> items;
        configs.reserve(source.size());
        items.reserve(source.size());
        for (const auto& item : source) {
            configs.emplace_back(&item.config);
            items.push_back({translation_request(item.input), configs.back().view()});
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::IBatchSpeechTranslation>(
            model, internal::IBatchSpeechTranslation::kTask);
        auto result = task.run_batch({{items.data(), items.size()}});
        if (result.size() != items.size())
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "family translation batch result count differs from input count"};
        *output = make_result<BatchTextResultStorage>(std::move(result));
    });
}

extern "C" trtmc_status TRTMC_CALL
mixed_batch_run(trtmc_model* model, const trtmc_mixed_batch_speech_to_text_request_v1* input,
                trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "mixed speech batch and result output are required");
        const auto source = checked_span(input->items, input->count);
        std::vector<ConvertedConfig> configs;
        std::vector<internal::MixedBatchSpeechToTextItem> items;
        configs.reserve(source.size());
        items.reserve(source.size());
        for (const auto& item : source) {
            configs.emplace_back(&item.config);
            switch (item.kind) {
            case TRTMC_SPEECH_TEXT_TRANSCRIPTION:
                items.push_back(
                    {transcription_request(item.input.transcription), configs.back().view()});
                break;
            case TRTMC_SPEECH_TEXT_TRANSLATION:
                items.push_back(
                    {translation_request(item.input.translation), configs.back().view()});
                break;
            default:
                throw ApiFailure{TRTMC_INVALID_ARGUMENT, "unknown mixed speech item kind"};
            }
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::IMixedBatchSpeechToText>(
            model, internal::IMixedBatchSpeechToText::kTask);
        auto result = task.run_batch({{items.data(), items.size()}});
        if (result.size() != items.size())
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "family mixed speech batch count differs from input count"};
        *output = make_result<BatchTextResultStorage>(std::move(result));
    });
}

struct BatchAudioStorage final : ResultStorage {
    explicit BatchAudioStorage(internal::BatchAudioResult results) {
        items.reserve(results.size());
        for (auto& result : results)
            items.push_back(std::make_unique<AudioResultStorage>(std::move(result)));
    }
    std::vector<std::unique_ptr<AudioResultStorage>> items;
};

trtmc_status TRTMC_CALL audio_batch_count(const trtmc_result* result, std::uint64_t* output,
                                          trtmc_error** error) noexcept {
    if (output)
        *output = 0;
    return guarded(error, [&] {
        require(output != nullptr, "audio batch count output is required");
        *output = require_result<BatchAudioStorage>(result).items.size();
    });
}

trtmc_status TRTMC_CALL audio_batch_item(const trtmc_result* result, std::uint64_t index,
                                         trtmc_audio_result_view_v1* output,
                                         trtmc_error** error) noexcept {
    if (output)
        *output = {};
    return guarded(error, [&] {
        require(output != nullptr, "audio item view output is required");
        const auto& batch = require_result<BatchAudioStorage>(result);
        require(index < batch.items.size(), "audio item index is out of range");
        fill_audio_result_view(*batch.items[static_cast<std::size_t>(index)], output);
    });
}

trtmc_status TRTMC_CALL history_audio_batch_run(
    trtmc_model* model, const trtmc_batch_text_audio_token_history_to_audio_request_v1* input,
    trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "audio history batch request and output are required");
        const auto history = token_history(input->history);
        const auto source = checked_span(input->items, input->count);
        require(!source.empty(), "audio history batch requires at least one item");
        std::vector<ConvertedConfig> configs;
        std::vector<internal::BatchTextToAudioItem> items;
        configs.reserve(source.size());
        items.reserve(source.size());
        for (size_t index = 0; index < source.size(); ++index) {
            try {
                configs.emplace_back(&source[index].config);
                items.push_back({{string_view(source[index].input.prompt)}, configs.back().view()});
            } catch (const ApiFailure& failure) {
                throw OwnedApiFailure{failure.status, "batch item[" + std::to_string(index) +
                                                          "]: " + failure.message};
            }
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::IBatchTextAudioTokenHistoryToAudio>(
            model, internal::IBatchTextAudioTokenHistoryToAudio::kTask);
        auto results = family.run_batch({history, {items.data(), items.size()}});
        if (results.size() != items.size())
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "history audio batch result count differs from input count"};
        *output = make_result<BatchAudioStorage>(std::move(results));
    });
}

trtmc_status TRTMC_CALL text_audio_batch_run(trtmc_model* model,
                                             const trtmc_batch_text_to_audio_request_v1* input,
                                             trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "audio batch request and result output are required");
        const auto source = checked_span(input->items, input->count);
        require(!source.empty(), "audio generation batch must contain at least one item");
        std::vector<ConvertedConfig> configs;
        std::vector<internal::BatchTextToAudioItem> items;
        configs.reserve(source.size());
        items.reserve(source.size());
        for (const auto& item : source) {
            configs.emplace_back(&item.config);
            items.push_back({{string_view(item.input.prompt)}, configs.back().view()});
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::IBatchTextToAudio>(
            model, internal::IBatchTextToAudio::kTask);
        auto result = task.run_batch({{items.data(), items.size()}});
        if (result.size() != items.size())
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "family audio batch count differs from input count"};
        *output = make_result<BatchAudioStorage>(std::move(result));
    });
}

trtmc_status TRTMC_CALL text_speech_batch_run(trtmc_model* model,
                                              const trtmc_batch_text_to_speech_request_v1* input,
                                              trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "speech batch request and result output are required");
        const auto source = checked_span(input->items, input->count);
        require(!source.empty(), "speech generation batch must contain at least one item");
        std::vector<ConvertedConfig> configs;
        std::vector<internal::BatchTextToSpeechItem> items;
        configs.reserve(source.size());
        items.reserve(source.size());
        for (const auto& item : source) {
            configs.emplace_back(&item.config);
            items.push_back({{string_view(item.input.text),
                              language(item.input.has_language, item.input.language)},
                             configs.back().view()});
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::IBatchTextToSpeech>(
            model, internal::IBatchTextToSpeech::kTask);
        auto result = task.run_batch({{items.data(), items.size()}});
        if (result.size() != items.size())
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "family speech batch count differs from input count"};
        *output = make_result<BatchAudioStorage>(std::move(result));
    });
}

const trtmc_text_to_audio_api_v1 text_audio_api{
    {1, 0, sizeof(text_audio_api)}, text_to_audio_run, audio_result_view};
const trtmc_text_audio_token_history_to_audio_api_v1 history_audio_api{
    {1, 0, sizeof(history_audio_api)}, history_audio_run, audio_result_view};
static_assert(offsetof(trtmc_text_audio_token_history_to_audio_api_v1, header) == 0);
const trtmc_text_to_speech_api_v1 text_speech_api{
    {1, 0, sizeof(text_speech_api)}, text_to_speech_run, audio_result_view};
const trtmc_speech_transcription_api_v1 transcription_api{
    {1, 0, sizeof(transcription_api)}, speech_transcription_run, text_result_view};
const trtmc_speech_translation_api_v1 translation_api{
    {1, 0, sizeof(translation_api)}, speech_translation_run, text_result_view};
const trtmc_audio_language_identification_api_v1 language_api{
    {1, 0, sizeof(language_api)}, language_identification_run, label_scores_result_view};
const trtmc_speech_to_speech_response_api_v1 response_api{
    {1, 0, sizeof(response_api)}, speech_response_run, audio_result_view};
const trtmc_batch_speech_transcription_api_v1 batch_transcription_api{
    {1, 0, sizeof(batch_transcription_api)},
    batch_transcription_run,
    text_batch_result_count,
    text_batch_result_item_view};
const trtmc_batch_speech_translation_api_v1 batch_translation_api{
    {1, 0, sizeof(batch_translation_api)},
    batch_translation_run,
    text_batch_result_count,
    text_batch_result_item_view};
const trtmc_mixed_batch_speech_to_text_api_v1 mixed_batch_api{{1, 0, sizeof(mixed_batch_api)},
                                                              mixed_batch_run,
                                                              text_batch_result_count,
                                                              text_batch_result_item_view};
const trtmc_batch_text_to_audio_api_v1 batch_text_audio_api{{1, 0, sizeof(batch_text_audio_api)},
                                                            text_audio_batch_run,
                                                            audio_batch_count,
                                                            audio_batch_item};
const trtmc_batch_text_to_speech_api_v1 batch_text_speech_api{{1, 0, sizeof(batch_text_speech_api)},
                                                              text_speech_batch_run,
                                                              audio_batch_count,
                                                              audio_batch_item};

const trtmc_batch_text_audio_token_history_to_audio_api_v1 history_audio_batch_api{
    {1, 0, sizeof(history_audio_batch_api)},
    history_audio_batch_run,
    audio_batch_count,
    audio_batch_item};
static_assert(offsetof(trtmc_batch_text_audio_token_history_to_audio_api_v1, header) == 0);

const TaskBinding bindings[] = {
    {internal::IBatchTextAudioTokenHistoryToAudio::kTask, 1, 0, &history_audio_batch_api.header,
     implements<internal::IBatchTextAudioTokenHistoryToAudio>},
    {internal::ITextAudioTokenHistoryToAudio::kTask, 1, 0, &history_audio_api.header,
     implements<internal::ITextAudioTokenHistoryToAudio>},
    {internal::ITextToAudio::kTask, 1, 0, &text_audio_api.header,
     implements<internal::ITextToAudio>},
    {internal::ITextToSpeech::kTask, 1, 0, &text_speech_api.header,
     implements<internal::ITextToSpeech>},
    {internal::ISpeechTranscription::kTask, 1, 0, &transcription_api.header,
     implements<internal::ISpeechTranscription>},
    {internal::ISpeechTranslation::kTask, 1, 0, &translation_api.header,
     implements<internal::ISpeechTranslation>},
    {internal::IAudioLanguageIdentification::kTask, 1, 0, &language_api.header,
     implements<internal::IAudioLanguageIdentification>},
    {internal::ISpeechToSpeechResponse::kTask, 1, 0, &response_api.header,
     implements<internal::ISpeechToSpeechResponse>},
    {internal::IBatchSpeechTranscription::kTask, 1, 0, &batch_transcription_api.header,
     implements<internal::IBatchSpeechTranscription>},
    {internal::IBatchSpeechTranslation::kTask, 1, 0, &batch_translation_api.header,
     implements<internal::IBatchSpeechTranslation>},
    {internal::IMixedBatchSpeechToText::kTask, 1, 0, &mixed_batch_api.header,
     implements<internal::IMixedBatchSpeechToText>},
    {internal::IBatchTextToAudio::kTask, 1, 0, &batch_text_audio_api.header,
     implements<internal::IBatchTextToAudio>},
    {internal::IBatchTextToSpeech::kTask, 1, 0, &batch_text_speech_api.header,
     implements<internal::IBatchTextToSpeech>},
};

} // namespace

Span<const TaskBinding> audio_task_bindings() noexcept {
    return bindings;
}

} // namespace trtmc::api
