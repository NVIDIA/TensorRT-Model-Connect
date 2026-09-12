/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/language.h"

#include "api_internal.h"
#include "language_internal.h"
#include "trtmc/language.h"

#include <cmath>
#include <set>
#include <type_traits>

namespace trtmc::api {
namespace {
void output_check(bool condition, const char* message) {
    if (!condition)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
}
using language_detail::LanguageInputs;

struct TextSpeechStorage final : ResultStorage {
    explicit TextSpeechStorage(internal::TextSpeechResult value)
        : text(std::move(value.text)), speech(std::move(value.speech)) {
        fill_text_result_view(text, &view.text);
        fill_audio_result_view(speech, &view.speech);
    }
    TextResultStorage text;
    AudioResultStorage speech;
    trtmc_text_speech_result_view_v1 view{};
};
using language_detail::ConversationStorage;
struct GeneratedLabelStorage final : ResultStorage {
    explicit GeneratedLabelStorage(internal::GeneratedLabelResult value)
        : result(std::move(value)) {
        output_check(!result.vocabulary.empty() && result.label_index >= -1 &&
                         (result.label_index == -1 ||
                          uint64_t(result.label_index) < result.vocabulary.size()),
                     "generated label index is outside its declared vocabulary");
        labels.reserve(result.vocabulary.size());
        for (const auto& label : result.vocabulary)
            labels.push_back(borrowed_string(label));
        view = {borrowed_string(result.raw_label),
                result.label_index,
                {labels.data(), labels.size()},
                {result.token_ids.data(), result.token_ids.size()}};
    }
    internal::GeneratedLabelResult result;
    std::vector<trtmc_string_view> labels;
    trtmc_generated_label_result_view_v1 view{};
};
struct TokenAxisSnapshot {
    explicit TokenAxisSnapshot(const internal::LanguageTokenSequence& source)
        : ids(source.token_ids.begin(), source.token_ids.end()) {
        if (source.attention_mask.empty())
            mask.assign(ids.size(), 1);
        else
            mask.assign(source.attention_mask.begin(), source.attention_mask.end());
    }
    std::vector<int32_t> ids;
    std::vector<uint8_t> mask;
};
struct EncoderDecoderStorage final : ResultStorage {
    EncoderDecoderStorage(internal::EncoderDecoderStatesResult value,
                          const internal::TextEncoderDecoderHiddenStatesRequest& input)
        : result(std::move(value)), source(input.source), decoder(input.decoder) {
        auto encoder_matrix = matrix_result_view(result.encoder_last_hidden_state);
        auto decoder_matrix = matrix_result_view(result.decoder_last_hidden_state);
        output_check(encoder_matrix.rows == source.ids.size() &&
                         decoder_matrix.rows == decoder.ids.size(),
                     "encoder/decoder output rows do not match their original token axes");
        view = {{encoder_matrix,
                 {source.ids.data(), source.ids.size()},
                 {source.mask.data(), source.mask.size()}},
                {decoder_matrix,
                 {decoder.ids.data(), decoder.ids.size()},
                 {decoder.mask.data(), decoder.mask.size()}}};
    }
    internal::EncoderDecoderStatesResult result;
    TokenAxisSnapshot source, decoder;
    trtmc_encoder_decoder_states_result_view_v1 view{};
};
template <class Storage, class View>
trtmc_status TRTMC_CALL result_view(const trtmc_result* result, View* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "language result view output is null");
        *out = require_result<Storage>(result).view;
    });
}
template <class Interface, class WireRequest, class Storage>
trtmc_status TRTMC_CALL run(trtmc_model* model, const WireRequest* input,
                            const trtmc_config_view_v1* config, trtmc_result** out,
                            trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(input && out, "language request or result output is null");
        LanguageInputs storage;
        const auto request = storage.convert(*input);
        const ConvertedConfig options(config);

        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<Interface>(model, Interface::kTask);
        validate_task_config(model_owner(model), internal::contract_key<Interface>(),
                             options.view());
        auto result = [&] {
            if constexpr (std::is_same_v<Interface, internal::IImagesTextConversation> ||
                          std::is_same_v<Interface, internal::IVideoTextConversation>)
                return family.run_conversation(request, options.view());
            else
                return family.run(request, options.view());
        }();
        if constexpr (std::is_same_v<Storage, EncoderDecoderStorage>)
            *out = make_result<Storage>(std::move(result), request);
        else
            *out = make_result<Storage>(std::move(result));
    });
}

const trtmc_images_text_to_text_api_v1 images_text_to_text_api = {
    {1, 0, sizeof(trtmc_images_text_to_text_api_v1)},
    run<internal::IImagesTextToText, trtmc_images_text_to_text_request_v1, TextResultStorage>,
    text_result_view};
static_assert(offsetof(trtmc_images_text_to_text_api_v1, header) == 0);

const trtmc_video_text_to_text_api_v1 video_text_to_text_api = {
    {1, 0, sizeof(trtmc_video_text_to_text_api_v1)},
    run<internal::IVideoTextToText, trtmc_video_text_to_text_request_v1, TextResultStorage>,
    text_result_view};
static_assert(offsetof(trtmc_video_text_to_text_api_v1, header) == 0);

const trtmc_image_video_text_to_text_api_v1 image_video_text_to_text_api = {
    {1, 0, sizeof(trtmc_image_video_text_to_text_api_v1)},
    run<internal::IImageVideoTextToText, trtmc_image_video_text_to_text_request_v1,
        TextResultStorage>,
    text_result_view};
static_assert(offsetof(trtmc_image_video_text_to_text_api_v1, header) == 0);

const trtmc_audio_text_to_text_api_v1 audio_text_to_text_api = {
    {1, 0, sizeof(trtmc_audio_text_to_text_api_v1)},
    run<internal::IAudioTextToText, trtmc_audio_text_to_text_request_v1, TextResultStorage>,
    text_result_view};
static_assert(offsetof(trtmc_audio_text_to_text_api_v1, header) == 0);

const trtmc_image_audio_to_text_api_v1 image_audio_to_text_api = {
    {1, 0, sizeof(trtmc_image_audio_to_text_api_v1)},
    run<internal::IImageAudioToText, trtmc_image_audio_to_text_request_v1, TextResultStorage>,
    text_result_view};
static_assert(offsetof(trtmc_image_audio_to_text_api_v1, header) == 0);

const trtmc_audio_video_text_to_text_api_v1 audio_video_text_to_text_api = {
    {1, 0, sizeof(trtmc_audio_video_text_to_text_api_v1)},
    run<internal::IAudioVideoTextToText, trtmc_audio_video_text_to_text_request_v1,
        TextResultStorage>,
    text_result_view};
static_assert(offsetof(trtmc_audio_video_text_to_text_api_v1, header) == 0);

const trtmc_image_audio_text_to_text_api_v1 image_audio_text_to_text_api = {
    {1, 0, sizeof(trtmc_image_audio_text_to_text_api_v1)},
    run<internal::IImageAudioTextToText, trtmc_image_audio_text_to_text_request_v1,
        TextResultStorage>,
    text_result_view};
static_assert(offsetof(trtmc_image_audio_text_to_text_api_v1, header) == 0);

const trtmc_image_audio_text_to_text_speech_response_api_v1
    image_audio_text_to_text_speech_response_api = {
        {1, 0, sizeof(trtmc_image_audio_text_to_text_speech_response_api_v1)},
        run<internal::IImageAudioTextToTextSpeechResponse,
            trtmc_image_audio_text_to_text_speech_response_request_v1, TextSpeechStorage>,
        result_view<TextSpeechStorage, trtmc_text_speech_result_view_v1>};
static_assert(offsetof(trtmc_image_audio_text_to_text_speech_response_api_v1, header) == 0);

const trtmc_text_conversation_api_v1 text_conversation_api = {
    {1, 0, sizeof(trtmc_text_conversation_api_v1)},
    run<internal::ITextConversation, trtmc_text_conversation_request_v1, ConversationStorage>,
    result_view<ConversationStorage, trtmc_conversation_result_view_v1>};
static_assert(offsetof(trtmc_text_conversation_api_v1, header) == 0);
const trtmc_images_text_conversation_api_v1 images_conversation_api{
    {1, 0, sizeof(images_conversation_api)},
    run<internal::IImagesTextConversation, trtmc_images_text_to_text_request_v1,
        ConversationStorage>,
    result_view<ConversationStorage, trtmc_conversation_result_view_v1>};
const trtmc_video_text_conversation_api_v1 video_conversation_api{
    {1, 0, sizeof(video_conversation_api)},
    run<internal::IVideoTextConversation, trtmc_video_text_to_text_request_v1, ConversationStorage>,
    result_view<ConversationStorage, trtmc_conversation_result_view_v1>};

template <class Function>
auto language_item(size_t index, Function function) {
    try {
        return function();
    } catch (const internal::ConfigError& failure) {
        throw OwnedApiFailure{TRTMC_INVALID_CONFIG,
                              "batch item[" + std::to_string(index) + "]: " + failure.what()};
    } catch (const ApiFailure& failure) {
        throw OwnedApiFailure{failure.status,
                              "batch item[" + std::to_string(index) + "]: " + failure.message};
    } catch (const OwnedApiFailure& failure) {
        throw OwnedApiFailure{failure.status,
                              "batch item[" + std::to_string(index) + "]: " + failure.message};
    }
    // bad_alloc reaches guarded directly, without allocating another diagnostic.
}
template <class Interface>
struct ConversationBatchStorage final : ResultStorage {
    explicit ConversationBatchStorage(std::vector<internal::ConversationResult> values) {
        items.reserve(values.size());
        for (size_t index = 0; index < values.size(); ++index)
            items.push_back(language_item(index, [&] {
                return std::make_unique<ConversationStorage>(std::move(values[index]));
            }));
    }
    std::vector<std::unique_ptr<ConversationStorage>> items;
};
template <class Interface>
trtmc_status TRTMC_CALL conversation_batch_count(const trtmc_result* result, uint64_t* out,
                                                 trtmc_error** error) noexcept {
    if (out)
        *out = 0;
    return guarded(error, [&] {
        require(out != nullptr, "conversation batch count output is null");
        *out = require_result<ConversationBatchStorage<Interface>>(result).items.size();
    });
}
template <class Interface>
trtmc_status TRTMC_CALL conversation_batch_item(const trtmc_result* result, uint64_t index,
                                                trtmc_conversation_result_view_v1* out,
                                                trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "conversation batch item view output is null");
        const auto& storage = require_result<ConversationBatchStorage<Interface>>(result);
        require(index < storage.items.size(), "conversation batch item index is out of range");
        *out = storage.items[index]->view;
    });
}
template <class Interface, class WireRequest>
trtmc_status TRTMC_CALL run_conversation_batch(trtmc_model* model, const WireRequest* input,
                                               trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(input && out, "conversation batch request and output are required");
        const auto source = checked_span(input->items, input->count);
        require(!source.empty(), "conversation batch requires at least one item");
        using Request = typename Interface::Request;
        std::vector<typename Request::Item> items;
        std::vector<LanguageInputs> storage;
        std::vector<ConvertedConfig> configs;
        items.reserve(source.size());
        storage.reserve(source.size());
        configs.reserve(source.size());
        for (size_t index = 0; index < source.size(); ++index)
            language_item(index, [&] {
                storage.emplace_back();
                configs.emplace_back(&source[index].config);

                items.push_back(
                    {storage.back().convert(source[index].input), configs.back().view()});
            });
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<Interface>(model, Interface::kTask);
        validate_batch_configs(model_owner(model), internal::contract_key<Interface>(), configs);
        auto results = family.run_batch(Request{{items.data(), items.size()}});
        output_check(results.size() == items.size(),
                     "family conversation batch result count differs from input count");
        *out = make_result<ConversationBatchStorage<Interface>>(std::move(results));
    });
}
const trtmc_batch_images_text_conversation_api_v1 batch_images_conversation_api{
    {1, 0, sizeof(trtmc_batch_images_text_conversation_api_v1)},
    run_conversation_batch<internal::IBatchImagesTextConversation,
                           trtmc_batch_images_text_conversation_request_v1>,
    conversation_batch_count<internal::IBatchImagesTextConversation>,
    conversation_batch_item<internal::IBatchImagesTextConversation>};
static_assert(offsetof(trtmc_batch_images_text_conversation_api_v1, header) == 0);

const trtmc_batch_text_conversation_api_v1 batch_text_conversation_api{
    {1, 0, sizeof(trtmc_batch_text_conversation_api_v1)},
    run_conversation_batch<internal::IBatchTextConversation,
                           trtmc_batch_text_conversation_request_v1>,
    conversation_batch_count<internal::IBatchTextConversation>,
    conversation_batch_item<internal::IBatchTextConversation>};
static_assert(offsetof(trtmc_batch_text_conversation_api_v1, header) == 0);

const trtmc_batch_video_text_conversation_api_v1 batch_video_text_conversation_api{
    {1, 0, sizeof(trtmc_batch_video_text_conversation_api_v1)},
    run_conversation_batch<internal::IBatchVideoTextConversation,
                           trtmc_batch_video_text_conversation_request_v1>,
    conversation_batch_count<internal::IBatchVideoTextConversation>,
    conversation_batch_item<internal::IBatchVideoTextConversation>};
static_assert(offsetof(trtmc_batch_video_text_conversation_api_v1, header) == 0);

const trtmc_batch_audio_text_conversation_api_v1 batch_audio_text_conversation_api{
    {1, 0, sizeof(trtmc_batch_audio_text_conversation_api_v1)},
    run_conversation_batch<internal::IBatchAudioTextConversation,
                           trtmc_batch_audio_text_conversation_request_v1>,
    conversation_batch_count<internal::IBatchAudioTextConversation>,
    conversation_batch_item<internal::IBatchAudioTextConversation>};
static_assert(offsetof(trtmc_batch_audio_text_conversation_api_v1, header) == 0);

const trtmc_batch_image_audio_text_conversation_api_v1 batch_image_audio_text_conversation_api{
    {1, 0, sizeof(trtmc_batch_image_audio_text_conversation_api_v1)},
    run_conversation_batch<internal::IBatchImageAudioTextConversation,
                           trtmc_batch_image_audio_text_conversation_request_v1>,
    conversation_batch_count<internal::IBatchImageAudioTextConversation>,
    conversation_batch_item<internal::IBatchImageAudioTextConversation>};
static_assert(offsetof(trtmc_batch_image_audio_text_conversation_api_v1, header) == 0);

const trtmc_batch_text_images_video_conversations_api_v1 batch_text_images_video_conversations_api{
    {1, 0, sizeof(trtmc_batch_text_images_video_conversations_api_v1)},
    run_conversation_batch<internal::IBatchTextImagesVideoConversations,
                           trtmc_batch_text_images_video_conversations_request_v1>,
    conversation_batch_count<internal::IBatchTextImagesVideoConversations>,
    conversation_batch_item<internal::IBatchTextImagesVideoConversations>};
static_assert(offsetof(trtmc_batch_text_images_video_conversations_api_v1, header) == 0);

const trtmc_batch_text_images_audio_conversations_api_v1 batch_text_images_audio_conversations_api{
    {1, 0, sizeof(trtmc_batch_text_images_audio_conversations_api_v1)},
    run_conversation_batch<internal::IBatchTextImagesAudioConversations,
                           trtmc_batch_text_images_audio_conversations_request_v1>,
    conversation_batch_count<internal::IBatchTextImagesAudioConversations>,
    conversation_batch_item<internal::IBatchTextImagesAudioConversations>};
static_assert(offsetof(trtmc_batch_text_images_audio_conversations_api_v1, header) == 0);
const trtmc_text_label_classification_api_v1 text_label_classification_api = {
    {1, 0, sizeof(trtmc_text_label_classification_api_v1)},
    run<internal::ITextLabelClassification, trtmc_text_label_classification_request_v1,
        GeneratedLabelStorage>,
    result_view<GeneratedLabelStorage, trtmc_generated_label_result_view_v1>};
static_assert(offsetof(trtmc_text_label_classification_api_v1, header) == 0);

const trtmc_text_pair_label_classification_api_v1 text_pair_label_classification_api = {
    {1, 0, sizeof(trtmc_text_pair_label_classification_api_v1)},
    run<internal::ITextPairLabelClassification, trtmc_text_pair_label_classification_request_v1,
        GeneratedLabelStorage>,
    result_view<GeneratedLabelStorage, trtmc_generated_label_result_view_v1>};
static_assert(offsetof(trtmc_text_pair_label_classification_api_v1, header) == 0);

const trtmc_text_encoder_decoder_hidden_states_api_v1 text_encoder_decoder_hidden_states_api = {
    {1, 0, sizeof(trtmc_text_encoder_decoder_hidden_states_api_v1)},
    run<internal::ITextEncoderDecoderHiddenStates,
        trtmc_text_encoder_decoder_hidden_states_request_v1, EncoderDecoderStorage>,
    result_view<EncoderDecoderStorage, trtmc_encoder_decoder_states_result_view_v1>};
static_assert(offsetof(trtmc_text_encoder_decoder_hidden_states_api_v1, header) == 0);

const TaskBinding bindings[] = {
    {internal::IBatchTextConversation::kTask, 1, 0, &batch_text_conversation_api.header},
    {internal::IBatchVideoTextConversation::kTask, 1, 0, &batch_video_text_conversation_api.header},
    {internal::IBatchAudioTextConversation::kTask, 1, 0, &batch_audio_text_conversation_api.header},
    {internal::IBatchImageAudioTextConversation::kTask, 1, 0,
     &batch_image_audio_text_conversation_api.header},
    {internal::IBatchTextImagesVideoConversations::kTask, 1, 0,
     &batch_text_images_video_conversations_api.header},
    {internal::IBatchTextImagesAudioConversations::kTask, 1, 0,
     &batch_text_images_audio_conversations_api.header},
    {internal::IImagesTextToText::kTask, 1, 0, &images_text_to_text_api.header},
    {internal::IVideoTextToText::kTask, 1, 0, &video_text_to_text_api.header},
    {internal::IImageVideoTextToText::kTask, 1, 0, &image_video_text_to_text_api.header},
    {internal::IAudioTextToText::kTask, 1, 0, &audio_text_to_text_api.header},
    {internal::IImageAudioToText::kTask, 1, 0, &image_audio_to_text_api.header},
    {internal::IAudioVideoTextToText::kTask, 1, 0, &audio_video_text_to_text_api.header},
    {internal::IImageAudioTextToText::kTask, 1, 0, &image_audio_text_to_text_api.header},
    {internal::IImageAudioTextToTextSpeechResponse::kTask, 1, 0,
     &image_audio_text_to_text_speech_response_api.header},
    {internal::ITextConversation::kTask, 1, 0, &text_conversation_api.header},
    {internal::IImagesTextConversation::kTask, 1, 0, &images_conversation_api.header},
    {internal::IVideoTextConversation::kTask, 1, 0, &video_conversation_api.header},
    {internal::IBatchImagesTextConversation::kTask, 1, 0, &batch_images_conversation_api.header},
    {internal::ITextLabelClassification::kTask, 1, 0, &text_label_classification_api.header},
    {internal::ITextPairLabelClassification::kTask, 1, 0,
     &text_pair_label_classification_api.header},
    {internal::ITextEncoderDecoderHiddenStates::kTask, 1, 0,
     &text_encoder_decoder_hidden_states_api.header},
};

} // namespace

Span<const TaskBinding> language_task_bindings() noexcept {
    return bindings;
}

} // namespace trtmc::api
