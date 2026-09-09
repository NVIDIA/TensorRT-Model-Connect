/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "api_internal.h"
#include "trtmc/text.h"

namespace trtmc::api {
namespace {

extern "C" trtmc_status TRTMC_CALL conditional_run(
    trtmc_model* model, const trtmc_conditional_text_generation_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "input and result output are required");
        internal::ConditionalTextGenerationRequest request{text_source(input->source)};
        ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::IConditionalTextGeneration>(
            model, internal::IConditionalTextGeneration::kTask);
        *output = make_result<TextResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL reconstruction_run(
    trtmc_model* model, const trtmc_corrupted_text_reconstruction_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "input and result output are required");
        internal::CorruptedTextReconstructionRequest request{string_view(input->corrupted_text)};
        ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::ICorruptedTextReconstruction>(
            model, internal::ICorruptedTextReconstruction::kTask);
        *output = make_result<TextResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL unconditional_run(trtmc_model* model,
                                                     const trtmc_config_view_v1* config,
                                                     trtmc_result** output,
                                                     trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(output != nullptr, "result output is required");
        ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::IUnconditionalTextGeneration>(
            model, internal::IUnconditionalTextGeneration::kTask);
        *output = make_result<TextResultStorage>(task.run(options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL translation_run(trtmc_model* model,
                                                   const trtmc_text_translation_request_v1* input,
                                                   const trtmc_config_view_v1* config,
                                                   trtmc_result** output,
                                                   trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "input and result output are required");
        require(input->has_target_language <= 1, "target-language presence must be zero or one");
        require(input->has_source_language <= 1, "source-language presence must be zero or one");
        internal::TextTranslationRequest request{string_view(input->source_text)};
        if (input->has_target_language) {
            request.target_language = string_view(input->target_language);
            require(!request.target_language->empty(), "present target language must not be empty");
        }
        if (input->has_source_language) {
            request.source_language = string_view(input->source_language);
            require(!request.source_language->empty(), "present source language must not be empty");
        }
        ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task =
            require_interface<internal::ITextTranslation>(model, internal::ITextTranslation::kTask);
        *output = make_result<TextResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL summarization_run(
    trtmc_model* model, const trtmc_text_summarization_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "input and result output are required");
        internal::TextSummarizationRequest request{string_view(input->document)};
        ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::ITextSummarization>(
            model, internal::ITextSummarization::kTask);
        *output = make_result<TextResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL infilling_run(
    trtmc_model* model, const trtmc_text_prefix_suffix_infilling_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "input and result output are required");
        internal::TextPrefixSuffixInfillingRequest request{string_view(input->prefix),
                                                           string_view(input->suffix)};
        ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::ITextPrefixSuffixInfilling>(
            model, internal::ITextPrefixSuffixInfilling::kTask);
        *output = make_result<TextResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL question_answering_run(
    trtmc_model* model, const trtmc_context_question_answering_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "input and result output are required");
        internal::ContextQuestionAnsweringRequest request{string_view(input->question),
                                                          string_view(input->context)};
        ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::IContextQuestionAnswering>(
            model, internal::IContextQuestionAnswering::kTask);
        *output = make_result<TextResultStorage>(task.run(request, options.view()));
    });
}

extern "C" trtmc_status TRTMC_CALL batch_run(trtmc_model* model,
                                             const trtmc_batch_text_continuation_request_v1* input,
                                             trtmc_result** output, trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "batch input and result output are required");
        const auto items = checked_span(input->items, input->count);
        std::vector<ConvertedConfig> configs;
        std::vector<internal::BatchTextContinuationItem> requests;
        configs.reserve(items.size());
        requests.reserve(items.size());
        for (const auto& item : items) {
            configs.emplace_back(&item.config);
            requests.push_back({{text_source(item.input.prefix)}, configs.back().view()});
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::IBatchTextContinuation>(
            model, internal::IBatchTextContinuation::kTask);
        auto result = task.run_batch({{requests.data(), requests.size()}});
        if (result.size() != items.size())
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "family batch result count differs from input count"};
        *output = make_result<BatchTextResultStorage>(std::move(result));
    });
}

const trtmc_conditional_text_generation_api_v1 conditional_api{
    {1, 0, sizeof(conditional_api)}, conditional_run, text_result_view};
const trtmc_corrupted_text_reconstruction_api_v1 reconstruction_api{
    {1, 0, sizeof(reconstruction_api)}, reconstruction_run, text_result_view};
const trtmc_unconditional_text_generation_api_v1 unconditional_api{
    {1, 0, sizeof(unconditional_api)}, unconditional_run, text_result_view};
const trtmc_text_translation_api_v1 translation_api{
    {1, 0, sizeof(translation_api)}, translation_run, text_result_view};
const trtmc_text_summarization_api_v1 summarization_api{
    {1, 0, sizeof(summarization_api)}, summarization_run, text_result_view};
const trtmc_text_prefix_suffix_infilling_api_v1 infilling_api{
    {1, 0, sizeof(infilling_api)}, infilling_run, text_result_view};
const trtmc_context_question_answering_api_v1 qa_api{
    {1, 0, sizeof(qa_api)}, question_answering_run, text_result_view};
const trtmc_batch_text_continuation_api_v1 batch_api{
    {1, 0, sizeof(batch_api)}, batch_run, text_batch_result_count, text_batch_result_item_view};

const TaskBinding bindings[] = {
    {internal::IConditionalTextGeneration::kTask, 1, 0, &conditional_api.header,
     implements<internal::IConditionalTextGeneration>},
    {internal::ICorruptedTextReconstruction::kTask, 1, 0, &reconstruction_api.header,
     implements<internal::ICorruptedTextReconstruction>},
    {internal::IUnconditionalTextGeneration::kTask, 1, 0, &unconditional_api.header,
     implements<internal::IUnconditionalTextGeneration>},
    {internal::ITextTranslation::kTask, 1, 0, &translation_api.header,
     implements<internal::ITextTranslation>},
    {internal::ITextSummarization::kTask, 1, 0, &summarization_api.header,
     implements<internal::ITextSummarization>},
    {internal::ITextPrefixSuffixInfilling::kTask, 1, 0, &infilling_api.header,
     implements<internal::ITextPrefixSuffixInfilling>},
    {internal::IContextQuestionAnswering::kTask, 1, 0, &qa_api.header,
     implements<internal::IContextQuestionAnswering>},
    {internal::IBatchTextContinuation::kTask, 1, 0, &batch_api.header,
     implements<internal::IBatchTextContinuation>},
};

} // namespace

Span<const TaskBinding> text_task_bindings() noexcept {
    return bindings;
}

} // namespace trtmc::api
