/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/core.hpp"
#include "trtmc/text.h"

namespace trtmc {

using TextSource = std::variant<std::string, std::vector<std::int32_t>>;
struct ConditionalTextGenerationRequest {
    TextSource source;
};
struct CorruptedTextReconstructionRequest {
    std::string corrupted_text;
};
struct TextTranslationRequest {
    std::string source_text;
    std::optional<std::string> target_language{};
    std::optional<std::string> source_language{};
};
struct TextSummarizationRequest {
    std::string document;
};
struct TextPrefixSuffixInfillingRequest {
    std::string prefix;
    std::string suffix;
};
struct ContextQuestionAnsweringRequest {
    std::string question;
    std::string context;
};

// The operation contracts differ; their owned text/token/timing storage is common.
using ConditionalTextGenerationResult = TextContinuationResult;
using CorruptedTextReconstructionResult = TextContinuationResult;
using UnconditionalTextGenerationResult = TextContinuationResult;
using TextTranslationResult = TextContinuationResult;
using TextSummarizationResult = TextContinuationResult;
using TextPrefixSuffixInfillingResult = TextContinuationResult;
using ContextQuestionAnsweringResult = TextContinuationResult;

namespace detail {

inline trtmc_text_source_v1 wire_text_source(const TextSource& source) {
    trtmc_text_source_v1 output{};
    if (const auto* text = std::get_if<std::string>(&source)) {
        output.kind = TRTMC_TEXT_UTF8;
        output.as.text = c_string(*text);
    } else {
        const auto& ids = std::get<std::vector<std::int32_t>>(source);
        output.kind = TRTMC_TEXT_TOKEN_IDS;
        output.as.token_ids = {ids.data(), ids.size()};
    }
    return output;
}

template <class Table>
void validate_text_table(const trtmc_api_header* table) {
    if (table == nullptr || table->major != 1 || table->minor != 0 ||
        table->byte_size < sizeof(Table))
        throw Error(TRTMC_VERSION_MISMATCH, "incompatible text Task API table");
}

template <class ViewFunction>
TextContinuationResult finish_text_call(const std::shared_ptr<ModelState>& model,
                                        trtmc_status status, trtmc_result* raw, trtmc_error* error,
                                        ViewFunction result_view) {
    auto result = TextResultAccess::adopt(model, raw);
    check(model->api, status, error);
    error = nullptr;
    status = result_view(raw, &TextResultAccess::view(result), &error);
    check(model->api, status, error);
    return result;
}

template <class Table, class Request>
TextContinuationResult run_text(const std::shared_ptr<ModelState>& model, const Table* table,
                                const Request& input, const Config& config) {
    auto entries = config.c_entries();
    auto options = entries.view();
    trtmc_result* raw = nullptr;
    trtmc_error* error = nullptr;
    auto status = table->run(model->handle, &input, &options, &raw, &error);
    return finish_text_call(model, status, raw, error, table->result_view);
}

} // namespace detail

class ConditionalTextGeneration {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_CONDITIONAL_TEXT_GENERATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    ConditionalTextGenerationResult run(const ConditionalTextGenerationRequest& input,
                                        const Config& config = {}) const {
        trtmc_conditional_text_generation_request_v1 request{
            detail::wire_text_source(input.source)};
        return detail::run_text(state_, api_, request, config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_text_table<trtmc_conditional_text_generation_api_v1>(table);
    }

  private:
    friend class Model;
    ConditionalTextGeneration(std::shared_ptr<detail::ModelState> state,
                              const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_conditional_text_generation_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_conditional_text_generation_api_v1* api_;
};

class CorruptedTextReconstruction {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_CORRUPTED_TEXT_RECONSTRUCTION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    CorruptedTextReconstructionResult run(const CorruptedTextReconstructionRequest& input,
                                          const Config& config = {}) const {
        trtmc_corrupted_text_reconstruction_request_v1 request{
            detail::c_string(input.corrupted_text)};
        return detail::run_text(state_, api_, request, config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_text_table<trtmc_corrupted_text_reconstruction_api_v1>(table);
    }

  private:
    friend class Model;
    CorruptedTextReconstruction(std::shared_ptr<detail::ModelState> state,
                                const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_corrupted_text_reconstruction_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_corrupted_text_reconstruction_api_v1* api_;
};

class UnconditionalTextGeneration {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_UNCONDITIONAL_TEXT_GENERATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    UnconditionalTextGenerationResult run(const Config& config = {}) const {
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run(state_->handle, &options, &raw, &error);
        return detail::finish_text_call(state_, status, raw, error, api_->result_view);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_text_table<trtmc_unconditional_text_generation_api_v1>(table);
    }

  private:
    friend class Model;
    UnconditionalTextGeneration(std::shared_ptr<detail::ModelState> state,
                                const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_unconditional_text_generation_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_unconditional_text_generation_api_v1* api_;
};

class TextTranslation {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_TRANSLATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    TextTranslationResult run(const TextTranslationRequest& input,
                              const Config& config = {}) const {
        trtmc_text_translation_request_v1 request{
            detail::c_string(input.source_text), input.target_language ? 1U : 0U,
            input.target_language ? detail::c_string(*input.target_language)
                                  : trtmc_string_view{nullptr, 0},
            input.source_language ? 1U : 0U,
            input.source_language ? detail::c_string(*input.source_language)
                                  : trtmc_string_view{nullptr, 0}};
        return detail::run_text(state_, api_, request, config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_text_table<trtmc_text_translation_api_v1>(table);
    }

  private:
    friend class Model;
    TextTranslation(std::shared_ptr<detail::ModelState> state,
                    const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_text_translation_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_text_translation_api_v1* api_;
};

class TextSummarization {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_SUMMARIZATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    TextSummarizationResult run(const TextSummarizationRequest& input,
                                const Config& config = {}) const {
        trtmc_text_summarization_request_v1 request{detail::c_string(input.document)};
        return detail::run_text(state_, api_, request, config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_text_table<trtmc_text_summarization_api_v1>(table);
    }

  private:
    friend class Model;
    TextSummarization(std::shared_ptr<detail::ModelState> state,
                      const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_text_summarization_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_text_summarization_api_v1* api_;
};

class TextPrefixSuffixInfilling {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_PREFIX_SUFFIX_INFILLING;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    TextPrefixSuffixInfillingResult run(const TextPrefixSuffixInfillingRequest& input,
                                        const Config& config = {}) const {
        trtmc_text_prefix_suffix_infilling_request_v1 request{detail::c_string(input.prefix),
                                                              detail::c_string(input.suffix)};
        return detail::run_text(state_, api_, request, config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_text_table<trtmc_text_prefix_suffix_infilling_api_v1>(table);
    }

  private:
    friend class Model;
    TextPrefixSuffixInfilling(std::shared_ptr<detail::ModelState> state,
                              const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_text_prefix_suffix_infilling_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_text_prefix_suffix_infilling_api_v1* api_;
};

class ContextQuestionAnswering {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_CONTEXT_QUESTION_ANSWERING;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    ContextQuestionAnsweringResult run(const ContextQuestionAnsweringRequest& input,
                                       const Config& config = {}) const {
        trtmc_context_question_answering_request_v1 request{detail::c_string(input.question),
                                                            detail::c_string(input.context)};
        return detail::run_text(state_, api_, request, config);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_text_table<trtmc_context_question_answering_api_v1>(table);
    }

  private:
    friend class Model;
    ContextQuestionAnswering(std::shared_ptr<detail::ModelState> state,
                             const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_context_question_answering_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_context_question_answering_api_v1* api_;
};

struct BatchTextContinuationItem {
    TextContinuationRequest input;
    Config config;
};
struct BatchTextContinuationRequest {
    std::vector<BatchTextContinuationItem> items;
};

namespace detail {
struct BatchTextResultAccess;
}

class BatchTextContinuationResult {
  public:
    BatchTextContinuationResult(const BatchTextContinuationResult&) = delete;
    BatchTextContinuationResult& operator=(const BatchTextContinuationResult&) = delete;
    BatchTextContinuationResult(BatchTextContinuationResult&& other) noexcept
        : owner_(std::move(other.owner_)), item_view_(other.item_view_),
          size_(std::exchange(other.size_, 0)) {}
    BatchTextContinuationResult& operator=(BatchTextContinuationResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            item_view_ = other.item_view_;
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    std::size_t size() const noexcept { return static_cast<std::size_t>(size_); }
    bool empty() const noexcept { return size_ == 0; }
    TextResultView at(std::size_t index) const {
        trtmc_text_result_view_v1 view{};
        trtmc_error* error = nullptr;
        const auto status = item_view_(owner_.get(), index, &view, &error);
        detail::check(owner_.api(), status, error);
        return detail::text_result_view(view);
    }
    TextResultView operator[](std::size_t index) const { return at(index); }

  private:
    friend class BatchTextContinuation;
    friend struct detail::BatchTextResultAccess;
    using ItemViewFunction = trtmc_status(TRTMC_CALL*)(const trtmc_result*, std::uint64_t,
                                                       trtmc_text_result_view_v1*, trtmc_error**);
    BatchTextContinuationResult(std::shared_ptr<detail::ModelState> state,
                                const trtmc_batch_text_continuation_api_v1* api,
                                trtmc_result* result) noexcept
        : owner_(std::move(state), result), item_view_(api->result_item_view) {}
    BatchTextContinuationResult(std::shared_ptr<detail::ModelState> state, trtmc_result* result,
                                ItemViewFunction item_view) noexcept
        : owner_(std::move(state), result), item_view_(item_view) {}
    detail::ResultOwner owner_;
    ItemViewFunction item_view_;
    std::uint64_t size_{0};
};

namespace detail {
struct BatchTextResultAccess {
    template <class CountFunction, class ItemViewFunction>
    static BatchTextContinuationResult
    finish(const std::shared_ptr<ModelState>& state, trtmc_status status, trtmc_result* raw,
           trtmc_error* error, CountFunction count, ItemViewFunction item_view) {
        BatchTextContinuationResult result(state, raw, item_view);
        check(state->api, status, error);
        error = nullptr;
        status = count(raw, &result.size_, &error);
        check(state->api, status, error);
        return result;
    }
};
} // namespace detail

class BatchTextContinuation {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_TEXT_CONTINUATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    BatchTextContinuationResult run(const BatchTextContinuationRequest& input) const {
        std::vector<Config::CEntries> configs;
        std::vector<trtmc_batch_text_continuation_item_v1> items;
        configs.reserve(input.items.size());
        items.reserve(input.items.size());
        for (const auto& item : input.items) {
            configs.push_back(item.config.c_entries());
            items.push_back({{detail::wire_text_source(item.input.prefix)}, configs.back().view()});
        }
        trtmc_batch_text_continuation_request_v1 request{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run_batch(state_->handle, &request, &raw, &error);
        BatchTextContinuationResult result(state_, api_, raw);
        detail::check(state_->api, status, error);
        error = nullptr;
        status = api_->result_count(raw, &result.size_, &error);
        detail::check(state_->api, status, error);
        return result;
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_text_table<trtmc_batch_text_continuation_api_v1>(table);
    }

  private:
    friend class Model;
    BatchTextContinuation(std::shared_ptr<detail::ModelState> state,
                          const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_batch_text_continuation_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_batch_text_continuation_api_v1* api_;
};

} // namespace trtmc
