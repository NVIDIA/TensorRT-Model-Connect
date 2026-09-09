/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "api_internal.h"

namespace trtmc::api {

TextResultStorage::TextResultStorage(internal::TextResult result) : text(std::move(result)) {
    segments.reserve(text.segments.size());
    for (const auto& segment : text.segments) {
        segments.push_back({segment.start_seconds,
                            segment.end_seconds,
                            borrowed_string(segment.text),
                            {segment.token_ids.data(), segment.token_ids.size()}});
    }
}

BatchTextResultStorage::BatchTextResultStorage(internal::BatchTextResult results) {
    items.reserve(results.size());
    for (auto& result : results)
        items.push_back(std::make_unique<TextResultStorage>(std::move(result)));
}

trtmc_status TRTMC_CALL text_batch_result_count(const trtmc_result* result, std::uint64_t* count,
                                                trtmc_error** error) noexcept {
    if (count)
        *count = 0;
    return guarded(error, [&] {
        require(count != nullptr, "result count output is required");
        *count = require_result<BatchTextResultStorage>(result).items.size();
    });
}

trtmc_status TRTMC_CALL text_batch_result_item_view(const trtmc_result* result, std::uint64_t index,
                                                    trtmc_text_result_view_v1* output,
                                                    trtmc_error** error) noexcept {
    if (output)
        *output = {};
    return guarded(error, [&] {
        require(output != nullptr, "item view output is required");
        const auto& storage = require_result<BatchTextResultStorage>(result);
        require(index < storage.items.size(), "batch result index is out of range");
        fill_text_result_view(*storage.items[static_cast<std::size_t>(index)], output);
    });
}

internal::TextSource text_source(const trtmc_text_source_v1& source) {
    switch (source.kind) {
    case TRTMC_TEXT_UTF8:
        return string_view(source.as.text);
    case TRTMC_TEXT_TOKEN_IDS:
        return checked_span(source.as.token_ids.data, source.as.token_ids.size);
    default:
        throw ApiFailure{TRTMC_INVALID_ARGUMENT, "unknown text input kind"};
    }
}

void fill_text_result_view(const TextResultStorage& result,
                           trtmc_text_result_view_v1* out) noexcept {
    const auto& text = result.text;
    *out = {borrowed_string(text.text),
            {text.token_ids.data(), text.token_ids.size()},
            text.setup_ms,
            text.prefill_ms,
            text.decode_ms,
            result.segments.data(),
            result.segments.size()};
}

trtmc_status TRTMC_CALL text_result_view(const trtmc_result* result, trtmc_text_result_view_v1* out,
                                         trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "result view is null");
        fill_text_result_view(require_result<TextResultStorage>(result), out);
    });
}

namespace {

trtmc_status TRTMC_CALL text_run(trtmc_model* model,
                                 const trtmc_text_continuation_request_v1* request,
                                 const trtmc_config_view_v1* config, trtmc_result** out,
                                 trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request != nullptr && out != nullptr, "request or result output is null");
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& task = require_interface<internal::ITextContinuation>(
            model, internal::ITextContinuation::kTask);
        const internal::TextContinuationRequest input{text_source(request->prefix)};
        const ConvertedConfig converted(config);
        *out = make_result<TextResultStorage>(task.run(input, converted.view()));
    });
}

const trtmc_text_continuation_api_v1 text_api = {
    {1, 0, sizeof(trtmc_text_continuation_api_v1)}, text_run, text_result_view};

const TaskBinding bindings[] = {{internal::ITextContinuation::kTask, 1, 0, &text_api.header,
                                 implements<internal::ITextContinuation>}};

static_assert(offsetof(trtmc_text_continuation_api_v1, header) == 0);

} // namespace

Span<const TaskBinding> text_continuation_bindings() noexcept {
    return bindings;
}

} // namespace trtmc::api
