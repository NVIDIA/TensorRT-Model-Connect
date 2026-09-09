/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/config.h"
#include "trtmc/task.h"

#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace trtmc::internal {

using TextSource = std::variant<std::string_view, Span<const std::int32_t>>;

struct TextContinuationRequest {
    TextSource prefix;
};

// Reuse the existing owned text/timing/segment data; it does not cross the
// public C ABI. More specific Task contracts may use the same result storage.
using TextResult = trtmc::TextResult;

class ITextContinuation {
  public:
    static constexpr std::string_view kTask = "text_continuation";
    virtual ~ITextContinuation() = default;
    virtual TextResult run(const TextContinuationRequest& request, ConfigView config) = 0;
};

struct ConditionalTextGenerationRequest {
    TextSource source;
};
struct CorruptedTextReconstructionRequest {
    std::string_view corrupted_text;
};
struct TextTranslationRequest {
    std::string_view source_text;
    // Absence selects only a default declared by this loaded family/bundle.
    // A family with no applicable default rejects the missing language.
    std::optional<std::string_view> target_language{};
    std::optional<std::string_view> source_language{};
};
struct TextSummarizationRequest {
    std::string_view document;
};
struct TextPrefixSuffixInfillingRequest {
    std::string_view prefix;
    std::string_view suffix;
};
struct ContextQuestionAnsweringRequest {
    std::string_view question;
    std::string_view context;
};

class IConditionalTextGeneration {
  public:
    static constexpr std::string_view kTask = "conditional_text_generation";
    virtual ~IConditionalTextGeneration() = default;
    // Return generated target text; source is conditioning, not a decoder prefix.
    virtual TextResult run(const ConditionalTextGenerationRequest&, ConfigView) = 0;
};

class ICorruptedTextReconstruction {
  public:
    static constexpr std::string_view kTask = "corrupted_text_reconstruction";
    virtual ~ICorruptedTextReconstruction() = default;
    // Return complete reconstructed text, not just the replacements.
    virtual TextResult run(const CorruptedTextReconstructionRequest&, ConfigView) = 0;
};

class IUnconditionalTextGeneration {
  public:
    static constexpr std::string_view kTask = "unconditional_text_generation";
    virtual ~IUnconditionalTextGeneration() = default;
    virtual TextResult run(ConfigView) = 0;
};

class ITextTranslation {
  public:
    static constexpr std::string_view kTask = "text_translation";
    virtual ~ITextTranslation() = default;
    virtual TextResult run(const TextTranslationRequest&, ConfigView) = 0;
};

class ITextSummarization {
  public:
    static constexpr std::string_view kTask = "text_summarization";
    virtual ~ITextSummarization() = default;
    virtual TextResult run(const TextSummarizationRequest&, ConfigView) = 0;
};

class ITextPrefixSuffixInfilling {
  public:
    static constexpr std::string_view kTask = "text_prefix_suffix_infilling";
    virtual ~ITextPrefixSuffixInfilling() = default;
    // Own any sentinel/template mapping here and return only the missing middle.
    virtual TextResult run(const TextPrefixSuffixInfillingRequest&, ConfigView) = 0;
};

class IContextQuestionAnswering {
  public:
    static constexpr std::string_view kTask = "context_question_answering";
    virtual ~IContextQuestionAnswering() = default;
    // This is a generated answer, not an extractive span or a score vector.
    virtual TextResult run(const ContextQuestionAnsweringRequest&, ConfigView) = 0;
};

struct BatchTextContinuationItem {
    TextContinuationRequest input;
    ConfigView config;
};
struct BatchTextContinuationRequest {
    Span<const BatchTextContinuationItem> items;
};
using BatchTextResult = std::vector<TextResult>;

class IBatchTextContinuation {
  public:
    static constexpr std::string_view kTask = "batch_text_continuation";
    virtual ~IBatchTextContinuation() = default;
    // Validate all items before execution; a successful result preserves order
    // and has exactly one item per input. Native batch is its own execution.
    virtual BatchTextResult run_batch(const BatchTextContinuationRequest&) = 0;
};

} // namespace trtmc::internal
