/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <string>

namespace {
using namespace trtmc::internal;

std::string source_text(const TextSource& source) {
    if (const auto* text = std::get_if<std::string_view>(&source))
        return std::string(*text);
    std::string output = "tokens";
    for (const auto token : std::get<trtmc::Span<const std::int32_t>>(source))
        output += ":" + std::to_string(token);
    return output;
}

TextResult result(std::string text, std::int32_t marker) {
    TextResult output;
    output.text = std::move(text);
    output.token_ids = {marker};
    output.setup_ms = 1;
    output.prefill_ms = 2;
    output.decode_ms = 3;
    output.segments.push_back({0, 0.5, "segment:" + output.text, {marker, 99}});
    return output;
}

class TextFixture final : public IModel,
                          public ITextContinuation,
                          public IConditionalTextGeneration,
                          public ICorruptedTextReconstruction,
                          public IUnconditionalTextGeneration,
                          public ITextTranslation,
                          public ITextSummarization,
                          public ITextPrefixSuffixInfilling,
                          public IContextQuestionAnswering,
                          public IBatchTextContinuation {
  public:
    explicit TextFixture(std::string mode)
        : mode_(std::move(mode)),
          default_target_(mode_ == "translation_explicit" ? std::nullopt
                                                          : std::optional<std::string>{"en"}) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInfo> task_info() const override {
        if (mode_ == "translation_only" || mode_ == "translation_explicit")
            return {{ITextTranslation::kTask, 1, 0}};
        if (mode_ == "broken_batch")
            return {{IBatchTextContinuation::kTask, 1, 0}};
        return {{ITextContinuation::kTask, 1, 0},
                {IConditionalTextGeneration::kTask, 1, 0},
                {ICorruptedTextReconstruction::kTask, 1, 0},
                {IUnconditionalTextGeneration::kTask, 1, 0},
                {ITextTranslation::kTask, 1, 0},
                {ITextSummarization::kTask, 1, 0},
                {ITextPrefixSuffixInfilling::kTask, 1, 0},
                {IContextQuestionAnswering::kTask, 1, 0},
                {IBatchTextContinuation::kTask, 1, 0}};
    }
    std::vector<ConfigField> config_fields(std::string_view id) const override {
        const auto supported = task_info();
        if (std::none_of(supported.begin(), supported.end(),
                         [&](const TaskInfo& info) { return info.id == id; }))
            throw UnsupportedTask("Task is not enabled in this fixture bundle");
        return {{"suffix", ConfigKind::String, ConfigValue{std::string_view{"!"}},
                 "Fixture output suffix"}};
    }

    TextResult run(const TextContinuationRequest& input, ConfigView config) override {
        const auto suffix = parse(config, ITextContinuation::kTask);
        ++single_calls_;
        return result("single:" + source_text(input.prefix) + suffix, single_calls_);
    }
    TextResult run(const ConditionalTextGenerationRequest& input, ConfigView config) override {
        return result("conditional:" + source_text(input.source) +
                          parse(config, IConditionalTextGeneration::kTask),
                      11);
    }
    TextResult run(const CorruptedTextReconstructionRequest& input, ConfigView config) override {
        return result("reconstructed:" + std::string(input.corrupted_text) +
                          parse(config, ICorruptedTextReconstruction::kTask),
                      12);
    }
    TextResult run(ConfigView config) override {
        return result("unconditional" + parse(config, IUnconditionalTextGeneration::kTask), 13);
    }
    TextResult run(const TextTranslationRequest& input, ConfigView config) override {
        const auto from = input.source_language.value_or("fixed-src");
        const auto to = input.target_language ? *input.target_language
                        : default_target_     ? std::string_view(*default_target_)
                                              : std::string_view{};
        if (to.empty())
            throw std::invalid_argument("this bundle requires an explicit target language");
        return result("translation:" + std::string(from) + "->" + std::string(to) + ":" +
                          std::string(input.source_text) + parse(config, ITextTranslation::kTask),
                      14);
    }
    TextResult run(const TextSummarizationRequest& input, ConfigView config) override {
        return result("summary:" + std::string(input.document) +
                          parse(config, ITextSummarization::kTask),
                      15);
    }
    TextResult run(const TextPrefixSuffixInfillingRequest& input, ConfigView config) override {
        return result("middle:" + std::string(input.prefix) + "|" + std::string(input.suffix) +
                          parse(config, ITextPrefixSuffixInfilling::kTask),
                      16);
    }
    TextResult run(const ContextQuestionAnsweringRequest& input, ConfigView config) override {
        return result("answer:" + std::string(input.question) + "|" + std::string(input.context) +
                          parse(config, IContextQuestionAnswering::kTask),
                      17);
    }
    BatchTextResult run_batch(const BatchTextContinuationRequest& input) override {
        std::vector<std::string> suffixes;
        for (const auto& item : input.items)
            suffixes.push_back(parse(item.config, IBatchTextContinuation::kTask));
        ++batch_calls_; // Only after every item's configuration passes.
        BatchTextResult output;
        for (std::size_t i = 0; i < input.items.size(); ++i) {
            auto item = result("batch:" + source_text(input.items[i].input.prefix) + suffixes[i],
                               batch_calls_);
            item.token_ids.push_back(static_cast<std::int32_t>(i));
            item.token_ids.push_back(single_calls_);
            output.push_back(std::move(item));
        }
        if (mode_ == "broken_batch" && !output.empty())
            output.pop_back();
        return output;
    }

  private:
    std::string parse(ConfigView config, std::string_view task) const {
        const auto fields = config_fields(task);
        std::string suffix(std::get<std::string_view>(*fields[0].default_value));
        bool seen = false;
        for (const auto& entry : config) {
            if (entry.name != fields[0].name)
                throw ConfigError("unknown config key");
            if (seen)
                throw ConfigError("duplicate config key");
            if (config_kind(entry.value) != fields[0].kind)
                throw ConfigError("suffix must be a string");
            seen = true;
            suffix = std::get<std::string_view>(entry.value);
        }
        return suffix;
    }
    std::string mode_;
    std::optional<std::string> default_target_;
    std::int32_t single_calls_{0};
    std::int32_t batch_calls_{0};
};
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "text_fixture")
        throw std::invalid_argument("unexpected text fixture family");
    return new TextFixture(context.reader.info().task);
}
