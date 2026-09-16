/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <set>
#include <type_traits>

namespace {
using namespace trtmc::internal;
using Json = nlohmann::json;

Json json_value(const ConfigValue& value) {
    return std::visit(
        [](const auto& item) -> Json {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_arithmetic_v<T>)
                return item;
            else if constexpr (std::is_same_v<T, std::string_view>)
                return std::string(item);
            else {
                Json values = Json::array();
                for (auto entry : item)
                    values.push_back(entry);
                return values;
            }
        },
        value);
}

class DatasetFixture final : public IModel,
                             public ITextContinuation,
                             public IConditionalTextGeneration,
                             public ITextTranslation,
                             public IImagesTextToText,
                             public trtmc::ITextGeneration {
  public:
    DatasetFixture(std::string mode, bool multiple, bool text)
        : mode_(std::move(mode)), multiple_(multiple), text_(text) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::int32_t default_max_new_tokens() const override { return 5; }
    std::vector<TaskInstance> task_bindings() override {
        std::vector<TaskInstance> tasks;
        if (text_)
            tasks.push_back(bind<ITextContinuation>(*this, fields_for(ITextContinuation::kTask)));
        if (mode_ == IImagesTextToText::kTask)
            tasks.push_back(bind<IImagesTextToText>(*this));
        if (mode_ == IConditionalTextGeneration::kTask ||
            (text_ && (multiple_ || mode_ == ITextTranslation::kTask)))
            tasks.push_back(bind<IConditionalTextGeneration>(
                *this, fields_for(IConditionalTextGeneration::kTask)));
        if (mode_ == ITextTranslation::kTask)
            tasks.push_back(bind<ITextTranslation>(*this, fields_for(ITextTranslation::kTask)));
        return tasks;
    }
    TextResult run(const ImagesTextToTextRequest&, ConfigView) override {
        throw std::logic_error("dataset must select the bound text-only Task before execution");
    }
    TextResult run(const ConditionalTextGenerationRequest& input, ConfigView config) override {
        auto result = run(TextContinuationRequest{input.source}, config);
        result.text = "conditioned:" + result.text;
        return result;
    }
    TextResult run(const TextTranslationRequest& input, ConfigView config) override {
        if (input.source_language || input.target_language)
            throw std::logic_error("absent dataset languages must reach the family as absent");
        auto result = run(TextContinuationRequest{input.source_text}, config);
        result.text = "translated:" + result.text;
        return result;
    }
    trtmc::Span<const ConfigField> fields_for(std::string_view id) const {
        if (id != ITextContinuation::kTask && id != IConditionalTextGeneration::kTask &&
            id != ITextTranslation::kTask)
            throw UnsupportedTask("fixture does not support this text Task");
        static const ConfigField declared[] = {
            {"max_new_tokens", ConfigKind::I64, ConfigValue{std::int64_t{5}}, "Token limit"},
            {"temperature", ConfigKind::F64, ConfigValue{0.75}, "Temperature"},
            {"top_k", ConfigKind::I64, ConfigValue{std::int64_t{7}}, "Top k"},
            {"top_p", ConfigKind::F64, ConfigValue{0.8}, "Top p"},
            {"min_p", ConfigKind::F64, ConfigValue{0.1}, "Min p"},
            {"seed", ConfigKind::I64, ConfigValue{std::int64_t{-1}}, "Seed"},
            {"use_chat_template", ConfigKind::Bool, ConfigValue{true}, "Chat template"},
            {"enable_thinking", ConfigKind::Bool, ConfigValue{true}, "Thinking"},
            {"stop_on_boxed_answer", ConfigKind::Bool, ConfigValue{false}, "Answer stop"},
            {"stop_check_interval", ConfigKind::I64, ConfigValue{std::int64_t{16}},
             "Stop interval"},
            {"suffix", ConfigKind::String, ConfigValue{std::string_view{"!"}}, "Suffix"},
            {"ids", ConfigKind::I64List, ConfigValue{trtmc::Span<const std::int64_t>{}}, "IDs"},
            {"schedule", ConfigKind::F64List, ConfigValue{trtmc::Span<const double>{}}, "Schedule"},
            {"labels", ConfigKind::StringList, ConfigValue{trtmc::Span<const std::string_view>{}},
             "Labels"},
        };
        return declared;
    }
    TextResult run(const TextContinuationRequest& input, ConfigView config) override {
        const auto fields = fields_for(ITextContinuation::kTask);
        Json values = Json::object();
        for (const auto& field : fields)
            values[std::string(field.name)] = json_value(*field.default_value);
        std::set<std::string_view> seen;
        for (const auto& entry : config) {
            if (!seen.insert(entry.name).second)
                throw ConfigError("duplicate: " + std::string(entry.name));
            const auto field = std::find_if(fields.begin(), fields.end(), [&](const auto& field) {
                return field.name == entry.name;
            });
            if (field == fields.end())
                throw ConfigError("unknown config");
            if (config_kind(entry.value) != field->kind)
                throw ConfigError("wrong config type");
            values[std::string(entry.name)] = json_value(entry.value);
        }
        if (values.at("max_new_tokens").get<std::int64_t>() > 20000)
            throw ConfigError("fixture token limit exceeded");
        const auto prompt = std::get<std::string_view>(input.prefix);
        TextResult result;
        result.text = std::string(prompt) + "\n\\boxed{42}\n" + values.dump();
        result.token_ids = {100, 101};
        result.setup_ms = 1;
        result.prefill_ms = 2;
        result.decode_ms = 4;
        return result;
    }
    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::TextGenerationConfig& config) override {
        // Existing path has its own method; it does not call the new interface.
        trtmc::TextResult result;
        result.text = prompt + "\n\\boxed{42}\n" +
                      Json{{"seed", config.seed},
                           {"max_new_tokens", config.max_new_tokens},
                           {"top_k", config.top_k},
                           {"use_chat_template", config.use_chat_template}}
                          .dump();
        result.token_ids = {100, 101};
        result.setup_ms = 1;
        result.prefill_ms = 2;
        result.decode_ms = 4;
        return result;
    }

  private:
    std::string mode_;
    bool multiple_;
    bool text_;
};
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "dataset_fixture")
        throw std::runtime_error("wrong fixture family");
    const auto plan = context.reader.read_section("engine.plan");
    return new DatasetFixture(context.reader.info().task,
                              plan == std::vector<char>{'B', 'O', 'T', 'H'},
                              plan != std::vector<char>{'N', 'O', 'N', 'E'});
}
