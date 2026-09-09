/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <map>
#include <set>

namespace {
using namespace trtmc;
using namespace trtmc::internal;

// Protocol fixture only: no model-quality or GPU-performance assertion.
class LanguageFixture final : public IModel,
                              public IImagesTextToText,
                              public IVideoTextToText,
                              public IImageVideoTextToText,
                              public IAudioTextToText,
                              public IImageAudioToText,
                              public IAudioVideoTextToText,
                              public IImageAudioTextToText,
                              public IImageAudioTextToTextSpeechResponse,
                              public ITextConversation,
                              public IBatchImagesTextConversation,
                              public IBatchTextConversation,
                              public IBatchVideoTextConversation,
                              public IBatchAudioTextConversation,
                              public IBatchImageAudioTextConversation,
                              public IBatchTextImagesVideoConversations,
                              public IBatchTextImagesAudioConversations,
                              public ITextLabelClassification,
                              public ITextPairLabelClassification,
                              public ITextEncoderDecoderHiddenStates {
  public:
    explicit LanguageFixture(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInfo> task_info() const override {
        if (mode_ == "none")
            return {};
        if (mode_ == "single_only")
            return {{IImagesTextToText::kTask, 1, 0}};
        return {{IImagesTextToText::kTask, 1, 0},
                {IVideoTextToText::kTask, 1, 0},
                {IImageVideoTextToText::kTask, 1, 0},
                {IAudioTextToText::kTask, 1, 0},
                {IImageAudioToText::kTask, 1, 0},
                {IAudioVideoTextToText::kTask, 1, 0},
                {IImageAudioTextToText::kTask, 1, 0},
                {IImageAudioTextToTextSpeechResponse::kTask, 1, 0},
                {ITextConversation::kTask, 1, 0},
                {IBatchImagesTextConversation::kTask, 1, 0},
                {IBatchTextConversation::kTask, 1, 0},
                {IBatchVideoTextConversation::kTask, 1, 0},
                {IBatchAudioTextConversation::kTask, 1, 0},
                {IBatchImageAudioTextConversation::kTask, 1, 0},
                {IBatchTextImagesVideoConversations::kTask, 1, 0},
                {IBatchTextImagesAudioConversations::kTask, 1, 0},
                {ITextLabelClassification::kTask, 1, 0},
                {ITextPairLabelClassification::kTask, 1, 0},
                {ITextEncoderDecoderHiddenStates::kTask, 1, 0}};
    }
    std::vector<ConfigField> config_fields(std::string_view task) const override {
        if (task.substr(0, 6) == "batch_")
            return batch_fields();
        return {{"suffix", ConfigKind::String, ConfigValue{std::string_view{"!"}},
                 "Synthetic response suffix."}};
    }
    TextResult run(const ImagesTextToTextRequest& input, ConfigView config) override {
        if (!input.tools.empty())
            throw std::invalid_argument("plain fixture does not consume tool declarations");
        return text(1, messages(input.messages), suffix(config));
    }
    TextResult run(const VideoTextToTextRequest& input, ConfigView config) override {
        if (!input.tools.empty())
            throw std::invalid_argument("plain fixture does not consume tool declarations");
        return text(2, messages(input.messages), suffix(config));
    }
    TextResult run(const ImageVideoTextToTextRequest& input, ConfigView config) override {
        return text(3, messages(input.messages), suffix(config));
    }
    TextResult run(const AudioTextToTextRequest& input, ConfigView config) override {
        return text(4, messages(input.messages), suffix(config));
    }
    TextResult run(const ImageAudioToTextRequest& input, ConfigView config) override {
        return text(5, messages(input.messages), suffix(config));
    }
    TextResult run(const AudioVideoTextToTextRequest& input, ConfigView config) override {
        return text(6, messages(input.messages), suffix(config));
    }
    TextResult run(const ImageAudioTextToTextRequest& input, ConfigView config) override {
        return text(7, messages(input.messages), suffix(config));
    }
    TextSpeechResult run(const ImageAudioTextToTextSpeechResponseRequest& input,
                         ConfigView config) override {
        return {text(8, messages(input.messages), suffix(config)),
                {{0.25F, -0.25F, 0.5F, -0.5F}, 16000, 2, 0.5, 1.5}};
    }
    ConversationResult run(const TextConversationRequest& input, ConfigView config) override {
        const auto ending = suffix(config);
        std::string probe;
        std::set<std::string_view> tool_names, pending_ids;
        for (const auto& tool : input.tools) {
            if (!tool_names.insert(tool.name).second || tool.parameters_schema_json.empty())
                throw std::invalid_argument(
                    "fixture tool declarations require unique names and schemas");
            probe += "D:" + std::string(tool.name) + ":" + std::string(tool.description) + ":" +
                     std::string(tool.parameters_schema_json) + ";";
        }
        for (const auto& message : input.messages) {
            probe += "[" + std::to_string(static_cast<uint32_t>(message.role)) + ":";
            for (const auto& part : message.parts) {
                if (const auto* value = std::get_if<TextPartView>(&part))
                    probe += "T:" + std::string(value->text) + ";";
                else if (const auto* value = std::get_if<ReasoningPartView>(&part))
                    probe += "R:" + std::string(value->text) + ";";
                else if (const auto* value = std::get_if<ToolCallView>(&part)) {
                    if (!tool_names.count(value->name) ||
                        !pending_ids.insert(value->call_id).second)
                        throw std::invalid_argument("fixture tool call is undeclared or duplicate");
                    probe += "C:" + std::string(value->call_id) + ":" + std::string(value->name) +
                             ":" + std::string(value->arguments_json) + ";";
                } else {
                    const auto& response = std::get<ToolResultView>(part);
                    if (!pending_ids.erase(response.call_id))
                        throw std::invalid_argument("fixture tool result has no matching call");
                    probe += "O:" + std::string(response.call_id) + ":" +
                             std::string(response.content_text) + ":" +
                             (response.is_error ? "error" : "ok") + ";";
                }
            }
            probe += "]";
        }
        ConversationResult result;
        result.parts = {ReasoningPart{"private reasoning marker"}, TextPart{"9:" + probe + ending}};
        if (!input.tools.empty()) {
            result.parts.emplace_back(
                ToolCall{"next-1", std::string(input.tools[0].name), R"({"x":2})"});
            if (mode_ == "bad_calls")
                result.parts.emplace_back(
                    ToolCall{"next-1", std::string(input.tools[0].name), "{}"});
        }
        result.token_ids = {9, 99};
        result.setup_ms = 0.25;
        result.prefill_ms = 0.5;
        result.decode_ms = 0.75;
        return result;
    }
    std::vector<ConversationResult>
    run_batch(const BatchImagesTextConversationRequest& input) override {
        return batch(input, 13);
    }
    std::vector<ConversationResult> run_batch(const BatchTextConversationRequest& input) override {
        return batch(input, 14);
    }
    std::vector<ConversationResult>
    run_batch(const BatchVideoTextConversationRequest& input) override {
        return batch(input, 15);
    }
    std::vector<ConversationResult>
    run_batch(const BatchAudioTextConversationRequest& input) override {
        return batch(input, 16);
    }
    std::vector<ConversationResult>
    run_batch(const BatchImageAudioTextConversationRequest& input) override {
        return batch(input, 17);
    }
    std::vector<ConversationResult>
    run_batch(const BatchTextImagesVideoConversationsRequest& input) override {
        return batch(input, 18);
    }
    std::vector<ConversationResult>
    run_batch(const BatchTextImagesAudioConversationsRequest& input) override {
        return batch(input, 19);
    }
    GeneratedLabelResult run(const TextLabelClassificationRequest& input,
                             ConfigView config) override {
        (void)suffix(config);
        if (input.text == "unknown")
            return {"not a label", -1, {"negative", "positive"}, {10, 99}};
        return {"positive",
                mode_ == "bad_label" ? 7 : 1,
                {"negative", "positive"},
                {10, static_cast<int32_t>(input.text.size())}};
    }
    GeneratedLabelResult run(const TextPairLabelClassificationRequest& input,
                             ConfigView config) override {
        (void)suffix(config);
        return {"related",
                1,
                {"unrelated", "related"},
                {11, static_cast<int32_t>(input.first.size()),
                 static_cast<int32_t>(input.second.size())}};
    }
    EncoderDecoderStatesResult run(const TextEncoderDecoderHiddenStatesRequest& input,
                                   ConfigView config) override {
        (void)suffix(config);
        EncoderDecoderStatesResult out;
        out.encoder_last_hidden_state = matrix(input.source, 2);
        out.decoder_last_hidden_state = matrix(input.decoder, 3);
        if (mode_ == "bad_axis") {
            out.encoder_last_hidden_state.rows += 1;
            out.encoder_last_hidden_state.values.insert(out.encoder_last_hidden_state.values.end(),
                                                        {0, 0});
        }
        return out;
    }

  private:
    static std::vector<ConfigField> batch_fields() {
        return {
            {"suffix", ConfigKind::String, ConfigValue{std::string_view{"!"}}, "Response suffix."},
            {"fixture.flag", ConfigKind::Bool, ConfigValue{true}, "Boolean marker."},
            {"fixture.number", ConfigKind::I64, ConfigValue{int64_t{7}}, "Integer marker."},
            {"fixture.scale", ConfigKind::F64, ConfigValue{0.5}, "Floating marker."},
            {"fixture.numbers", ConfigKind::I64List, ConfigValue{Span<const int64_t>{}},
             "Integer markers."},
            {"fixture.scales", ConfigKind::F64List, ConfigValue{Span<const double>{}},
             "Floating markers."},
            {"fixture.labels", ConfigKind::StringList, ConfigValue{Span<const std::string_view>{}},
             "String markers."}};
    }
    static std::string batch_config(ConfigView config) {
        const auto fields = batch_fields();
        std::set<std::string_view> seen;
        std::string ending{"!"}, markers;
        for (const auto& entry : config) {
            const auto rule =
                std::find_if(fields.begin(), fields.end(),
                             [&](const ConfigField& field) { return field.name == entry.name; });
            if (rule == fields.end() || !seen.insert(entry.name).second ||
                rule->kind != config_kind(entry.value))
                throw ConfigError("unknown, duplicate or mistyped batch config");
            if (entry.name == "suffix")
                ending = std::get<std::string_view>(entry.value);
            else {
                markers += std::string(entry.name) + "=";
                std::visit(
                    [&](const auto& value) {
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, bool>)
                            markers += value ? "true" : "false";
                        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>)
                            markers += std::to_string(value);
                        else if constexpr (std::is_same_v<T, std::string_view>)
                            markers += std::string(value);
                        else {
                            markers += "[";
                            for (const auto& item : value) {
                                if constexpr (std::is_same_v<std::decay_t<decltype(item)>,
                                                             std::string_view>)
                                    markers += std::string(item);
                                else
                                    markers += std::to_string(item);
                                markers += ",";
                            }
                            markers += "]";
                        }
                    },
                    entry.value);
                markers += ";";
            }
        }
        return ending + markers;
    }
    template <class Request>
    static std::string request_probe(const Request& input) {
        std::set<std::string_view> names, pending;
        std::string probe;
        for (const auto& tool : input.tools) {
            if (!names.insert(tool.name).second || tool.parameters_schema_json.empty())
                throw std::invalid_argument("fixture tools require unique names and schemas");
            probe += "D:" + std::string(tool.name) + ":" + std::string(tool.description) + ":" +
                     std::string(tool.parameters_schema_json) + ";";
        }
        for (const auto& message : input.messages) {
            for (const auto& part : message.parts) {
                std::visit(
                    [&](const auto& value) {
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, ToolCallView>) {
                            if (!names.count(value.name) || !pending.insert(value.call_id).second)
                                throw std::invalid_argument(
                                    "fixture tool call is undeclared or duplicate");
                        } else if constexpr (std::is_same_v<T, ToolResultView>) {
                            if (!pending.erase(value.call_id))
                                throw std::invalid_argument(
                                    "fixture tool result has no matching call");
                        }
                    },
                    part);
            }
        }
        return probe + messages(input.messages);
    }
    static std::string request_probe(const TextImagesVideoConversationRequest& input) {
        return std::visit([](const auto& value) { return request_probe(value); }, input);
    }
    static std::string request_probe(const TextImagesAudioConversationRequest& input) {
        return std::visit([](const auto& value) { return request_probe(value); }, input);
    }
    template <class Request>
    std::vector<ConversationResult> batch(const Request& input, int32_t marker) {
        std::vector<std::string> probes, endings;
        for (size_t index = 0; index < input.items.size(); ++index) {
            try {
                endings.push_back(batch_config(input.items[index].config));
                probes.push_back(request_probe(input.items[index].input));
            } catch (const ConfigError& error) {
                throw ConfigError("batch item[" + std::to_string(index) + "]: " + error.what());
            } catch (const std::invalid_argument& error) {
                throw std::invalid_argument("batch item[" + std::to_string(index) +
                                            "]: " + error.what());
            }
        }
        if (mode_ == "equal_batch_config" &&
            std::any_of(endings.begin(), endings.end(),
                        [&](const std::string& value) { return value != endings.front(); }))
            throw ConfigError("fixture batch requires equal resolved generation options");
        ++batch_calls_;
        if (mode_ == "batch_failure")
            throw std::runtime_error("fixture batch execution failed");
        std::vector<ConversationResult> results;
        for (size_t index = 0; index < input.items.size(); ++index) {
            ConversationResult result;
            result.parts = {ReasoningPart{"batch reasoning"},
                            TextPart{probes[index] + endings[index]}};
            result.token_ids = {marker, batch_calls_, static_cast<int32_t>(index)};
            result.setup_ms = 0.25;
            result.prefill_ms = 0.5;
            result.decode_ms = 0.75;
            if (mode_ == "batch_structured") {
                result.parts.emplace_back(ToolCall{"generated-" + std::to_string(index), "lookup",
                                                   "{broken", ToolCallState::Malformed});
                result.finish_reason = FinishReason::Length;
                result.usage.output_tokens = 0;
            }
            results.push_back(std::move(result));
        }
        if (mode_ == "bad_batch_count")
            results.pop_back();
        return results;
    }
    static std::string suffix(ConfigView config) {
        std::string output = "!";
        bool seen = false;
        for (const auto& entry : config) {
            if (entry.name != "suffix" || seen ||
                !std::holds_alternative<std::string_view>(entry.value))
                throw ConfigError("unknown, duplicate or mistyped language config");
            output = std::get<std::string_view>(entry.value);
            seen = true;
        }
        return output;
    }
    static TextResult text(int marker, std::string probe, const std::string& suffix) {
        TextResult out{std::to_string(marker) + ":" + probe + suffix, {marker, 99}, 0.5, 0.75};
        out.setup_ms = 0.25;
        return out;
    }
    static std::string part(const TextPartView& value) {
        return "T:" + std::string(value.text) + ";";
    }
    static std::string part(const ReasoningPartView& value) {
        return "R:" + std::string(value.text) + ";";
    }
    static std::string part(const ToolCallView& value) {
        return "C:" + std::string(value.call_id) + ":" + std::string(value.name) + ":" +
               std::string(value.arguments_json) + ";";
    }
    static std::string part(const ToolResultView& value) {
        return "O:" + std::string(value.call_id) + ":" + std::string(value.content_text) + ";";
    }
    static std::string part(const ImageView& value) {
        const int pixel = value.format == ImageFormat::Float32
                              ? static_cast<int>(static_cast<const float*>(value.data)[0])
                              : static_cast<const uint8_t*>(value.data)[0];
        return "I:" + std::to_string(pixel) + ";";
    }
    static std::string part(const VideoView& value) {
        std::string out = "V:" + std::to_string(value.frames.size()) + ":";
        for (const auto& frame : value.frames)
            out += part(frame);
        out += value.timestamps_seconds.empty()
                   ? "untimed;"
                   : "t" +
                         std::to_string(
                             value.timestamps_seconds[value.timestamps_seconds.size() - 1]) +
                         ";";
        return out;
    }
    static std::string part(const AudioView& value) {
        // This fixture deliberately has no implicit sampling rate.
        if (!value.sample_rate)
            throw std::invalid_argument("fixture requires an explicit audio rate");
        return "A:" + std::to_string(*value.sample_rate) + ":" + std::to_string(value.channels) +
               ":" + std::to_string(value.samples.size()) + ":" + std::to_string(value.samples[0]) +
               ";";
    }
    static std::string part(const AlignedAudioVideoView& value) {
        if (!value.audio_start_seconds || value.video.timestamps_seconds.empty())
            throw std::invalid_argument("fixture requires explicit aligned-media clock metadata");
        return "AV:" + part(value.video) + part(value.audio) + ":" +
               std::to_string(*value.audio_start_seconds) + ";";
    }
    template <class Message>
    static std::string messages(Span<const Message> input) {
        std::string out;
        for (const auto& message : input) {
            out += "[" + std::to_string(static_cast<uint32_t>(message.role)) + ":";
            for (const auto& item : message.parts)
                out += std::visit([](const auto& value) { return part(value); }, item);
            out += "]";
        }
        return out;
    }
    static FloatMatrix matrix(const LanguageTokenSequence& input, uint64_t columns) {
        FloatMatrix out{{}, input.token_ids.size(), columns};
        for (size_t row = 0; row < input.token_ids.size(); ++row)
            for (uint64_t column = 0; column < columns; ++column)
                out.values.push_back(
                    static_cast<float>(input.token_ids[row]) + static_cast<float>(column) / 10 +
                    ((input.attention_mask.empty() || input.attention_mask[row]) ? 0 : 1000));
        return out;
    }
    std::string mode_;
    int32_t batch_calls_{0};
};
class DeclaredOnly final : public IModel {
  public:
    const char* task() const noexcept override { return "missing"; }
    std::vector<TaskInfo> task_info() const override {
        return {{IImagesTextToText::kTask, 1, 0},
                {IBatchTextConversation::kTask, 1, 0},
                {IBatchTextImagesAudioConversations::kTask, 1, 0}};
    }
    std::vector<ConfigField> config_fields(std::string_view) const override { return {}; }
};
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "language_fixture")
        throw std::runtime_error("wrong fixture family");
    if (context.reader.info().task == "missing")
        return new DeclaredOnly;
    return new LanguageFixture(context.reader.info().task);
}
