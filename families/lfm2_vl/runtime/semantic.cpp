/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_vl/runtime/conversation.h"
#include "families/lfm2_vl/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>

namespace trtmc {
namespace {
namespace api = internal;
Span<const api::ConfigField> fields(bool images = true) {
    static const std::vector<api::ConfigField> values{
        {"max_new_tokens", api::ConfigKind::I64, std::int64_t{128}, "Maximum generated tokens"},
        {"temperature", api::ConfigKind::F64, 0.1, "Sampling temperature"},
        {"top_k", api::ConfigKind::I64, std::int64_t{1},
         "Candidate count; one selects greedy decoding"},
        {"top_p", api::ConfigKind::F64, 1.0, "Nucleus probability mass"},
        {"min_p", api::ConfigKind::F64, 0.15,
         "Minimum probability relative to the most likely token"},
        {"repetition_penalty", api::ConfigKind::F64, 1.05,
         "Penalty applied to previously seen tokens"},
        {"seed", api::ConfigKind::I64, std::int64_t{42}, "Sampling seed"},
        {"keep_past_thinking", api::ConfigKind::Bool, false,
         "Keep reasoning in assistant turns before the latest assistant message"},
        {"min_image_tokens", api::ConfigKind::I64, std::int64_t{64},
         "Minimum tokens per resized image"},
        {"max_image_tokens", api::ConfigKind::I64, std::int64_t{256},
         "Maximum tokens per resized image"},
        {"max_tiles", api::ConfigKind::I64, std::int64_t{10}, "Maximum number of image tiles"},
        {"do_image_splitting", api::ConfigKind::Bool, true,
         "Split high resolution images into tiles"},
    };
    const auto image_fields = std::find_if(values.begin(), values.end(), [](const auto& field) {
        return field.name == "min_image_tokens";
    });
    return {values.data(),
            images ? values.size() : static_cast<std::size_t>(image_fields - values.begin())};
}
std::size_t python_space_width(const std::string& value, std::size_t at) {
    const auto first = static_cast<unsigned char>(value[at]);
    // Python str.strip additionally recognizes U+001C..U+001F, which are not
    // members of the Unicode White_Space property.
    if ((first >= 0x09 && first <= 0x0D) || (first >= 0x1C && first <= 0x20))
        return 1;
    if (at + 1 < value.size() && first == 0xC2) {
        const auto second = static_cast<unsigned char>(value[at + 1]);
        if (second == 0x85 || second == 0xA0)
            return 2;
    }
    if (at + 2 < value.size()) {
        const auto second = static_cast<unsigned char>(value[at + 1]);
        const auto third = static_cast<unsigned char>(value[at + 2]);
        if ((first == 0xE1 && second == 0x9A && third == 0x80) ||
            (first == 0xE2 && second == 0x80 &&
             ((third >= 0x80 && third <= 0x8A) || third == 0xA8 || third == 0xA9 ||
              third == 0xAF)) ||
            (first == 0xE2 && second == 0x81 && third == 0x9F) ||
            (first == 0xE3 && second == 0x80 && third == 0x80))
            return 3;
    }
    return 0;
}

std::string trim(const std::string& value) {
    std::size_t begin = 0, end = value.size();
    while (begin < value.size()) {
        const auto width = python_space_width(value, begin);
        if (!width)
            break;
        begin += width;
    }
    std::size_t last_content = begin;
    for (std::size_t position = begin; position < end;) {
        const auto width = python_space_width(value, position);
        if (width)
            position += width;
        else
            last_content = ++position;
    }
    return value.substr(begin, last_content - begin);
}

std::string json_text(const nlohmann::ordered_json& value, bool python = false) {
    if (value.is_object()) {
        std::string out = "{";
        bool first = true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!first)
                out += ", ";
            first = false;
            out += nlohmann::ordered_json(it.key()).dump(-1, ' ', false) + ": " +
                   json_text(it.value(), python);
        }
        return out + "}";
    }
    if (value.is_array()) {
        std::string out = "[";
        bool first = true;
        for (const auto& entry : value) {
            if (!first)
                out += ", ";
            first = false;
            out += json_text(entry, python);
        }
        return out + "]";
    }
    if (python && value.is_null())
        return "None";
    if (python && value.is_boolean())
        return value.get<bool>() ? "True" : "False";
    return value.dump(-1, ' ', false);
}
std::string tool_call(const api::ToolCallView& call) {
    if (call.name.empty() || call.state == api::ToolCallState::Malformed ||
        call.state == api::ToolCallState::Incomplete)
        throw std::invalid_argument("LFM2-VL requires a complete named tool call");
    const auto arguments = nlohmann::ordered_json::parse(call.arguments_json);
    if (!arguments.is_object())
        throw std::invalid_argument("LFM2-VL tool arguments must be an object");
    std::string text = "<|tool_call_start|>[" + std::string(call.name) + "(";
    bool first = true;
    for (auto it = arguments.begin(); it != arguments.end(); ++it) {
        if (!first)
            text += ", ";
        first = false;
        text += it.key() + "=" + json_text(it.value(), true);
    }
    return text + ")]<|tool_call_end|>";
}
std::string role(api::MessageRole role) {
    switch (role) {
    case api::MessageRole::System:
        return "system";
    case api::MessageRole::Developer:
        return "developer";
    case api::MessageRole::User:
        return "user";
    case api::MessageRole::Assistant:
        return "assistant";
    case api::MessageRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("invalid LFM2-VL message role");
}
ImageResult image(const api::ImageView& source) {
    if (!source.data || !source.width || !source.height || source.channels != 3 ||
        source.width > static_cast<unsigned>(std::numeric_limits<int>::max()) ||
        source.height > static_cast<unsigned>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("LFM2-VL requires an RGB image");
    const auto count = static_cast<std::size_t>(source.width) * source.height * 3;
    const bool bytes = source.format == api::ImageFormat::UInt8;
    if (!bytes && source.format != api::ImageFormat::Float32)
        throw std::invalid_argument("unsupported LFM2-VL image format");
    if (source.byte_size != count * (bytes ? 1 : sizeof(float)))
        throw std::invalid_argument("invalid LFM2-VL image byte size");
    ImageResult out;
    out.width = source.width;
    out.height = source.height;
    out.pixels.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float value = bytes ? static_cast<const std::uint8_t*>(source.data)[i]
                                  : static_cast<const float*>(source.data)[i] * 255.0F;
        if (!std::isfinite(value) || value < 0 || value > 255)
            throw std::invalid_argument("invalid LFM2-VL pixel value");
        // The public processor consumes RGB byte images. The SDK's decoded
        // Float32 representation transports those byte values normalized to [0,1].
        out.pixels[i] = std::nearbyint(value);
    }
    return out;
}
} // namespace
std::vector<internal::TaskInstance> Lfm2VlTextGenerationPipeline::task_bindings() {
    return {internal::bind<internal::IImagesTextToText>(*this, fields()),
            internal::bind<internal::ITextConversation>(*this, fields(false))};
}
internal::ConversationResult
Lfm2VlTextGenerationPipeline::run(const internal::TextConversationRequest& input,
                                  internal::ConfigView supplied) {
    internal::validate_config(fields(false), supplied);
    std::vector<std::vector<internal::ImagesTextPart>> parts;
    for (const auto& message : input.messages) {
        parts.emplace_back();
        for (const auto& part : message.parts)
            std::visit([&](const auto& value) { parts.back().emplace_back(value); }, part);
    }
    std::vector<internal::ImagesTextMessage> messages;
    for (std::size_t i = 0; i < parts.size(); ++i)
        messages.push_back({input.messages[i].role, {parts[i].data(), parts[i].size()}});
    const auto text =
        run(internal::ImagesTextToTextRequest{{messages.data(), messages.size()}, input.tools},
            supplied);
    auto result = lfm2_vl_parse_conversation(text.text, false, !input.tools.empty());
    lfm2_vl_assign_tool_call_ids(result, input);
    result.token_ids = text.token_ids;
    result.setup_ms = text.setup_ms;
    result.prefill_ms = text.prefill_ms;
    result.decode_ms = text.decode_ms;
    result.usage.output_tokens = text.token_ids.size();
    const bool stopped =
        !text.token_ids.empty() && std::find(config_.id_eos_ids.begin(), config_.id_eos_ids.end(),
                                             text.token_ids.back()) != config_.id_eos_ids.end();
    result.finish_reason = stopped ? internal::FinishReason::Stop : internal::FinishReason::Length;
    if (stopped && std::any_of(result.parts.begin(), result.parts.end(), [](const auto& part) {
            const auto* call = std::get_if<internal::ToolCall>(&part);
            return call && call->state == internal::ToolCallState::Complete;
        }))
        result.finish_reason = internal::FinishReason::ToolCalls;
    return result;
}
TextResult Lfm2VlTextGenerationPipeline::run(const internal::ImagesTextToTextRequest& input,
                                             internal::ConfigView supplied) {
    return generate(lfm2_vl_prepare_request(input, supplied));
}
Lfm2VlRequest lfm2_vl_prepare_request(const internal::ImagesTextToTextRequest& input,
                                      internal::ConfigView supplied) {
    const auto& config_fields = fields();
    internal::validate_config(config_fields, supplied);
    auto integer = [&](const char* key) {
        auto value = internal::config_get<std::int64_t>(supplied, config_fields, key).value();
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument(std::string("LFM2-VL option out of range: ") + key);
        return static_cast<std::int32_t>(value);
    };
    auto real = [&](const char* key) {
        return static_cast<float>(
            internal::config_get<double>(supplied, config_fields, key).value());
    };
    Lfm2VlRequest request;
    request.generation.max_new_tokens = integer("max_new_tokens");
    request.generation.top_k = integer("top_k");
    request.generation.seed = integer("seed");
    request.generation.temperature = real("temperature");
    request.generation.top_p = real("top_p");
    request.generation.min_p = real("min_p");
    request.generation.repetition_penalty = real("repetition_penalty");
    request.min_image_tokens = integer("min_image_tokens");
    request.max_image_tokens = integer("max_image_tokens");
    request.max_tiles = integer("max_tiles");
    request.split_images =
        internal::config_get<bool>(supplied, config_fields, "do_image_splitting").value();
    if (request.generation.max_new_tokens <= 0 || request.min_image_tokens <= 0 ||
        request.max_image_tokens < request.min_image_tokens || request.max_image_tokens > 1024 ||
        request.max_tiles < 1 || request.max_tiles > 10 ||
        (request.split_images && request.max_tiles < 2))
        throw std::invalid_argument("LFM2-VL generation or image budget is out of range");
    if (input.messages.empty())
        throw std::invalid_argument("LFM2-VL requires messages");
    std::string tools;
    if (!input.tools.empty()) {
        tools = "List of tools: [";
        bool first = true;
        for (const auto& tool : input.tools) {
            if (!first)
                tools += ", ";
            first = false;
            nlohmann::ordered_json entry = {
                {"name", tool.name},
                {"description", tool.description},
                {"parameters", nlohmann::ordered_json::parse(tool.parameters_schema_json)}};
            tools += json_text(entry);
        }
        tools += "]";
    }
    request.prompt = "<|startoftext|>";
    if (!tools.empty() && input.messages[0].role != internal::MessageRole::System)
        request.prompt += "<|im_start|>system\n" + tools + "<|im_end|>\n";
    const bool keep_past_thinking =
        internal::config_get<bool>(supplied, config_fields, "keep_past_thinking").value();
    std::size_t last_assistant = input.messages.size();
    for (std::size_t i = 0; i < input.messages.size(); ++i)
        if (input.messages[i].role == internal::MessageRole::Assistant)
            last_assistant = i;
    for (std::size_t index = 0; index < input.messages.size(); ++index) {
        const auto& message = input.messages[index];
        std::string content;
        for (const auto& part : message.parts) {
            if (const auto* text = std::get_if<internal::TextPartView>(&part))
                content += text->text;
            else if (const auto* media = std::get_if<internal::ImageView>(&part)) {
                request.images.push_back(image(*media));
                request.integer_pixels.push_back(true);
                content += "<image>";
            } else if (const auto* reasoning = std::get_if<internal::ReasoningPartView>(&part)) {
                if (message.role != internal::MessageRole::Assistant)
                    throw std::invalid_argument("LFM2-VL reasoning requires an assistant role");
                content += "<think>" + std::string(reasoning->text) + "</think>";
            } else if (const auto* result = std::get_if<internal::ToolResultView>(&part)) {
                if (message.role != internal::MessageRole::Tool)
                    throw std::invalid_argument("LFM2-VL tool results require a tool role");
                content += result->content_text;
            } else if (const auto* call = std::get_if<internal::ToolCallView>(&part)) {
                if (message.role != internal::MessageRole::Assistant)
                    throw std::invalid_argument("LFM2-VL tool calls require an assistant role");
                content += tool_call(*call);
            }
        }
        if (message.role == internal::MessageRole::Assistant && index != last_assistant &&
            !keep_past_thinking) {
            const auto end = content.rfind("</think>");
            if (end != std::string::npos)
                content = trim(content.substr(end + 8));
        }
        if (index == 0 && message.role == internal::MessageRole::System) {
            if (!tools.empty())
                content += (content.empty() ? "" : "\n") + tools;
            if (content.empty())
                continue;
        }
        request.prompt += "<|im_start|>" + role(message.role) + "\n" + content + "<|im_end|>\n";
    }
    request.prompt += "<|im_start|>assistant\n";
    return request;
}
} // namespace trtmc
