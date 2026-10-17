/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_moe/runtime/conversation.h"

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace trtmc {
namespace {
using Json = nlohmann::ordered_json;
using namespace internal;

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

// Transformers' template tojson uses json.dumps(..., ensure_ascii=False), with
// insertion order and spaces after separators. JSON schema order affects IDs.
std::string template_json(const Json& value) {
    if (!value.is_structured())
        return value.dump();
    std::string output = value.is_object() ? "{" : "[";
    bool first = true;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!first)
            output += ", ";
        first = false;
        if (value.is_object())
            output += Json(it.key()).dump() + ": ";
        output += template_json(it.value());
    }
    return output + (value.is_object() ? "}" : "]");
}

std::string python_repr(const Json& value) {
    if (value.is_string()) {
        const auto text = value.get<std::string>();
        const char quote =
            text.find('\'') != std::string::npos && text.find('"') == std::string::npos ? '"'
                                                                                        : '\'';
        std::string output(1, quote);
        for (char c : text) {
            if (c == quote) {
                output += '\\';
                output += c;
                continue;
            }
            switch (c) {
            case '\\':
                output += "\\\\";
                break;
            case '\n':
                output += "\\n";
                break;
            case '\r':
                output += "\\r";
                break;
            case '\t':
                output += "\\t";
                break;
            default:
                output += c;
            }
        }
        return output + quote;
    }
    if (value.is_structured()) {
        std::string output = value.is_object() ? "{" : "[";
        bool first = true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!first)
                output += ", ";
            first = false;
            if (value.is_object())
                output += python_repr(it.key()) + ": ";
            output += python_repr(it.value());
        }
        return output + (value.is_object() ? "}" : "]");
    }
    if (value.is_boolean())
        return value.get<bool>() ? "True" : "False";
    if (value.is_null())
        return "None";
    return value.dump();
}

std::string argument(const Json& value) {
    // The pinned MoE template concatenates string arguments without escaping,
    // JSON-serializes mappings, and uses Python str() for other values. Preserve
    // this checkpoint's historical format, including its string limitations.
    if (value.is_string())
        return "'" + value.get<std::string>() + "'";
    if (value.is_object())
        return template_json(value);
    return python_repr(value);
}

const char* role_name(MessageRole role) {
    switch (role) {
    case MessageRole::System:
        return "system";
    case MessageRole::Developer:
        return "developer";
    case MessageRole::User:
        return "user";
    case MessageRole::Assistant:
        return "assistant";
    case MessageRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("LFM2-MoE conversation has an invalid role");
}

struct Message {
    MessageRole role;
    std::string content, reasoning;
    std::vector<ToolCallView> calls;
};

Message read_message(const ConversationMessageView& input) {
    Message output{input.role, {}, {}, {}};
    for (const auto& part : input.parts) {
        if (const auto* text = std::get_if<TextPartView>(&part))
            output.content += text->text;
        else if (const auto* reasoning = std::get_if<ReasoningPartView>(&part)) {
            if (input.role != MessageRole::Assistant)
                throw std::invalid_argument("LFM2-MoE reasoning requires an assistant role");
            output.reasoning += reasoning->text;
        } else if (const auto* call = std::get_if<ToolCallView>(&part)) {
            if (input.role != MessageRole::Assistant)
                throw std::invalid_argument("LFM2-MoE tool calls require an assistant role");
            output.calls.push_back(*call);
        } else if (const auto* result = std::get_if<ToolResultView>(&part)) {
            if (input.role != MessageRole::Tool)
                throw std::invalid_argument("LFM2-MoE tool results require a tool role");
            output.content += result->content_text;
        }
    }
    return output;
}

std::string render_calls(const std::vector<ToolCallView>& calls) {
    std::string output = "<|tool_call_start|>[";
    for (std::size_t index = 0; index < calls.size(); ++index) {
        if (index)
            output += ", ";
        const auto& call = calls[index];
        if (call.name.empty() || call.state == ToolCallState::Malformed ||
            call.state == ToolCallState::Incomplete)
            throw std::invalid_argument(
                "LFM2-MoE cannot render a malformed or incomplete tool call");
        const auto args = Json::parse(call.arguments_json);
        if (!args.is_object())
            throw std::invalid_argument("LFM2-MoE tool call arguments must be a JSON object");
        output += std::string(call.name) + "(";
        bool first = true;
        for (auto it = args.begin(); it != args.end(); ++it) {
            if (!first)
                output += ", ";
            first = false;
            output += it.key() + "=" + argument(it.value());
        }
        output += ")";
    }
    return output + "]<|tool_call_end|>";
}

// Parse literal Pythonic arguments without evaluating code. The model protocol
// allows JSON-style nested containers and Python's True/False/None literals.
class LiteralParser {
  public:
    explicit LiteralParser(std::string_view text) : text_(text) {}
    void whitespace() {
        while (position_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[position_])))
            ++position_;
    }
    bool consume(char value) {
        whitespace();
        if (position_ < text_.size() && text_[position_] == value) {
            ++position_;
            return true;
        }
        return false;
    }
    void require(char value) {
        if (!consume(value))
            throw std::invalid_argument("invalid Pythonic tool literal");
    }
    std::string identifier(bool qualified = false) {
        whitespace();
        const auto start = position_;
        bool segment_start = true;
        while (position_ < text_.size()) {
            const auto c = static_cast<unsigned char>(text_[position_]);
            if (segment_start) {
                if (!std::isalpha(c) && c != '_')
                    throw std::invalid_argument("invalid tool identifier");
                segment_start = false;
                ++position_;
            } else if (std::isalnum(c) || c == '_')
                ++position_;
            else if (qualified && c == '.') {
                segment_start = true;
                ++position_;
            } else
                break;
        }
        if (position_ == start || segment_start)
            throw std::invalid_argument("invalid tool identifier");
        return std::string(text_.substr(start, position_ - start));
    }
    bool done() {
        whitespace();
        return position_ == text_.size();
    }
    Json value(unsigned depth = 0) {
        if (depth > 64)
            throw std::invalid_argument("tool literal nesting exceeds 64 levels");
        whitespace();
        if (position_ == text_.size())
            throw std::invalid_argument("missing tool literal");
        const char c = text_[position_];
        if (c == '\'' || c == '"') {
            const auto start = position_++;
            std::string output;
            while (position_ < text_.size()) {
                char ch = text_[position_++];
                if (ch == c)
                    return output;
                if (ch == '\\') {
                    if (position_ == text_.size())
                        break;
                    ch = text_[position_++];
                    if (ch == 'n')
                        ch = '\n';
                    else if (ch == 'r')
                        ch = '\r';
                    else if (ch == 't')
                        ch = '\t';
                    else if (ch == 'b')
                        ch = '\b';
                    else if (ch == 'f')
                        ch = '\f';
                    else if (ch == 'a')
                        ch = '\a';
                    else if (ch == 'v')
                        ch = '\v';
                    else if (ch == 'u' || ch == 'U' || ch == 'x') {
                        auto codepoint = hex(ch == 'u' ? 4 : (ch == 'U' ? 8 : 2));
                        if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                            if (text_.substr(position_, 2) != "\\u")
                                throw std::invalid_argument("unpaired tool string surrogate");
                            position_ += 2;
                            const auto low = hex(4);
                            if (low < 0xDC00 || low > 0xDFFF)
                                throw std::invalid_argument("invalid tool string surrogate");
                            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + low - 0xDC00;
                        }
                        if (codepoint > 0x10FFFF || (codepoint >= 0xD800 && codepoint <= 0xDFFF))
                            throw std::invalid_argument("invalid tool string code point");
                        append_utf8(output, codepoint);
                        continue;
                    } else if (ch != '\\' && ch != '\'' && ch != '"' && ch != '/')
                        throw std::invalid_argument("unsupported tool string escape");
                } else if (ch == '\n' || ch == '\r')
                    throw std::invalid_argument("unescaped newline in tool string");
                output += ch;
            }
            (void)start;
            throw std::invalid_argument("unterminated tool string");
        }
        if (consume('[')) {
            Json result = Json::array();
            if (!consume(']')) {
                do {
                    result.push_back(value(depth + 1));
                } while (consume(','));
                require(']');
            }
            return result;
        }
        if (consume('{')) {
            Json result = Json::object();
            if (!consume('}')) {
                do {
                    auto key = value(depth + 1);
                    if (!key.is_string())
                        throw std::invalid_argument("tool object keys must be strings");
                    require(':');
                    result[key.get<std::string>()] = value(depth + 1);
                } while (consume(','));
                require('}');
            }
            return result;
        }
        const auto start = position_;
        while (position_ < text_.size()) {
            const char ch = text_[position_];
            if (ch == ',' || ch == ')' || ch == ']' || ch == '}' ||
                std::isspace(static_cast<unsigned char>(ch)))
                break;
            ++position_;
        }
        const auto literal = std::string(text_.substr(start, position_ - start));
        if (literal == "True")
            return true;
        if (literal == "False")
            return false;
        if (literal == "None")
            return nullptr;
        return Json::parse(literal);
    }

  private:
    std::uint32_t hex(unsigned count) {
        std::uint32_t value = 0;
        for (unsigned index = 0; index < count; ++index) {
            if (position_ == text_.size())
                throw std::invalid_argument("incomplete tool string escape");
            const auto c = text_[position_++];
            const int digit = c >= '0' && c <= '9'   ? c - '0'
                              : c >= 'a' && c <= 'f' ? c - 'a' + 10
                              : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                                     : -1;
            if (digit < 0)
                throw std::invalid_argument("invalid tool string escape");
            value = (value << 4) | static_cast<std::uint32_t>(digit);
        }
        return value;
    }
    static void append_utf8(std::string& output, std::uint32_t value) {
        if (value <= 0x7F)
            output += static_cast<char>(value);
        else if (value <= 0x7FF) {
            output += static_cast<char>(0xC0 | (value >> 6));
            output += static_cast<char>(0x80 | (value & 0x3F));
        } else if (value <= 0xFFFF) {
            output += static_cast<char>(0xE0 | (value >> 12));
            output += static_cast<char>(0x80 | ((value >> 6) & 0x3F));
            output += static_cast<char>(0x80 | (value & 0x3F));
        } else {
            output += static_cast<char>(0xF0 | (value >> 18));
            output += static_cast<char>(0x80 | ((value >> 12) & 0x3F));
            output += static_cast<char>(0x80 | ((value >> 6) & 0x3F));
            output += static_cast<char>(0x80 | (value & 0x3F));
        }
    }
    std::string_view text_;
    std::size_t position_{0};
};

std::vector<ToolCall> parse_calls(const std::string& text, bool complete) {
    if (!complete)
        return {{"", "", text, ToolCallState::Incomplete}};
    try {
        const auto json = Json::parse(text, nullptr, false);
        std::vector<ToolCall> calls;
        if (!json.is_discarded()) {
            const auto values = json.is_array() ? json : Json::array({json});
            for (const auto& item : values) {
                const auto& function = item.contains("function") ? item.at("function") : item;
                const auto name = function.at("name").get<std::string>();
                auto args = function.at("arguments");
                if (args.is_string())
                    args = Json::parse(args.get<std::string>());
                if (name.empty() || !args.is_object())
                    throw std::invalid_argument("invalid JSON tool call");
                calls.push_back({item.value("id", ""), name, args.dump(), ToolCallState::Complete});
            }
        } else {
            LiteralParser parser(text);
            parser.require('[');
            if (!parser.consume(']')) {
                do {
                    const auto name = parser.identifier(true);
                    parser.require('(');
                    Json args = Json::object();
                    if (!parser.consume(')')) {
                        do {
                            const auto key = parser.identifier();
                            parser.require('=');
                            if (args.contains(key))
                                throw std::invalid_argument("duplicate tool argument");
                            args[key] = parser.value();
                        } while (parser.consume(','));
                        parser.require(')');
                    }
                    calls.push_back({"", name, args.dump(), ToolCallState::Complete});
                } while (parser.consume(','));
                parser.require(']');
            }
            if (!parser.done())
                throw std::invalid_argument("trailing tool text");
        }
        if (calls.empty())
            throw std::invalid_argument("empty tool call list");
        return calls;
    } catch (const std::exception&) {
        return {{"", "", text, ToolCallState::Malformed}};
    }
}
} // namespace

std::string lfm2_moe_render_conversation(const internal::TextConversationRequest& request,
                                         bool preserve_thinking) {
    if (request.messages.empty())
        throw std::invalid_argument("LFM2-MoE conversation requires at least one message");
    std::vector<Message> messages;
    for (const auto& message : request.messages)
        messages.push_back(read_message(message));
    std::string system;
    if (messages.front().role == MessageRole::System) {
        system = messages.front().content;
        messages.erase(messages.begin());
    }
    if (!request.tools.empty()) {
        if (!system.empty())
            system += '\n';
        system += "List of tools: [";
        for (std::size_t index = 0; index < request.tools.size(); ++index) {
            if (index)
                system += ", ";
            const auto& tool = request.tools[index];
            auto parameters = Json::parse(tool.parameters_schema_json);
            if (tool.name.empty() || !parameters.is_object())
                throw std::invalid_argument(
                    "LFM2-MoE tool definition requires name and object schema");
            system += template_json(Json{{"name", tool.name},
                                         {"description", tool.description},
                                         {"parameters", parameters}});
        }
        system += "]";
    }
    std::string output = "<|startoftext|>";
    if (!system.empty())
        output += "<|im_start|>system\n" + system + "<|im_end|>\n";
    std::ptrdiff_t last_user = -1;
    for (std::size_t index = 0; index < messages.size(); ++index)
        if (messages[index].role == MessageRole::User)
            last_user = static_cast<std::ptrdiff_t>(index);
    for (std::size_t index = 0; index < messages.size(); ++index) {
        auto& message = messages[index];
        output += std::string("<|im_start|>") + role_name(message.role) + "\n";
        if (message.role == MessageRole::Assistant) {
            const bool keep = preserve_thinking || static_cast<std::ptrdiff_t>(index) > last_user;
            if (keep && !message.reasoning.empty())
                output += "<think>" + message.reasoning + "</think>";
            const auto end = message.content.rfind("</think>");
            if (!keep && end != std::string::npos)
                message.content = trim(message.content.substr(end + 8));
        }
        output += message.content;
        if (!message.calls.empty())
            output += render_calls(message.calls);
        output += "<|im_end|>\n";
    }
    return output + "<|im_start|>assistant\n";
}

internal::ConversationResult lfm2_moe_parse_conversation(const std::string& decoded,
                                                         bool starts_in_reasoning,
                                                         bool allow_json_tools) {
    internal::ConversationResult result;
    const std::string source = (starts_in_reasoning ? "<think>" : "") + decoded;
    const std::string tool_start = "<|tool_call_start|>", tool_end = "<|tool_call_end|>";
    std::size_t position = 0;
    while (position < source.size()) {
        if (source.compare(position, 7, "<think>") == 0) {
            const auto end = source.find("</think>", position + 7);
            result.parts.push_back(ReasoningPart{
                source.substr(position + 7, end == std::string::npos ? end : end - position - 7)});
            if (end == std::string::npos)
                break;
            position = end + 8;
        } else if (source.compare(position, tool_start.size(), tool_start) == 0) {
            const auto begin = position + tool_start.size();
            const auto end = source.find(tool_end, begin);
            for (auto& call :
                 parse_calls(source.substr(begin, end == std::string::npos ? end : end - begin),
                             end != std::string::npos))
                result.parts.push_back(std::move(call));
            if (end == std::string::npos)
                break;
            position = end + tool_end.size();
        } else {
            const auto next =
                std::min(source.find("<think>", position), source.find(tool_start, position));
            const auto text =
                source.substr(position, next == std::string::npos ? next : next - position);
            bool json_tools = false;
            if (allow_json_tools) {
                const auto json = Json::parse(text, nullptr, false);
                const auto first = json.is_array() && !json.empty() ? json.front() : json;
                json_tools =
                    first.is_object() && (first.contains("function") ||
                                          (first.contains("name") && first.contains("arguments")));
            }
            if (json_tools) {
                for (auto& call : parse_calls(text, true))
                    result.parts.push_back(std::move(call));
            } else if (!text.empty())
                result.parts.push_back(TextPart{text});
            if (next == std::string::npos)
                break;
            position = next;
        }
    }
    return result;
}
void lfm2_moe_assign_tool_call_ids(internal::ConversationResult& result,
                                   const internal::TextConversationRequest& request) {
    std::unordered_set<std::string> history;
    for (const auto& message : request.messages) {
        for (const auto& part : message.parts) {
            if (const auto* call = std::get_if<internal::ToolCallView>(&part)) {
                if (!call->call_id.empty())
                    history.emplace(call->call_id);
            } else if (const auto* response = std::get_if<internal::ToolResultView>(&part)) {
                if (!response->call_id.empty())
                    history.emplace(response->call_id);
            }
        }
    }
    auto reserved = history;
    std::unordered_map<std::string, std::size_t> counts;
    for (const auto& part : result.parts) {
        if (const auto* call = std::get_if<internal::ToolCall>(&part);
            call && !call->call_id.empty()) {
            ++counts[call->call_id];
            reserved.insert(call->call_id);
        }
    }
    std::size_t next = 0;
    for (auto& part : result.parts) {
        auto* call = std::get_if<internal::ToolCall>(&part);
        if (!call)
            continue;
        if (!call->call_id.empty() && counts.at(call->call_id) == 1 &&
            !history.count(call->call_id))
            continue;
        if (call->state != internal::ToolCallState::Complete) {
            // An absent/ambiguous ID on incomplete output is still unknown.
            call->call_id.clear();
            continue;
        }
        // Pythonic calls contain no IDs. These identifiers belong to the Task
        // transport and never alter the model's name, arguments, or token IDs.
        do {
            call->call_id = "call_" + std::to_string(next++);
        } while (reserved.count(call->call_id));
        reserved.insert(call->call_id);
    }
}
} // namespace trtmc
