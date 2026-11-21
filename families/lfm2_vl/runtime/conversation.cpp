/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_vl/runtime/conversation.h"

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

internal::ConversationResult lfm2_vl_parse_conversation(const std::string& decoded,
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
void lfm2_vl_assign_tool_call_ids(internal::ConversationResult& result,
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
