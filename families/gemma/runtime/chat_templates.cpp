/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gemma/runtime/chat_templates.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace trtmc {
namespace {

std::string apply_chatml(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (!enable_thinking)
        r += "<think>\n\n</think>\n\n";
    return r;
}

std::string apply_mistral(const std::string& prompt, bool /*enable_thinking*/) {
    return "[INST] " + prompt + " [/INST]";
}

std::string apply_phi(const std::string& prompt, bool /*enable_thinking*/) {
    return "<|user|>\n" + prompt + "<|end|>\n<|assistant|>\n";
}

std::string apply_gemma(const std::string& prompt, bool /*enable_thinking*/) {
    return "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";
}

// Match the pinned provider template's Jinja/Python Unicode trim semantics.
std::string trim_chat_text(const std::string& text) {
    static constexpr std::string_view whitespace[]{
        "\x09",         "\x0a",         "\x0b",         "\x0c",         "\x0d",
        "\x1c",         "\x1d",         "\x1e",         "\x1f",         "\x20",
        "\xc2\x85",     "\xc2\xa0",     "\xe1\x9a\x80", "\xe2\x80\x80", "\xe2\x80\x81",
        "\xe2\x80\x82", "\xe2\x80\x83", "\xe2\x80\x84", "\xe2\x80\x85", "\xe2\x80\x86",
        "\xe2\x80\x87", "\xe2\x80\x88", "\xe2\x80\x89", "\xe2\x80\x8a", "\xe2\x80\xa8",
        "\xe2\x80\xa9", "\xe2\x80\xaf", "\xe2\x81\x9f", "\xe3\x80\x80"};
    std::size_t first = 0, last = text.size();
    bool trimmed = true;
    while (trimmed && first < last) {
        trimmed = false;
        for (const auto space : whitespace) {
            if (space.size() <= last - first && text.compare(first, space.size(), space) == 0) {
                first += space.size();
                trimmed = true;
                break;
            }
        }
    }
    trimmed = true;
    while (trimmed && first < last) {
        trimmed = false;
        for (const auto space : whitespace) {
            if (space.size() <= last - first &&
                text.compare(last - space.size(), space.size(), space) == 0) {
                last -= space.size();
                trimmed = true;
                break;
            }
        }
    }
    return text.substr(first, last - first);
}

std::string apply_gemma4(const std::string& prompt, bool enable_thinking,
                         const std::string& system_prompt) {
    std::string result = "<bos>";
    if (enable_thinking || !system_prompt.empty()) {
        result += "<|turn>system\n";
        if (enable_thinking)
            result += "<|think|>\n";
        result += trim_chat_text(system_prompt) + "<turn|>\n";
    }
    return result + "<|turn>user\n" + trim_chat_text(prompt) + "<turn|>\n<|turn>model\n";
}

std::string apply_llama3(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|begin_of_text|>";
    if (!enable_thinking)
        r += "<|start_header_id|>system<|end_header_id|>\n\ndetailed thinking off<|eot_id|>";
    r += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt +
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return r;
}

std::string apply_nemotron(const std::string& prompt, bool /*enable_thinking*/) {
    return "<extra_id_0>System\n\n<extra_id_1>User\n" + prompt + "\n<extra_id_1>Assistant\n";
}

std::string apply_nemotron_h(const std::string& prompt, bool enable_thinking) {
    std::string r =
        "<SPECIAL_10>System\n\n<SPECIAL_11>User\n" + prompt + "\n<SPECIAL_11>Assistant\n";
    r += enable_thinking ? "<think>\n" : "<think></think>";
    return r;
}

} // namespace

std::string gemma4_generation_suffix(const std::string& format, bool enable_thinking) {
    return format == "gemma4_closed_thought" && !enable_thinking ? "<|channel>thought\n<channel|>"
                                                                 : "";
}

std::string gemma_trim_chat_text(const std::string& text) {
    return trim_chat_text(text);
}

std::string gemma_detect_chat_template_format(const std::string& jinja_template) {
    if (jinja_template.empty())
        return {};
    if (jinja_template.find("<|im_start|>") != std::string::npos)
        return "chatml";
    if (jinja_template.find("[INST]") != std::string::npos)
        return "mistral";
    if (jinja_template.find("<|user|>") != std::string::npos ||
        jinja_template.find("<|assistant|>") != std::string::npos)
        return "phi";
    if (jinja_template.find("<|turn>") != std::string::npos) {
        const auto generation = jinja_template.rfind("if add_generation_prompt");
        const auto suffix =
            generation == std::string::npos ? jinja_template : jinja_template.substr(generation);
        if (suffix.find("<|channel>thought\\n<channel|>") != std::string::npos ||
            suffix.find("<|channel>thought\n<channel|>") != std::string::npos)
            return "gemma4_closed_thought";
        return "gemma4";
    }
    if (jinja_template.find("<start_of_turn>") != std::string::npos)
        return "gemma";
    if (jinja_template.find("<|start_header_id|>") != std::string::npos)
        return "llama3";
    if (jinja_template.find("<extra_id_0>") != std::string::npos)
        return "nemotron";
    if (jinja_template.find("<SPECIAL_10>") != std::string::npos)
        return "nemotron_h";
    return {};
}

std::string gemma_apply_chat_template(const std::string& format, const std::string& prompt,
                                      bool enable_thinking, const std::string& system_prompt) {
    if (!system_prompt.empty() && format != "gemma4" && format != "gemma4_closed_thought")
        throw std::invalid_argument("Gemma native system prompts require the Gemma4 template");
    if (format.empty())
        return prompt;
    if (format == "chatml")
        return apply_chatml(prompt, enable_thinking);
    if (format == "mistral")
        return apply_mistral(prompt, enable_thinking);
    if (format == "phi")
        return apply_phi(prompt, enable_thinking);
    if (format == "gemma4" || format == "gemma4_closed_thought")
        return apply_gemma4(prompt, enable_thinking, system_prompt) +
               gemma4_generation_suffix(format, enable_thinking);
    if (format == "gemma")
        return apply_gemma(prompt, enable_thinking);
    if (format == "llama3")
        return apply_llama3(prompt, enable_thinking);
    if (format == "nemotron")
        return apply_nemotron(prompt, enable_thinking);
    if (format == "nemotron_h")
        return apply_nemotron_h(prompt, enable_thinking);
    return prompt;
}

} // namespace trtmc
