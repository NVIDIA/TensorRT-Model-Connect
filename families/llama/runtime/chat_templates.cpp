/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/llama/runtime/chat_templates.h"

#include <ctime>
#include <iomanip>
#include <locale>
#include <sstream>
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

// Zephyr / TinyLlama share role tags with Phi but end turns with eos (</s>), not <|end|>.
std::string apply_zephyr(const std::string& prompt, bool /*enable_thinking*/) {
    return "<|user|>\n" + prompt + "</s>\n<|assistant|>\n";
}

std::string apply_gemma(const std::string& prompt, bool /*enable_thinking*/) {
    return "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";
}

std::string apply_llama3(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|begin_of_text|>";
    if (!enable_thinking)
        r += "<|start_header_id|>system<|end_header_id|>\n\ndetailed thinking off<|eot_id|>";
    r += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt +
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return r;
}

std::string trim_chat_content(const std::string& prompt) {
    // The checkpoint uses Jinja's trim filter (Python str.strip), including
    // Unicode spaces. Compare UTF-8 sequences without decoding model input.
    static constexpr std::string_view spaces[] = {
        " ",      "\t",     "\r",     "\n",     "\v",     "\f",     "\x1c",   "\x1d",
        "\x1e",   "\x1f",   "\u0085", "\u00a0", "\u1680", "\u2000", "\u2001", "\u2002",
        "\u2003", "\u2004", "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200a",
        "\u2028", "\u2029", "\u202f", "\u205f", "\u3000"};
    std::string_view content(prompt);
    for (auto size = content.size(); !content.empty();) {
        for (auto space : spaces) {
            if (content.substr(0, space.size()) == space)
                content.remove_prefix(space.size());
            if (content.size() >= space.size() &&
                content.substr(content.size() - space.size()) == space)
                content.remove_suffix(space.size());
        }
        if (size == content.size())
            break;
        size = content.size();
    }
    return std::string(content);
}

std::string local_chat_date() {
    const auto now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    if (localtime_s(&local, &now) != 0)
#else
    if (localtime_r(&now, &local) == nullptr)
#endif
        throw std::runtime_error("Cannot render the Llama chat template date");
    std::ostringstream date;
    date.imbue(std::locale::classic());
    date << std::put_time(&local, "%d %b %Y");
    return date.str();
}

std::string apply_dated_llama3(const std::string& prompt, const std::string& date) {
    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
           "Cutting Knowledge Date: December 2023\nToday Date: " +
           date + "\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
           trim_chat_content(prompt) +
           "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
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

std::string llama_detect_chat_template_format(const std::string& jinja_template) {
    if (jinja_template.empty())
        return {};
    if (jinja_template.find("<|im_start|>") != std::string::npos)
        return "chatml";
    if (jinja_template.find("[INST]") != std::string::npos)
        return "mistral";
    // Phi and Zephyr/TinyLlama both use <|user|>/<|assistant|>. Phi is distinguished by
    // literal <|end|> turn separators; Zephyr/TinyLlama concatenate eos_token (</s>) instead.
    if (jinja_template.find("<|user|>") != std::string::npos ||
        jinja_template.find("<|assistant|>") != std::string::npos) {
        if (jinja_template.find("<|end|>") != std::string::npos)
            return "phi";
        return "zephyr";
    }
    if (jinja_template.find("<start_of_turn>") != std::string::npos)
        return "gemma";
    if (jinja_template.find("<|start_header_id|>") != std::string::npos) {
        if (jinja_template.find("Cutting Knowledge Date: December 2023") != std::string::npos &&
            jinja_template.find("Today Date: ") != std::string::npos &&
            jinja_template.find("strftime_now(\"%d %b %Y\")") != std::string::npos)
            return "llama3_dated";
        if (jinja_template.find("Cutting Knowledge Date: December 2023") != std::string::npos &&
            jinja_template.find("Today Date: ") != std::string::npos &&
            jinja_template.find("set date_string = \"26 Jul 2024\"") != std::string::npos &&
            jinja_template.find("strftime_now") == std::string::npos)
            return "llama3_fixed_date";
        if (jinja_template.find("set loop_messages = messages") != std::string::npos &&
            jinja_template.find("message['content'] | trim") != std::string::npos &&
            jinja_template.find("detailed thinking") == std::string::npos &&
            jinja_template.find("system") == std::string::npos)
            return "llama3_user_header";
        return "llama3";
    }
    if (jinja_template.find("<extra_id_0>") != std::string::npos)
        return "nemotron";
    if (jinja_template.find("<SPECIAL_10>") != std::string::npos)
        return "nemotron_h";
    return {};
}

std::string llama_apply_chat_template(const std::string& format, const std::string& prompt,
                                      bool enable_thinking) {
    if (format.empty())
        return prompt;
    if (format == "chatml")
        return apply_chatml(prompt, enable_thinking);
    if (format == "mistral")
        return apply_mistral(prompt, enable_thinking);
    if (format == "phi")
        return apply_phi(prompt, enable_thinking);
    if (format == "zephyr")
        return apply_zephyr(prompt, enable_thinking);
    if (format == "gemma")
        return apply_gemma(prompt, enable_thinking);
    if (format == "llama3_dated")
        return apply_dated_llama3(prompt, local_chat_date());
    if (format == "llama3_fixed_date")
        return apply_dated_llama3(prompt, "26 Jul 2024");
    if (format == "llama3_user_header")
        return apply_llama3(trim_chat_content(prompt), true);
    if (format == "llama3")
        return apply_llama3(prompt, enable_thinking);
    if (format == "nemotron")
        return apply_nemotron(prompt, enable_thinking);
    if (format == "nemotron_h")
        return apply_nemotron_h(prompt, enable_thinking);
    return prompt;
}

} // namespace trtmc
