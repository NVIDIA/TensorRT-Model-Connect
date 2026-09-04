/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/smollm3/chat_templates.h"

#include <cstdio>
#include <ctime>
#include <string>

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

// SmolLM3 always emits a system block, even when the caller supplies no system
// message. The two Custom Instruction bodies below are the upstream defaults for
// the two reasoning modes; reproducing them verbatim is what keeps the served
// prompt identical to the Hugging Face chat template.
constexpr char kSmollm3ThinkInstructions[] =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an "
    "assistant involves thoroughly exploring questions through a systematic thinking process "
    "before providing the final precise and accurate solutions. This requires engaging in a "
    "comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, "
    "backtracking, and iteration to develop well-considered thinking process. Please structure "
    "your response into two main sections: Thought and Solution using the specified format: "
    "<think> Thought section </think> Solution section. In the Thought section, detail your "
    "reasoning process in steps. Each step should include detailed considerations such as "
    "analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the "
    "accuracy of the current steps, refining any errors, and revisiting previous steps. In the "
    "Solution section, based on various attempts, explorations, and reflections from the Thought "
    "section, systematically present the final solution that you deem correct. The Solution "
    "section should be logical, accurate, and concise and detail necessary steps needed to reach "
    "the conclusion.";
constexpr char kSmollm3NoThinkInstructions[] =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face.";

std::string smollm3_today() {
    return smollm3_format_date(std::time(nullptr));
}

std::string apply_smollm3(const std::string& prompt, bool enable_thinking,
                          const std::string& today) {
    const std::string mode = enable_thinking ? "/think" : "/no_think";
    std::string r = "<|im_start|>system\n";
    r += "## Metadata\n\n";
    r += "Knowledge Cutoff Date: June 2025\n";
    r += "Today Date: " + (today.empty() ? smollm3_today() : today) + "\n";
    r += "Reasoning Mode: " + mode + "\n\n";
    r += "## Custom Instructions\n\n";
    r += enable_thinking ? kSmollm3ThinkInstructions : kSmollm3NoThinkInstructions;
    r += "\n\n";
    // Upstream does NOT close the system block with <|im_end|>: the instructions
    // run straight into the user turn. Verified against apply_chat_template.
    r += "<|im_start|>user\n" + prompt + "<|im_end|>\n";
    r += "<|im_start|>assistant\n";
    if (!enable_thinking)
        r += "<think>\n\n</think>\n";
    return r;
}

} // namespace

std::string smollm3_format_date(std::time_t when) {
    // Upstream renders strftime_now("%d %B %Y"), e.g. "04 September 2026". The
    // month name comes from this table rather than from strftime's %B, which
    // follows LC_TIME and yields a translated name under a non-English locale.
    // The served prompt would then stop matching the reference for those users.
    static constexpr const char* kMonths[] = {"January",   "February", "March",    "April",
                                              "May",       "June",     "July",     "August",
                                              "September", "October",  "November", "December"};

    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &when);
#else
    gmtime_r(&when, &tm_utc);
#endif
    if (tm_utc.tm_mon < 0 || tm_utc.tm_mon > 11)
        return {};

    char buf[32];
    const int written = std::snprintf(buf, sizeof(buf), "%02d %s %d", tm_utc.tm_mday,
                                      kMonths[tm_utc.tm_mon], tm_utc.tm_year + 1900);
    if (written <= 0 || static_cast<std::size_t>(written) >= sizeof(buf))
        return {};
    return std::string(buf);
}

std::string smollm3_detect_chat_template_format(const std::string& jinja_template) {
    // Ordered marker table: the first row whose markers are all present wins.
    // SmolLM3's template is ChatML-framed, so it is matched on its mandatory
    // system block before the generic ChatML row below. A row's second marker
    // is either an additional requirement (`require_both`) or an alternative.
    struct Rule {
        const char* first;
        const char* second;
        bool require_both;
        const char* format;
    };
    static constexpr Rule kRules[] = {
        {"<|im_start|>", "Reasoning Mode:", true, "smollm3"},
        {"<|im_start|>", nullptr, false, "chatml"},
        {"[INST]", nullptr, false, "mistral"},
        {"<|user|>", "<|assistant|>", false, "phi"},
        {"<start_of_turn>", nullptr, false, "gemma"},
        {"<|start_header_id|>", nullptr, false, "llama3"},
        {"<extra_id_0>", nullptr, false, "nemotron"},
        {"<SPECIAL_10>", nullptr, false, "nemotron_h"},
    };

    for (const Rule& rule : kRules) {
        const bool has_first = jinja_template.find(rule.first) != std::string::npos;
        const bool has_second =
            rule.second != nullptr && jinja_template.find(rule.second) != std::string::npos;
        const bool matched =
            rule.require_both ? (has_first && has_second) : (has_first || has_second);
        if (matched)
            return rule.format;
    }
    return {};
}

std::string smollm3_apply_chat_template(const std::string& format, const std::string& prompt,
                                        bool enable_thinking, const std::string& today) {
    if (format.empty())
        return prompt;
    if (format == "smollm3")
        return apply_smollm3(prompt, enable_thinking, today);
    if (format == "chatml")
        return apply_chatml(prompt, enable_thinking);
    if (format == "mistral")
        return apply_mistral(prompt, enable_thinking);
    if (format == "phi")
        return apply_phi(prompt, enable_thinking);
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
