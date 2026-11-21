/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/chat_templates.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

// Minimal single-user ChatML templates. Expected prefixes match the pinned
// Instruct-2507 and hybrid Qwen3 checkpoint examples in issue #1615.
const std::string chatml =
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
const std::string hybrid =
    chatml + "{% if add_generation_prompt and enable_thinking is defined "
             "and enable_thinking is false %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}";
const std::string prefix = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n";
const std::string empty_think = "<think>\n\n</think>\n\n";
const std::string system_prefix = "<|im_start|>system\nS<|im_end|>\n";

void check_render(const std::string& source, bool thinking, const std::string& expected,
                  const char* name, const std::string& system_prompt = {}) {
    const auto format = trtmc::qwen_detect_chat_template_format(source);
    const auto rendered = trtmc::qwen_apply_chat_template(format, "Q", thinking, system_prompt);
    if (rendered != expected) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        ++failures;
    }
}

} // namespace

int main() {
    check_render(chatml, false, prefix, "non-thinking checkpoint with thinking disabled");
    check_render(chatml, true, prefix, "non-thinking checkpoint with thinking enabled");
    check_render(hybrid, false, prefix + empty_think, "hybrid checkpoint with thinking disabled");
    check_render(hybrid, true, prefix, "hybrid checkpoint with thinking enabled");
    check_render(chatml, false, system_prefix + prefix,
                 "non-thinking checkpoint with system prompt and thinking disabled", "S");
    check_render(chatml, true, system_prefix + prefix,
                 "non-thinking checkpoint with system prompt and thinking enabled", "S");
    check_render(hybrid, false, system_prefix + prefix + empty_think,
                 "hybrid checkpoint with system prompt and thinking disabled", "S");
    check_render(hybrid, true, system_prefix + prefix,
                 "hybrid checkpoint with system prompt and thinking enabled", "S");
    check_render(chatml + "{% if enable_thinking is false %}"
                          "{{ '<think>\n\n</think>\n\n' }}{% endif %}",
                 false, prefix + empty_think, "multiline hybrid suffix");
    check_render(chatml + "{# Previous messages may contain <think>\\n\\n</think>\\n\\n #}", false,
                 prefix, "think text without a thinking option");
    check_render(chatml + "{# enable_thinking is an unused option #}", false, prefix,
                 "thinking option without an empty block");
    check_render("", false, "Q", "no template preserves the prompt");
    check_render("{{ messages[0]['content'] }}", false, "Q",
                 "unsupported template preserves the prompt");

    const std::vector<std::pair<std::string, std::string>> history{
        {"system", "  system\n"},
        {"user", "Remember ORCHID."},
        {"assistant", "<think>\nreasoning\n</think>\n\nOK"},
        {"user", "What word?"}};
    const std::string history_prefix = "<|im_start|>system\n  system\n<|im_end|>\n"
                                       "<|im_start|>user\nRemember ORCHID.<|im_end|>\n";
    const std::string history_suffix =
        "<|im_end|>\n<|im_start|>user\nWhat word?<|im_end|>\n<|im_start|>assistant\n";
    if (trtmc::qwen_apply_conversation_template("chatml_thinking", history, false) !=
        history_prefix + "<|im_start|>assistant\nOK" + history_suffix + empty_think) {
        std::fprintf(
            stderr,
            "FAIL: hybrid conversation retains roles/history and removes prior reasoning\n");
        ++failures;
    }
    if (trtmc::qwen_apply_conversation_template("chatml", history, false) !=
        history_prefix + "<|im_start|>assistant\n<think>\nreasoning\n</think>\n\nOK" +
            history_suffix) {
        std::fprintf(stderr, "FAIL: ordinary ChatML preserves assistant content\n");
        ++failures;
    }
    for (const auto& invalid : std::vector<std::vector<std::pair<std::string, std::string>>>{
             {},
             {{"assistant", "unfinished"}},
             {{"tool", "unsupported"}, {"user", "Q"}},
             {{"user", "Q"}, {"system", "misplaced"}, {"user", "Q"}}}) {
        bool rejected = false;
        try {
            trtmc::qwen_apply_conversation_template("chatml_thinking", invalid, false);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) {
            std::fprintf(stderr, "FAIL: invalid conversation accepted\n");
            ++failures;
        }
    }

    if (failures == 0)
        std::fprintf(stderr, "All Qwen single-turn and conversation template cases passed.\n");
    return failures;
}
