/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Model-owned chat-template coverage for smollm3 decoder formats.

#include "runtime/models/smollm3/chat_templates.h"

#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_detect_nemotron_h() {
    std::string tpl = "{% if add_generation_prompt %}<SPECIAL_10>System\n"
                      "<SPECIAL_11>User\n{{ message.content }}\n"
                      "<SPECIAL_11>Assistant\n<think>{% endif %}";
    auto fmt = trtmc::smollm3_detect_chat_template_format(tpl);
    check(fmt == "nemotron_h", "nemotron-h detection");
}

static void test_detect_chatml() {
    std::string tpl = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ "
                      "message.content }}<|im_end|>\n{% endfor %}";
    auto fmt = trtmc::smollm3_detect_chat_template_format(tpl);
    check(fmt == "chatml", "chatml detection");
}

static void test_detect_mistral() {
    std::string tpl = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' "
                      "%}[INST] {{ message['content'] }} [/INST]{% endif %}{% endfor %}";
    auto fmt = trtmc::smollm3_detect_chat_template_format(tpl);
    check(fmt == "mistral", "mistral detection");
}

static void test_detect_phi() {
    std::string tpl = "{% for message in messages %}<|user|>\n{{ message.content "
                      "}}<|end|>\n<|assistant|>\n{% endfor %}";
    auto fmt = trtmc::smollm3_detect_chat_template_format(tpl);
    check(fmt == "phi", "phi detection");
}

static void test_detect_gemma() {
    std::string tpl = "{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ "
                      "message.content }}<end_of_turn>\n{% endfor %}";
    auto fmt = trtmc::smollm3_detect_chat_template_format(tpl);
    check(fmt == "gemma", "gemma detection");
}

static void test_detect_llama3() {
    std::string tpl = "{% for message in messages %}<|start_header_id|>{{ message.role "
                      "}}<|end_header_id|>\n{{ message.content }}<|eot_id|>{% endfor %}";
    auto fmt = trtmc::smollm3_detect_chat_template_format(tpl);
    check(fmt == "llama3", "llama3 detection");
}

static void test_apply_nemotron_h_no_thinking() {
    auto result = trtmc::smollm3_apply_chat_template("nemotron_h", "hello", false);
    check(result == "<SPECIAL_10>System\n\n<SPECIAL_11>User\nhello\n"
                    "<SPECIAL_11>Assistant\n<think></think>",
          "nemotron-h no-thinking application");
}

static void test_apply_chatml_no_thinking() {
    auto result = trtmc::smollm3_apply_chat_template("chatml", "What is 2+2?", false);
    check(result == "<|im_start|>user\nWhat is "
                    "2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
          "chatml no-thinking application");
}

static void test_apply_mistral_no_thinking_ignored() {
    auto result = trtmc::smollm3_apply_chat_template("mistral", "hello", false);
    check(result == "[INST] hello [/INST]", "mistral no-thinking ignored");
}

static void test_apply_phi() {
    auto result = trtmc::smollm3_apply_chat_template("phi", "hello");
    check(result == "<|user|>\nhello<|end|>\n<|assistant|>\n", "phi application");
}

static void test_apply_gemma() {
    auto result = trtmc::smollm3_apply_chat_template("gemma", "hello");
    check(result == "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n",
          "gemma application");
}

static void test_apply_llama3() {
    auto result = trtmc::smollm3_apply_chat_template("llama3", "hello");
    check(result == "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|"
                    "start_header_id|>assistant<|end_header_id|>\n\n",
          "llama3 application");
}


// ── SmolLM3's own chat template ─────────────────────────────────────────────
// The expected strings below are the verbatim output of the upstream
// tokenizer's apply_chat_template() for HuggingFaceTB/SmolLM3-3B at revision
// a07cc9a0, captured with the date pinned. Asserting against the real
// rendering, rather than a reading of the Jinja source, is what makes this a
// contract: SmolLM3 always emits a system block, and it does *not* close that
// block with <|im_end|> before the user turn.

static const char* kSmolLM3Date = "04 September 2026";

static void test_detect_smollm3_over_chatml() {
    // SmolLM3's template is ChatML-framed, so generic ChatML detection must not
    // claim it -- the SmolLM3 branch carries the mandatory system block.
    std::string tpl =
        "{%- if enable_thinking %}{%- set reasoning_mode = \"/think\" %}{%- endif %}"
        "{{- \"<|im_start|>system\\n\" -}}"
        "{{- \"Reasoning Mode: \" + reasoning_mode + \"\\n\\n\" -}}";
    check(trtmc::smollm3_detect_chat_template_format(tpl) == "smollm3",
          "smollm3 detection wins over chatml");
}

static void test_apply_smollm3_no_thinking() {
    const std::string expected =
        "<|im_start|>system\n"
        "## Metadata\n"
        "\n"
        "Knowledge Cutoff Date: June 2025\n"
        "Today Date: 04 September 2026\n"
        "Reasoning Mode: /no_think\n"
        "\n"
        "## Custom Instructions\n"
        "\n"
        "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n"
        "\n"
        "<|im_start|>user\n"
        "What is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "\n"
        "</think>\n";
    auto result = trtmc::smollm3_apply_chat_template(
        "smollm3", "What is 2+2?", false, kSmolLM3Date);
    check(result == expected, "smollm3 /no_think matches upstream byte for byte");
    check(result.size() == 297, "smollm3 /no_think length matches upstream (297)");
}

static void test_apply_smollm3_thinking() {
    const std::string expected =
        "<|im_start|>system\n"
        "## Metadata\n"
        "\n"
        "Knowledge Cutoff Date: June 2025\n"
        "Today Date: 04 September 2026\n"
        "Reasoning Mode: /think\n"
        "\n"
        "## Custom Instructions\n"
        "\n"
        "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.\n"
        "\n"
        "<|im_start|>user\n"
        "What is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n";
    auto result = trtmc::smollm3_apply_chat_template(
        "smollm3", "What is 2+2?", true, kSmolLM3Date);
    check(result == expected, "smollm3 /think matches upstream byte for byte");
    check(result.size() == 1369, "smollm3 /think length matches upstream (1369)");
}

int main() {

    test_detect_smollm3_over_chatml();
    test_detect_chatml();
    test_detect_mistral();
    test_detect_phi();
    test_detect_gemma();
    test_detect_llama3();
    test_detect_nemotron_h();
    test_apply_smollm3_no_thinking();
    test_apply_smollm3_thinking();
    test_apply_chatml_no_thinking();
    test_apply_mistral_no_thinking_ignored();
    test_apply_phi();
    test_apply_gemma();
    test_apply_llama3();
    test_apply_nemotron_h_no_thinking();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All smollm3 chat_template tests passed.\n";
    return 0;
}
