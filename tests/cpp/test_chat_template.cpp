// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CHAT-TPL-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-CHAT-01
// Intent:         Chat template detection and application for known formats
// Preconditions:  None (pure string logic, no GPU or TRT required)
// Postconditions: Detected formats match expected enum, applied templates
//                 produce correct formatted strings
// =============================================================================

#include "runtime/core/chat_template.h"

#include <cstdlib>
#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// --- Detection tests ---

static void test_detect_empty() {
    auto fmt = trtmc::detect_chat_template_format("");
    check(fmt == trtmc::ChatTemplateFormat::kNone, "empty -> kNone");
}

static void test_detect_chatml() {
    std::string tpl = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ "
                      "message.content }}<|im_end|>\n{% endfor %}";
    auto fmt = trtmc::detect_chat_template_format(tpl);
    check(fmt == trtmc::ChatTemplateFormat::kChatML, "chatml detection");
}

static void test_detect_internlm() {
    std::string tpl = "{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + "
                      "message['role'] + '\\n' + message['content'] + '<|im_end|>' + "
                      "'\\n'}}{% endfor %}{% if add_generation_prompt %}{{ "
                      "'<|im_start|>assistant\\n' }}{% endif %}";
    auto fmt = trtmc::detect_chat_template_format(tpl);
    check(fmt == trtmc::ChatTemplateFormat::kInternLM, "internlm detection");
}

static void test_detect_mistral() {
    std::string tpl = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' "
                      "%}[INST] {{ message['content'] }} [/INST]{% endif %}{% endfor %}";
    auto fmt = trtmc::detect_chat_template_format(tpl);
    check(fmt == trtmc::ChatTemplateFormat::kMistral, "mistral detection");
}

static void test_detect_phi() {
    std::string tpl = "{% for message in messages %}<|user|>\n{{ message.content "
                      "}}<|end|>\n<|assistant|>\n{% endfor %}";
    auto fmt = trtmc::detect_chat_template_format(tpl);
    check(fmt == trtmc::ChatTemplateFormat::kPhi, "phi detection");
}

static void test_detect_gemma() {
    std::string tpl = "{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ "
                      "message.content }}<end_of_turn>\n{% endfor %}";
    auto fmt = trtmc::detect_chat_template_format(tpl);
    check(fmt == trtmc::ChatTemplateFormat::kGemma, "gemma detection");
}

static void test_detect_llama3() {
    std::string tpl = "{% for message in messages %}<|start_header_id|>{{ message.role "
                      "}}<|end_header_id|>\n{{ message.content }}<|eot_id|>{% endfor %}";
    auto fmt = trtmc::detect_chat_template_format(tpl);
    check(fmt == trtmc::ChatTemplateFormat::kLlama3, "llama3 detection");
}

static void test_detect_nemotron_h() {
    std::string tpl = "{% if add_generation_prompt %}<SPECIAL_10>System\n"
                      "<SPECIAL_11>User\n{{ message.content }}\n"
                      "<SPECIAL_11>Assistant\n<think>{% endif %}";
    auto fmt = trtmc::detect_chat_template_format(tpl);
    check(fmt == trtmc::ChatTemplateFormat::kNemotronH, "nemotron-h detection");
}

static void test_detect_unknown() {
    auto fmt = trtmc::detect_chat_template_format("some random jinja template");
    check(fmt == trtmc::ChatTemplateFormat::kNone, "unknown -> kNone");
}

// --- Application tests ---

static void test_apply_none() {
    auto result = trtmc::apply_chat_template(trtmc::ChatTemplateFormat::kNone, "hello");
    check(result == "hello", "kNone passthrough");
}

static void test_apply_chatml() {
    auto result = trtmc::apply_chat_template(trtmc::ChatTemplateFormat::kChatML, "What is 2+2?");
    check(result == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
          "chatml application");
}

static void test_apply_internlm() {
    auto result = trtmc::apply_chat_template(trtmc::ChatTemplateFormat::kInternLM, "What is 2+2?");
    check(result == "<s><|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
          "internlm application");
}

static void test_apply_chatml_no_thinking() {
    auto result =
        trtmc::apply_chat_template(trtmc::ChatTemplateFormat::kChatML, "What is 2+2?", false);
    check(result == "<|im_start|>user\nWhat is "
                    "2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
          "chatml no-thinking application");
}

static void test_apply_mistral() {
    auto result = trtmc::apply_chat_template(trtmc::ChatTemplateFormat::kMistral, "hello");
    check(result == "[INST] hello [/INST]", "mistral application");
}

static void test_apply_phi() {
    auto result = trtmc::apply_chat_template(trtmc::ChatTemplateFormat::kPhi, "hello");
    check(result == "<|user|>\nhello<|end|>\n<|assistant|>\n", "phi application");
}

static void test_apply_gemma() {
    auto result = trtmc::apply_chat_template(trtmc::ChatTemplateFormat::kGemma, "hello");
    check(result == "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n",
          "gemma application");
}

static void test_apply_llama3() {
    auto result = trtmc::apply_chat_template(trtmc::ChatTemplateFormat::kLlama3, "hello");
    check(result == "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|"
                    "start_header_id|>assistant<|end_header_id|>\n\n",
          "llama3 application");
}

static void test_apply_nemotron_h_no_thinking() {
    auto result = trtmc::apply_chat_template(trtmc::ChatTemplateFormat::kNemotronH, "hello", false);
    check(result == "<SPECIAL_10>System\n\n<SPECIAL_11>User\nhello\n"
                    "<SPECIAL_11>Assistant\n<think></think>",
          "nemotron-h no-thinking application");
}

static void test_apply_mistral_no_thinking_ignored() {
    auto result = trtmc::apply_chat_template(trtmc::ChatTemplateFormat::kMistral, "hello", false);
    check(result == "[INST] hello [/INST]", "mistral no-thinking ignored");
}

int main() {
    // Detection
    test_detect_empty();
    test_detect_chatml();
    test_detect_internlm();
    test_detect_mistral();
    test_detect_phi();
    test_detect_gemma();
    test_detect_llama3();
    test_detect_nemotron_h();
    test_detect_unknown();

    // Application
    test_apply_none();
    test_apply_chatml();
    test_apply_internlm();
    test_apply_chatml_no_thinking();
    test_apply_mistral();
    test_apply_phi();
    test_apply_gemma();
    test_apply_llama3();
    test_apply_nemotron_h_no_thinking();
    test_apply_mistral_no_thinking_ignored();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All chat_template tests passed.\n";
    return 0;
}
