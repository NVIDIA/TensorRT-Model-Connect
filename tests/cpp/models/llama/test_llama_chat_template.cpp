// Model-owned chat-template coverage for llama decoder formats.

#include "runtime/models/llama/chat_templates.h"

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
    auto fmt = trtmc::llama_detect_chat_template_format(tpl);
    check(fmt == "nemotron_h", "nemotron-h detection");
}

static void test_detect_chatml() {
    std::string tpl = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ "
                      "message.content }}<|im_end|>\n{% endfor %}";
    auto fmt = trtmc::llama_detect_chat_template_format(tpl);
    check(fmt == "chatml", "chatml detection");
}

static void test_detect_mistral() {
    std::string tpl = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' "
                      "%}[INST] {{ message['content'] }} [/INST]{% endif %}{% endfor %}";
    auto fmt = trtmc::llama_detect_chat_template_format(tpl);
    check(fmt == "mistral", "mistral detection");
}

static void test_detect_phi() {
    std::string tpl = "{% for message in messages %}<|user|>\n{{ message.content "
                      "}}<|end|>\n<|assistant|>\n{% endfor %}";
    auto fmt = trtmc::llama_detect_chat_template_format(tpl);
    check(fmt == "phi", "phi detection");
}

static void test_detect_gemma() {
    std::string tpl = "{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ "
                      "message.content }}<end_of_turn>\n{% endfor %}";
    auto fmt = trtmc::llama_detect_chat_template_format(tpl);
    check(fmt == "gemma", "gemma detection");
}

static void test_detect_llama3() {
    std::string tpl = "{% for message in messages %}<|start_header_id|>{{ message.role "
                      "}}<|end_header_id|>\n{{ message.content }}<|eot_id|>{% endfor %}";
    auto fmt = trtmc::llama_detect_chat_template_format(tpl);
    check(fmt == "llama3", "llama3 detection");
}

static void test_apply_nemotron_h_no_thinking() {
    auto result = trtmc::llama_apply_chat_template("nemotron_h", "hello", false);
    check(result == "<SPECIAL_10>System\n\n<SPECIAL_11>User\nhello\n"
                    "<SPECIAL_11>Assistant\n<think></think>",
          "nemotron-h no-thinking application");
}

static void test_apply_chatml_no_thinking() {
    auto result = trtmc::llama_apply_chat_template("chatml", "What is 2+2?", false);
    check(result == "<|im_start|>user\nWhat is "
                    "2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
          "chatml no-thinking application");
}

static void test_apply_mistral_no_thinking_ignored() {
    auto result = trtmc::llama_apply_chat_template("mistral", "hello", false);
    check(result == "[INST] hello [/INST]", "mistral no-thinking ignored");
}

static void test_apply_phi() {
    auto result = trtmc::llama_apply_chat_template("phi", "hello");
    check(result == "<|user|>\nhello<|end|>\n<|assistant|>\n", "phi application");
}

static void test_apply_gemma() {
    auto result = trtmc::llama_apply_chat_template("gemma", "hello");
    check(result == "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n",
          "gemma application");
}

static void test_apply_llama3() {
    auto result = trtmc::llama_apply_chat_template("llama3", "hello");
    check(result == "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|"
                    "start_header_id|>assistant<|end_header_id|>\n\n",
          "llama3 application");
}

int main() {

    test_detect_chatml();
    test_detect_mistral();
    test_detect_phi();
    test_detect_gemma();
    test_detect_llama3();
    test_detect_nemotron_h();
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
    std::cerr << "All llama chat_template tests passed.\n";
    return 0;
}
