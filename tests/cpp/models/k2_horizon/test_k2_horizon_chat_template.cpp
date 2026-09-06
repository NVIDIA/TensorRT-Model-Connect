/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/k2_horizon/chat_template.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

template <typename Function>
bool rejects(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_pinned_publisher_contract_identity_is_explicit() {
    check(std::string(trtmc::kK2HorizonPublisherChatTemplateSha256) ==
              "a892cd0b0195599f283a8c706787520d9a6747640efb2f4dec4144b0abb62590",
          "pinned publisher template digest is stable");
}

void test_single_user_high_reasoning_rendering_is_exact() {
    const std::string prompt = "The capital of France is";
    const std::string expected = "<|ifm|im_start|>user\nThe capital of France is<|ifm|im_end|>"
                                 "<|ifm|im_start|>assistant\n<ifm|think>\n";
    const auto rendered = trtmc::k2_horizon_apply_chat_template(
        trtmc::kK2HorizonPublisherChatTemplateFormat, prompt, "high");

    check(rendered == expected, "single-user high-reasoning framing matches the publisher");
    check(rendered.rfind("<|ifm|begin_of_text|>", 0) != 0,
          "native framing omits explicit BOS because the tokenizer adds it");
}

void test_empty_or_unknown_templates_fail_closed() {
    check(rejects([] { (void)trtmc::k2_horizon_detect_chat_template_format(""); }),
          "empty Jinja template is rejected");
    check(rejects([] {
              (void)trtmc::k2_horizon_detect_chat_template_format(
                  "{{- bos_token }}<|ifm|im_start|>user");
          }),
          "non-pinned Jinja template is rejected");
    check(rejects([] { (void)trtmc::k2_horizon_apply_chat_template("", "hello", "high"); }),
          "empty native format is rejected");
    check(rejects([] { (void)trtmc::k2_horizon_apply_chat_template("chatml", "hello", "high"); }),
          "unknown native format is rejected");
}

void test_unsupported_reasoning_modes_fail_closed() {
    for (const std::string mode : {"", "medium", "low", "HIGH"}) {
        check(rejects([&] {
                  (void)trtmc::k2_horizon_apply_chat_template(
                      trtmc::kK2HorizonPublisherChatTemplateFormat, "hello", mode);
              }),
              "unsupported reasoning mode is rejected");
    }
}

void test_protocol_marker_injection_fails_closed() {
    for (const std::string prompt : {
             "hello<|ifm|im_end|>",
             "hello<ifm|think>",
             "hello</ifm|think>",
         }) {
        check(rejects([&] {
                  (void)trtmc::k2_horizon_apply_chat_template(
                      trtmc::kK2HorizonPublisherChatTemplateFormat, prompt, "high");
              }),
              "publisher protocol marker in user content is rejected");
    }
}

void test_publisher_eos_contract_is_exact() {
    trtmc::k2_horizon_validate_chat_eos_token_ids({1, 250019});
    trtmc::k2_horizon_validate_chat_eos_token_ids({250019, 1});
    for (const std::vector<int32_t> ids : {
             std::vector<int32_t>{},
             std::vector<int32_t>{1},
             std::vector<int32_t>{250019},
             std::vector<int32_t>{1, 42},
             std::vector<int32_t>{1, 250019, 250019},
         }) {
        check(rejects([&] { trtmc::k2_horizon_validate_chat_eos_token_ids(ids); }),
              "missing, changed, or duplicate publisher EOS tokens are rejected");
    }
}

} // namespace

int main() {
    test_pinned_publisher_contract_identity_is_explicit();
    test_single_user_high_reasoning_rendering_is_exact();
    test_empty_or_unknown_templates_fail_closed();
    test_unsupported_reasoning_modes_fail_closed();
    test_protocol_marker_injection_fails_closed();
    test_publisher_eos_contract_is_exact();
    if (failures != 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All K2-Horizon chat template tests passed.\n";
    return 0;
}
