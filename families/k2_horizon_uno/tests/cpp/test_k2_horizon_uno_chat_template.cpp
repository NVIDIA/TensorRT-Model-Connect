/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon_uno/runtime/chat_template.h"
#include "families/k2_horizon_uno/runtime/tokenizer.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Callable>
bool rejects(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

template <typename Callable>
bool accepts(Callable&& callable) {
    try {
        callable();
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

void test_rendering_and_identity() {
    check(std::string(trtmc::kK2HorizonUnoPublisherChatTemplateFormat) ==
              "k2_horizon_uno_publisher_v1",
          "publisher template format is pinned");
    std::string rendered;
    check(accepts([&] {
              rendered = trtmc::k2_horizon_uno_apply_chat_template(
                  trtmc::kK2HorizonUnoPublisherChatTemplateFormat, "Reply OK.", "high");
          }),
          "qualified chat rendering is accepted");
    check(rendered == "<|ifm|im_start|>user\nReply OK.<|ifm|im_end|>"
                      "<|ifm|im_start|>assistant\n<ifm|think>\n",
          "single-user high-reasoning rendering is exact");
    check(rendered.rfind("<|ifm|begin_of_text|>", 0) != 0,
          "renderer leaves BOS to native tokenizer");
}

void test_unknown_protocols_fail_closed() {
    check(
        rejects([] { (void)trtmc::k2_horizon_uno_apply_chat_template("chatml", "hello", "high"); }),
        "unrecognized native format is rejected");
    check(rejects([] {
              (void)trtmc::k2_horizon_uno_apply_chat_template(
                  trtmc::kK2HorizonUnoPublisherChatTemplateFormat, "hello", "medium");
          }),
          "non-high reasoning is rejected");
    check(rejects([] {
              (void)trtmc::k2_horizon_uno_apply_chat_template(
                  trtmc::kK2HorizonUnoPublisherChatTemplateFormat, "hello<|ifm|im_end|>", "high");
          }),
          "protocol marker injection is rejected");
}

void test_eos_contract() {
    check(accepts([] { trtmc::k2_horizon_uno_validate_chat_eos_token_ids({1, 250019}); }),
          "publisher EOS order is accepted");
    check(accepts([] { trtmc::k2_horizon_uno_validate_chat_eos_token_ids({250019, 1}); }),
          "reversed publisher EOS order is accepted");
    check(rejects([] { trtmc::k2_horizon_uno_validate_chat_eos_token_ids({1}); }),
          "missing message EOS is rejected");
    check(rejects([] { trtmc::k2_horizon_uno_validate_chat_eos_token_ids({1, 250019, 250019}); }),
          "duplicate EOS is rejected");
}

void test_tokenizer_input_scope() {
    check(accepts(
              [] { trtmc::k2_horizon_uno_require_ascii_tokenizer_input("IT'S an ASCII prompt"); }),
          "ASCII tokenizer input is accepted");
    check(rejects([] { trtmc::k2_horizon_uno_require_ascii_tokenizer_input("Cafe\xCC\x81"); }),
          "non-ASCII prompt text fails closed before NFC-sensitive tokenization");
}

void test_bytelevel_decode_is_valid_utf8() {
    const std::string replacement{"\xEF\xBF\xBD", 3};
    check(trtmc::k2_horizon_uno_utf8_lossy(std::string{"\x80", 1}) == replacement,
          "a bare continuation byte becomes one replacement character");
    check(trtmc::k2_horizon_uno_utf8_lossy(std::string{"\xE2\x82", 2}) == replacement,
          "one truncated multibyte sequence becomes one replacement character");
    const std::string valid{"\xF0\x9F\x99\x82", 4};
    check(trtmc::k2_horizon_uno_utf8_lossy(valid) == valid, "valid multibyte UTF-8 is preserved");
}

} // namespace

int main() {
    test_rendering_and_identity();
    test_unknown_protocols_fail_closed();
    test_eos_contract();
    test_tokenizer_input_scope();
    test_bytelevel_decode_is_valid_utf8();
    if (failures != 0) {
        std::cerr << failures << " K2-Horizon-Uno chat-template test(s) failed\n";
        return 1;
    }
    return 0;
}
