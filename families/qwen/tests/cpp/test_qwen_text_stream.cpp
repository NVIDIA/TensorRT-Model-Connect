/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen/runtime/chat_templates.h"
#include "families/qwen/runtime/text_stream.h"

#include <future>
#include <iostream>
#include <stdexcept>

using namespace trtmc;
using namespace trtmc::internal;
void require(bool ok) {
    if (!ok)
        throw std::runtime_error("Qwen stream contract failed");
}

int main() {
    try {
        require(qwen_utf8_text("a\xe4", false) == "a");
        require(qwen_utf8_text("a\xe4\xb8", false) == "a");
        require(qwen_utf8_text("a\xe4\xb8\xad", false) == "a中");
        require(qwen_utf8_text("\xf0\x9f\x98\x80", false) == "😀");
        require(qwen_utf8_text("\xed\xa0\x80", true) == "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd");
        require(qwen_utf8_text("\xe4", true) == "\xef\xbf\xbd");
        require(qwen_utf8_text("\xe4\xb8", true) == "\xef\xbf\xbd");
        require(qwen_utf8_text("\xe4!", false) == "\xef\xbf\xbd!");
        require(qwen_apply_chat_template("chatml", "user", false, "system")
                    .find("<|im_start|>system\nsystem<|im_end|>\n<|im_start|>user\nuser") == 0);
        std::promise<void> release;
        auto barrier = release.get_future().share();
        std::atomic<bool> complete{false};
        QwenTextStream stream([&](const auto& emit, const auto&) {
            emit({StreamEventKind::Delta, "中", {10, 11}, {}});
            barrier.wait();
            emit({StreamEventKind::Delta, "!", {12}, {}});
            complete = true;
            return TextResult{"中!", {10, 11, 12}};
        });
        auto first = stream.next(-1);
        require(first && first->kind == StreamEventKind::Delta && first->text_delta == "中");
        require(!complete); // first delta is observable before producer completion
        require(!stream.next(0));
        release.set_value();
        auto second = stream.next(-1);
        require(second && second->text_delta == "!");
        auto final = stream.next(-1);
        require(final && final->kind == StreamEventKind::Complete &&
                final->final_result->text == "中!");
        require(final->final_result->token_ids == std::vector<int32_t>({10, 11, 12}));
        try {
            stream.next(0);
            require(false);
        } catch (const std::logic_error&) {
        }

        // A producer filling its bounded queue must stop when nobody consumes.
        std::promise<void> filled;
        QwenTextStream slow([&](const auto& emit, const auto&) {
            for (int i = 0; i < 100; ++i) {
                if (i == 8)
                    filled.set_value();
                if (!emit({StreamEventKind::Delta, "x", {i}, {}}))
                    break;
            }
            return TextResult{};
        });
        filled.get_future().wait();
        slow.cancel();
        slow.cancel();
        require(slow.next(-1)->kind == StreamEventKind::Cancelled);

        QwenTextStream failed([](const auto&, const auto&) -> TextResult {
            throw std::runtime_error("producer failed");
        });
        try {
            failed.next(-1);
            require(false);
        } catch (const std::runtime_error& error) {
            require(std::string(error.what()) == "producer failed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
