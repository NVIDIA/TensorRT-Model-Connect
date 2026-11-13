/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Exact-bundle GPU validation through the public Task SDK. This complements
// the family's existing native/reference correctness tests; it does not set
// or replace their numerical acceptance criteria.
#include "trtmc/core.hpp"
#include "trtmc/stream.hpp"
#include "trtmc/text.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: qwen_text_stream_consumer BUNDLE RUNTIME_ROOT\n";
        return 2;
    }
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        const auto model = trtmc::Model::load(argv[1], options);
        require(model.info().family == "qwen", "expected the Qwen owning family");
        require(model.supports<trtmc::StreamingTextContinuation>(),
                "bundle has no single-process streaming task");
        const auto text = model.task<trtmc::TextContinuation>();
        const auto streaming = model.task<trtmc::StreamingTextContinuation>();
        for (const bool chat : {false, true}) {
            const std::string prompt = "Continue this sentence in Chinese: 春天的花园里";
            trtmc::Config config{{"max_new_tokens", std::int64_t{16}}, {"temperature", 0.0},
                                 {"top_k", std::int64_t{1}},           {"seed", std::int64_t{0}},
                                 {"use_chat_template", chat},          {"enable_thinking", false}};
            if (chat)
                config.add("system_prompt", std::string("Answer briefly."));
            const trtmc::TextContinuationRequest request{prompt};
            const auto expected = text.run(request, config);
            auto stream = streaming.start(request, config);
            std::string joined;
            std::vector<std::int32_t> tokens;
            bool complete = false;
            while (auto event = stream.next()) {
                if (event->kind() == trtmc::StreamEventKind::Delta) {
                    joined += event->text_delta();
                    const auto delta = event->token_ids();
                    tokens.insert(tokens.end(), delta.begin(), delta.end());
                } else {
                    require(event->kind() == trtmc::StreamEventKind::Complete,
                            "unexpected cancellation");
                    const auto final = event->final_result();
                    require(final.has_value() && final->text == expected.text(),
                            "stream final text differs from run");
                    require(std::vector<std::int32_t>(final->token_ids.begin(),
                                                      final->token_ids.end()) ==
                                std::vector<std::int32_t>(expected.token_ids().begin(),
                                                          expected.token_ids().end()),
                            "stream final token IDs differ from run");
                    complete = true;
                    break;
                }
            }
            require(complete && joined == expected.text(),
                    "deltas do not reconstruct generated text");
            require(tokens == std::vector<std::int32_t>(expected.token_ids().begin(),
                                                        expected.token_ids().end()),
                    "deltas do not reconstruct generated token IDs");
            // Confirm cancellation, then reuse this same model for a full call.
            auto cancelled = streaming.start(request, config);
            auto first = cancelled.next();
            if (first && first->kind() == trtmc::StreamEventKind::Delta) {
                cancelled.cancel();
                const auto terminal = cancelled.next();
                require(terminal && terminal->kind() == trtmc::StreamEventKind::Cancelled,
                        "native cancellation was not confirmed");
            }
            cancelled.close();
            const auto after = text.run(request, config);
            require(after.text() == expected.text() &&
                        std::vector<std::int32_t>(after.token_ids().begin(),
                                                  after.token_ids().end()) == tokens,
                    "cancellation changed subsequent generation");
        }
        std::cout << "Qwen raw/chat streaming parity and cancellation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
