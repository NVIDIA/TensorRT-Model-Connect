/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bloom/runtime/task_config.h"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool value) {
    if (!value)
        throw std::runtime_error("bloom Task configuration contract failed");
}
template <class Function>
void rejects(Function call) {
    try {
        call();
    } catch (const trtmc::internal::ConfigError&) {
        return;
    }
    throw std::runtime_error("invalid Task configuration was accepted");
}
} // namespace

int main() {
    using namespace trtmc::internal;
    using trtmc::bloom::parse_text_config;
    try {
        using trtmc::bloom::copy_token_prefix;
        require(copy_token_prefix({nullptr, 0}).empty());
        std::int32_t tokens[] = {3, 1, 4};
        require(copy_token_prefix({tokens, 0}).empty());
        const auto copied = copy_token_prefix(tokens);
        require(copied == std::vector<std::int32_t>({3, 1, 4}));
        tokens[0] = 9;
        require(copied[0] == 3);
        const auto capacity = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
        // Null storage avoids allocating or reading huge arrays if the guard regresses.
        for (const auto count :
             {std::size_t{1}, capacity, capacity + 1, std::numeric_limits<std::size_t>::max()}) {
            bool rejected = false;
            try {
                (void)copy_token_prefix({nullptr, count});
            } catch (const std::invalid_argument& error) {
                const auto* expected = count > capacity
                                           ? "prefix token count exceeds int32 capacity"
                                           : "token input has no storage";
                rejected = std::string_view(error.what()) == expected;
            }
            require(rejected);
        }
        const auto defaults = parse_text_config({});
        require(defaults.max_new_tokens == 128 && defaults.temperature == 1 &&
                defaults.top_k == 1 && defaults.top_p == 1 && defaults.min_p == 0 &&
                defaults.seed == -1 && defaults.eos_token_id == -1 &&
                defaults.text_generation_mode == "auto" && !defaults.use_chat_template &&
                defaults.enable_thinking && !defaults.stop_on_boxed_answer &&
                defaults.stop_check_interval == 16 && defaults.repetition_penalty == 1);
        const ConfigEntry supplied[] = {
            {"max_new_tokens", std::int64_t{0}},
            {"temperature", 0.5},
            {"top_k", std::int64_t{4}},
            {"top_p", 0.75},
            {"min_p", 0.125},
            {"seed", std::int64_t{7}},
            {"eos_token_id", std::int64_t{2}},
            {"generation_mode", std::string_view{"ar"}},
            {"use_chat_template", true},
            {"enable_thinking", false},
            {"stop_on_boxed_answer", true},
            {"stop_check_interval", std::int64_t{3}},
            {"repetition_penalty", 1.0},
        };
        const auto configured = parse_text_config(supplied);
        require(configured.max_new_tokens == 0 && configured.temperature == 0.5F &&
                configured.top_k == 4 && configured.top_p == 0.75F && configured.min_p == 0.125F &&
                configured.seed == 7 && configured.eos_token_id == 2 &&
                configured.text_generation_mode == "ar" && configured.use_chat_template &&
                !configured.enable_thinking && configured.stop_on_boxed_answer &&
                configured.stop_check_interval == 3);
        const ConfigEntry boundaries[] = {{"temperature", 0.0},
                                          {"top_k", std::int64_t{0}},
                                          {"top_p", 0.0},
                                          {"min_p", 1.0},
                                          {"seed", std::int64_t{-99}}};
        const auto boundary = parse_text_config(boundaries);
        require(boundary.temperature == 0 && boundary.top_k == 0 && boundary.top_p == 0 &&
                boundary.min_p == 1 && boundary.seed == -99);
        const ConfigEntry duplicate[] = {{"top_k", std::int64_t{1}}, {"top_k", std::int64_t{2}}};
        rejects([&] { parse_text_config(duplicate); });
        for (const ConfigEntry bad : {
                 ConfigEntry{"unknown", true},
                 ConfigEntry{"top_k", 1.0},
                 ConfigEntry{"max_new_tokens", std::int64_t{-1}},
                 ConfigEntry{"seed", std::numeric_limits<std::int64_t>::max()},
                 ConfigEntry{"temperature", std::numeric_limits<double>::infinity()},
                 ConfigEntry{"temperature", -0.1},
                 ConfigEntry{"top_k", std::int64_t{-1}},
                 ConfigEntry{"top_p", -0.1},
                 ConfigEntry{"top_p", 1.1},
                 ConfigEntry{"min_p", -0.1},
                 ConfigEntry{"min_p", 1.1},
                 ConfigEntry{"repetition_penalty", 1.2},
             })
            rejects([&] { parse_text_config({&bad, 1}); });
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
