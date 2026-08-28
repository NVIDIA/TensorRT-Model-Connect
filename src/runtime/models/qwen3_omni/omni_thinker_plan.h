/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace trtmc {

inline std::string format_omni_chat_prompt(const std::string& prompt) {
    return "<|im_start|>system\n"
           "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
           "perceiving auditory and visual inputs, as well as generating text and speech."
           "<|im_end|>\n<|im_start|>user\n" +
           prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

inline bool omni_thinker_should_stop(int32_t token_id, int32_t eos_token_id) {
    return eos_token_id >= 0 && token_id == eos_token_id;
}

inline bool omni_thinker_request_fits_cache(std::size_t prompt_tokens, int32_t max_tokens,
                                            int32_t cache_capacity) {
    if (cache_capacity < 0)
        return true;
    if (max_tokens < 0)
        return false;
    const auto capacity = static_cast<std::size_t>(cache_capacity);
    if (prompt_tokens > capacity)
        return false;
    return static_cast<std::size_t>(max_tokens) <= capacity - prompt_tokens;
}

} // namespace trtmc
