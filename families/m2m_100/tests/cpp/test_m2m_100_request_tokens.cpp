/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/m2m_100/runtime/request_tokens.h"

#include <stdexcept>
#include <vector>

int main() {
    const std::vector<int32_t> content{1617, 34151, 248, 157826, 248075};
    auto ids = content;
    trtmc::m2m_100_apply_source_language_token(ids, 2, 256047);
    if (ids != std::vector<int32_t>{256047, 1617, 34151, 248, 157826, 248075, 2})
        throw std::runtime_error("default NLLB source frame must be language, content, EOS");

    ids = content;
    trtmc::m2m_100_apply_source_language_token(ids, 2, 256047, false);
    if (ids != std::vector<int32_t>{1617, 34151, 248, 157826, 248075, 2, 256047})
        throw std::runtime_error("explicit legacy NLLB source frame must retain its suffix");

    ids = content;
    trtmc::m2m_100_apply_source_language_token(ids, 2, 256057);
    if (ids.front() != 256057 || ids.back() != 2 || ids.size() != content.size() + 2)
        throw std::runtime_error("each request must use its own source language");

    ids = content;
    trtmc::m2m_100_apply_source_language_token(ids, 2, -1);
    if (ids != content)
        throw std::runtime_error("an omitted source override must leave token framing alone");

    ids = {2};
    trtmc::m2m_100_apply_source_language_token(ids, 2, 0);
    if (ids != std::vector<int32_t>{0, 2, 2})
        throw std::runtime_error("content and zero-valued language IDs must be preserved");
    ids.clear();
    trtmc::m2m_100_apply_source_language_token(ids, 2, 256047);
    if (ids != std::vector<int32_t>{256047, 2})
        throw std::runtime_error("empty content still needs the source-language frame");
    if (trtmc::m2m_100_apply_forced_bos_token(5, 0, 0) != 0 ||
        trtmc::m2m_100_apply_forced_bos_token(5, 1, 0) != 5)
        throw std::runtime_error("target language forcing must remain first-step only");
}
