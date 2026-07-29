/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/m2m_100/request_tokens.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

} // namespace

int main() {
    {
        std::vector<int32_t> ids{11, 12, 2, 3};
        trtmc::m2m_100_apply_source_language_token(ids, 2, 256047);
        check(ids == std::vector<int32_t>({11, 12, 2, 256047}), "replaces unknown language suffix");
    }
    {
        std::vector<int32_t> ids{11, 12};
        trtmc::m2m_100_apply_source_language_token(ids, 2, 256047);
        check(ids == std::vector<int32_t>({11, 12, 2, 256047}), "appends EOS and source language");
    }
    {
        std::vector<int32_t> ids{11, 12, 2};
        trtmc::m2m_100_apply_source_language_token(ids, 2, -1);
        check(ids == std::vector<int32_t>({11, 12, 2}),
              "negative source language preserves legacy input");
    }
    check(trtmc::m2m_100_apply_forced_bos_token(17, 0, 256057) == 256057,
          "forces target BOS on the first decoder step");
    check(trtmc::m2m_100_apply_forced_bos_token(17, 1, 256057) == 17,
          "preserves selected tokens after the first decoder step");
    check(trtmc::m2m_100_apply_forced_bos_token(17, 0, -1) == 17,
          "negative forced BOS preserves legacy decoding");
    std::cout << "All M2M100 request token tests passed.\n";
    return 0;
}
