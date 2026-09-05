/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/magpie_tts/runtime/pipeline.h"

#include <exception>
#include <iostream>

int main() {
    trtmc::MagpieTTSConfig config;
    config.num_codebooks = 0;
    bool rejected = false;
    try {
        trtmc::MagpiePipeline pipeline(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, {}, {},
                                       {}, {}, trtmc::MagpieCudaBuffer(0),
                                       trtmc::MagpieCudaBuffer(0), {}, {}, {}, {}, {}, {}, {}, {},
                                       0, config, nullptr, nullptr, "test");
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "FAIL: magpie constructor accepts a null decoder\n";
        return 1;
    }
    return 0;
}
