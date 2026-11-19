/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip/runtime/fp32_kernels.h"

#include <new>
#include <stdexcept>
#include <string>

extern "C" const char* trtmc_blip_initialize_fp32_kernels() noexcept {
    static thread_local std::string error;
    try {
        trtmc::blip::initialize_fp32_kernels();
        return nullptr;
    } catch (const std::bad_alloc&) {
        return "BLIP kernel initialization ran out of memory";
    } catch (const std::exception& exception) {
        try {
            error = exception.what();
            return error.c_str();
        } catch (...) {
            return "BLIP kernel initialization ran out of memory";
        }
    } catch (...) {
        return "BLIP kernel initialization failed";
    }
}
