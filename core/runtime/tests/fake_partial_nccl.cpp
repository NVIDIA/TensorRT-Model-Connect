/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A shared library that exports only part of the NCCL entry points the
// family runtimes resolve, to test the missing-symbol diagnostics.

#if defined(_WIN32)
#define TRTMC_FAKE_EXPORT __declspec(dllexport)
#else
#define TRTMC_FAKE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" TRTMC_FAKE_EXPORT int ncclGetVersion(int* version) {
    *version = 23007;
    return 0;
}

extern "C" TRTMC_FAKE_EXPORT int ncclGetUniqueId(void* id) {
    return id == nullptr ? 4 : 0;
}
