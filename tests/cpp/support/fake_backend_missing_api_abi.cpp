/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Simulates a backend built before the TRTMC backend API ABI handshake existed.
// Its legacy factory must never be called by a new core.
extern "C" void* trtmc_create_backend() {
    return nullptr;
}

extern "C" void trtmc_destroy_backend(void*) {}
