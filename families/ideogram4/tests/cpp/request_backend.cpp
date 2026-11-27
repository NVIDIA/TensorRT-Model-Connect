/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "request_backend.h"

extern "C" trtmc::IBackend* trtmc_create_backend() {
    return new trtmc::ideogram4::test::RequestBackend;
}
extern "C" void trtmc_destroy_backend(trtmc::IBackend* backend) {
    delete backend;
}
