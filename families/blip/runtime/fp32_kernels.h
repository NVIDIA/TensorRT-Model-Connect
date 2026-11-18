/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace trtmc::blip {
void initialize_fp32_kernels();
}

extern "C" const char* trtmc_blip_initialize_fp32_kernels() noexcept;
