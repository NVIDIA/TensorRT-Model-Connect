/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

struct TrtmcSam2VideoSession;

namespace trtmc::sam2 {

class NativeVideoProcessor;

TrtmcSam2VideoSession* makeVideoSessionHandle(std::unique_ptr<NativeVideoProcessor> processor);

namespace c_api_internal {
void clearLastError() noexcept;
void setLastError(const char* message) noexcept;
} // namespace c_api_internal

} // namespace trtmc::sam2
