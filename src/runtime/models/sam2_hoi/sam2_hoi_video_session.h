/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

struct TrtmcSam2HoiVideoSession;

namespace trtmc::sam2_hoi {

class Sam2HoiPipeline;

TrtmcSam2HoiVideoSession* makeVideoSessionHandle(std::unique_ptr<Sam2HoiPipeline> pipeline);

namespace c_api_internal {
void clearLastError() noexcept;
void setLastError(const char* message) noexcept;
} // namespace c_api_internal

} // namespace trtmc::sam2_hoi
