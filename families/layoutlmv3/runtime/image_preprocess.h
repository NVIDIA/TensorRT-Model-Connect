/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/image.h"
namespace trtmc::layoutlmv3 {
std::vector<float> image_pixels(const internal::ImageView&);
}
