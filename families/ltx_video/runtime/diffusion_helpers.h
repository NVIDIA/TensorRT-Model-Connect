/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "families/ltx_video/runtime/ltx_video_diffusion_types.h"
#include "plugin_helpers.h"

namespace trtmc {

LTXVideoDiffusionConfig make_diffusion_config(const std::string& json);

} // namespace trtmc
