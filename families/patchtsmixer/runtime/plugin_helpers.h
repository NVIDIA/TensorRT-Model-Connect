/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::patchtsmixer {

std::vector<char> require_section(const BundleReader& bundle, const char* name);
std::string require_text_section(const BundleReader& bundle, const char* name);
std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan);
std::int32_t require_rank(std::int32_t tensor_parallel_size);

} // namespace trtmc::patchtsmixer
