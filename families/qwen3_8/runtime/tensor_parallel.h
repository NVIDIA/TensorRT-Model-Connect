/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/task.h"

#include <functional>

namespace trtmc::qwen3_8 {
using NativeRankFactory = std::function<ITask*(int, const ModuleCreateOptions&)>;
ITask* create_native_tensor_parallel_task(NativeRankFactory factory, bool media);
} // namespace trtmc::qwen3_8
