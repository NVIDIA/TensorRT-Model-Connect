/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/bundle.h"
#include "trtmc/task.h"

namespace trtmc::qwen3_8::edge_llm {

/// Create a persistent Edge task from a self-contained bundle; throws on load failure.
ITask* create(const BundleReader& bundle);

} // namespace trtmc::qwen3_8::edge_llm
