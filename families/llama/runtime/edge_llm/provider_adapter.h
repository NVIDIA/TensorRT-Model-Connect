/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/bundle.h"
#include "trtmc/task.h"
namespace trtmc::llama::edge_llm {
ITask* create_provider(const BundleReader& bundle);
}
