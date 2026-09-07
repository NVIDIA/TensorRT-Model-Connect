/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace trtmc {
class ITask;
struct FamilyOnlyContext;
} // namespace trtmc

namespace trtmc::qwen::edge_llm {

ITask* create(const FamilyOnlyContext& context);

} // namespace trtmc::qwen::edge_llm
