/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/task.h"

#include <filesystem>
namespace trtmc::clef {
StructuredDecisionRequest read_request(const std::filesystem::path& path);
}
