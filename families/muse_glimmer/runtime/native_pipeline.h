/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/runtime/family_factory.h"
namespace trtmc::muse_glimmer {
ITask* create_native(const FamilyContext& context);
}
