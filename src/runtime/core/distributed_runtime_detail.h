/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace trtmc::distributed_runtime_detail {

int detect_world_size();
int detect_rank();
int detect_local_rank(int global_rank);

} // namespace trtmc::distributed_runtime_detail
