/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace trtmc::wan_t2v {

struct DistributedRuntimeGroup {
    int world_size{1};
    int rank{0};
    int parallel_size{1};
    void* communicator{nullptr};
    std::shared_ptr<void> owner;
};

// Initialize the NCCL communicator consumed by TensorRT distributed layers.
// Launcher discovery and communicator ownership remain local to Wan.
DistributedRuntimeGroup initialize_parallel_group(int parallel_size);

} // namespace trtmc::wan_t2v
