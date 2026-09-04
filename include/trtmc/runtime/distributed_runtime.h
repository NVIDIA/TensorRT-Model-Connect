/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>

namespace trtmc {

struct DistributedRuntimeGroup {
    int world_size{1};
    // Global tensor-parallel rank. Used for communicator position and engine
    // shard selection. CUDA device binding uses launcher-provided local rank.
    int rank{0};
    int tp_size{1};
    void* communicator{nullptr};
    std::shared_ptr<void> owner;
};

// Initialize an NCCL communicator for TensorRT 11.0+ distributed collective layers.
//
// This intentionally avoids compile-time MPI/NCCL dependencies: ranks are
// discovered from common launcher environment variables, and NCCL is loaded
// with dlopen at runtime. Rank 0 writes the NCCL unique ID to a small
// rendezvous file under /tmp unless TRTMC_NCCL_RENDEZVOUS points elsewhere.
// Multi-node launchers must set that variable to a unique path shared by every
// rank; node-local /tmp is not a cross-node rendezvous.
DistributedRuntimeGroup initialize_tensor_parallel_group(int tp_size);

} // namespace trtmc
