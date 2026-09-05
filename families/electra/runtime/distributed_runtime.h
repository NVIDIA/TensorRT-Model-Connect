/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>

namespace trtmc::electra {

struct DistributedRuntimeGroup {
    int world_size{1};
    // Single-node tensor-parallel rank. Used for communicator position,
    // engine shard selection, and CUDA device binding in this initial runtime.
    int rank{0};
    int tp_size{1};
    void* communicator{nullptr};
    std::shared_ptr<void> owner;
};

// Initialize an NCCL communicator for TensorRT 11.0+ distributed collective layers.
//
// This intentionally avoids compile-time MPI/NCCL dependencies: ranks are
// discovered from common mpirun environment variables, and NCCL is loaded with
// dlopen at runtime. Rank 0 writes the NCCL unique ID to a small rendezvous
// file at the path named by TRTMC_NCCL_RENDEZVOUS.
DistributedRuntimeGroup initialize_tensor_parallel_group(int tp_size);

} // namespace trtmc::electra
