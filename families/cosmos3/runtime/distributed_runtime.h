/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace trtmc::cosmos3 {

struct DistributedRuntimeGroup {
    int world_size{1};
    int rank{0};
    int cp_size{1};
    void* communicator{nullptr};
    std::shared_ptr<void> owner;
};

// Initialize an NCCL communicator for TensorRT 11.0+ distributed collective layers.
//
// This intentionally avoids compile-time MPI/NCCL dependencies: ranks are
// discovered from exact OpenMPI environment variables, and NCCL is loaded with
// dlopen at runtime. Rank 0 can publish its unique ID to stdout for an external
// MPI launcher by setting TRTMC_NCCL_UNIQUE_ID_STDOUT=1; rank 1 then consumes
// the MPI-broadcast value from TRTMC_NCCL_UNIQUE_ID_HEX. The original
// TRTMC_NCCL_RENDEZVOUS file contract remains available for callers that do not
// use the MPI launcher.
DistributedRuntimeGroup initialize_context_parallel_group(int cp_size);

} // namespace trtmc::cosmos3
