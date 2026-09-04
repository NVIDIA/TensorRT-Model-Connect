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
// dlopen at runtime. TRTMC_NCCL_RENDEZVOUS must name the file shared or
// transferred between ranks.
DistributedRuntimeGroup initialize_context_parallel_group(int cp_size);

} // namespace trtmc::cosmos3
