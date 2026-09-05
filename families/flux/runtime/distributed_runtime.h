/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace trtmc::flux_runtime {

struct DistributedGroup {
    int world_size{1};
    int rank{0};
    int size{1};
    void* communicator{nullptr};
    std::shared_ptr<void> owner;
};

// Initialize the communicator used by FLUX TP and Ulysses CP engines. Exact
// OpenMPI rank variables and TRTMC_NCCL_RENDEZVOUS are required; NCCL is loaded
// at runtime so the family target has no compile-time MPI or NCCL dependency.
DistributedGroup initialize_group(int size);

} // namespace trtmc::flux_runtime
