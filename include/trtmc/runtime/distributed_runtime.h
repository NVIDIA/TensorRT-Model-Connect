/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <string>

namespace trtmc {

struct DistributedRuntimeGroup {
    int world_size{1};
    // Rank within this communicator. Used for communicator position and
    // engine shard selection.
    int rank{0};
    // Original launcher coordinates. CUDA device binding always follows the
    // global rank, including when this object represents a subgroup.
    int global_world_size{1};
    int global_rank{0};
    int tp_size{1};
    void* communicator{nullptr};
    std::shared_ptr<void> owner;
    std::function<void(const void*, void*, std::size_t, cudaStream_t)> all_gather_float;
};

// Initialize an NCCL communicator for TensorRT 11.0+ distributed collective layers.
//
// This intentionally avoids compile-time MPI/NCCL dependencies: ranks are
// discovered from common mpirun environment variables, and NCCL is loaded with
// dlopen at runtime. Rank 0 writes the NCCL unique ID to a small rendezvous
// file under /tmp unless TRTMC_NCCL_RENDEZVOUS points elsewhere.
DistributedRuntimeGroup initialize_tensor_parallel_group(int tp_size);

// Initialize one explicitly mapped NCCL subgroup inside a larger launcher
// world. Every process supplies its rank within the subgroup and a key shared
// only by the members of that communicator. The CUDA device remains selected
// by the process's global launcher rank.
DistributedRuntimeGroup initialize_distributed_subgroup(int group_size, int group_rank,
                                                        const std::string& group_key);

// Enqueue a float32 all-gather on the group's NCCL communicator.
void distributed_all_gather_float(const DistributedRuntimeGroup& group, const void* send_buffer,
                                  void* receive_buffer, std::size_t element_count,
                                  cudaStream_t stream);

int distributed_process_world_size();
int distributed_process_rank();

} // namespace trtmc
