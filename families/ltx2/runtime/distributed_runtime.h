/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <vector>

namespace trtmc::ltx2 {

// One point-to-point copy of a device buffer between this rank and `peer`.
struct PeerTransfer {
    int peer{0};
    void* device{nullptr};
    std::size_t bytes{0};
    bool send{false};
};

// Point-to-point transfers on the communicator the TensorRT engines use. Runs only between
// engine executions, so it never interleaves with an engine's collectives.
class PeerChannel {
  public:
    virtual ~PeerChannel() = default;
    // Runs the transfers as one group and waits for them. When they do not finish within
    // `timeout`, aborts the communicator (in-flight transfers exit) and throws.
    virtual void run(const std::vector<PeerTransfer>& transfers,
                     std::chrono::milliseconds timeout) = 0;
    // Returns once every rank has called it (same timeout and abort behavior as run).
    virtual void barrier(std::chrono::milliseconds timeout) = 0;
};

struct DistributedRuntimeGroup {
    int world_size{1};
    int rank{0};
    int parallel_size{1};
    void* communicator{nullptr};
    std::shared_ptr<void> owner;
    std::shared_ptr<PeerChannel> channel; // null on a single device
};

// Initialize the NCCL communicator consumed by TensorRT distributed layers.
// Launcher discovery and communicator ownership remain local to LTX-2.5.
DistributedRuntimeGroup initialize_parallel_group(int parallel_size);

} // namespace trtmc::ltx2
