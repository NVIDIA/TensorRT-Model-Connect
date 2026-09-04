/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace trtmc::internvl {

struct DistributedRuntimeGroup {
    int rank{0};
    int tp_size{1};
    void* communicator{nullptr};
    std::shared_ptr<void> owner;
};

DistributedRuntimeGroup initialize_tensor_parallel_group(int tp_size);

} // namespace trtmc::internvl
