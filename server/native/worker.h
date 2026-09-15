/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iosfwd>

namespace trtmc {
class ITask;
}

namespace trtmc::serve {

// Run the private, serialized JSONL data-plane protocol for one resident task.
int run_worker_protocol(ITask& task, std::istream& input, std::ostream& output);

} // namespace trtmc::serve
