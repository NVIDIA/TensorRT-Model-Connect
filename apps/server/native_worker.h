/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iosfwd>

namespace trtmc {
class ITask;
}

namespace trtmc::server {

// Runs the private serialized protocol used by the Python serving control plane.
int run_text_worker(ITask& task, std::istream& input, std::ostream& output);

} // namespace trtmc::server
