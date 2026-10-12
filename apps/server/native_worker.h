/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iosfwd>
#include <string>

namespace trtmc {
class ITask;
class Model;
struct LoadOptions;
} // namespace trtmc

namespace trtmc::server {

// Runs the private serialized protocol used by the Python serving control plane.
int run_text_worker(ITask& task, std::istream& input, std::ostream& output);
int run_text_worker(const Model& model, std::istream& input, std::ostream& output);
int run_bundle_worker(const std::string& bundle, const LoadOptions& options, std::istream& input,
                      std::ostream& output);

} // namespace trtmc::server
