/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace trtmc {
struct LoadOptions;
}

namespace trtmc::server {

// Private native data-plane entrypoint used by `trtmc serve`'s Python facade.
int run_native_worker(const std::string& bundle_path, const LoadOptions& options,
                      const std::string& kernel_bindings_path);

} // namespace trtmc::server
