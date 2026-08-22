/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace trtmc::cli {

struct CliArgs;

// Private native data-plane entrypoint used by `trtmc serve`'s Python facade.
int run_serve_worker(const CliArgs& args);

} // namespace trtmc::cli
