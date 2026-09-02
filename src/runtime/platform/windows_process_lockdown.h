/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef _WIN32
#error "The locked MiniMax-H3 process policy is Windows-only"
#endif

namespace trtmc::internal {

// Reject every environment variable that could otherwise redirect a runtime
// backend, model plugin, or kernel-binding manifest.
void reject_locked_runtime_override_environment();

// Assign this process to a private one-process Job Object. The Job handle is
// retained for the lifetime of the process so no later child process can
// escape the active-process limit.
void enforce_single_process_job();

// Apply the complete fail-closed policy. Both the packaged CLI and public
// pipeline entry points call this before loading TensorRT-RTX.
void enforce_locked_h3_process_policy();

// Exposed for the Windows unit test; production startup treats false as fatal.
bool single_process_job_is_active() noexcept;

} // namespace trtmc::internal
