/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

namespace trtmc {

class PipelineRegistry;

namespace detail {

using RegisterModelPluginFn = void (*)(PipelineRegistry*);

// Opens a candidate and resolves its identity and registration entrypoints only
// after validating the current model plugin ABI. The caller owns the returned
// handle on success.
bool open_model_plugin_entrypoints(const std::string& path, const std::string& expected_model_id,
                                   void*& handle, RegisterModelPluginFn& register_fn,
                                   std::vector<std::string>& errors);

} // namespace detail
} // namespace trtmc
