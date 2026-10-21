/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/qwen3_8/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::qwen3_8 {

std::vector<char> require_section(const BundleReader& bundle, std::string_view name);
std::string require_text_section(const BundleReader& bundle, std::string_view name);
std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* label);

// Load libtrtmc_qwen3_8_native_plugin.so (the cuTile GDN TensorRT plugins) so its plugin
// creators are registered before an engine that uses them is deserialized. The library is
// looked up next to this runtime library first, then on the default library search path.
// Returns false if it is not available; engines without the plugins do not need it.
// Safe to call repeatedly; the library stays loaded for the life of the process.
bool load_native_plugin_library();
std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle);

} // namespace trtmc::qwen3_8
