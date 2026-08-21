/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct LoadedModule {
    std::unique_ptr<ITrtModule> module;
};

LoadedModule load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan,
                                       const char* label, const ModuleCreateOptions& options = {});

std::shared_ptr<ITokenizer> try_create_native_tokenizer(const BundleFile& bundle,
                                                        bool add_special_tokens);

std::vector<float> section_to_floats(const std::vector<char>* section);
std::vector<int32_t> section_to_int32s(const std::vector<char>* section);

struct MelFilterbank {
    std::vector<float> data;
    int32_t n_freq_bins{0};
    int32_t n_mel_bins{0};
};

MelFilterbank load_mel_filterbank(const BundleFile& bundle);

} // namespace trtmc
