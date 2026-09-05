/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/magpie_tts/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct LoadedModule {
    std::unique_ptr<ITrtModule> module;
};

LoadedModule load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan,
                                       const char* label, const ModuleCreateOptions& options = {});

std::vector<float> section_to_floats(const std::vector<char>* section);
std::vector<std::int32_t> section_to_int32s(const std::vector<char>* section);
bool has_section_data(const std::vector<char>* section);

struct MelFilterbank {
    std::vector<float> data;
    std::int32_t n_freq_bins{0};
    std::int32_t n_mel_bins{0};
};

MelFilterbank load_mel_filterbank(const BundleReader& bundle);

} // namespace trtmc
