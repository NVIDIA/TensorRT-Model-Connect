/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "plugin_helpers.h"
#include "runtime/models/magpie/cuda_common.h"
#include "runtime/models/magpie/magpie_config.h"
#include "utils/json_helpers.h"

namespace trtmc {

MagpieTTSConfig build_magpie_config(const std::string& json, const BaseConfig& base);

void allocate_cross_kv_buffers(int32_t num_layers, std::size_t buf_size,
                               std::vector<MagpieCudaBuffer>& cross_k,
                               std::vector<MagpieCudaBuffer>& cross_v);

std::shared_ptr<ITokenizer> make_ipa_tok(const BundleFile& bundle);

} // namespace trtmc
