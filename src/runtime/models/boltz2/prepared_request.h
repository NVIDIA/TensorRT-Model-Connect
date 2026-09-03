/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/boltz2/feature_bundle.h"
#include "runtime/models/boltz2/random_samples.h"

#include <cstddef>
#include <string>

namespace trtmc::boltz2 {

struct PreparedRequest {
    FeatureBundle features;
    RandomSamples random_samples;
    std::string request;
    std::string structure_metadata_json;

    static bool isPrepared(const void* data, std::size_t size);
    static PreparedRequest parse(const void* data, std::size_t size);
};

} // namespace trtmc::boltz2
