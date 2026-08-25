/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cli/args.h"
#include "trtmc/pipeline.h"

#include <vector>

namespace trtmc::cli {

// Load CLI media references into the public AudioVideoRequest representation.
// Reference order is preserved exactly. Throws std::runtime_error when a media
// file or strict video manifest is invalid.
std::vector<AudioVideoReference> load_reference_inputs(const std::vector<ReferenceInput>& inputs);

} // namespace trtmc::cli
