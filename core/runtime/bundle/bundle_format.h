/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Internal bundle format: read .bundle files.
// Format:
//   Bytes 0-7:    Magic "BUNDLE\x01\x00"
//   Bytes 8-15:   uint64_t json_header_length (LE)
//   Bytes 16..N:  JSON metadata header (UTF-8)
//   Bytes N..EOF: Binary sections referenced by offset in the header

#include "trtmc/bundle.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace trtmc {

// Magic bytes for .bundle files.
static constexpr unsigned char kBundleMagic[8] = {'B', 'U', 'N', 'D', 'L', 'E', '\x01', '\0'};
static constexpr std::size_t kBundleHeaderOffset = 16; // 8 magic + 8 length

} // namespace trtmc
