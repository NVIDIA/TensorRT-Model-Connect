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
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc {

// Magic bytes for .bundle files.
static constexpr unsigned char kBundleMagic[8] = {'B', 'U', 'N', 'D', 'L', 'E', '\x01', '\0'};
static constexpr std::size_t kBundleHeaderOffset = 16; // 8 magic + 8 length

struct BundleSection {
    std::string name;
    std::vector<char> data;
};

struct BundleFile {
    BundleInfo info;
    std::vector<BundleSection> sections;
};

struct BundleSectionFileRange {
    std::uint64_t offset{0};
    std::uint64_t size{0};
};

// Read a complete bundle from disk.
BundleFile ReadBundleFile(const std::string& path);

// Read just the header metadata (no section data loaded).
BundleInfo ReadBundleHeader(const std::string& path);

// Read one section identified by header metadata. Only the requested bytes are
// read; other bundle payloads are never materialized. The section metadata must
// come from ReadBundleHeader(path).
std::vector<char> ReadBundleSection(const std::string& path, const BundleSectionInfo& section);

// Resolve and validate the absolute byte range for a section without loading
// its payload. This supports stream-based backends for model-sized plans.
BundleSectionFileRange ResolveBundleSectionFileRange(const std::string& path,
                                                     const BundleSectionInfo& section);

// Copy one section to an output stream in bounded-size chunks. This is the
// preferred path for model-sized payloads that must not be buffered in memory.
void CopyBundleSection(const std::string& path, const BundleSectionInfo& section,
                       std::ostream& output);

// Stream one section through the repository SHA-256 implementation and fail
// closed unless the digest exactly matches the lowercase attestation. The
// section payload is never materialized as one allocation.
void ValidateBundleSectionSha256(const std::string& path, const BundleSectionInfo& section,
                                 std::string_view expected_sha256);

// Check magic bytes without reading full file.
bool HasBundleMagic(const std::string& path);

} // namespace trtmc
