/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Internal bundle format: read .trtfb files.
// Format:
//   Bytes 0-7:    Magic "TRTFB\x00\x01\x00"
//   Bytes 8-15:   uint64_t json_header_length (LE)
//   Bytes 16..N:  JSON metadata header (UTF-8)
//   Bytes N..EOF: Binary sections referenced by offset in the header

#include "trtmc/bundle.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iosfwd>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc {

// Magic bytes for .trtfb files.
static constexpr unsigned char kBundleMagic[8] = {'T', 'R', 'T', 'F', 'B', '\0', '\x01', '\0'};
static constexpr std::size_t kBundleHeaderOffset = 16; // 8 magic + 8 length

struct BundleSection {
    std::string name;
    std::vector<char> data;
};

struct BundleFile {
    BundleInfo info;
    std::vector<BundleSection> sections;
};

// Keeps the opened bundle file pinned while reading individual payloads on
// demand.  This prevents staged runtimes from accidentally mixing metadata
// loaded from one file with plans from a later path replacement, and remains
// usable after a POSIX rename/unlink of the original pathname.
class BundleSectionReader {
  public:
    explicit BundleSectionReader(const std::string& path);

    BundleSectionReader(const BundleSectionReader&) = delete;
    BundleSectionReader& operator=(const BundleSectionReader&) = delete;

    std::vector<char> read(const std::string& name);
    // Visit a section through a bounded scratch buffer. The callback runs
    // while the reader lock is held and must not recursively access this
    // reader. This preserves staged materialization while allowing integrity
    // checks over multi-GiB payloads.
    void for_each_chunk(const std::string& name, std::size_t chunk_size,
                        const std::function<void(const char*, std::size_t)>& visitor);
    BundleFile read_all();
    const std::string& path() const { return path_; }
    const BundleInfo& info() const { return info_; }

  private:
    std::vector<char> read_locked(const BundleSectionInfo& section);

    std::string path_;
    std::ifstream stream_;
    std::uint64_t data_start_{0};
    std::uint64_t file_size_{0};
    BundleInfo info_;
    std::mutex mutex_;
};

// Read a complete bundle from disk.
BundleFile ReadBundleFile(const std::string& path);

// Materialize a complete bundle through an already-open reader. The caller
// may retain the reader for later lazy reads from the exact same open file.
BundleFile ReadBundleFile(BundleSectionReader& reader);

// Read one named section without materializing any other section payload.
// Throws when the section is absent or its declared byte range is outside the
// bundle file.
std::vector<char> ReadBundleSection(const std::string& path, const std::string& name);

// Read just the header metadata (no section data loaded).
BundleInfo ReadBundleHeader(const std::string& path);

// Read one section identified by header metadata. Only the requested bytes are
// read; other bundle payloads are never materialized. The section metadata must
// come from ReadBundleHeader(path).
std::vector<char> ReadBundleSection(const std::string& path, const BundleSectionInfo& section);

// Copy one section to an output stream in bounded-size chunks. This is the
// preferred path for model-sized payloads that must not be buffered in memory.
void CopyBundleSection(const std::string& path, const BundleSectionInfo& section,
                       std::ostream& output);

// Check magic bytes without reading full file.
bool HasBundleMagic(const std::string& path);

} // namespace trtmc
