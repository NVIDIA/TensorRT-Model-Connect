/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {

enum class PlanSeekOrigin { begin, current, end };

// A bounded, seekable source of serialized engine bytes. Destinations are host
// memory; a backend owns any device transfer required by its deserializer.
class IPlanReader {
  public:
    virtual ~IPlanReader() = default;
    virtual std::uint64_t size() const noexcept = 0;
    // Return at most count bytes, or zero at EOF. Throw on an I/O failure.
    virtual std::size_t read(void* destination, std::size_t count) = 0;
    // Invalid seeks leave the cursor unchanged. I/O failures may throw.
    virtual bool seek(std::int64_t offset, PlanSeekOrigin origin) = 0;
};

struct BundleSectionInfo {
    std::string name;
    std::uint64_t offset{0};
    std::uint64_t length{0};

    BundleSectionInfo() = default;
    BundleSectionInfo(std::string section_name, std::uint64_t section_offset,
                      std::uint64_t section_length)
        : name(std::move(section_name)), offset(section_offset), length(section_length) {}
};

struct BundleInfo {
    std::int32_t format{0};
    std::string family;
    std::string task;
    std::string backend;
    std::vector<BundleSectionInfo> sections;
};

// Validates a bundle once and reads named sections directly from the file.
// The reader owns its path and metadata, so copies remain valid after the
// caller that created them returns.
class BundleReader {
  public:
    explicit BundleReader(std::string bundle_path);

    const std::string& path() const noexcept { return path_; }
    const BundleInfo& info() const noexcept { return info_; }
    const BundleSectionInfo* find_section(std::string_view name) const noexcept;
    std::vector<char> read_section(std::string_view name) const;

    // Open an independent bounded reader without allocating a section-sized
    // buffer. The opened regular file must retain its identity while reading.
    std::unique_ptr<IPlanReader> open_section(std::string_view name) const;

    /// Copy a named section to an output stream using bounded working memory.
    /// @param name Validated bundle section name.
    /// @param output Caller-owned stream; may contain partial data on failure.
    /// @throws std::runtime_error If the section is absent or input/output fails.
    void copy_section(std::string_view name, std::ostream& output) const;

  private:
    std::string path_;
    BundleInfo info_;
    std::uint64_t data_offset_{0};
    std::uint64_t file_size_{0};
};

// Read metadata without loading the engine.
BundleInfo InspectBundle(const std::string& bundle_path);

} // namespace trtmc
