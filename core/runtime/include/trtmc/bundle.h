/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {

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

  private:
    std::string path_;
    BundleInfo info_;
    std::uint64_t data_offset_{0};
    std::uint64_t file_size_{0};
};

// Read metadata without loading the engine.
BundleInfo InspectBundle(const std::string& bundle_path);

} // namespace trtmc
