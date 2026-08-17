/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::sam2::native {

class BundleWriteError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct BundleSectionView {
    std::string_view name;
    const void* data{nullptr};
    std::size_t size{0};
};

struct BundleMetadata {
    std::string model_id;
    std::string trt_version;
    std::string trt_abi;
    std::string gpu_name;
    std::string created_at;
};

// Facts derived from the exact completed descriptor before it is published.
// The full-bundle SHA-256 is the cross-snapshot trust primitive consumed by
// the authenticated loader. Linux device/inode facts additionally bind it to
// the inode linked at the requested destination.
struct BundlePublicationFacts {
    std::string sha256;
    std::uint64_t size_bytes{0U};
    std::uint64_t device{0U};
    std::uint64_t inode{0U};
    bool filesystem_identity_available{false};
};

// Writes the exact SAM2 native bundle atomically. The destination must not
// already exist. sections must contain each of the six plan sections plus
// config.json and sam2_build_receipt.json exactly once, and no other section.
// The canonical header binds every section's offset, size, and SHA-256. Section
// payloads are borrowed only for this call.
BundlePublicationFacts writeSam2NativeBundle(const std::filesystem::path& destination,
                                             const BundleMetadata& metadata,
                                             const std::vector<BundleSectionView>& sections);

} // namespace trtmc::sam2::native
