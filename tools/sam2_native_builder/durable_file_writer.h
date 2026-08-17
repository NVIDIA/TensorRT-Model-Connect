/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string_view>

namespace trtmc::sam2::durable_file {

// Internal SAM2 tool primitive. This header is not installed and is not part
// of the runtime or bundle-format ABI.
class WriteError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct PublicationIdentity {
    std::uint64_t size_bytes{0U};
    std::uint64_t device{0U};
    std::uint64_t inode{0U};
};

using DescriptorAction = std::function<void(int)>;

// Create a same-directory exclusive regular temporary file, invoke writer,
// fsync the completed descriptor, invoke validate_after_sync, and atomically
// publish the exact descriptor without replacing an existing destination.
// The temporary name is removed before the parent directory is fsynced.
// Failures remove only names still authenticated as this call's inode and
// durably roll back an exact destination published by this call.
PublicationIdentity writeExclusiveDurably(const std::filesystem::path& destination,
                                          std::string_view artifact, const DescriptorAction& writer,
                                          const DescriptorAction& validate_after_sync = {});

} // namespace trtmc::sam2::durable_file
