/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace trtmc::installer {

struct PayloadEntry {
    std::array<std::uint8_t, 32> sha256{};
    std::uintmax_t size{0};
    std::filesystem::path relative_path;
};

// Manifest rows use: lowercase-sha256<TAB>decimal-size<TAB>UTF-8-relative-path.
// Paths must use forward slashes and satisfy Windows canonical-name rules.
std::vector<PayloadEntry> read_payload_manifest(const std::filesystem::path& manifest_path);

bool is_safe_payload_path(const std::string& utf8_path);

std::array<std::uint8_t, 32> sha256_file(const std::filesystem::path& path);
std::string sha256_hex(const std::array<std::uint8_t, 32>& digest);

void verify_payload(const std::filesystem::path& payload_root,
                    const std::vector<PayloadEntry>& entries);

// Reject manifests that contain anything outside the fixed native H3 runtime,
// one versioned TensorRT-RTX DLL, and the optional repository legal notices.
void validate_minimax_h3_runtime_payload(const std::vector<PayloadEntry>& entries);

// Verify first, materialize into a sibling staging directory, then atomically
// replace an existing installation. Existing destinations must carry the
// exact install marker, which prevents an installer invocation from replacing
// an unrelated directory.
void install_payload_transactional(const std::filesystem::path& payload_root,
                                   const std::filesystem::path& install_root,
                                   const std::vector<PayloadEntry>& entries,
                                   const std::string& marker_name,
                                   const std::string& marker_contents);

bool installation_marker_matches(const std::filesystem::path& install_root,
                                 const std::string& marker_name,
                                 const std::string& marker_contents);

} // namespace trtmc::installer
