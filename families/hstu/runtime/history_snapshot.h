/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "families/hstu/runtime/pipeline.h"

#include <array>
#include <cstddef>

namespace trtmc::hstu {
// Incremental CPU SHA-256; snapshot integrity never depends on process-local hashes.
class HistoryHash {
  public:
    void update(const void* data, std::size_t size);
    std::string finish();

  private:
    void block(const std::uint8_t* data);
    std::array<std::uint32_t, 8> state_{{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f,
                                         0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t count_{};
    std::uint64_t bytes_{};
};
std::string history_sha256(const void* data, std::size_t size);
std::string history_file_sha256(const std::string& path);
struct HistorySnapshot {
    nlohmann::json metadata;
    std::vector<FeatureTensor> tensors;
};
std::vector<std::uint8_t> encode_history_snapshot(HistorySnapshot snapshot);
HistorySnapshot decode_history_snapshot(const std::vector<std::uint8_t>& data,
                                        const std::string& bundle_identity,
                                        const nlohmann::json& specs, std::int64_t max_batch);
inline constexpr std::size_t kHistorySnapshotLimit = 64 * 1024 * 1024;
} // namespace trtmc::hstu
