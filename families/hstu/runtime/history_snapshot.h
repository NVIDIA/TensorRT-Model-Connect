/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "families/hstu/runtime/validation.h"

#include <array>
#include <cstddef>

namespace trtmc::hstu {
// Deterministic CPU data fingerprint for row deduplication and accidental
// corruption detection. Snapshots are trusted inputs, not authenticated messages.
class HistoryHash {
  public:
    void update(const void* data, std::size_t size);
    std::string finish() const;

  private:
    std::uint64_t first_{14695981039346656037ull}, second_{7809847782465536322ull};
};
std::string history_data_key(const void* data, std::size_t size);
struct HistorySnapshot {
    nlohmann::json metadata;
    std::vector<FeatureTensor> tensors;
};
std::vector<std::uint8_t> encode_history_snapshot(HistorySnapshot snapshot);
HistorySnapshot decode_history_snapshot(const std::vector<std::uint8_t>& data,
                                        const std::string& bundle_identity,
                                        const nlohmann::json& specs, std::int64_t max_batch);
inline constexpr std::size_t kHistorySnapshotLimit = 1024ull * 1024 * 1024;
} // namespace trtmc::hstu
