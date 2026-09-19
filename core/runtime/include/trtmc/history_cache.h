/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class DeviceTensor;

// Source-level contract version, not a stable compiler or standard-library ABI.
inline constexpr std::uint32_t kHistoryCacheInterfaceVersion = 1;

struct HistoryCacheKey {
    std::string artifact_id;
    std::string feature_version;
    std::string subject_id;
    std::string history_epoch;

    bool operator==(const HistoryCacheKey& other) const;
    bool operator!=(const HistoryCacheKey& other) const { return !(*this == other); }
};

struct HistoryCacheTensor {
    std::string name;
    std::vector<std::int64_t> shape;
    DType dtype{DType::kFloat32};
    std::vector<std::uint8_t> host_data;
    std::shared_ptr<const DeviceTensor> device;
};

struct HistoryCacheValue {
    std::uint32_t schema_version{kHistoryCacheInterfaceVersion};
    std::string format;
    std::vector<std::uint8_t> metadata;
    std::vector<HistoryCacheTensor> tensors;
};

class IHistoryCacheStorage {
  public:
    virtual ~IHistoryCacheStorage() = default;
    // Return a host-only immutable snapshot, or nullptr on a miss. Implementors
    // report failures by throwing; the cache records them and recomputes.
    virtual std::shared_ptr<const HistoryCacheValue> load(const HistoryCacheKey& key) = 0;
    virtual void store(const HistoryCacheKey& key, const HistoryCacheValue& host_value) = 0;
    virtual void erase(const HistoryCacheKey& key) = 0;
};

// Bounded native CPU storage. Snapshots held by readers retain their budget
// charge after eviction. An admission that cannot fit throws std::length_error.
class InMemoryHistoryCacheStorage final : public IHistoryCacheStorage {
  public:
    explicit InMemoryHistoryCacheStorage(std::size_t max_bytes, std::size_t max_entries = 1024);
    ~InMemoryHistoryCacheStorage() override;
    std::shared_ptr<const HistoryCacheValue> load(const HistoryCacheKey& key) override;
    void store(const HistoryCacheKey& key, const HistoryCacheValue& host_value) override;
    void erase(const HistoryCacheKey& key) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct HistoryCacheOptions {
    // A positive explicit budget is required. Live retired snapshots, host and
    // device payloads, and owned metadata count against this budget.
    std::size_t max_bytes{0};
    std::size_t max_entries{1024};
    std::shared_ptr<IHistoryCacheStorage> storage;
    bool write_through{false};
};

enum class HistoryCacheSource { kMiss, kMemory, kStorage };

struct HistoryCacheStats {
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    std::uint64_t storage_hits{0};
    std::uint64_t load_failures{0};
    std::uint64_t store_failures{0};
    std::uint64_t erase_failures{0};
    std::uint64_t publications{0};
    std::uint64_t rejected_publications{0};
    std::uint64_t stale_publications{0};
    std::uint64_t evictions{0};
    std::size_t resident_entries{0};
    std::size_t resident_bytes{0};
    std::size_t live_bytes{0};
};

class HistoryCache {
  public:
    struct Lease {
        HistoryCacheKey key;
        std::uint64_t generation{0};
        std::shared_ptr<const HistoryCacheValue> value;
        HistoryCacheSource source{HistoryCacheSource::kMiss};

      private:
        friend class HistoryCache;
        std::shared_ptr<const void> owner_;
    };

    explicit HistoryCache(HistoryCacheOptions options);
    ~HistoryCache();
    HistoryCache(const HistoryCache&) = delete;
    HistoryCache& operator=(const HistoryCache&) = delete;

    Lease lookup(const HistoryCacheKey& key);
    // Compare-and-swap consumes the lease generation. Obtain another lease for
    // another publication. Device writes must be complete before publication,
    // and device buffers must remain immutable while any snapshot owns them.
    bool publish(const Lease& lease, HistoryCacheValue value);
    // Invalidate both tiers. Concurrent old loads/publications cannot restore
    // the invalidated value. A failing erase disables lower-tier reads until a
    // new cache instance is created, so stale storage is never silently reused.
    void invalidate(const HistoryCacheKey& key);
    // Evict native entries/reservations, preserving the external storage tier.
    // Outstanding snapshots remain readable; outstanding publications fail CAS.
    void clear_memory();
    HistoryCacheStats stats() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Optional task capability for attaching a shared platform-owned cache service.
class IHistoryCacheConsumer {
  public:
    virtual ~IHistoryCacheConsumer() = default;
    virtual void set_history_cache(std::shared_ptr<HistoryCache> cache) = 0;
    virtual std::string history_cache_artifact_id() const = 0;
};

} // namespace trtmc
