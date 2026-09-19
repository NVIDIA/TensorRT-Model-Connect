/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/hstu/runtime/cache_policy.h"
#include "trtmc/history_cache.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::hstu {

struct PagedCacheGeometry {
    std::int32_t layers{0}, heads{0}, head_dim{0};
    // Capacity bounds both persistent history and each logical history+query
    // attention span, matching the model's max_sequence_length.
    std::int32_t slots{0}, max_history_tokens{0}, page_size{128};
    DType dtype{DType::kBFloat16};
    std::size_t max_bytes{0};
};

struct PagedCacheRequest {
    std::int32_t slot{0};
    // Subject/cache key or session identity, independent of the history tokens.
    std::string owner;
    HistorySignature history;
    // Full uncached sequence length, including request-local candidates.
    std::int32_t total_tokens{0};
};

struct PagedCachePlan {
    // Request order; repeated offsets denote inactive zero-query users.
    std::vector<std::int32_t> reused_tokens, query_offsets, active_requests;
    std::vector<CacheReuse> reuse;
    // Native PACKED_NHD update metadata, indexed over every allocated page.
    // Real rows address compact current QKV. -1 requests a Select-zero row.
    std::vector<std::int32_t> page_write_indices, page_update_rows, page_update_lengths;
    // Only active requests enter the original attention metadata.
    std::vector<std::int32_t> q_offsets, k_offsets, targets, page_indptrs, page_ids;
    // Optional original-kernel transport: INT32[5,active_requests+1,8].
    // Empty for zero total queries. Requires page128/capacity<=1024.
    std::vector<std::int32_t> packed_attention_metadata;
    std::int32_t total_queries() const { return query_offsets.empty() ? 0 : query_offsets.back(); }
    // Smallest nonempty contiguous page view containing every read and write.
    // Validates the full-page extent metadata; never changes storage, ownership,
    // page IDs, or update rows. Bind the existing pointer with this leading size.
    std::int32_t binding_pages() const;
};

// Private family primitive, not a stable public ABI. Each arena owns contiguous
// [layers,slots*pages_per_slot,2,page_size,heads*head_dim] storage. Each slot maps
// logical history pages to demand-assigned physical IDs, retained on append.
// A lease is an exclusive transaction, not an immutable public HistoryCache snapshot.
class PagedCacheArena {
    struct Impl;

  public:
    class Lease {
      public:
        Lease();
        ~Lease();
        Lease(Lease&&) noexcept;
        Lease& operator=(Lease&&) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        const PagedCachePlan& plan() const;
        void* layer_data(std::int32_t layer) const;
        std::uint64_t generation() const { return generation_; }

      private:
        friend class PagedCacheArena;
        std::shared_ptr<Impl> owner_;
        std::uint64_t generation_{0};
    };

    // Both factories perform real zero initialization. The host arena supports
    // native CPU staging/verification; it is not a mock CUDA implementation.
    static PagedCacheArena create_host(PagedCacheGeometry geometry);
    // The caller retains this stream and its executor until all arena leases
    // have finished. All bound kernels must execute on this same stream.
    static PagedCacheArena create_cuda(PagedCacheGeometry geometry, cudaStream_t stream);
    ~PagedCacheArena();
    PagedCacheArena(PagedCacheArena&&) noexcept;
    PagedCacheArena& operator=(PagedCacheArena&&) noexcept;
    PagedCacheArena(const PagedCacheArena&) = delete;
    PagedCacheArena& operator=(const PagedCacheArena&) = delete;

    const PagedCacheGeometry& geometry() const;
    std::int32_t pages_per_slot() const; // Logical capacity, not a physical slot stride.
    std::size_t nbytes() const;
    bool is_cuda() const;

    // Clear assigned destination pages, copy only compact validated history,
    // then complete the copy before establishing zero-tail trust. Compact KV
    // shape is [layers,2,heads,N,head_dim], preserving the current cache format.
    // Device sources follow HistoryCache's completed immutable-writer contract;
    // the primitive validates their extent/device, without a full D2H scan.
    void initialize_history(std::int32_t slot, std::string owner, HistorySignature signature,
                            const HistoryCacheTensor& compact_kv);
    // Unknown page restore preserves every source byte; it never certifies
    // unused tails. Source shape is [layers,pages_per_slot,2,page_size,H*D], in
    // logical slot-page order; all pages, including unused ones, are copied.
    void restore_unknown(std::int32_t slot, std::string owner, HistorySignature signature,
                         const HistoryCacheTensor& pages);
    // Reassignment/correction/truncation cannot reuse the prior prefix. A new
    // prepare recomputes that slot and uses safe tail-clearing metadata.
    void invalidate(std::int32_t slot);

    Lease prepare(const std::vector<PagedCacheRequest>& requests, const CachePolicy& policy);
    // Call only after the family executor and output validation succeed. These
    // methods synchronize CUDA work before advancing/revoking the generation.
    // A failed stream drain quarantines the arena; replace it before further use.
    // The admitted graph must perform exactly plan().page_update_* writes;
    // the primitive does not scan the full GPU arena on every request.
    void commit(Lease& lease);
    void fail(Lease& lease);
    // Copies history into the existing compact contract; never exposes a
    // mutable arena allocation through the public platform cache.
    HistoryCacheTensor copy_history(std::int32_t slot) const;

  private:
    explicit PagedCacheArena(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
};

} // namespace trtmc::hstu
