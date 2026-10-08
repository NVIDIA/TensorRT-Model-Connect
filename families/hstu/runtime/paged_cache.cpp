/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/paged_cache.h"

#include "families/hstu/runtime/attention_metadata.h"
#include "trtmc/runtime/device_tensor.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace trtmc::hstu {
namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("hstu paged cache: ") + cudaGetErrorString(status));
}

std::size_t product(const std::vector<std::int64_t>& shape, std::size_t width) {
    for (const auto dimension : shape) {
        require(dimension >= 0, "negative paged cache dimension");
        require(!width || static_cast<std::uint64_t>(dimension) <=
                              std::numeric_limits<std::size_t>::max() / width,
                "paged cache byte extent overflow");
        width *= static_cast<std::size_t>(dimension);
    }
    return width;
}

std::int32_t integer(std::size_t value) {
    require(value <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "paged cache metadata exceeds INT32");
    return static_cast<std::int32_t>(value);
}

struct Slot {
    std::string owner;
    HistorySignature signature;
    bool valid{false};
    bool zero_tail{true};
    std::vector<std::int32_t> pages;
};

struct Pending {
    PagedCacheRequest request;
    std::int32_t prefix{0}, queries{0}, added{0};
    bool zero_tail{false};
    std::vector<std::int32_t> pages;
};

struct Source {
    const std::uint8_t* data{nullptr};
    bool device{false};
    std::vector<std::uint8_t> staging;
};

void finite_host(const void* pointer, std::size_t count, DType dtype) {
    const auto* bytes = static_cast<const std::uint8_t*>(pointer);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, bytes + i * dtype_size(dtype), dtype_size(dtype));
        const auto mask = dtype == DType::kFloat32   ? 0x7f800000U
                          : dtype == DType::kFloat16 ? 0x7c00U
                                                     : 0x7f80U;
        require((bits & mask) != mask, "nonfinite live history in paged cache");
    }
}

} // namespace

std::int32_t PagedCachePlan::binding_pages() const {
    const auto pages = integer(page_write_indices.size());
    require(pages > 0 && page_update_lengths.size() == page_write_indices.size() + 1,
            "invalid full-page update metadata length");
    require(page_update_lengths.front() == 0 &&
                page_update_lengths.back() == integer(page_update_rows.size()),
            "invalid cumulative page update endpoints");
    std::int32_t bound = 1;
    for (std::int32_t page = 0; page < pages; ++page) {
        require(page_write_indices[page] >= 0, "negative page write offset");
        require(page_update_lengths[page] >= 0 &&
                    page_update_lengths[page] <= page_update_lengths[page + 1],
                "invalid cumulative page update counts");
        if (page_update_lengths[page] != page_update_lengths[page + 1])
            bound = page + 1;
    }
    for (const auto page : page_ids) {
        require(page >= 0 && page < pages, "attention page exceeds owned arena");
        bound = std::max(bound, page + 1);
    }
    return bound;
}

struct PagedCacheArena::Impl {
    PagedCacheGeometry geometry;
    std::int32_t pages_per_slot{0}, total_pages{0};
    std::size_t element_bytes{0}, token_bytes{0}, page_bytes{0}, layer_bytes{0}, bytes{0};
    std::vector<std::uint8_t> host;
    std::unique_ptr<DeviceTensor> device_storage;
    cudaStream_t stream{nullptr};
    int device{-1};
    mutable std::mutex mutex;
    std::vector<Slot> slots;
    // -1 is free. Clean free pages still contain the allocation's original zeros.
    // A failed stream drain quarantines the arena instead of recycling pages.
    std::vector<std::int32_t> page_owners;
    std::vector<bool> clean_pages;
    std::vector<Pending> pending;
    PagedCachePlan plan;
    std::uint64_t generation{0};
    bool inflight{false};
    mutable bool failed_sync{false};

    explicit Impl(PagedCacheGeometry value) : geometry(value) {
        const auto& g = geometry;
        require(g.layers > 0 && g.heads > 0 && g.head_dim > 0 && g.slots > 0 && g.page_size > 0 &&
                    g.max_history_tokens > 0 && g.max_bytes > 0,
                "positive paged cache geometry and explicit budget required");
        require(g.dtype == DType::kFloat32 || g.dtype == DType::kFloat16 ||
                    g.dtype == DType::kBFloat16,
                "paged cache requires floating point storage");
        pages_per_slot = integer(
            (static_cast<std::size_t>(g.max_history_tokens) + g.page_size - 1) / g.page_size);
        total_pages = integer(static_cast<std::size_t>(g.slots) * pages_per_slot);
        element_bytes = dtype_size(g.dtype);
        token_bytes = product({g.heads, g.head_dim}, element_bytes);
        page_bytes = product({2, g.page_size}, token_bytes);
        layer_bytes = product({total_pages}, page_bytes);
        bytes = product({g.layers}, layer_bytes);
        require(bytes <= g.max_bytes &&
                    bytes / element_bytes <=
                        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
                "paged cache allocation exceeds budget or tensor extent");
        slots.resize(g.slots);
        page_owners.assign(total_pages, -1);
        clean_pages.assign(total_pages, true);
    }

    ~Impl() {
        if (!device_storage)
            return;
        int previous = -1;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device);
        (void)cudaStreamSynchronize(stream);
        device_storage.reset();
        if (previous >= 0 && previous != device)
            (void)cudaSetDevice(previous);
    }

    std::uint8_t* data() const {
        return device_storage ? static_cast<std::uint8_t*>(device_storage->data())
                              : const_cast<std::uint8_t*>(host.data());
    }

    void current_device() const {
        if (!device_storage)
            return;
        int current = -1;
        cuda_check(cudaGetDevice(&current));
        require(current == device, "paged cache belongs to another CUDA device");
    }

    void sync() const {
        if (device_storage) {
            current_device();
            const auto status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess)
                failed_sync = true;
            cuda_check(status);
        }
    }

    void idle() const {
        require(!inflight, "paged cache lease is in flight");
        require(!failed_sync, "paged cache arena is quarantined after failed stream drain");
    }
    void check(std::uint64_t value) const {
        require(inflight && generation == value, "stale or consumed paged cache lease");
    }
    void check_slot(std::int32_t slot) const {
        require(slot >= 0 && slot < geometry.slots, "invalid paged cache slot");
    }
    void advance() {
        require(generation != std::numeric_limits<std::uint64_t>::max(),
                "paged cache generation exhausted");
        ++generation;
    }

    void validate_signature(const std::string& owner, const HistorySignature& signature) const {
        require(!owner.empty(), "paged cache owner identity is required");
        require(signature.token_rows.size() <=
                    static_cast<std::size_t>(geometry.max_history_tokens),
                "history exceeds paged cache capacity");
        require(plan_cache_reuse(nullptr, signature, {}).reason !=
                    CacheReuseReason::kInvalidSignature,
                "invalid paged cache history signature");
    }

    Source source(const HistoryCacheTensor& value, const std::vector<std::int64_t>& shape) const {
        const auto size = product(shape, element_bytes);
        require(value.shape == shape && value.dtype == geometry.dtype,
                "paged cache source shape/dtype mismatch");
        Source result;
        if (!value.device) {
            require(value.host_data.size() == size, "paged cache source byte count mismatch");
            result.data = value.host_data.data();
            return result;
        }
        require(value.device->ok() && value.device->shape() == shape &&
                    value.device->dtype() == geometry.dtype && value.device->nbytes() == size,
                "invalid paged cache source device tensor");
        require(value.host_data.empty() || value.host_data.size() == size,
                "invalid paged cache source host mirror");
        if (!device_storage) {
            result.staging.resize(size);
            require(value.device->copy_to_host(result.staging.data()),
                    "paged history download failed");
            result.data = result.staging.data();
        } else {
            cudaPointerAttributes attributes{};
            cuda_check(cudaPointerGetAttributes(&attributes, value.device->data()));
            require(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
                    "paged cache source belongs to another device");
            result.data = static_cast<const std::uint8_t*>(value.device->data());
            result.device = true;
        }
        return result;
    }

    void copy_rows(void* dst, std::size_t dst_pitch, const void* src, std::size_t src_pitch,
                   std::size_t width, std::size_t count, bool source_device) const {
        if (!count || !width)
            return;
        if (device_storage) {
            cuda_check(cudaMemcpy2DAsync(
                dst, dst_pitch, src, src_pitch, width, count,
                source_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));
        } else {
            for (std::size_t row = 0; row < count; ++row)
                std::memcpy(static_cast<std::uint8_t*>(dst) + row * dst_pitch,
                            static_cast<const std::uint8_t*>(src) + row * src_pitch, width);
        }
    }

    std::vector<std::int32_t> reserve_pages(std::int32_t slot, std::size_t count,
                                            std::vector<std::int32_t>& owners) const {
        require(count <= static_cast<std::size_t>(pages_per_slot), "slot page capacity exceeded");
        auto result = slots[slot].pages;
        result.resize(std::min(result.size(), count));
        result.reserve(count);
        for (std::int32_t page = 0; page < total_pages && result.size() < count; ++page) {
            if (owners[page] != -1)
                continue;
            owners[page] = slot;
            result.push_back(page);
        }
        require(result.size() == count, "paged cache has no free physical pages");
        return result;
    }

    void clear_pages(const std::vector<std::int32_t>& pages) const {
        for (std::int32_t layer = 0; layer < geometry.layers; ++layer)
            for (std::size_t first = 0; first < pages.size();) {
                auto end = first + 1;
                while (end < pages.size() && pages[end] == pages[end - 1] + 1)
                    ++end;
                auto* destination = data() + layer * layer_bytes + pages[first] * page_bytes;
                const auto size = (end - first) * page_bytes;
                if (device_storage)
                    cuda_check(cudaMemsetAsync(destination, 0, size, stream));
                else
                    std::memset(destination, 0, size);
                first = end;
            }
    }

    std::size_t page_offset(std::int32_t layer, const std::vector<std::int32_t>& pages,
                            std::int32_t kind, std::size_t token, std::int32_t head) const {
        return layer * layer_bytes + pages.at(token / geometry.page_size) * page_bytes +
               (kind * geometry.page_size + token % geometry.page_size) * token_bytes +
               head * geometry.head_dim * element_bytes;
    }

    void copy_compact_in(const std::vector<std::int32_t>& pages, std::size_t length,
                         const Source& value) const {
        const auto width = geometry.head_dim * element_bytes;
        for (std::int32_t layer = 0; layer < geometry.layers; ++layer)
            for (std::int32_t kind = 0; kind < 2; ++kind)
                for (std::int32_t head = 0; head < geometry.heads; ++head)
                    for (std::size_t token = 0; token < length; token += geometry.page_size) {
                        const auto count =
                            std::min<std::size_t>(geometry.page_size, length - token);
                        const auto offset =
                            ((static_cast<std::size_t>(layer) * 2 + kind) * geometry.heads + head) *
                            length * width;
                        copy_rows(data() + page_offset(layer, pages, kind, token, head),
                                  token_bytes, value.data + offset + token * width, width, width,
                                  count, value.device);
                    }
    }

    void copy_pages_in(const std::vector<std::int32_t>& pages, const Source& value) const {
        for (std::int32_t layer = 0; layer < geometry.layers; ++layer)
            for (std::size_t page = 0; page < pages.size(); ++page)
                copy_rows(data() + layer * layer_bytes + pages[page] * page_bytes, page_bytes,
                          value.data + (static_cast<std::size_t>(layer) * pages_per_slot + page) *
                                           page_bytes,
                          page_bytes, page_bytes, 1, value.device);
    }

    bool new_pages_clean(std::int32_t slot, const std::vector<std::int32_t>& pages) const {
        for (std::size_t index = slots[slot].pages.size(); index < pages.size(); ++index)
            if (!clean_pages[pages[index]])
                return false;
        return true;
    }

    void begin_restore(std::int32_t slot, const std::vector<std::int32_t>& pages,
                       std::vector<std::int32_t>& owners) {
        advance();
        slots[slot].valid = false;
        slots[slot].zero_tail = false;
        page_owners.swap(owners);
        for (const auto page : pages)
            clean_pages[page] = false;
    }

    void release_new_pages(std::int32_t slot, const std::vector<std::int32_t>& pages) {
        if (failed_sync)
            return;
        for (std::size_t index = slots[slot].pages.size(); index < pages.size(); ++index)
            page_owners[pages[index]] = -1;
    }

    void install_pages(std::int32_t slot, std::vector<std::int32_t> pages) {
        for (std::size_t index = pages.size(); index < slots[slot].pages.size(); ++index)
            page_owners[slots[slot].pages[index]] = -1;
        slots[slot].pages = std::move(pages);
    }

    void failed_restore(std::int32_t slot, const std::vector<std::int32_t>& pages) noexcept {
        try {
            sync();
        } catch (...) {
            failed_sync = true;
        }
        release_new_pages(slot, pages);
    }

    void publish_plan(PagedCachePlan value, std::vector<Pending> items,
                      std::vector<std::int32_t> owners) {
        advance();
        for (std::int32_t page = 0; page < total_pages; ++page)
            if (value.page_update_lengths[page] != value.page_update_lengths[page + 1])
                clean_pages[page] = false;
        page_owners.swap(owners);
        plan = std::move(value);
        pending = std::move(items);
        inflight = true;
    }

    void invalidate_pending() {
        for (const auto& item : pending) {
            auto& slot = slots[item.request.slot];
            slot.valid = false;
            slot.zero_tail = false;
            release_new_pages(item.request.slot, item.pages);
        }
        inflight = false;
        pending.clear();
    }

    void finish(std::uint64_t value, bool success) {
        std::lock_guard<std::mutex> lock(mutex);
        check(value);
        try {
            sync();
        } catch (...) {
            failed_sync = true;
            invalidate_pending();
            throw;
        }
        if (!success) {
            invalidate_pending();
            return;
        }
        for (auto& item : pending) {
            auto& slot = slots[item.request.slot];
            install_pages(item.request.slot, std::move(item.pages));
            slot.owner = std::move(item.request.owner);
            slot.signature = std::move(item.request.history);
            slot.valid = true;
            slot.zero_tail = item.zero_tail;
        }
        pending.clear();
        inflight = false;
    }

    void abandon(std::uint64_t value) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (!inflight || value != generation)
                return;
            try {
                sync();
            } catch (...) {
                failed_sync = true;
            }
            invalidate_pending();
        } catch (...) {
        }
    }
};

namespace {

void pack_attention(PagedCachePlan& plan, const PagedCacheGeometry& geometry) {
    if (plan.active_requests.empty() || geometry.page_size != 128 ||
        geometry.max_history_tokens > 1024)
        return;
    const auto length = plan.active_requests.size() + 1;
    const auto stride = 8 * length;
    plan.packed_attention_metadata.assign(5 * stride, 0);
    const std::vector<std::int32_t>* vectors[] = {&plan.q_offsets, &plan.k_offsets, &plan.targets,
                                                  &plan.page_indptrs, &plan.page_ids};
    for (std::size_t plane = 0; plane < 5; ++plane) {
        require(vectors[plane]->size() <= stride, "paged attention metadata plane overflow");
        std::copy(vectors[plane]->begin(), vectors[plane]->end(),
                  plan.packed_attention_metadata.begin() + plane * stride);
    }
    fill_attention_page_lengths(plan.packed_attention_metadata.data(), plan.active_requests.size(),
                                plan.packed_attention_metadata.size());
}

void plan_user(PagedCachePlan& plan, const Pending& item, std::size_t request_index,
               std::int32_t page_size, std::vector<std::vector<std::int32_t>>& page_rows) {
    const auto first_query = plan.query_offsets.back();
    plan.query_offsets.push_back(integer(static_cast<std::size_t>(first_query) + item.queries));
    if (!item.queries)
        return;
    plan.active_requests.push_back(integer(request_index));
    plan.q_offsets.push_back(plan.query_offsets.back());
    plan.k_offsets.push_back(
        integer(static_cast<std::size_t>(plan.k_offsets.back()) + item.request.total_tokens));
    plan.targets.push_back(item.queries - item.added);
    const auto history = item.request.history.token_rows.size();
    const auto used = (history + page_size - 1) / page_size;
    for (std::size_t page = 0; page < used; ++page)
        plan.page_ids.push_back(item.pages.at(page));
    plan.page_indptrs.push_back(integer(plan.page_ids.size()));
    const auto stop = item.zero_tail ? history : used * page_size;
    for (std::size_t token = item.prefix; token < stop; ++token) {
        const auto page = item.pages.at(token / page_size);
        auto& rows = page_rows[page];
        if (rows.empty())
            plan.page_write_indices[page] = static_cast<std::int32_t>(token % page_size);
        rows.push_back(token < history ? integer(first_query + token - item.prefix) : -1);
    }
}

} // namespace

PagedCacheArena::PagedCacheArena(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
PagedCacheArena::~PagedCacheArena() = default;
PagedCacheArena::PagedCacheArena(PagedCacheArena&&) noexcept = default;
PagedCacheArena& PagedCacheArena::operator=(PagedCacheArena&&) noexcept = default;

PagedCacheArena PagedCacheArena::create_host(PagedCacheGeometry geometry) {
    auto impl = std::make_shared<Impl>(geometry);
    impl->host.resize(impl->bytes, 0);
    return PagedCacheArena(std::move(impl));
}

PagedCacheArena PagedCacheArena::create_cuda(PagedCacheGeometry geometry, cudaStream_t stream) {
    auto impl = std::make_shared<Impl>(geometry);
    impl->stream = stream;
    cuda_check(cudaGetDevice(&impl->device));
    impl->device_storage = std::make_unique<DeviceTensor>(
        std::vector<std::int64_t>{geometry.layers, impl->total_pages, 2, geometry.page_size,
                                  static_cast<std::int64_t>(geometry.heads) * geometry.head_dim},
        geometry.dtype, stream);
    require(impl->device_storage->ok(), "paged cache allocation failed");
    cuda_check(cudaMemsetAsync(impl->data(), 0, impl->bytes, stream));
    impl->sync();
    return PagedCacheArena(std::move(impl));
}

const PagedCacheGeometry& PagedCacheArena::geometry() const {
    return impl_->geometry;
}
std::int32_t PagedCacheArena::pages_per_slot() const {
    return impl_->pages_per_slot;
}
std::size_t PagedCacheArena::nbytes() const {
    return impl_->bytes;
}
bool PagedCacheArena::is_cuda() const {
    return bool(impl_->device_storage);
}

void PagedCacheArena::initialize_history(std::int32_t slot, std::string owner,
                                         HistorySignature signature,
                                         const HistoryCacheTensor& compact) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->idle();
    impl_->check_slot(slot);
    impl_->current_device();
    impl_->validate_signature(owner, signature);
    const auto& g = impl_->geometry;
    const auto length = signature.token_rows.size();
    auto source = impl_->source(
        compact, {g.layers, 2, g.heads, static_cast<std::int64_t>(length), g.head_dim});
    if (!source.device)
        finite_host(source.data, product(compact.shape, 1), g.dtype);
    auto owners = impl_->page_owners;
    auto pages = impl_->reserve_pages(slot, (length + g.page_size - 1) / g.page_size, owners);
    auto& saved = impl_->slots[slot];
    impl_->begin_restore(slot, pages, owners);
    try {
        impl_->clear_pages(pages);
        impl_->copy_compact_in(pages, length, source);
        impl_->sync();
    } catch (...) {
        impl_->failed_restore(slot, pages);
        throw;
    }
    impl_->install_pages(slot, std::move(pages));
    saved.owner = std::move(owner);
    saved.signature = std::move(signature);
    saved.valid = true;
    saved.zero_tail = true;
}

void PagedCacheArena::restore_unknown(std::int32_t slot, std::string owner,
                                      HistorySignature signature, const HistoryCacheTensor& pages) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->idle();
    impl_->check_slot(slot);
    impl_->current_device();
    impl_->validate_signature(owner, signature);
    const auto& g = impl_->geometry;
    auto source = impl_->source(pages, {g.layers, impl_->pages_per_slot, 2, g.page_size,
                                        static_cast<std::int64_t>(g.heads) * g.head_dim});
    if (!source.device) {
        const auto layer_bytes = impl_->pages_per_slot * impl_->page_bytes;
        for (std::int32_t layer = 0; layer < g.layers; ++layer)
            for (std::size_t token = 0; token < signature.token_rows.size(); ++token)
                for (std::int32_t kind = 0; kind < 2; ++kind) {
                    const auto offset =
                        layer * layer_bytes + (token / g.page_size) * impl_->page_bytes +
                        (kind * g.page_size + token % g.page_size) * impl_->token_bytes;
                    finite_host(source.data + offset, impl_->token_bytes / impl_->element_bytes,
                                g.dtype);
                }
    }
    auto owners = impl_->page_owners;
    auto assigned = impl_->reserve_pages(slot, impl_->pages_per_slot, owners);
    impl_->begin_restore(slot, assigned, owners);
    try {
        impl_->copy_pages_in(assigned, source);
        impl_->sync();
    } catch (...) {
        impl_->failed_restore(slot, assigned);
        throw;
    }
    impl_->install_pages(slot, std::move(assigned));
    auto& saved = impl_->slots[slot];
    saved.owner = std::move(owner);
    saved.signature = std::move(signature);
    saved.valid = true;
}

void PagedCacheArena::invalidate(std::int32_t slot) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->idle();
    impl_->check_slot(slot);
    impl_->advance();
    impl_->slots[slot].valid = false;
    // A fresh (or history-empty) verified slot has never stored any history or
    // candidate KV. Invalidating its identity does not change its all-zero bytes.
    // Failed writes already set zero_tail=false, including an empty old prefix.
    impl_->slots[slot].zero_tail =
        impl_->slots[slot].zero_tail && impl_->slots[slot].signature.token_rows.empty();
}

PagedCacheArena::Lease PagedCacheArena::prepare(const std::vector<PagedCacheRequest>& requests,
                                                const CachePolicy& policy) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->idle();
    impl_->current_device();
    require(!requests.empty(), "paged cache requires at least one request");
    std::vector<bool> used(impl_->geometry.slots, false);
    auto owners = impl_->page_owners;
    std::vector<Pending> pending;
    PagedCachePlan plan;
    plan.query_offsets = plan.q_offsets = plan.k_offsets = plan.page_indptrs = {0};
    plan.page_write_indices.resize(impl_->total_pages, 0);
    std::vector<std::vector<std::int32_t>> page_rows(impl_->total_pages);
    for (const auto& request : requests) {
        impl_->check_slot(request.slot);
        require(!used[request.slot], "one request per paged cache slot required");
        used[request.slot] = true;
        impl_->validate_signature(request.owner, request.history);
        require(request.total_tokens >= 0 &&
                    request.history.token_rows.size() <=
                        static_cast<std::size_t>(request.total_tokens) &&
                    request.total_tokens <= impl_->geometry.max_history_tokens,
                "invalid complete request length");
        const auto& slot = impl_->slots[request.slot];
        const auto* cached = slot.valid && slot.owner == request.owner ? &slot.signature : nullptr;
        auto reuse = plan_cache_reuse(cached, request.history, policy);
        auto assigned = impl_->reserve_pages(
            request.slot,
            (request.history.token_rows.size() + impl_->geometry.page_size - 1) /
                impl_->geometry.page_size,
            owners);
        Pending item{request,
                     integer(reuse.reused_tokens),
                     request.total_tokens - integer(reuse.reused_tokens),
                     integer(request.history.token_rows.size() - reuse.reused_tokens),
                     slot.zero_tail &&
                         (reuse.reused_tokens != 0 || slot.signature.token_rows.empty()) &&
                         impl_->new_pages_clean(request.slot, assigned),
                     std::move(assigned)};
        plan.reused_tokens.push_back(item.prefix);
        plan.reuse.push_back(reuse);
        plan_user(plan, item, pending.size(), impl_->geometry.page_size, page_rows);
        pending.push_back(std::move(item));
    }
    plan.page_update_lengths.push_back(0);
    for (const auto& rows : page_rows) {
        plan.page_update_rows.insert(plan.page_update_rows.end(), rows.begin(), rows.end());
        plan.page_update_lengths.push_back(integer(plan.page_update_rows.size()));
    }
    pack_attention(plan, impl_->geometry);
    impl_->publish_plan(std::move(plan), std::move(pending), std::move(owners));
    Lease lease;
    lease.owner_ = impl_;
    lease.generation_ = impl_->generation;
    return lease;
}

void PagedCacheArena::commit(Lease& lease) {
    require(lease.owner_ == impl_, "foreign paged cache lease");
    impl_->finish(lease.generation_, true);
    lease.owner_.reset();
}
void PagedCacheArena::fail(Lease& lease) {
    require(lease.owner_ == impl_, "foreign paged cache lease");
    impl_->finish(lease.generation_, false);
    lease.owner_.reset();
}

PagedCacheArena::Lease::Lease() = default;
PagedCacheArena::Lease::~Lease() {
    if (owner_)
        owner_->abandon(generation_);
}
PagedCacheArena::Lease::Lease(Lease&& other) noexcept = default;
PagedCacheArena::Lease& PagedCacheArena::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        if (owner_)
            owner_->abandon(generation_);
        owner_ = std::move(other.owner_);
        generation_ = other.generation_;
    }
    return *this;
}
const PagedCachePlan& PagedCacheArena::Lease::plan() const {
    require(bool(owner_), "empty paged cache lease");
    std::lock_guard<std::mutex> lock(owner_->mutex);
    owner_->check(generation_);
    return owner_->plan;
}
void* PagedCacheArena::Lease::layer_data(std::int32_t layer) const {
    require(bool(owner_), "empty paged cache lease");
    std::lock_guard<std::mutex> lock(owner_->mutex);
    owner_->check(generation_);
    owner_->current_device();
    require(layer >= 0 && layer < owner_->geometry.layers, "invalid paged cache layer");
    return owner_->data() + layer * owner_->layer_bytes;
}

HistoryCacheTensor PagedCacheArena::copy_history(std::int32_t slot) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->idle();
    impl_->check_slot(slot);
    impl_->current_device();
    require(impl_->slots[slot].valid, "cannot snapshot invalid paged history");
    const auto& g = impl_->geometry;
    const auto length = impl_->slots[slot].signature.token_rows.size();
    HistoryCacheTensor result;
    result.name = "history_kv";
    result.dtype = g.dtype;
    result.shape = {g.layers, 2, g.heads, static_cast<std::int64_t>(length), g.head_dim};
    const auto size = product(result.shape, impl_->element_bytes);
    if (!size)
        return result;
    std::shared_ptr<DeviceTensor> device;
    if (impl_->device_storage) {
        device = std::make_shared<DeviceTensor>(result.shape, result.dtype, impl_->stream);
        require(device->ok(), "paged history snapshot allocation failed");
    } else {
        result.host_data.resize(size);
    }
    auto* destination =
        device ? static_cast<std::uint8_t*>(device->data()) : result.host_data.data();
    const auto width = g.head_dim * impl_->element_bytes;
    try {
        for (std::int32_t layer = 0; layer < g.layers; ++layer)
            for (std::int32_t kind = 0; kind < 2; ++kind)
                for (std::int32_t head = 0; head < g.heads; ++head)
                    for (std::size_t token = 0; token < length; token += g.page_size) {
                        const auto offset =
                            ((static_cast<std::size_t>(layer) * 2 + kind) * g.heads + head) *
                            length * width;
                        impl_->copy_rows(
                            destination + offset + token * width, width,
                            impl_->data() + impl_->page_offset(layer, impl_->slots[slot].pages,
                                                               kind, token, head),
                            impl_->token_bytes, width,
                            std::min<std::size_t>(g.page_size, length - token), bool(device));
                    }
        impl_->sync();
    } catch (...) {
        try {
            impl_->sync();
        } catch (...) {
        }
        throw;
    }
    result.device = std::move(device);
    return result;
}

} // namespace trtmc::hstu
