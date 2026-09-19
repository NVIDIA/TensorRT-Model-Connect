/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/history_cache.h"

#include "trtmc/runtime/device_tensor.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace trtmc {
namespace {

using Key = HistoryCacheKey;
using Value = HistoryCacheValue;

auto key_parts(const Key& key) {
    return std::tie(key.artifact_id, key.feature_version, key.subject_id, key.history_epoch);
}

struct KeyLess {
    bool operator()(const Key& left, const Key& right) const {
        return key_parts(left) < key_parts(right);
    }
};

std::size_t add_size(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left)
        throw std::length_error("history cache byte count overflow");
    return left + right;
}

std::size_t multiply_size(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
        throw std::length_error("history cache tensor size overflow");
    return left * right;
}

std::size_t tensor_bytes(const HistoryCacheTensor& tensor) {
    std::size_t count = 1;
    if (tensor.shape.empty() || dtype_size(tensor.dtype) == 0)
        throw std::invalid_argument("history cache tensor requires shape and valid dtype");
    for (const auto dimension : tensor.shape) {
        if (dimension < 0)
            throw std::invalid_argument("history cache tensor dimensions must be nonnegative");
        count = multiply_size(count, static_cast<std::size_t>(dimension));
    }
    return multiply_size(count, dtype_size(tensor.dtype));
}

void validate_device_tensor(const HistoryCacheTensor& tensor, std::size_t expected,
                            bool host_only) {
    if (host_only || tensor.device->shape() != tensor.shape ||
        tensor.device->dtype() != tensor.dtype || tensor.device->nbytes() != expected ||
        (expected != 0 && !tensor.device->ok()))
        throw std::invalid_argument("history cache device tensor does not match metadata");
}

std::size_t owned_tensor_bytes(const HistoryCacheTensor& tensor, bool host_only) {
    const auto expected = tensor_bytes(tensor);
    auto bytes = add_size(tensor.name.capacity(), tensor.host_data.capacity());
    bytes = add_size(bytes, multiply_size(tensor.shape.capacity(), sizeof(std::int64_t)));
    if (tensor.device) {
        validate_device_tensor(tensor, expected, host_only);
        if (!tensor.host_data.empty() && tensor.host_data.size() != expected)
            throw std::invalid_argument("history cache host mirror has incorrect size");
        bytes = add_size(bytes, add_size(sizeof(DeviceTensor), expected));
    } else if (tensor.host_data.size() != expected) {
        throw std::invalid_argument("history cache host tensor has incorrect size");
    }
    return bytes;
}

// Conservatively count owned container capacities, not only populated lengths.
// Shared device allocations count in each snapshot until that snapshot retires.
std::size_t value_bytes(const Value& value, bool host_only = false) {
    if (value.schema_version != kHistoryCacheInterfaceVersion || value.format.empty())
        throw std::invalid_argument("history cache value has unsupported schema or empty format");
    auto bytes = add_size(sizeof(Value), value.format.capacity());
    bytes = add_size(bytes, value.metadata.capacity());
    bytes = add_size(bytes, multiply_size(value.tensors.capacity(), sizeof(HistoryCacheTensor)));
    std::set<std::string> names;
    for (const auto& tensor : value.tensors) {
        if (tensor.name.empty() || !names.insert(tensor.name).second)
            throw std::invalid_argument("history cache tensor names must be nonempty and unique");
        bytes = add_size(bytes, owned_tensor_bytes(tensor, host_only));
    }
    return bytes;
}

Value host_snapshot(const Value& source) {
    Value host;
    host.schema_version = source.schema_version;
    host.format = source.format;
    host.metadata = source.metadata;
    host.tensors.reserve(source.tensors.size());
    for (const auto& tensor : source.tensors) {
        HistoryCacheTensor copy;
        copy.name = tensor.name;
        copy.shape = tensor.shape;
        copy.dtype = tensor.dtype;
        if (tensor.device) {
            copy.host_data.resize(tensor_bytes(tensor));
            if (!copy.host_data.empty() && !tensor.device->copy_to_host(copy.host_data.data()))
                throw std::runtime_error("history cache device-to-host copy failed");
        } else {
            copy.host_data = tensor.host_data;
        }
        host.tensors.push_back(std::move(copy));
    }
    return host;
}

struct Accounting {
    std::atomic<std::size_t> live_bytes{0};
};

std::shared_ptr<const Value> tracked_value(Value value, std::size_t bytes,
                                           const std::shared_ptr<Accounting>& accounting) {
    auto* raw = new Value(std::move(value));
    accounting->live_bytes.fetch_add(bytes);
    return std::shared_ptr<const Value>(raw, [accounting, bytes](const Value* snapshot) {
        delete snapshot;
        accounting->live_bytes.fetch_sub(bytes);
    });
}

struct Slot {
    std::uint64_t generation{0};
    std::shared_ptr<const Value> value;
    std::size_t value_bytes{0};
    std::size_t key_bytes{0};
    std::list<Key>::iterator lru;
};

std::size_t entry_bytes(const Key& key) {
    auto bytes = sizeof(Slot) + 2 * sizeof(Key) + 5 * sizeof(void*);
    for (const auto* part :
         {&key.artifact_id, &key.feature_version, &key.subject_id, &key.history_epoch})
        bytes = add_size(bytes, multiply_size(2, part->capacity()));
    return bytes;
}

// Callers hold their own mutex. Retired value accounting is atomic because a
// lease may release its last reference independently of that mutex.
struct Entries {
    std::size_t max_bytes;
    std::size_t max_entries;
    std::shared_ptr<Accounting> accounting{std::make_shared<Accounting>()};
    std::map<Key, Slot, KeyLess> slots;
    std::list<Key> lru;
    std::uint64_t next_generation{1};
    std::uint64_t evictions{0};

    Entries(std::size_t bytes, std::size_t entries) : max_bytes(bytes), max_entries(entries) {
        if (bytes == 0 || entries == 0)
            throw std::invalid_argument("history cache requires positive byte and entry budgets");
    }

    std::uint64_t generation() {
        if (next_generation == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("history cache generation exhausted");
        return next_generation++;
    }

    void touch(std::map<Key, Slot, KeyLess>::iterator slot) {
        lru.splice(lru.begin(), lru, slot->second.lru);
    }

    void erase(std::map<Key, Slot, KeyLess>::iterator slot) {
        accounting->live_bytes.fetch_sub(slot->second.key_bytes);
        lru.erase(slot->second.lru);
        slots.erase(slot);
    }

    void clear() {
        while (!slots.empty())
            erase(slots.begin());
    }

    std::map<Key, Slot, KeyLess>::iterator eviction_candidate(const Key* protected_key) {
        // Prefer an unpinned LRU entry; deleting a pinned entry is safe but
        // its snapshot continues consuming capacity until readers release it.
        for (bool allow_pinned : {false, true}) {
            for (auto cursor = lru.rbegin(); cursor != lru.rend(); ++cursor) {
                if (protected_key && *cursor == *protected_key)
                    continue;
                auto slot = slots.find(*cursor);
                if (allow_pinned || slot->second.value.use_count() <= 1)
                    return slot;
            }
        }
        return slots.end();
    }

    bool reserve(std::size_t bytes, std::size_t entries, const Key* protected_key = nullptr) {
        if (bytes > max_bytes || entries > max_entries)
            return false;
        while (accounting->live_bytes.load() > max_bytes - bytes ||
               slots.size() > max_entries - entries) {
            const auto victim = eviction_candidate(protected_key);
            if (victim == slots.end())
                return false;
            erase(victim);
            ++evictions;
        }
        return true;
    }

    std::map<Key, Slot, KeyLess>::iterator create(const Key& key) {
        const auto bytes = entry_bytes(key);
        if (!reserve(bytes, 1))
            return slots.end();
        const auto next = generation();
        lru.push_front(key);
        Slot slot;
        slot.generation = next;
        slot.key_bytes = bytes;
        slot.lru = lru.begin();
        try {
            auto inserted = slots.emplace(key, std::move(slot)).first;
            accounting->live_bytes.fetch_add(bytes);
            return inserted;
        } catch (...) {
            lru.pop_front();
            throw;
        }
    }

    std::size_t resident_bytes() const {
        std::size_t result = 0;
        for (const auto& [key, slot] : slots)
            result += slot.key_bytes + slot.value_bytes;
        return result;
    }

    bool admit_value(const Key& key, std::size_t bytes) {
        auto& slot = slots.at(key);
        if (slot.key_bytes > max_bytes || bytes > max_bytes - slot.key_bytes)
            return false;
        // Replacing an unpinned snapshot may reclaim its capacity immediately.
        // Readers of a pinned old version continue paying for that version.
        if (slot.value.use_count() == 1) {
            slot.value.reset();
            slot.value_bytes = 0;
        }
        return reserve(bytes, 0, &key);
    }
};

} // namespace

bool HistoryCacheKey::operator==(const HistoryCacheKey& other) const {
    return key_parts(*this) == key_parts(other);
}

struct InMemoryHistoryCacheStorage::Impl {
    std::mutex mutex;
    Entries entries;
    Impl(std::size_t bytes, std::size_t count) : entries(bytes, count) {}
};

InMemoryHistoryCacheStorage::InMemoryHistoryCacheStorage(std::size_t max_bytes,
                                                         std::size_t max_entries)
    : impl_(std::make_unique<Impl>(max_bytes, max_entries)) {}

InMemoryHistoryCacheStorage::~InMemoryHistoryCacheStorage() = default;

std::shared_ptr<const Value> InMemoryHistoryCacheStorage::load(const Key& key) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto slot = impl_->entries.slots.find(key);
    if (slot == impl_->entries.slots.end())
        return nullptr;
    impl_->entries.touch(slot);
    return slot->second.value;
}

void InMemoryHistoryCacheStorage::store(const Key& key, const Value& host_value) {
    auto owned = host_value;
    const auto bytes = value_bytes(owned, true);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& entries = impl_->entries;
    auto slot = entries.slots.find(key);
    if (slot == entries.slots.end())
        slot = entries.create(key);
    if (slot == entries.slots.end() || !entries.admit_value(key, bytes))
        throw std::length_error("history cache storage budget exhausted");
    slot->second.value = tracked_value(std::move(owned), bytes, entries.accounting);
    slot->second.value_bytes = bytes;
    slot->second.generation = entries.generation();
    entries.touch(slot);
}

void InMemoryHistoryCacheStorage::erase(const Key& key) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto slot = impl_->entries.slots.find(key);
    if (slot != impl_->entries.slots.end())
        impl_->entries.erase(slot);
}

struct HistoryCache::Impl {
    explicit Impl(HistoryCacheOptions settings)
        : options(std::move(settings)), entries(options.max_bytes, options.max_entries) {}

    HistoryCacheOptions options;
    mutable std::mutex mutex;
    // Serializes this manager's storage operations, never GPU copies or model
    // execution. Invalidations revoke generations before waiting for storage.
    std::mutex storage_mutex;
    Entries entries;
    HistoryCacheStats counters;
    std::size_t pending_invalidations{0};
    bool storage_reads_disabled{false};

    bool current(const Lease& lease) const {
        const auto found = entries.slots.find(lease.key);
        return lease.owner_.get() == entries.accounting.get() && lease.generation != 0 &&
               found != entries.slots.end() && found->second.generation == lease.generation;
    }

    bool can_load() const {
        return options.storage && pending_invalidations == 0 && !storage_reads_disabled;
    }

    void restore(Lease& lease) {
        std::lock_guard<std::mutex> storage_lock(storage_mutex);
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!current(lease) || !can_load())
                return;
        }
        try {
            const auto loaded = options.storage->load(lease.key);
            if (!loaded)
                return;
            auto owned = *loaded;
            const auto bytes = value_bytes(owned, true);
            std::lock_guard<std::mutex> lock(mutex);
            if (current(lease) && can_load() && entries.reserve(bytes, 0, &lease.key)) {
                auto& slot = entries.slots.at(lease.key);
                slot.value = tracked_value(std::move(owned), bytes, entries.accounting);
                slot.value_bytes = bytes;
                slot.generation = entries.generation();
                lease.generation = slot.generation;
                lease.value = slot.value;
                lease.source = HistoryCacheSource::kStorage;
                ++counters.storage_hits;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            ++counters.load_failures;
        }
    }
};

HistoryCache::HistoryCache(HistoryCacheOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

HistoryCache::~HistoryCache() = default;

HistoryCache::Lease HistoryCache::lookup(const Key& key) {
    Lease lease;
    lease.key = key;
    lease.owner_ = impl_->entries.accounting;
    bool load = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto slot = impl_->entries.slots.find(key);
        if (slot == impl_->entries.slots.end())
            slot = impl_->entries.create(key);
        if (slot == impl_->entries.slots.end()) {
            ++impl_->counters.misses;
            return lease;
        }
        impl_->entries.touch(slot);
        lease.generation = slot->second.generation;
        lease.value = slot->second.value;
        if (lease.value) {
            lease.source = HistoryCacheSource::kMemory;
            ++impl_->counters.hits;
            return lease;
        }
        load = impl_->can_load();
    }
    if (load)
        impl_->restore(lease);
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!lease.value)
            ++impl_->counters.misses;
    }
    return lease;
}

bool HistoryCache::publish(const Lease& lease, Value value) {
    std::size_t bytes = 0;
    try {
        bytes = value_bytes(value);
    } catch (...) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        ++impl_->counters.rejected_publications;
        return false;
    }
    Lease committed;
    committed.key = lease.key;
    committed.owner_ = lease.owner_;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->current(lease)) {
            ++impl_->counters.stale_publications;
            return false;
        }
        if (!impl_->entries.admit_value(lease.key, bytes)) {
            ++impl_->counters.rejected_publications;
            return false;
        }
        auto& slot = impl_->entries.slots.at(lease.key);
        slot.value = tracked_value(std::move(value), bytes, impl_->entries.accounting);
        slot.value_bytes = bytes;
        slot.generation = impl_->entries.generation();
        impl_->entries.touch(impl_->entries.slots.find(lease.key));
        committed.generation = slot.generation;
        committed.value = slot.value;
        ++impl_->counters.publications;
    }
    if (impl_->options.storage && impl_->options.write_through) {
        try {
            // Copies/synchronization can be expensive. Keep them outside both
            // the manager mutex and the storage serialization mutex.
            const auto host = host_snapshot(*committed.value);
            std::lock_guard<std::mutex> storage_lock(impl_->storage_mutex);
            bool current = false;
            {
                std::lock_guard<std::mutex> lock(impl_->mutex);
                current = impl_->current(committed);
            }
            if (current)
                impl_->options.storage->store(committed.key, host);
        } catch (...) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            ++impl_->counters.store_failures;
        }
    }
    return true;
}

void HistoryCache::invalidate(const Key& key) {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto slot = impl_->entries.slots.find(key);
        if (slot != impl_->entries.slots.end())
            impl_->entries.erase(slot);
        ++impl_->pending_invalidations;
    }
    bool failed = false;
    if (impl_->options.storage) {
        std::lock_guard<std::mutex> storage_lock(impl_->storage_mutex);
        try {
            impl_->options.storage->erase(key);
        } catch (...) {
            failed = true;
        }
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        --impl_->pending_invalidations;
        if (failed) {
            ++impl_->counters.erase_failures;
            impl_->storage_reads_disabled = true;
        }
    }
}

void HistoryCache::clear_memory() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->entries.clear();
}

HistoryCacheStats HistoryCache::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto result = impl_->counters;
    result.evictions = impl_->entries.evictions;
    result.resident_entries = impl_->entries.slots.size();
    result.resident_bytes = impl_->entries.resident_bytes();
    result.live_bytes = impl_->entries.accounting->live_bytes.load();
    return result;
}

} // namespace trtmc
