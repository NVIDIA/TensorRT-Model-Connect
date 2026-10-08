/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/history_cache.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace trtmc;
using namespace std::chrono_literals;

std::size_t checks = 0;

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
    ++checks;
}

void rejects(const std::function<void()>& operation, const char* message) {
    bool threw = false;
    try {
        operation();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, message);
}

HistoryCacheKey key(std::string subject = "user") {
    return {"artifact-v1", "features-v2", std::move(subject), "history-v3"};
}

HistoryCacheValue value(std::uint8_t marker = 7, std::size_t size = 32) {
    HistoryCacheValue result;
    result.format = "test-native-context-v1";
    result.metadata = {marker, 1};
    HistoryCacheTensor tensor;
    tensor.name = "context";
    tensor.shape = {static_cast<std::int64_t>(size)};
    tensor.dtype = DType::kInt8;
    tensor.host_data.assign(size, marker);
    result.tensors.push_back(std::move(tensor));
    return result;
}

class Gate {
  public:
    void enable() {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = true;
    }

    void pause() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!enabled_)
            return;
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [&] { return released_; });
    }

    bool entered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, 5s, [&] { return entered_; });
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool enabled_{false};
    bool entered_{false};
    bool released_{false};
};

// A controlled test adapter around the real bounded native storage. Gates make
// invalidation races deterministic; this is not a production storage backend.
class ControlledStorage final : public IHistoryCacheStorage {
  public:
    InMemoryHistoryCacheStorage backing{1 << 20};
    Gate load_gate;
    Gate store_gate;
    bool fail_load{false};
    bool fail_store{false};
    bool fail_erase{false};

    std::shared_ptr<const HistoryCacheValue> load(const HistoryCacheKey& item) override {
        if (fail_load)
            throw std::runtime_error("injected load failure");
        auto loaded = backing.load(item);
        load_gate.pause();
        return loaded;
    }

    void store(const HistoryCacheKey& item, const HistoryCacheValue& snapshot) override {
        store_gate.pause();
        if (fail_store)
            throw std::runtime_error("injected store failure");
        backing.store(item, snapshot);
    }

    void erase(const HistoryCacheKey& item) override {
        if (fail_erase)
            throw std::runtime_error("injected erase failure");
        backing.erase(item);
    }
};

bool wait_for_empty(HistoryCache& cache) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
        if (cache.stats().resident_entries == 0)
            return true;
        std::this_thread::yield();
    }
    return false;
}

void test_identity_and_cas() {
    rejects([] { HistoryCache invalid({}); }, "zero budget must fail");
    rejects([] { InMemoryHistoryCacheStorage invalid(1024, 0); }, "zero entries must fail");
    HistoryCache cache({1 << 20, 8, nullptr, false});
    const auto original_key = key();
    const auto first = cache.lookup(original_key);
    const auto racing = cache.lookup(original_key);
    check(!first.value && first.source == HistoryCacheSource::kMiss, "first lookup is miss");
    check(first.generation == racing.generation && first.generation != 0,
          "concurrent misses share current slot generation");
    HistoryCache other({1 << 20, 8, nullptr, false});
    const auto foreign = other.lookup(original_key);
    check(!cache.publish(foreign, value()), "lease from another manager cannot publish");
    auto owned = value();
    check(cache.publish(first, owned), "first writer publishes");
    owned.metadata[0] = 99;
    owned.tensors[0].host_data[0] = 99;
    check(!cache.publish(racing, value(8)), "second writer must fail CAS");
    const auto hit = cache.lookup(original_key);
    check(hit.source == HistoryCacheSource::kMemory && hit.value->metadata[0] == 7,
          "publication snapshots own immutable metadata");
    check(hit.value->tensors[0].host_data[0] == 7, "publication snapshots own host payload");
    check(hit.generation != first.generation, "publication consumes generation");
    check(cache.publish(hit, value(9)), "fresh hit lease can replace value");
    check(hit.value->metadata[0] == 7, "replaced snapshot remains readable");

    for (int component = 0; component < 4; ++component) {
        auto different = original_key;
        std::string* fields[] = {&different.artifact_id, &different.feature_version,
                                 &different.subject_id, &different.history_epoch};
        *fields[component] += "-different";
        check(different != original_key && !cache.lookup(different).value,
              "each canonical identity component partitions cache");
    }
    cache.invalidate(original_key);
    check(!cache.publish(hit, value()), "invalidated lease cannot republish");
    check(!cache.lookup(original_key).value, "invalidated key misses");
    check(cache.stats().stale_publications == 3, "stale writes are observable");
}

void test_lru_and_pinned_budget() {
    HistoryCache cache({1 << 20, 2, nullptr, false});
    check(cache.publish(cache.lookup(key("a")), value(1)), "publish LRU a");
    check(cache.publish(cache.lookup(key("b")), value(2)), "publish LRU b");
    check(cache.lookup(key("a")).value != nullptr, "touch a");
    check(cache.publish(cache.lookup(key("c")), value(3)), "publish c evicts oldest b");
    check(cache.lookup(key("a")).value != nullptr, "recent a retained");
    check(cache.lookup(key("c")).value != nullptr, "new c retained");
    check(!cache.lookup(key("b")).value, "least recently used b evicted");
    check(cache.stats().resident_entries <= 2, "entry bound includes miss reservations");
    check(cache.stats().evictions > 0, "evictions counted");
    for (int index = 0; index < 1000; ++index) {
        auto item = key(std::to_string(index));
        const auto expired = cache.lookup(item);
        cache.invalidate(item);
        check(!cache.publish(expired, value()), "generation cannot survive erase and reuse");
    }
    check(cache.stats().resident_entries <= 2, "invalidations do not accumulate tombstones");

    HistoryCache bounded({8192, 8, nullptr, false});
    check(bounded.publish(bounded.lookup(key("large")), value(4, 4096)), "publish large entry");
    auto pinned = bounded.lookup(key("large"));
    bounded.clear_memory();
    check(bounded.stats().resident_entries == 0 && bounded.stats().live_bytes >= 4096,
          "retired pinned snapshot still consumes budget");
    check(pinned.value->tensors[0].host_data[0] == 4, "clear keeps reader snapshot alive");
    auto replacement = bounded.lookup(key("replacement"));
    check(!bounded.publish(replacement, value(5, 4096)), "pinned bytes prevent oversubscription");
    pinned.value.reset();
    check(bounded.publish(replacement, value(5, 4096)), "released pin returns capacity");
    check(bounded.stats().live_bytes <= 8192, "live byte budget remains bounded");
    check(!bounded.publish(bounded.lookup(key("oversize")), value(6, 8192)),
          "oversize payload is rejected");
}

void test_storage_roundtrip() {
    auto storage = std::make_shared<InMemoryHistoryCacheStorage>(1 << 20, 4);
    HistoryCache cache({1 << 20, 2, storage, true});
    auto lease = cache.lookup(key());
    check(cache.publish(lease, value(11)), "write-through publication succeeds");
    const auto lower = storage->load(key());
    check(lower && lower->metadata[0] == 11 && !lower->tensors[0].device,
          "lower-tier snapshot is host-only");
    cache.clear_memory();
    check(!cache.publish(lease, value()), "clear revokes prior generation");
    auto restored = cache.lookup(key());
    check(restored.source == HistoryCacheSource::kStorage && restored.value->metadata[0] == 11,
          "clear_memory preserves storage and restores snapshot");
    check(cache.stats().storage_hits == 1, "storage hit is observable");
    cache.invalidate(key());
    check(!storage->load(key()) && !cache.lookup(key()).value, "invalidate erases both tiers");
    check(restored.value->metadata[0] == 11, "invalidated pinned lower snapshot remains readable");

    HistoryCache read_only({1 << 20, 2, storage, false});
    check(read_only.publish(read_only.lookup(key("local")), value(12)), "local publish succeeds");
    check(!storage->load(key("local")), "write-through false does not publish lower copies");
    storage->store(key("a"), value(1));
    storage->store(key("b"), value(2));
    storage->store(key("c"), value(3));
    storage->store(key("d"), value(4));
    check(storage->load(key("a")) != nullptr, "touch storage a");
    storage->store(key("e"), value(5));
    check(!storage->load(key("b")), "native storage obeys entry LRU");
    check(storage->load(key("a")) != nullptr, "native storage retains recent entry");
}

void test_storage_pins_and_failures() {
    InMemoryHistoryCacheStorage bounded(8192);
    bounded.store(key(), value(1, 4096));
    bounded.store(key(), value(2, 4096));
    check(bounded.load(key())->metadata[0] == 2,
          "unpinned storage replacement can reclaim old value capacity");
    auto pinned = bounded.load(key());
    bounded.erase(key());
    rejects([&] { bounded.store(key("next"), value(2, 4096)); },
            "storage pins continue consuming budget after erase");
    pinned.reset();
    bounded.store(key("next"), value(2, 4096));
    check(bounded.load(key("next")) != nullptr, "storage pin release returns capacity");

    auto storage = std::make_shared<ControlledStorage>();
    storage->fail_load = true;
    HistoryCache cache({1 << 20, 8, storage, true});
    auto miss = cache.lookup(key());
    check(!miss.value && cache.stats().load_failures == 1, "load failure falls back to miss");
    storage->fail_store = true;
    check(cache.publish(miss, value()), "write-through error does not fail native publication");
    check(cache.stats().store_failures == 1 && cache.lookup(key()).value,
          "write-through error is counted while local value remains usable");
    storage->fail_load = false;
    storage->backing.store(key(), value(88));
    storage->fail_erase = true;
    cache.invalidate(key());
    check(cache.stats().erase_failures == 1, "erase error counted");
    check(!cache.lookup(key()).value, "failed erase cannot resurrect stale lower snapshot");
    cache.clear_memory();
    check(!cache.lookup(key()).value, "clear cannot undo failed-erase safety barrier");
}

void test_invalidation_races() {
    auto storage = std::make_shared<ControlledStorage>();
    storage->backing.store(key(), value(31));
    storage->load_gate.enable();
    HistoryCache cache({1 << 20, 8, storage, true});
    auto loading = std::async(std::launch::async, [&] { return cache.lookup(key()); });
    const auto load_entered = storage->load_gate.entered();
    auto invalidating = std::async(std::launch::async, [&] { cache.invalidate(key()); });
    const auto revoked = wait_for_empty(cache);
    storage->load_gate.release();
    auto loaded = loading.get();
    invalidating.get();
    check(load_entered && revoked, "invalidate revokes slot while old storage load is pending");
    check(!loaded.value, "in-flight stale load does not become a hit");
    check(!cache.publish(loaded, value()), "in-flight stale load lease cannot publish");
    check(!cache.lookup(key()).value, "invalidated storage value stays absent");

    storage->store_gate.enable();
    auto before = cache.lookup(key());
    auto publishing =
        std::async(std::launch::async, [&] { return cache.publish(before, value(32)); });
    const auto store_entered = storage->store_gate.entered();
    auto erasing = std::async(std::launch::async, [&] { cache.invalidate(key()); });
    const auto write_revoked = wait_for_empty(cache);
    storage->store_gate.release();
    const auto published = publishing.get();
    erasing.get();
    check(store_entered && write_revoked && published,
          "invalidate crosses in-flight write-through");
    check(!storage->backing.load(key()) && !cache.lookup(key()).value,
          "in-flight write cannot survive completed invalidation");
}

void test_clear_during_load_and_racing_publishers() {
    auto storage = std::make_shared<ControlledStorage>();
    storage->backing.store(key(), value(41));
    storage->load_gate.enable();
    HistoryCache cache({1 << 20, 8, storage, false});
    auto loading = std::async(std::launch::async, [&] { return cache.lookup(key()); });
    const auto entered = storage->load_gate.entered();
    cache.clear_memory();
    storage->load_gate.release();
    const auto old = loading.get();
    check(entered && !old.value, "clear revokes an in-flight storage load");
    const auto fresh = cache.lookup(key());
    check(fresh.source == HistoryCacheSource::kStorage && fresh.value->metadata[0] == 41,
          "new lookup can restore lower tier after clear");
    auto first = std::async(std::launch::async, [&] { return cache.publish(fresh, value(42)); });
    auto second = std::async(std::launch::async, [&] { return cache.publish(fresh, value(43)); });
    check(first.get() != second.get(), "exactly one concurrent publisher wins CAS");
}

void test_value_validation() {
    HistoryCache cache({1 << 20, 8, nullptr, false});
    const auto lease = cache.lookup(key());
    auto invalid = value();
    invalid.schema_version = 2;
    check(!cache.publish(lease, invalid), "unknown schema cannot be cached");
    invalid = value();
    invalid.tensors[0].host_data.pop_back();
    check(!cache.publish(lease, invalid), "malformed tensor bytes rejected");
    invalid = value();
    invalid.tensors[0].shape = {-1};
    check(!cache.publish(lease, invalid), "negative tensor shape rejected");
    invalid = value();
    invalid.tensors[0].shape = {std::numeric_limits<std::int64_t>::max(), 8};
    check(!cache.publish(lease, invalid), "overflowing tensor shape rejected");
    invalid = value();
    invalid.tensors.push_back(invalid.tensors[0]);
    check(!cache.publish(lease, invalid), "duplicate tensor names rejected");
    invalid = value();
    invalid.format.clear();
    check(!cache.publish(lease, invalid), "unidentified format rejected");
    check(cache.publish(lease, value()), "failed validation does not consume lease generation");
}

} // namespace

int main() {
    try {
        test_identity_and_cas();
        test_lru_and_pinned_budget();
        test_storage_roundtrip();
        test_storage_pins_and_failures();
        test_invalidation_races();
        test_clear_during_load_and_racing_publishers();
        test_value_validation();
        std::cout << "Native history cache: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Native history cache failed: " << error.what() << '\n';
        return 1;
    }
}
