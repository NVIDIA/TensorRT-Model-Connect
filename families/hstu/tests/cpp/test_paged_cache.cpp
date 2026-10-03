/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/paged_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
using namespace trtmc;
using namespace trtmc::hstu;
std::size_t checks = 0;

void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
    ++checks;
}
template <class Function>
void rejects(Function&& function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, message);
}

PagedCacheGeometry geometry(int slots = 3) {
    return {2, 2, 3, slots, 1024, 128, DType::kFloat32, 4 * 1024 * 1024};
}
HistorySignature signature(int tokens) {
    HistorySignature value;
    value.model_namespace = "model-v1";
    value.feature_version = "features-v2";
    value.history_revision = "lineage-3";
    value.effective_scale = 1024;
    for (int i = 0; i < tokens; ++i) {
        value.token_rows.push_back(i + 1);
        value.position_ids.push_back(i);
    }
    return value;
}
float element(int layer, int kind, int head, int token, int column) {
    return static_cast<float>(layer * 20000 + kind * 10000 + head * 2000 + token * 3 + column);
}

HistoryCacheTensor compact(const PagedCacheGeometry& g, int length) {
    HistoryCacheTensor value;
    value.dtype = g.dtype;
    value.shape = {g.layers, 2, g.heads, length, g.head_dim};
    std::vector<float> data;
    for (int layer = 0; layer < g.layers; ++layer)
        for (int kind = 0; kind < 2; ++kind)
            for (int head = 0; head < g.heads; ++head)
                for (int token = 0; token < length; ++token)
                    for (int column = 0; column < g.head_dim; ++column)
                        data.push_back(element(layer, kind, head, token, column));
    value.host_data.resize(data.size() * sizeof(float));
    if (!data.empty())
        std::memcpy(value.host_data.data(), data.data(), value.host_data.size());
    return value;
}

PagedCacheRequest request(int slot, const char* owner, int history, int candidates) {
    return {slot, owner, signature(history), history + candidates};
}

void expect_compact(PagedCacheArena& arena, int slot, int length) {
    const auto result = arena.copy_history(slot);
    const auto expected = compact(arena.geometry(), length);
    check(result.shape == expected.shape, "compact snapshot shape");
    check(result.dtype == expected.dtype && !result.device, "host compact dtype/storage");
    check(result.host_data == expected.host_data, "all compact history components exact");
}

// Acts as a native PACKED_NHD engine on actual owned host storage. Every current
// candidate K/V is NaN, so any accidentally persisted candidate is observable.
void apply(PagedCacheArena& arena, PagedCacheArena::Lease& lease,
           const std::vector<PagedCacheRequest>& requests) {
    const auto& g = arena.geometry();
    const auto& p = lease.plan();
    const auto width = g.heads * g.head_dim;
    for (int layer = 0; layer < g.layers; ++layer) {
        std::vector<float> current(static_cast<std::size_t>(p.total_queries()) * 2 * width,
                                   std::numeric_limits<float>::quiet_NaN());
        for (std::size_t user = 0; user < requests.size(); ++user) {
            const auto added = requests[user].history.token_rows.size() - p.reused_tokens[user];
            for (std::size_t row = 0; row < added; ++row)
                for (int kind = 0; kind < 2; ++kind)
                    for (int head = 0; head < g.heads; ++head)
                        for (int column = 0; column < g.head_dim; ++column)
                            current[(p.query_offsets[user] + row) * 2 * width + kind * width +
                                    head * g.head_dim + column] =
                                element(layer, kind, head,
                                        p.reused_tokens[user] + static_cast<int>(row), column);
        }
        auto* pages = static_cast<float*>(lease.layer_data(layer));
        for (std::size_t page = 0; page < p.page_write_indices.size(); ++page) {
            const auto begin = p.page_update_lengths[page];
            const auto end = p.page_update_lengths[page + 1];
            for (int index = begin; index < end; ++index) {
                const int source = p.page_update_rows[index];
                const int destination = p.page_write_indices[page] + index - begin;
                check(destination >= 0 && destination < g.page_size, "page update within capacity");
                for (int kind = 0; kind < 2; ++kind)
                    for (int component = 0; component < width; ++component)
                        pages[((page * 2 + kind) * g.page_size + destination) * width + component] =
                            source < 0 ? 0 : current[(source * 2 + kind) * width + component];
            }
        }
    }
}

void test_initialization_mixed_compact_rows_and_zero_queries() {
    auto arena = PagedCacheArena::create_host(geometry());
    check(!arena.is_cuda() && arena.pages_per_slot() == 8, "real host geometry");
    for (const auto& item : std::vector<std::pair<int, int>>{{0, 127}, {1, 128}, {2, 400}}) {
        const auto owner = "user-" + std::to_string(item.first);
        arena.initialize_history(item.first, owner, signature(item.second),
                                 compact(arena.geometry(), item.second));
        expect_compact(arena, item.first, item.second);
    }
    const std::vector<PagedCacheRequest> requests = {
        request(2, "user-2", 402, 5), request(0, "user-0", 130, 2), request(1, "user-1", 128, 0)};
    auto lease = arena.prepare(requests, {});
    const auto& p = lease.plan();
    check(p.reused_tokens == std::vector<std::int32_t>({400, 127, 128}),
          "reuse is independent of request order");
    check(p.query_offsets == std::vector<std::int32_t>({0, 7, 12, 12}),
          "compact per-request offsets");
    check(p.active_requests == std::vector<std::int32_t>({0, 1}), "zero-query user omitted");
    check(p.q_offsets == std::vector<std::int32_t>({0, 7, 12}), "active compact offsets");
    check(p.k_offsets == std::vector<std::int32_t>({0, 407, 539}), "logical key lengths");
    check(p.targets == std::vector<std::int32_t>({5, 2}), "request-local candidate counts");
    check(p.page_ids == std::vector<std::int32_t>({2, 3, 4, 5, 0, 6}),
          "retained owner mappings and lowest-free append page survive request reorder");
    check(p.page_update_rows == std::vector<std::int32_t>({7, 0, 1, 8, 9}),
          "updates grouped by destination page");
    check(p.packed_attention_metadata.size() == 5 * 3 * 8, "packed transport active batch");
    check(p.packed_attention_metadata[4 * 24] == 2, "page ID plane begins at original vector");
    rejects([&] { arena.invalidate(0); }, "in-flight invalidation rejected");
    rejects([&] { arena.copy_history(0); }, "in-flight snapshot rejected");
    apply(arena, lease, requests);
    arena.commit(lease);
    rejects([&] { (void)lease.plan(); }, "consumed lease metadata rejected");
    expect_compact(arena, 0, 130);
    expect_compact(arena, 1, 128);
    expect_compact(arena, 2, 402);
    auto idle = arena.prepare({request(2, "user-2", 402, 0)}, {});
    check(idle.plan().total_queries() == 0 && idle.plan().active_requests.empty() &&
              idle.plan().packed_attention_metadata.empty() && idle.plan().page_update_rows.empty(),
          "zero total queries bypass attention/update");
    arena.commit(idle);
}

HistoryCacheTensor unknown_pages(PagedCacheArena& arena, int length) {
    const auto& g = arena.geometry();
    const auto saved = arena.copy_history(0);
    check(saved.shape == std::vector<std::int64_t>({g.layers, 2, g.heads, length, g.head_dim}),
          "unknown restore fixture uses actual compact history");
    std::vector<float> history(saved.host_data.size() / sizeof(float));
    if (!history.empty())
        std::memcpy(history.data(), saved.host_data.data(), saved.host_data.size());
    const auto layer_elements = arena.pages_per_slot() * 2 * g.page_size * g.heads * g.head_dim;
    HistoryCacheTensor value;
    value.dtype = g.dtype;
    value.shape = {g.layers, arena.pages_per_slot(), 2, g.page_size, g.heads * g.head_dim};
    value.host_data.resize(static_cast<std::size_t>(g.layers) * layer_elements * sizeof(float));
    for (int layer = 0; layer < g.layers; ++layer) {
        std::vector<float> copied(layer_elements, std::numeric_limits<float>::quiet_NaN());
        for (int token = 0; token < length; ++token)
            for (int kind = 0; kind < 2; ++kind)
                for (int head = 0; head < g.heads; ++head)
                    for (int column = 0; column < g.head_dim; ++column)
                        copied[((token / g.page_size * 2 + kind) * g.page_size +
                                token % g.page_size) *
                                   g.heads * g.head_dim +
                               head * g.head_dim + column] =
                            history[(((layer * 2 + kind) * g.heads + head) * length + token) *
                                        g.head_dim +
                                    column];
        std::memcpy(value.host_data.data() + layer * layer_elements * sizeof(float), copied.data(),
                    layer_elements * sizeof(float));
    }
    return value;
}

void test_unknown_tail_clear_and_unused_page_preservation() {
    auto arena = PagedCacheArena::create_host(geometry(1));
    arena.initialize_history(0, "user", signature(400), compact(arena.geometry(), 400));
    const auto raw = unknown_pages(arena, 400);
    arena.restore_unknown(0, "user", signature(400), raw);
    auto lease = arena.prepare({request(0, "user", 400, 256)}, {});
    check(lease.plan().page_update_rows == std::vector<std::int32_t>(112, -1),
          "unknown hit clears original partial tail");
    const std::vector<PagedCacheRequest> requests{request(0, "user", 400, 256)};
    apply(arena, lease, requests);
    for (int layer = 0; layer < 2; ++layer) {
        const auto* data = static_cast<const float*>(lease.layer_data(layer));
        for (int kind = 0; kind < 2; ++kind) {
            for (int row = 16; row < 128; ++row)
                check(data[((3 * 2 + kind) * 128 + row) * 6] == 0, "all used tail rows zero");
            check(std::isnan(data[(4 * 2 + kind) * 128 * 6]), "unused whole page remains poisoned");
        }
    }
    arena.commit(lease);
    expect_compact(arena, 0, 400);
    auto again = arena.prepare({request(0, "user", 402, 256)}, {});
    check(again.plan().page_update_rows.size() == 112 && again.plan().page_update_rows[0] == 0 &&
              again.plan().page_update_rows[1] == 1,
          "safe completion does not certify future pages");
    apply(arena, again, {request(0, "user", 402, 256)});
    arena.commit(again);
    expect_compact(arena, 0, 402);
}

void test_identity_truncation_failure_and_reinitialization() {
    auto arena = PagedCacheArena::create_host(geometry(1));
    arena.initialize_history(0, "A", signature(400), compact(arena.geometry(), 400));
    auto changed = arena.prepare({request(0, "B", 400, 256)}, {});
    check(changed.plan().reused_tokens[0] == 0 && changed.plan().page_update_rows.size() == 512,
          "same tokens for another subject cannot reuse slot");
    apply(arena, changed, {request(0, "B", 400, 256)});
    arena.commit(changed);
    auto shrunk = arena.prepare({request(0, "B", 128, 2)}, {});
    check(shrunk.plan().reused_tokens[0] == 0, "truncation recomputes");
    apply(arena, shrunk, {request(0, "B", 128, 2)});
    arena.commit(shrunk);
    auto crossing = arena.prepare({request(0, "B", 129, 2)}, {});
    check(crossing.plan().page_update_rows.size() == 128 &&
              crossing.plan().page_update_rows[0] == 0,
          "cross-page append after truncation clears stale future page");
    apply(arena, crossing, {request(0, "B", 129, 2)});
    arena.fail(crossing);
    rejects([&] { arena.copy_history(0); }, "failed writes revoke snapshot validity");
    auto recovery = arena.prepare({request(0, "B", 129, 2)}, {});
    check(recovery.plan().reused_tokens[0] == 0 && recovery.plan().page_update_rows.size() == 256,
          "failure recovers by full history recomputation with safe clears");
    apply(arena, recovery, {request(0, "B", 129, 2)});
    arena.commit(recovery);
    expect_compact(arena, 0, 129);
    arena.initialize_history(0, "B", signature(129), compact(arena.geometry(), 129));
    auto trusted = arena.prepare({request(0, "B", 129, 3)}, {});
    check(trusted.plan().page_update_rows.empty(), "real zero reinitialization restores trust");
    arena.commit(trusted);
}

void test_abort_lifetime_and_invalid_admission() {
    auto arena = PagedCacheArena::create_host(geometry(1));
    arena.initialize_history(0, "user", signature(10), compact(arena.geometry(), 10));
    const auto original = arena.copy_history(0).host_data;
    auto wrong = compact(arena.geometry(), 10);
    wrong.host_data.pop_back();
    rejects([&] { arena.initialize_history(0, "new", signature(10), wrong); },
            "short input rejected");
    check(arena.copy_history(0).host_data == original, "invalid input preserves prior state");
    rejects([&] { arena.prepare({request(0, "user", 10, 2), request(0, "user", 10, 3)}, {}); },
            "duplicate slot rejected");
    rejects([&] { arena.prepare({request(1, "user", 10, 2)}, {}); }, "invalid slot rejected");
    rejects([&] { arena.prepare({request(0, "user", 1024, 1)}, {}); },
            "logical key capacity enforced");
    {
        auto abandoned = arena.prepare({request(0, "user", 11, 2)}, {});
        rejects([&] { arena.prepare({request(0, "user", 11, 2)}, {}); }, "second writer rejected");
    }
    rejects([&] { arena.copy_history(0); }, "abandoned transaction revokes validity");
    auto lease = arena.prepare({request(0, "user", 10, 2)}, {});
    auto other = PagedCacheArena::create_host(geometry(1));
    rejects([&] { other.commit(lease); }, "foreign commit rejected");
    arena.fail(lease);
    PagedCacheArena::Lease retained;
    {
        auto temporary = PagedCacheArena::create_host(geometry(1));
        retained = temporary.prepare({request(0, "temp", 1, 0)}, {});
    }
    check(retained.layer_data(0) != nullptr && retained.plan().total_queries() == 1,
          "lease retains actual allocation after arena handle destruction");
    auto bad = geometry();
    bad.max_bytes = 0;
    rejects([&] { PagedCacheArena::create_host(bad); }, "explicit nonzero budget required");
    bad = geometry();
    bad.max_bytes = 1;
    rejects([&] { PagedCacheArena::create_host(bad); }, "allocation budget enforced");
    bad = geometry();
    bad.layers = std::numeric_limits<std::int32_t>::max();
    bad.heads = std::numeric_limits<std::int32_t>::max();
    rejects([&] { PagedCacheArena::create_host(bad); },
            "shape overflow rejected before allocation");
}

void test_semantic_policy_and_initial_cold_history_only() {
    auto arena = PagedCacheArena::create_host(geometry(1));
    arena.invalidate(0);
    auto cold = arena.prepare({request(0, "user", 400, 256)}, {});
    check(cold.plan().page_update_rows.size() == 400 &&
              std::find(cold.plan().page_update_rows.begin(), cold.plan().page_update_rows.end(),
                        -1) == cold.plan().page_update_rows.end(),
          "invalidating a fresh zero slot retains proof and writes history only");
    apply(arena, cold, {request(0, "user", 400, 256)});
    arena.commit(cold);
    auto incoming = request(0, "user", 402, 256);
    incoming.history.effective_scale = 658;
    auto changed = arena.prepare({incoming}, {});
    check(changed.plan().reused_tokens[0] == 0 &&
              changed.plan().reuse[0].reason == CacheReuseReason::kScaleChanged,
          "existing semantic reuse policy remains authoritative");
    arena.fail(changed);

    auto failed_empty = PagedCacheArena::create_host(geometry(1));
    auto partial = failed_empty.prepare({request(0, "user", 400, 256)}, {});
    apply(failed_empty, partial, {request(0, "user", 400, 256)});
    failed_empty.fail(partial);
    failed_empty.invalidate(0);
    auto recovery = failed_empty.prepare({request(0, "user", 400, 256)}, {});
    check(recovery.plan().page_update_rows.size() == 512,
          "failed empty committed state cannot regain zero proof on invalidation");
    failed_empty.fail(recovery);
}

void test_half_storage_bits_and_invalid_live_restore() {
    for (const auto dtype : {DType::kFloat16, DType::kBFloat16}) {
        auto g = geometry(1);
        g.dtype = dtype;
        auto arena = PagedCacheArena::create_host(g);
        HistoryCacheTensor value;
        value.dtype = dtype;
        value.shape = {g.layers, 2, g.heads, 129, g.head_dim};
        std::vector<std::uint16_t> compact_bits(g.layers * 2 * g.heads * 129 * g.head_dim);
        for (std::size_t i = 0; i < compact_bits.size(); ++i)
            compact_bits[i] = static_cast<std::uint16_t>(0x3c00 + i % 64);
        value.host_data.resize(compact_bits.size() * sizeof(std::uint16_t));
        std::memcpy(value.host_data.data(), compact_bits.data(), value.host_data.size());
        arena.initialize_history(0, "half", signature(129), value);
        check(arena.copy_history(0).host_data == value.host_data,
              "half compact roundtrip preserves every bit");
        auto lease = arena.prepare({request(0, "half", 129, 0)}, {});
        const auto layer_elements = arena.pages_per_slot() * 2 * g.page_size * g.heads * g.head_dim;
        std::vector<std::uint16_t> raw(g.layers * layer_elements);
        for (int layer = 0; layer < g.layers; ++layer)
            std::memcpy(raw.data() + layer * layer_elements, lease.layer_data(layer),
                        layer_elements * sizeof(std::uint16_t));
        arena.commit(lease);
        HistoryCacheTensor unknown;
        unknown.dtype = dtype;
        unknown.shape = {g.layers, arena.pages_per_slot(), 2, g.page_size, g.heads * g.head_dim};
        unknown.host_data.resize(raw.size() * sizeof(std::uint16_t));
        const auto first = raw[0];
        raw[0] = 0x7fc1;
        std::memcpy(unknown.host_data.data(), raw.data(), unknown.host_data.size());
        rejects([&] { arena.restore_unknown(0, "half", signature(129), unknown); },
                "nonfinite live half prefix rejected before restore");
        check(arena.copy_history(0).host_data == value.host_data,
              "invalid half restore preserves prior owned history");
        raw[0] = first;
        for (int layer = 0; layer < g.layers; ++layer)
            for (int token = 129; token < 1024; ++token)
                for (int kind = 0; kind < 2; ++kind)
                    for (int component = 0; component < 6; ++component)
                        raw[layer * layer_elements +
                            ((token / 128 * 2 + kind) * 128 + token % 128) * 6 + component] =
                            0x7fc1;
        std::memcpy(unknown.host_data.data(), raw.data(), unknown.host_data.size());
        arena.restore_unknown(0, "half", signature(129), unknown);
        auto clear = arena.prepare({request(0, "half", 129, 256)}, {});
        check(clear.plan().page_update_rows == std::vector<std::int32_t>(127, -1),
              "half unknown tail uses safe sentinel metadata");
        for (int layer = 0; layer < g.layers; ++layer) {
            auto* actual = static_cast<std::uint16_t*>(clear.layer_data(layer));
            check(actual[0] == raw[layer * layer_elements], "half old prefix bits retained");
            check(actual[(2 * 2) * 128 * 6] == 0x7fc1, "half unused page poison remains intact");
        }
        arena.fail(clear);
    }
}

void test_binding_pages_preserves_owned_arena() {
    auto arena = PagedCacheArena::create_host(geometry(8));
    const auto bytes = arena.nbytes();
    {
        auto empty = arena.prepare({request(0, "user-0", 0, 0)}, {});
        check(empty.plan().binding_pages() == 1, "empty attention/update view retains one page");
        check(empty.plan().page_write_indices.size() == 64 &&
                  empty.plan().page_update_lengths.size() == 65,
              "empty view retains full physical update metadata");
        arena.commit(empty);
    }
    std::vector<PagedCacheRequest> requests;
    for (int slot = 0; slot < 8; ++slot) {
        const auto owner = "user-" + std::to_string(slot);
        arena.initialize_history(slot, owner, signature(400), compact(arena.geometry(), 400));
        requests.push_back(request(slot, owner.c_str(), 400, 256));
    }
    {
        auto single = arena.prepare({requests.front()}, {});
        check(single.plan().binding_pages() == 4, "one400-token history reads four pages");
        check(single.plan().page_update_rows.empty(), "verified exact hit performs no writes");
        auto* storage = single.layer_data(0);
        const auto ids = single.plan().page_ids;
        (void)single.plan().binding_pages();
        check(single.layer_data(0) == storage && single.plan().page_ids == ids &&
                  single.plan().page_write_indices.size() == 64 && arena.nbytes() == bytes,
              "binding extent does not alter ownership, IDs or allocation");
        arena.commit(single);
    }
    auto batch = arena.prepare(requests, {});
    check(batch.plan().binding_pages() == 32,
          "eight400-token histories demand exactly32 physical pages");
    check(batch.plan().page_ids.size() == 32 && batch.plan().page_ids.back() == 31,
          "existing assignments remain stable without relocating pages");
    arena.commit(batch);
    auto appended = requests.front();
    appended.history = signature(513);
    appended.total_tokens = 769;
    auto crossing = arena.prepare({appended}, {});
    check(crossing.plan().binding_pages() == 33,
          "cross-page append retains old IDs and includes lowest-free page32");
    check(arena.nbytes() == bytes && arena.pages_per_slot() == 8,
          "append binding retains original allocation and per-slot logical capacity");
    arena.fail(crossing);
}

void test_binding_pages_includes_unreferenced_writes_and_rejects_invalid_metadata() {
    PagedCachePlan valid;
    valid.page_write_indices.assign(64, 0);
    valid.page_update_lengths.assign(65, 0);
    valid.page_ids = {3, 0};
    check(valid.binding_pages() == 4, "read maximum is independent of page ID ordering");
    valid.page_update_rows = {-1};
    valid.page_write_indices[63] = 127;
    valid.page_update_lengths.back() = 1;
    check(valid.binding_pages() == 64, "tail clear beyond all reads must remain bound");
    check(valid.page_update_lengths[valid.binding_pages()] ==
              static_cast<std::int32_t>(valid.page_update_rows.size()),
          "sliced cumulative lengths include every update row");
    auto bad = valid;
    bad.page_update_lengths.pop_back();
    rejects([&] { bad.binding_pages(); }, "short cumulative metadata rejected");
    bad = valid;
    bad.page_update_lengths.push_back(1);
    rejects([&] { bad.binding_pages(); }, "extra cumulative metadata rejected");
    bad = valid;
    bad.page_update_lengths.front() = 1;
    rejects([&] { bad.binding_pages(); }, "nonzero cumulative origin rejected");
    bad = valid;
    bad.page_update_lengths.back() = 2;
    rejects([&] { bad.binding_pages(); }, "update count exceeds row vector");
    bad = valid;
    bad.page_update_lengths.back() = 0;
    rejects([&] { bad.binding_pages(); }, "unaccounted update row rejected");
    bad = valid;
    bad.page_update_lengths[32] = 2;
    rejects([&] { bad.binding_pages(); }, "nonmonotonic cumulative count rejected");
    bad = valid;
    bad.page_update_lengths[32] = -1;
    rejects([&] { bad.binding_pages(); }, "negative cumulative count rejected");
    bad = valid;
    bad.page_write_indices[32] = -1;
    rejects([&] { bad.binding_pages(); }, "negative write offset rejected even on inactive page");
    bad = valid;
    bad.page_ids.push_back(-1);
    rejects([&] { bad.binding_pages(); }, "negative page ID rejected");
    bad = valid;
    bad.page_ids.push_back(64);
    rejects([&] { bad.binding_pages(); }, "page ID outside physical allocation rejected");
    rejects([] { PagedCachePlan{}.binding_pages(); }, "missing physical extent metadata rejected");
}

void test_demand_capacity_and_zero_query_release() {
    for (const int batch : {1, 2, 4, 8}) {
        auto arena = PagedCacheArena::create_host(geometry(8));
        std::vector<PagedCacheRequest> requests;
        for (int slot = 0; slot < batch; ++slot)
            requests.push_back(request(slot, "owner", 400, 256));
        auto lease = arena.prepare(requests, {});
        const auto& plan = lease.plan();
        check(plan.binding_pages() == batch * 4, "fresh histories receive only required pages");
        check(plan.page_update_rows.size() == static_cast<std::size_t>(batch * 400),
              "new clean pages preserve zero-tail proof without redundant clears");
        auto ids = plan.page_ids;
        std::sort(ids.begin(), ids.end());
        check(std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
              "active owners never share physical pages");
        check(arena.nbytes() == 8 * 8 * 2 * 128 * 6 * sizeof(float) * 2,
              "demand assignment retains complete eight-slot backing allocation");
        arena.fail(lease);
    }
    auto arena = PagedCacheArena::create_host(geometry(2));
    arena.initialize_history(0, "old", signature(400), compact(arena.geometry(), 400));
    auto empty = arena.prepare({request(0, "old", 0, 0)}, {});
    check(empty.plan().total_queries() == 0 && empty.plan().page_ids.empty(),
          "zero-query truncation requires no attention or persistent pages");
    arena.commit(empty);
    auto replacement = arena.prepare({request(1, "new", 1, 1)}, {});
    check(replacement.plan().page_ids == std::vector<std::int32_t>({0}),
          "committed zero-history owner releases lowest page");
    check(replacement.plan().page_update_rows.size() == 128,
          "released written page is dirty and uses safe partial-tail clearing");
    arena.fail(replacement);
    std::vector<PagedCacheRequest> maximum{request(0, "A", 1024, 0), request(1, "B", 1024, 0)};
    auto full = arena.prepare(maximum, {});
    check(full.plan().page_ids.size() == 16 && full.plan().binding_pages() == 16,
          "every slot can grow to maximum despite delayed page release");
    auto ids = full.plan().page_ids;
    std::sort(ids.begin(), ids.end());
    check(std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
          "full-capacity assignments remain disjoint");
    arena.fail(full);
}

void test_demand_rejected_prepare_and_failed_append() {
    auto arena = PagedCacheArena::create_host(geometry(3));
    arena.initialize_history(0, "A", signature(127), compact(arena.geometry(), 127));
    const auto prior = arena.copy_history(0).host_data;
    rejects([&] { arena.prepare({request(0, "A", 257, 1), request(3, "bad", 1, 0)}, {}); },
            "later invalid request rejects staged page reservations");
    check(arena.copy_history(0).host_data == prior, "rejected prepare preserves committed owner");
    auto other = arena.prepare({request(1, "B", 128, 0)}, {});
    check(other.plan().page_ids == std::vector<std::int32_t>({1}),
          "rejected staged reservation does not consume lowest free page");
    apply(arena, other, {request(1, "B", 128, 0)});
    arena.commit(other);
    auto grow = arena.prepare({request(0, "A", 129, 1)}, {});
    check(grow.plan().page_ids == std::vector<std::int32_t>({0, 2}),
          "append retains prior ID and allocates around another owner");
    apply(arena, grow, {request(0, "A", 129, 1)});
    rejects([&] { arena.prepare({request(2, "C", 1, 0)}, {}); },
            "reserved append pages cannot be reassigned before drain");
    arena.fail(grow);
    rejects([&] { arena.copy_history(0); }, "failed append invalidates prior session history");
    auto recycled = arena.prepare({request(2, "C", 1, 0)}, {});
    check(recycled.plan().page_ids == std::vector<std::int32_t>({2}) &&
              recycled.plan().page_update_rows.size() == 128,
          "drained failed append releases its new page dirty");
    apply(arena, recycled, {request(2, "C", 1, 0)});
    arena.commit(recycled);
    expect_compact(arena, 1, 128);
    auto recover = arena.prepare({request(0, "A", 129, 1)}, {});
    check(recover.plan().reused_tokens[0] == 0 &&
              recover.plan().page_ids == std::vector<std::int32_t>({0, 3}),
          "failed session recomputes into retained plus newly assigned pages");
    apply(arena, recover, {request(0, "A", 129, 1)});
    arena.commit(recover);
    expect_compact(arena, 0, 129);
    const auto snapshot = arena.copy_history(0);
    auto restored = PagedCacheArena::create_host(geometry(1));
    restored.initialize_history(0, "A", signature(129), snapshot);
    check(restored.copy_history(0).host_data == snapshot.host_data,
          "fragmented physical mapping preserves compact snapshot restore bits");
}

void test_demand_dirty_page_poison_and_logical_unknown_restore() {
    auto arena = PagedCacheArena::create_host(geometry(3));
    arena.initialize_history(0, "old", signature(0), compact(arena.geometry(), 0));
    const auto poison = unknown_pages(arena, 0);
    arena.restore_unknown(0, "old", signature(0), poison);
    arena.initialize_history(1, "protected", signature(128), compact(arena.geometry(), 128));
    const auto protected_history = arena.copy_history(1).host_data;
    auto shrink = arena.prepare({request(0, "old", 1, 0)}, {});
    apply(arena, shrink, {request(0, "old", 1, 0)});
    arena.commit(shrink);
    auto dirty = arena.prepare({request(2, "new", 129, 1)}, {});
    check(dirty.plan().page_ids == std::vector<std::int32_t>({1, 2}),
          "new owner receives lowest released physical pages");
    check(dirty.plan().page_update_rows.size() == 256 && dirty.plan().page_update_rows.back() == -1,
          "poisoned reused pages cannot inherit fresh allocation zero proof");
    const auto* before = static_cast<const float*>(dirty.layer_data(0));
    check(std::isnan(before[1 * 2 * 128 * 6]), "dirty page poison remains until native writes");
    apply(arena, dirty, {request(2, "new", 129, 1)});
    const auto* after = static_cast<const float*>(dirty.layer_data(0));
    check(after[((2 * 2 + 1) * 128 + 127) * 6] == 0,
          "same native update plan clears poisoned last-page V tail");
    check(std::isnan(after[3 * 2 * 128 * 6]), "unallocated poison is not cleared implicitly");
    arena.commit(dirty);
    check(arena.copy_history(1).host_data == protected_history,
          "another owner's history stays exact");
    expect_compact(arena, 2, 129);
    auto append = arena.prepare({request(0, "old", 129, 1)}, {});
    apply(arena, append, {request(0, "old", 129, 1)});
    arena.commit(append);
    const auto source = unknown_pages(arena, 129);
    arena.restore_unknown(0, "old", signature(129), source);
    auto inspect = arena.prepare({request(0, "old", 1024, 0)}, {});
    const auto ids = inspect.plan().page_ids;
    check(ids.size() == 8 && ids[0] == 0 && ids[1] == 3,
          "unknown restore retains existing noncontiguous logical page mapping");
    const auto page_bytes = 2 * 128 * 6 * sizeof(float);
    for (int layer = 0; layer < 2; ++layer)
        for (std::size_t page = 0; page < ids.size(); ++page)
            check(std::memcmp(static_cast<const std::uint8_t*>(inspect.layer_data(layer)) +
                                  ids[page] * page_bytes,
                              source.host_data.data() + (layer * 8 + page) * page_bytes,
                              page_bytes) == 0,
                  "unknown restore scatters every logical page including future poison exactly");
    arena.fail(inspect);
    check(arena.copy_history(1).host_data == protected_history,
          "unknown scatter and failed append do not corrupt another owner");
}
} // namespace

int main() {
    try {
        test_initialization_mixed_compact_rows_and_zero_queries();
        test_unknown_tail_clear_and_unused_page_preservation();
        test_identity_truncation_failure_and_reinitialization();
        test_abort_lifetime_and_invalid_admission();
        test_semantic_policy_and_initial_cold_history_only();
        test_half_storage_bits_and_invalid_live_restore();
        test_binding_pages_preserves_owned_arena();
        test_binding_pages_includes_unreferenced_writes_and_rejects_invalid_metadata();
        test_demand_capacity_and_zero_query_release();
        test_demand_rejected_prepare_and_failed_append();
        test_demand_dirty_page_poison_and_logical_unknown_restore();
        std::cout << checks << " paged-cache checks passed without CUDA execution\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
