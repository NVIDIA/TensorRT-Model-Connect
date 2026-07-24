/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_kv_setup.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class HostAllocator final : public trtmc::IRuntimeDeviceAllocator {
  public:
    struct Request {
        std::uint64_t bytes;
        std::uint64_t alignment;
        std::uint32_t device;
        std::uint64_t live_bytes_before;
    };
    struct GuardedAllocation {
        std::byte* base{nullptr};
        std::byte* usable{nullptr};
        std::uint64_t bytes{0};
        std::uint64_t guard_bytes{0};
        bool guards_ok{true};
        bool released{false};
    };

    trtmc::RuntimeDeviceAllocation allocate(std::uint64_t bytes, std::uint64_t alignment,
                                            std::uint32_t device, void*) override {
        void* base_pointer = nullptr;
        const auto total = static_cast<std::size_t>(bytes + 2 * alignment);
        if (posix_memalign(&base_pointer, static_cast<std::size_t>(alignment), total) != 0) {
            throw std::bad_alloc();
        }
        auto* base = static_cast<std::byte*>(base_pointer);
        std::memset(base, 0xA5, static_cast<std::size_t>(alignment));
        std::memset(base + alignment + bytes, 0xA5, static_cast<std::size_t>(alignment));
        auto usable = base + alignment;
        requests.push_back(Request{bytes, alignment, device, live_bytes()});
        auto releases = releases_;
        auto record =
            std::make_shared<GuardedAllocation>(GuardedAllocation{base, usable, bytes, alignment});
        guarded.push_back(record);
        auto owner = std::shared_ptr<void>(usable, [record, releases](void*) {
            for (std::uint64_t index = 0; index < record->guard_bytes; ++index) {
                if (record->base[index] != std::byte{0xA5} ||
                    record->usable[record->bytes + index] != std::byte{0xA5}) {
                    record->guards_ok = false;
                }
            }
            record->released = true;
            std::free(record->base);
            ++*releases;
        });
        return trtmc::RuntimeDeviceAllocation{usable, bytes, device, alignment, std::move(owner)};
    }

    bool guards_intact() const {
        for (const auto& allocation : guarded) {
            if (!allocation->guards_ok)
                return false;
            if (allocation->released)
                continue;
            for (std::uint64_t index = 0; index < allocation->guard_bytes; ++index) {
                if (allocation->base[index] != std::byte{0xA5} ||
                    allocation->usable[allocation->bytes + index] != std::byte{0xA5}) {
                    return false;
                }
            }
        }
        return true;
    }

    int releases() const { return *releases_; }
    std::uint64_t live_bytes() const {
        std::uint64_t result = 0;
        for (const auto& allocation : guarded) {
            if (!allocation->released)
                result += allocation->bytes;
        }
        return result;
    }

    std::vector<Request> requests;
    std::vector<std::shared_ptr<GuardedAllocation>> guarded;

  private:
    std::shared_ptr<int> releases_{std::make_shared<int>(0)};
};

class FakeRuntimeModuleBase : public trtmc::ITrtModule, public trtmc::IRuntimeMemoryModuleV1 {
  public:
    FakeRuntimeModuleBase(std::uint64_t token_max, std::uint64_t cache_max,
                          std::size_t context_base, std::size_t alignment = 256, int32_t device = 0)
        : token_max_(token_max), cache_max_(cache_max), context_base_(context_base),
          alignment_(alignment), device_(device) {}

    trtmc::TensorMap forward(const trtmc::TensorMap&) override { return {}; }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "token_id" || name == "position_id" || name == "history_length" ||
               name.rfind("cache_k_", 0) == 0 || name.rfind("cache_v_", 0) == 0;
    }
    bool has_output(const std::string& name) const override {
        return name.rfind("present_k_", 0) == 0 || name.rfind("present_v_", 0) == 0;
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return name.rfind("cache_", 0) == 0 || name.rfind("present_", 0) == 0
                   ? trtmc::DType::kBFloat16
                   : trtmc::DType::kInt32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name.rfind("cache_", 0) == 0)
            return {1, 1, -1, 128};
        if (name.rfind("present_", 0) == 0)
            return {-1, 128};
        if (name == "history_length")
            return {1};
        return {-1};
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector selector) const override {
        const auto extent = selector == trtmc::ProfileShapeSelector::kMin
                                ? std::uint64_t{1}
                                : (name.rfind("cache_", 0) == 0 ? cache_max_ : token_max_);
        if (name.rfind("cache_", 0) == 0)
            return {1, 1, static_cast<int64_t>(extent), 128};
        return {static_cast<int64_t>(extent)};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    int32_t input_rank(const std::string& name) const override {
        return name.rfind("cache_", 0) == 0 ? 4 : 1;
    }
    bool input_is_dynamic(const std::string&) const override { return true; }
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    void set_runtime_binding_shape(const trtmc::RuntimeMemoryShapeV1& shape) override {
        if (shape.name.rfind("cache_", 0) == 0)
            last_bound_tokens_ = shape.bound_tokens;
    }
    void set_runtime_alias_pair_shape(const trtmc::RuntimeMemoryAliasShapeV1&) override {
        throw std::logic_error("alias path is not used by fused attention");
    }
    void set_runtime_input_shape(const trtmc::RuntimeInputShapeV1& shape) override {
        if (shape.name == "token_id" && shape.shape.size() == 1)
            last_query_tokens_ = static_cast<std::uint64_t>(shape.shape.front());
    }
    void bind_runtime_memory(const trtmc::RuntimeMemoryBindingV1& binding) override {
        if (binding.name.rfind("cache_", 0) == 0) {
            assert(binding.capacity_tokens == expected_cache_capacity_tokens);
            assert(binding.capacity_bytes == expected_cache_binding_bytes);
            assert(binding.bound_tokens == expected_history_bound);
            assert(binding.valid_tokens == expected_history_tokens);
        } else {
            assert(binding.name.rfind("present_", 0) == 0);
            assert(binding.capacity_tokens == expected_present_capacity_tokens);
            assert(binding.capacity_bytes == expected_present_binding_bytes);
            assert(binding.bound_tokens == expected_query_tokens);
            assert(binding.valid_tokens == expected_query_tokens);
        }
        ++binding_count;
    }
    void bind_runtime_memory_alias_pair(const trtmc::RuntimeMemoryAliasBindingV1&) override {
        throw std::logic_error("alias path is not used by fused attention");
    }
    trtmc::RuntimeMemoryContextRequirementV1 context_memory_requirement() override {
        // Model the production module: ordinary dynamic I/O is absent at
        // construction and becomes visible only after this context is planned.
        engine_stats_.ordinary_device_input_bytes = 16;
        engine_stats_.ordinary_device_output_bytes = 32;
        engine_stats_.device_output_bytes = 32;
        engine_stats_.host_output_staging_bytes = 64;
        trtmc::RuntimeMemoryContextRequirementV1 requirement;
        requirement.capacity_bytes =
            context_requirement_
                ? context_requirement_(last_bound_tokens_, last_query_tokens_)
                : context_base_ + static_cast<std::size_t>(last_bound_tokens_) * 64;
        context_requirement_probes.emplace_back(last_bound_tokens_, last_query_tokens_);
        requirement.alignment = alignment_;
        requirement.device = device_;
        return requirement;
    }
    void bind_context_memory(const trtmc::RuntimeMemoryContextBlockV1& block) override {
        assert(block.pointer != nullptr);
        const auto required = context_requirement_
                                  ? context_requirement_(last_bound_tokens_, last_query_tokens_)
                                  : context_base_ + last_bound_tokens_ * 64;
        assert(block.capacity_bytes >= required);
        context_bound = true;
    }
    bool runtime_memory_ready() const noexcept override {
        return context_bound && binding_count >= expected_binding_count;
    }
    trtmc::TensorMap forward_selected(const trtmc::TensorMap&,
                                      const std::vector<std::string>&) override {
        return {};
    }
    trtmc::RuntimeMemoryEngineStatsV1 runtime_memory_engine_stats() const noexcept {
        return engine_stats_;
    }

    void set_engine_stats(std::uintptr_t identity, std::uint64_t weight_bytes,
                          std::uint64_t streamable_bytes = 0,
                          std::uint64_t streaming_budget_bytes = 0,
                          bool streaming_budget_available = true) {
        engine_stats_.engine_identity = identity;
        engine_stats_.total_weight_bytes = weight_bytes;
        engine_stats_.total_weight_bytes_available = true;
        engine_stats_.streamable_weight_bytes = streamable_bytes;
        engine_stats_.weight_streaming_budget_bytes = streaming_budget_bytes;
        engine_stats_.weight_streaming_budget_available = streaming_budget_available;
    }

    void set_context_requirement(
        std::function<std::size_t(std::uint64_t, std::uint64_t)> context_requirement) {
        context_requirement_ = std::move(context_requirement);
    }

    int binding_count{0};
    bool context_bound{false};
    std::uint64_t expected_history_tokens{0};
    std::uint64_t expected_history_bound{1};
    std::uint64_t expected_query_tokens{4};
    std::uint64_t expected_cache_capacity_tokens{4};
    std::uint64_t expected_present_capacity_tokens{4};
    std::size_t expected_cache_binding_bytes{1024};
    std::size_t expected_present_binding_bytes{1024};
    int expected_binding_count{4};
    std::vector<std::pair<std::uint64_t, std::uint64_t>> context_requirement_probes;

  private:
    std::uint64_t token_max_;
    std::uint64_t cache_max_;
    std::size_t context_base_;
    std::size_t alignment_;
    int32_t device_;
    std::uint64_t last_bound_tokens_{1};
    std::uint64_t last_query_tokens_{1};
    std::function<std::size_t(std::uint64_t, std::uint64_t)> context_requirement_;
    trtmc::RuntimeMemoryEngineStatsV1 engine_stats_;
};

class FakeRuntimeModule final : public FakeRuntimeModuleBase,
                                public trtmc::IRuntimeMemoryEngineIntrospectionV1 {
  public:
    using FakeRuntimeModuleBase::FakeRuntimeModuleBase;

    trtmc::RuntimeMemoryEngineStatsV1 runtime_memory_engine_stats() const noexcept override {
        return FakeRuntimeModuleBase::runtime_memory_engine_stats();
    }
};

trtmc::RuntimeKvSetupRequest make_request(FakeRuntimeModuleBase& prefill,
                                          FakeRuntimeModuleBase& decode4,
                                          FakeRuntimeModuleBase& decode16,
                                          const std::shared_ptr<HostAllocator>& allocator) {
    // Prefill is one engine. Decode profile contexts share a second engine.
    // The accounting path must not count that decoder engine twice.
    prefill.set_engine_stats(/*identity=*/1001, /*weight_bytes=*/600);
    decode4.set_engine_stats(/*identity=*/2001, /*weight_bytes=*/500);
    decode16.set_engine_stats(/*identity=*/2001, /*weight_bytes=*/500);

    trtmc::RuntimeKvSetupRequest request;
    request.layout.layer_count = 1;
    request.layout.kv_head_count = 1;
    request.layout.head_dim = 128;
    request.layout.capacity_tokens = 16;
    request.layout.prefill_chunk_limit = 8;
    request.layout.dtype = trtmc::DType::kBFloat16;
    request.layout.names.token_id = "token_id";
    request.layout.names.position_id = "position_id";
    request.layout.names.history_length = "history_length";
    request.layout.names.cache_k = {"cache_k_0"};
    request.layout.names.cache_v = {"cache_v_0"};
    request.layout.names.cache_k_output = {"present_k_0"};
    request.layout.names.cache_v_output = {"present_v_0"};
    request.roles = {
        {&prefill, trtmc::RuntimeKvExecutionRoleKind::kPrefill, 16},
        {&decode4, trtmc::RuntimeKvExecutionRoleKind::kDecode, 4},
        {&decode16, trtmc::RuntimeKvExecutionRoleKind::kDecode, 16},
    };
    request.expected_active_kv_profile_limits = {4, 16};
    request.policy = trtmc::RuntimeKvPolicy{trtmc::RuntimeKvPolicyKind::kBytes, 0.0, 2048};
    request.expected_kv_bytes_per_token = 512;
    request.safety_reserve_bytes = 0;
    request.pre_load_memory_snapshot =
        trtmc::RuntimeDeviceMemorySnapshot{0, (5ULL << 18), (2ULL << 20)};
    request.pre_load_memory_snapshot_available = true;
    request.allocator = allocator;
    request.device_copy = [](void* destination, std::size_t destination_pitch, const void* source,
                             std::size_t source_pitch, std::size_t width_bytes, std::size_t height,
                             void*) {
        auto* destination_bytes = static_cast<std::byte*>(destination);
        const auto* source_bytes = static_cast<const std::byte*>(source);
        for (std::size_t row = 0; row < height; ++row) {
            std::memcpy(destination_bytes + row * destination_pitch,
                        source_bytes + row * source_pitch, width_bytes);
        }
        return cudaSuccess;
    };
    request.stream_synchronize = [](void*) { return cudaSuccess; };
    int observations = 0;
    request.query_device_memory = [observations](const char*) mutable {
        ++observations;
        constexpr std::uint64_t total = 2ULL << 20;
        constexpr std::uint64_t initial_free = 1ULL << 20;
        // R=4 is below C=8, so cold prefill uses the T=1 sentinel.
        constexpr std::uint64_t context_bytes = 4096 + 64;
        constexpr std::uint64_t staging_bytes = 2048;
        const auto free =
            observations == 1
                ? initial_free
                : (observations == 2 ? initial_free - context_bytes - staging_bytes
                                     : initial_free - context_bytes - staging_bytes - 2048);
        return trtmc::RuntimeDeviceMemorySnapshot{0, free, total};
    };
    return request;
}

template <typename Fn>
void expect_throw(Fn&& fn, const std::string& needle) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception& error) {
        threw = true;
        assert(std::string(error.what()).find(needle) != std::string::npos);
    }
    assert(threw);
}

void test_small_r_below_prefill_chunk_and_lifetime() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(/*token_max=*/8, /*cache_max=*/16,
                              /*context_base=*/4096, /*alignment=*/512);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    std::size_t copy_calls = 0;
    std::size_t observed_destination_pitch = 0;
    std::size_t observed_source_pitch = 0;
    std::size_t observed_width_bytes = 0;
    std::size_t observed_height = 0;
    int synchronize_calls = 0;
    request.device_copy = [&](void* destination, std::size_t destination_pitch, const void* source,
                              std::size_t source_pitch, std::size_t width_bytes, std::size_t height,
                              void*) {
        ++copy_calls;
        observed_destination_pitch = destination_pitch;
        observed_source_pitch = source_pitch;
        observed_width_bytes = width_bytes;
        observed_height = height;
        auto* destination_bytes = static_cast<std::byte*>(destination);
        const auto* source_bytes = static_cast<const std::byte*>(source);
        for (std::size_t row = 0; row < height; ++row) {
            std::memcpy(destination_bytes + row * destination_pitch,
                        source_bytes + row * source_pitch, width_bytes);
        }
        return cudaSuccess;
    };
    request.stream_synchronize = [&](void*) {
        ++synchronize_calls;
        return cudaSuccess;
    };

    {
        auto state = trtmc::create_runtime_kv_state(request);
        assert(state->valid());
        assert(state->capacity_tokens() == 4);
        assert(state->layout().prefill_chunk_limit == 8);
        assert(state->allocation_bytes() == 2048);
        assert(state->staging_capacity_tokens() == 4);
        assert(state->staging_bytes() == 2048);
        assert(state->context_allocation_bytes() == 4160);
        assert(state->receipt().context_device_memory_bytes == 4160);
        assert(state->receipt().pre_load_snapshot_available);
        assert(state->receipt().pre_load_free_bytes == (5ULL << 18));
        assert(state->receipt().post_load_total_bytes_available);
        assert(state->receipt().post_load_device_used_bytes == (1ULL << 20));
        assert(state->receipt().capacity_decision_snapshot_available);
        assert(state->receipt().capacity_decision_free_bytes == (1ULL << 20) - (4096 + 64) - 2048);
        assert(state->receipt().settled_snapshot_available);
        assert(state->receipt().settled_free_bytes == (1ULL << 20) - (4096 + 64) - 2048 - 2048);
        assert(state->receipt().final_snapshot_available);
        assert(state->receipt().final_free_bytes == state->receipt().capacity_decision_free_bytes);
        assert(state->receipt().engine_weight_bytes_available);
        assert(state->receipt().engine_weight_bytes == 1100);
        assert(state->receipt().resident_weight_bytes_available);
        assert(state->receipt().resident_weight_bytes == 1100);
        assert(state->receipt().resident_weight_copy_count_available);
        assert(state->receipt().resident_weight_copy_count == 2);
        assert(state->receipt().external_device_output_bytes_available);
        assert(state->receipt().ordinary_device_input_bytes_available);
        assert(state->receipt().ordinary_device_input_bytes == 32);
        assert(state->receipt().ordinary_device_output_bytes_available);
        assert(state->receipt().ordinary_device_output_bytes == 64);
        assert(state->receipt().external_device_output_bytes == 2048);
        assert(state->receipt().host_staging_bytes_available);
        assert(state->receipt().host_staging_bytes == 128);
        assert(state->receipt().peak_device_bytes_available);
        assert(state->receipt().peak_device_bytes == 270592);
        assert(state->receipt().peak_device_sample_count == 1);
        assert(state->receipt().peak_sampled_at_load_completion);
        assert(!state->receipt().peak_sampled_at_request_completion);
        const auto receipt_json = state->receipt_json();
        assert(receipt_json.find("\"receipt_schema_version\":3") != std::string::npos);
        assert(receipt_json.find("\"engine_weight_bytes\":"
                                 "\"tensorrt_engine_stat_total_weights_size\"") !=
               std::string::npos);
        assert(receipt_json.find("\"resident_weight_bytes\":1100") != std::string::npos);
        assert(receipt_json.find("\"resident_weight_copy_count\":2") != std::string::npos);
        assert(receipt_json.find("\"ordinary_device_input_bytes\":32") != std::string::npos);
        assert(receipt_json.find("\"ordinary_device_output_bytes\":64") != std::string::npos);
        assert(receipt_json.find("\"external_device_output_bytes\":2048") != std::string::npos);
        assert(receipt_json.find("\"external_device_output_bytes\":"
                                 "\"runtime_exact_sq_staging_allocation_ledger\"") !=
               std::string::npos);
        assert(receipt_json.find("\"capacity_decision_free_bytes\":"
                                 "\"cuda_mem_get_info_after_tentative_context_and_output_"
                                 "reservation\"") != std::string::npos);
        assert(receipt_json.find("\"settled_free_bytes\":"
                                 "\"cuda_mem_get_info_after_final_context_output_and_kv_"
                                 "allocation\"") != std::string::npos);
        assert(receipt_json.find("\"final_free_bytes\":"
                                 "\"deprecated_alias_of_capacity_decision_free_bytes\"") !=
               std::string::npos);
        assert(receipt_json.find("\"peak_device_bytes\":270592") != std::string::npos);
        assert(receipt_json.find("\"peak_device_bytes_scope\":\"device_wide\"") !=
               std::string::npos);
        assert(receipt_json.find("\"peak_device_sample_boundaries\":"
                                 "[\"after_runtime_kv_allocation\"]") != std::string::npos);
        assert(receipt_json.find("\"peak_device_bytes_unavailable_reason\":null") !=
               std::string::npos);
        assert(allocator->requests.size() == 3);
        assert(allocator->requests[0].bytes == 4160);
        assert(allocator->requests[0].alignment == 512);
        assert(allocator->requests[1].bytes == 2048);
        assert(allocator->requests[1].alignment == 256);
        assert(allocator->requests[2].bytes == 2048);
        assert(allocator->requests[2].alignment == 256);
        assert(allocator->live_bytes() == 8256);
        assert(allocator->live_bytes() == state->receipt().context_device_memory_bytes +
                                              state->staging_bytes() +
                                              state->receipt().kv_reserved_bytes);
        assert(allocator->releases() == 0);

        expect_throw(
            [&] { (void)trtmc::plan_runtime_kv_invocation(prefill, state->layout(), 0, 1, 2); },
            "cold-sentinel");
        expect_throw(
            [&] { (void)trtmc::plan_runtime_kv_invocation(prefill, state->layout(), 1, 1, 1); },
            "cold-sentinel");

        std::memset(state->cache_key_pointer(0), 0x11, 1024);
        std::memset(state->cache_value_pointer(0), 0x22, 1024);
        std::memset(state->staging_key_pointer(0), 0x33, 1024);
        std::memset(state->staging_value_pointer(0), 0x44, 1024);
        prefill.expected_history_tokens = 0;
        prefill.expected_history_bound = 1;
        prefill.expected_query_tokens = 4;
        state->prepare_invocation(prefill, 0, 4, 1);
        assert(prefill.binding_count == 4);
        assert(prefill.context_bound);
        assert(state->allocation_base_address() != 0);
        assert(state->allocation_base_address() % 256 == 0);
        assert(state->last_context_device_memory_bytes() == 4160);
        const auto before = state->commit_snapshot();
        state->commit_current_rows(0, 4);
        const auto after = state->commit_snapshot();
        assert(after.device_to_device_bytes - before.device_to_device_bytes == 2048);
        assert(after.device_to_device_events - before.device_to_device_events == 1);
        assert(copy_calls == 1);
        assert(observed_destination_pitch == 1024);
        assert(observed_source_pitch == 1024);
        assert(observed_width_bytes == 1024);
        assert(observed_height == 2);
        state->synchronize_commits();
        state->synchronize_commits();
        assert(synchronize_calls == 1);
        assert(std::memcmp(state->cache_key_pointer(0), state->staging_key_pointer(0), 1024) == 0);
        assert(std::memcmp(state->cache_value_pointer(0), state->staging_value_pointer(0), 1024) ==
               0);
        assert(allocator->guards_intact());
    }
    assert(allocator->releases() == 3);
    assert(allocator->guards_intact());
}

void test_multilayer_nonzero_history_uses_pitched_append_geometry() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(/*token_max=*/4, /*cache_max=*/16,
                              /*context_base=*/4096, /*alignment=*/512);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    request.layout.layer_count = 3;
    request.layout.prefill_chunk_limit = 4;
    request.layout.names.cache_k = {"cache_k_0", "cache_k_1", "cache_k_2"};
    request.layout.names.cache_v = {"cache_v_0", "cache_v_1", "cache_v_2"};
    request.layout.names.cache_k_output = {"present_k_0", "present_k_1", "present_k_2"};
    request.layout.names.cache_v_output = {"present_v_0", "present_v_1", "present_v_2"};
    request.expected_kv_bytes_per_token = 1536;
    request.policy = trtmc::RuntimeKvPolicy{trtmc::RuntimeKvPolicyKind::kBytes, 0.0, 12288};

    std::size_t copy_calls = 0;
    std::size_t observed_destination_pitch = 0;
    std::size_t observed_source_pitch = 0;
    std::size_t observed_width_bytes = 0;
    std::size_t observed_height = 0;
    request.device_copy = [&](void* destination, std::size_t destination_pitch, const void* source,
                              std::size_t source_pitch, std::size_t width_bytes, std::size_t height,
                              void*) {
        ++copy_calls;
        observed_destination_pitch = destination_pitch;
        observed_source_pitch = source_pitch;
        observed_width_bytes = width_bytes;
        observed_height = height;
        auto* destination_bytes = static_cast<std::byte*>(destination);
        const auto* source_bytes = static_cast<const std::byte*>(source);
        for (std::size_t row = 0; row < height; ++row) {
            std::memcpy(destination_bytes + row * destination_pitch,
                        source_bytes + row * source_pitch, width_bytes);
        }
        return cudaSuccess;
    };

    auto state = trtmc::create_runtime_kv_state(request);
    assert(state->capacity_tokens() == 8);
    assert(state->staging_capacity_tokens() == 4);
    assert(state->allocation_bytes() == 12288);
    assert(state->staging_bytes() == 6144);

    prefill.expected_history_tokens = 2;
    prefill.expected_history_bound = 4;
    prefill.expected_query_tokens = 2;
    prefill.expected_cache_capacity_tokens = 8;
    prefill.expected_present_capacity_tokens = 4;
    prefill.expected_cache_binding_bytes = 2048;
    prefill.expected_present_binding_bytes = 1024;
    prefill.expected_binding_count = 12;

    constexpr std::size_t row_bytes = 256;
    constexpr std::size_t cache_span_bytes = 8 * row_bytes;
    constexpr std::size_t staging_span_bytes = 4 * row_bytes;
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        auto* cache_key = static_cast<std::byte*>(state->cache_key_pointer(layer));
        auto* cache_value = static_cast<std::byte*>(state->cache_value_pointer(layer));
        auto* staging_key = static_cast<std::byte*>(state->staging_key_pointer(layer));
        auto* staging_value = static_cast<std::byte*>(state->staging_value_pointer(layer));
        std::memset(cache_key, 0x51 + static_cast<int>(layer), cache_span_bytes);
        std::memset(cache_value, 0x61 + static_cast<int>(layer), cache_span_bytes);
        for (std::size_t row = 0; row < 4; ++row) {
            std::memset(staging_key + row * row_bytes, 0x10 + static_cast<int>(layer * 8 + row),
                        row_bytes);
            std::memset(staging_value + row * row_bytes, 0x30 + static_cast<int>(layer * 8 + row),
                        row_bytes);
        }
    }

    state->prepare_invocation(prefill, /*history_tokens=*/2, /*query_tokens=*/2,
                              /*bound_tokens=*/4);
    assert(prefill.binding_count == 12);
    state->commit_current_rows(/*history_tokens=*/2, /*query_tokens=*/2);
    state->synchronize_commits();

    assert(copy_calls == 1);
    assert(observed_destination_pitch == cache_span_bytes);
    assert(observed_source_pitch == staging_span_bytes);
    assert(observed_width_bytes == 2 * row_bytes);
    assert(observed_height == 6);
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        const auto* cache_key = static_cast<const std::byte*>(state->cache_key_pointer(layer));
        const auto* cache_value = static_cast<const std::byte*>(state->cache_value_pointer(layer));
        const auto* staging_key = static_cast<const std::byte*>(state->staging_key_pointer(layer));
        const auto* staging_value =
            static_cast<const std::byte*>(state->staging_value_pointer(layer));
        for (std::size_t row = 0; row < 8; ++row) {
            if (row >= 2 && row < 4) {
                assert(std::memcmp(cache_key + row * row_bytes, staging_key + (row - 2) * row_bytes,
                                   row_bytes) == 0);
                assert(std::memcmp(cache_value + row * row_bytes,
                                   staging_value + (row - 2) * row_bytes, row_bytes) == 0);
            } else {
                const auto expected_key =
                    std::byte{static_cast<unsigned char>(0x51 + static_cast<int>(layer))};
                const auto expected_value =
                    std::byte{static_cast<unsigned char>(0x61 + static_cast<int>(layer))};
                for (std::size_t column = 0; column < row_bytes; ++column) {
                    assert(cache_key[row * row_bytes + column] == expected_key);
                    assert(cache_value[row * row_bytes + column] == expected_value);
                }
            }
        }
    }
    assert(allocator->guards_intact());
}

void test_copy_failure_poison_is_fail_closed() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(8, 16, 4096, 512);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    request.device_copy = [](void*, std::size_t, const void*, std::size_t, std::size_t, std::size_t,
                             void*) { return cudaErrorInvalidValue; };
    auto state = trtmc::create_runtime_kv_state(request);
    prefill.expected_history_tokens = 0;
    prefill.expected_history_bound = 1;
    prefill.expected_query_tokens = 4;
    state->prepare_invocation(prefill, 0, 4, 1);
    expect_throw([&] { state->commit_current_rows(0, 4); }, "D2D commit failed");
    assert(!state->valid());
    assert(state->commit_snapshot().device_to_device_bytes == 0);
    assert(state->commit_snapshot().device_to_device_events == 0);
    expect_throw([&] { state->reset_request_state(); }, "poisoned");
    expect_throw([&] { state->prepare_invocation(prefill, 0, 4, 1); }, "poisoned");
    assert(allocator->guards_intact());
}

void test_delayed_copy_failure_poison_is_fail_closed() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(8, 16, 4096, 512);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    int synchronize_calls = 0;
    request.stream_synchronize = [&synchronize_calls](void*) {
        ++synchronize_calls;
        return cudaErrorLaunchFailure;
    };
    auto state = trtmc::create_runtime_kv_state(request);
    prefill.expected_history_tokens = 0;
    prefill.expected_history_bound = 1;
    prefill.expected_query_tokens = 4;
    state->prepare_invocation(prefill, 0, 4, 1);
    state->commit_current_rows(0, 4);
    expect_throw([&] { state->synchronize_commits(); }, "asynchronous D2D commit failed");
    assert(synchronize_calls == 1);
    assert(!state->valid());
    expect_throw([&] { state->reset_request_state(); }, "poisoned");
    expect_throw([&] { state->prepare_invocation(prefill, 0, 4, 1); }, "poisoned");
    assert(allocator->guards_intact());
}

void test_final_fraction_shrink_reallocates_exact_overhead_below_c() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(8, 16, 4096, 512);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    request.policy = trtmc::RuntimeKvPolicy{trtmc::RuntimeKvPolicyKind::kAuto, 1.0, 0};
    int observations = 0;
    std::vector<std::string> phases;
    request.query_device_memory = [&observations, &phases](const char* phase) {
        ++observations;
        phases.emplace_back(phase);
        constexpr std::uint64_t total = 2ULL << 20;
        if (observations == 1)
            return trtmc::RuntimeDeviceMemorySnapshot{0, 1ULL << 20, total};
        if (observations == 2)
            return trtmc::RuntimeDeviceMemorySnapshot{0, 2048, total};
        // The smaller settled envelope may make substantially more memory
        // visible, but that later reading must never increase the R=4 decision.
        return trtmc::RuntimeDeviceMemorySnapshot{0, 1ULL << 20, total};
    };

    {
        auto state = trtmc::create_runtime_kv_state(request);
        assert(state->capacity_tokens() == 4);
        assert(state->staging_capacity_tokens() == 4);
        assert(state->staging_bytes() == 2048);
        assert(state->context_allocation_bytes() == 4160);
        assert(state->receipt().context_device_memory_bytes == 4160);
        assert(state->receipt().external_device_output_bytes == 2048);
        assert(state->receipt().kv_reserved_bytes == 2048);
        assert(state->receipt().capacity_decision_snapshot_available);
        assert(state->receipt().capacity_decision_free_bytes == 2048);
        assert(state->receipt().settled_snapshot_available);
        assert(state->receipt().settled_free_bytes == (1ULL << 20));
        assert(state->receipt().final_free_bytes == state->receipt().capacity_decision_free_bytes);
        assert(phases == std::vector<std::string>({
                             "before runtime KV planning",
                             "after shared context and output allocation",
                             "after runtime KV allocation",
                         }));
        assert(allocator->requests.size() == 5);
        assert(allocator->requests[0].bytes == 9216);
        assert(allocator->requests[1].bytes == 4096);
        assert(allocator->requests[2].bytes == 4160);
        assert(allocator->requests[2].live_bytes_before == 0);
        assert(allocator->requests[3].bytes == 2048);
        assert(allocator->requests[3].live_bytes_before == 4160);
        assert(allocator->requests[4].bytes == 2048);
        assert(allocator->requests[4].live_bytes_before == 6208);
        assert(allocator->releases() == 2);
        assert(allocator->live_bytes() == 8256);
        assert(allocator->live_bytes() == state->receipt().context_device_memory_bytes +
                                              state->staging_bytes() +
                                              state->receipt().kv_reserved_bytes);
        assert(allocator->guards_intact());
    }
    assert(allocator->releases() == 5);
    assert(allocator->live_bytes() == 0);
    assert(allocator->guards_intact());
}

void test_final_fraction_shrink_reallocates_larger_context_from_f2_delta() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(8, 16, 64);
    FakeRuntimeModule decode4(1, 4, 64);
    FakeRuntimeModule decode16(1, 16, 64);
    auto request = make_request(prefill, decode4, decode16, allocator);
    request.policy = trtmc::RuntimeKvPolicy{trtmc::RuntimeKvPolicyKind::kAuto, 1.0, 0};

    const auto discontinuous_context = [](std::uint64_t bound_tokens, std::uint64_t) {
        // O(16)=64, while the final 5..15 history envelope requires a
        // larger USER_MANAGED context block.
        return static_cast<std::size_t>(bound_tokens > 4 && bound_tokens < 16 ? 576 : 64);
    };
    prefill.set_context_requirement(discontinuous_context);
    decode4.set_context_requirement(discontinuous_context);
    decode16.set_context_requirement(discontinuous_context);

    int observations = 0;
    request.query_device_memory = [&observations](const char*) {
        ++observations;
        constexpr std::uint64_t total = 2ULL << 20;
        if (observations == 1)
            return trtmc::RuntimeDeviceMemorySnapshot{0, 1ULL << 20, total};
        if (observations == 2)
            return trtmc::RuntimeDeviceMemorySnapshot{0, 6144, total};
        // A settled observation is receipt-only and cannot increase R.
        return trtmc::RuntimeDeviceMemorySnapshot{0, 1ULL << 20, total};
    };

    {
        auto state = trtmc::create_runtime_kv_state(request);
        // At F2, 12 rows fit before accounting for O(12)-O(16)=512.
        // Charging that positive delta resolves exactly to 11 rows.
        assert(state->capacity_tokens() == 11);
        assert(state->context_allocation_bytes() == 576);
        assert(state->staging_bytes() == 4096);
        assert(state->receipt().context_device_memory_bytes == 576);
        assert(state->receipt().kv_reserved_bytes == 5632);
        assert(state->receipt().kv_budget_bytes == 5632);
        assert(state->receipt().capacity_decision_free_bytes == 6144);
        assert(state->receipt().settled_free_bytes == (1ULL << 20));

        assert(allocator->requests.size() == 4);
        assert(allocator->requests[0].bytes == 64);
        assert(allocator->requests[1].bytes == 4096);
        // The old 64-byte context is released before allocating the larger
        // exact final envelope, so both contexts are never live together.
        assert(allocator->requests[2].bytes == 576);
        assert(allocator->requests[2].live_bytes_before == 4096);
        assert(allocator->requests[3].bytes == 5632);
        assert(allocator->requests[3].live_bytes_before == 4672);
        assert(allocator->releases() == 1);
        assert(allocator->live_bytes() == 10304);
        assert(allocator->live_bytes() == state->receipt().context_device_memory_bytes +
                                              state->staging_bytes() +
                                              state->receipt().kv_reserved_bytes);
        assert(allocator->guards_intact());
    }
    assert(allocator->releases() == 4);
    assert(allocator->live_bytes() == 0);
    assert(allocator->guards_intact());
}

void test_streamed_weight_residency_is_not_invented() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(8, 16, 4096, 512);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    prefill.set_engine_stats(/*identity=*/1001, /*weight_bytes=*/600,
                             /*streamable_bytes=*/400,
                             /*streaming_budget_bytes=*/200);

    auto state = trtmc::create_runtime_kv_state(request);
    assert(state->receipt().weight_streaming_active);
    assert(state->receipt().engine_weight_bytes_available);
    assert(state->receipt().engine_weight_bytes == 1100);
    assert(!state->receipt().resident_weight_bytes_available);
    assert(state->receipt().resident_weight_copy_count_available);
    assert(state->receipt_json().find("\"resident_weight_bytes\":null") != std::string::npos);
    assert(state->receipt_json().find("\"measurement_sources\":{\"pre_load_free_bytes\":") !=
           std::string::npos);
}

void test_engine_profiles_must_match_bundle_contract() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(8, 16, 4096);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    request.roles.pop_back();
    expect_throw([&] { (void)trtmc::create_runtime_kv_state(request); },
                 "do not match the bundle contract");
    assert(allocator->requests.empty());
}

void test_context_roles_must_share_selected_device() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(8, 16, 4096, 256, 1);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    expect_throw([&] { (void)trtmc::create_runtime_kv_state(request); }, "different CUDA devices");
    assert(allocator->requests.empty());
}

void test_high_water_observability_failure_does_not_fail_load() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(8, 16, 4096, 512);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    int observations = 0;
    request.query_device_memory = [&observations](const char*) {
        ++observations;
        if (observations == 3)
            throw std::runtime_error("injected observability failure");
        constexpr std::uint64_t total = 2ULL << 20;
        constexpr std::uint64_t initial_free = 1ULL << 20;
        const auto free = observations == 1 ? initial_free : initial_free - 4160 - 2048;
        return trtmc::RuntimeDeviceMemorySnapshot{0, free, total};
    };

    auto state = trtmc::create_runtime_kv_state(request);

    assert(state->valid());
    assert(!state->receipt().settled_snapshot_available);
    assert(state->receipt_json().find("\"settled_free_bytes\":null") != std::string::npos);
    assert(state->receipt_json().find("\"settled_snapshot_unavailable_reason\":"
                                      "\"settled_cuda_mem_get_info_failed\"") != std::string::npos);
    assert(!state->receipt().peak_device_bytes_available);
    assert(state->receipt_json().find("\"peak_device_bytes_unavailable_reason\":"
                                      "\"load_completion_cuda_mem_get_info_failed\"") !=
           std::string::npos);
}

void test_settled_snapshot_does_not_require_a_peak_baseline() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(8, 16, 4096, 512);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    request.pre_load_memory_snapshot_available = false;

    auto state = trtmc::create_runtime_kv_state(request);

    assert(state->valid());
    assert(state->receipt().settled_snapshot_available);
    assert(state->receipt().settled_free_bytes > 0);
    assert(!state->receipt().peak_device_bytes_available);
    assert(state->receipt_json().find("\"peak_device_bytes_unavailable_reason\":"
                                      "\"pre_load_cuda_memory_snapshot_unavailable\"") !=
           std::string::npos);
}

void test_external_staging_accounting_does_not_require_engine_introspection() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModuleBase prefill(8, 16, 4096, 512);
    FakeRuntimeModuleBase decode4(1, 4, 2048);
    FakeRuntimeModuleBase decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);

    auto state = trtmc::create_runtime_kv_state(request);

    assert(state->receipt().external_device_output_bytes_available);
    assert(state->receipt().external_device_output_bytes == 2048);
    assert(!state->receipt().ordinary_device_input_bytes_available);
    assert(!state->receipt().ordinary_device_output_bytes_available);
    assert(!state->receipt().host_staging_bytes_available);
    const auto receipt_json = state->receipt_json();
    assert(receipt_json.find("\"external_device_output_bytes\":2048") != std::string::npos);
    assert(receipt_json.find("\"ordinary_device_input_bytes\":null") != std::string::npos);
    assert(receipt_json.find("\"ordinary_device_output_bytes\":null") != std::string::npos);
    assert(receipt_json.find("\"host_staging_bytes\":null") != std::string::npos);
}

void test_context_envelope_sweeps_every_reachable_prefill_shape() {
    auto allocator = std::make_shared<HostAllocator>();
    FakeRuntimeModule prefill(/*token_max=*/8, /*cache_max=*/16, /*context_base=*/4096);
    FakeRuntimeModule decode4(1, 4, 2048);
    FakeRuntimeModule decode16(1, 16, 8192);
    auto request = make_request(prefill, decode4, decode16, allocator);
    request.policy = trtmc::RuntimeKvPolicy{trtmc::RuntimeKvPolicyKind::kBytes, 0.0, 8192};
    request.query_device_memory = [](const char*) {
        return trtmc::RuntimeDeviceMemorySnapshot{0, 1ULL << 20, 2ULL << 20};
    };

    constexpr std::size_t discontinuous_middle_shape_bytes = 65536;
    prefill.set_context_requirement([](std::uint64_t bound_tokens, std::uint64_t query_tokens) {
        if (bound_tokens == 16 && query_tokens == 3)
            return discontinuous_middle_shape_bytes;
        return static_cast<std::size_t>(4096 + bound_tokens * 64);
    });

    auto state = trtmc::create_runtime_kv_state(request);
    assert(state->capacity_tokens() == 16);
    assert(state->context_allocation_bytes() == discontinuous_middle_shape_bytes);
    assert(state->receipt().context_device_memory_bytes == discontinuous_middle_shape_bytes);
    assert(std::find(prefill.context_requirement_probes.begin(),
                     prefill.context_requirement_probes.end(),
                     std::pair<std::uint64_t, std::uint64_t>{16, 3}) !=
           prefill.context_requirement_probes.end());

    prefill.expected_history_tokens = 8;
    prefill.expected_history_bound = 16;
    prefill.expected_query_tokens = 3;
    prefill.expected_cache_capacity_tokens = 16;
    prefill.expected_present_capacity_tokens = 8;
    prefill.expected_cache_binding_bytes = 4096;
    prefill.expected_present_binding_bytes = 2048;
    state->prepare_invocation(prefill, /*history_tokens=*/8, /*query_tokens=*/3,
                              /*bound_tokens=*/16);
    assert(state->last_context_device_memory_bytes() == discontinuous_middle_shape_bytes);
}

} // namespace

int main() {
    test_small_r_below_prefill_chunk_and_lifetime();
    test_multilayer_nonzero_history_uses_pitched_append_geometry();
    test_copy_failure_poison_is_fail_closed();
    test_delayed_copy_failure_poison_is_fail_closed();
    test_final_fraction_shrink_reallocates_exact_overhead_below_c();
    test_final_fraction_shrink_reallocates_larger_context_from_f2_delta();
    test_streamed_weight_residency_is_not_invented();
    test_engine_profiles_must_match_bundle_contract();
    test_context_roles_must_share_selected_device();
    test_high_water_observability_failure_does_not_fail_load();
    test_settled_snapshot_does_not_require_a_peak_baseline();
    test_external_staging_accounting_does_not_require_engine_introspection();
    test_context_envelope_sweeps_every_reachable_prefill_shape();
    return 0;
}
