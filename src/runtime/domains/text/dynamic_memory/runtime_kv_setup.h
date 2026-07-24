/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/domains/text/dynamic_memory/runtime_kv_state.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class IRuntimeMemoryBackendV1;
class ITrtModule;

enum class RuntimeKvExecutionRoleKind {
    kPrefill,
    kDecode,
};

// One already-deserialized USER_MANAGED execution context. Decode
// profile_limit is the profile's largest legal T; prefill ignores it because
// the qualified prefill profile spans the full model context envelope.
struct RuntimeKvExecutionRole {
    ITrtModule* module{nullptr};
    RuntimeKvExecutionRoleKind kind{RuntimeKvExecutionRoleKind::kDecode};
    std::uint64_t profile_limit{0};
};

struct RuntimeDeviceMemorySnapshot {
    std::uint32_t device{0};
    std::uint64_t free_bytes{0};
    std::uint64_t total_bytes{0};
};

// Empty in production requests, where setup synchronizes and queries CUDA
// directly. Tests can inject deterministic post-load/final observations
// without allocating GPU memory.
using RuntimeDeviceMemoryQuery = std::function<RuntimeDeviceMemorySnapshot(const char* phase)>;

// Internal qualification hook. Production leaves this unset. The real-bundle
// runner uses it to take an independent process-memory sample at the exact
// cudaMemGetInfo boundary used by the runtime receipt; it is intentionally not
// part of the installed public API or C ABI.
using RuntimeDeviceMemoryQualificationObserver =
    std::function<void(const char* phase, const RuntimeDeviceMemorySnapshot& snapshot)>;
using RuntimeDeviceMemoryQualificationPreSnapshotAction = std::function<void(const char* phase)>;

struct RuntimeKvSetupRequest {
    RuntimeKvGraphLayout layout;
    std::vector<RuntimeKvExecutionRole> roles;
    // Persistent bundle authority. The runtime rejects an engine whose
    // deserialized decode profiles do not exactly implement these limits,
    // including the terminal M profile.
    std::vector<std::uint64_t> expected_active_kv_profile_limits;
    RuntimeKvPolicy policy;
    std::uint64_t request_context_limit{0};
    std::uint64_t expected_kv_bytes_per_token{0};
    std::uint64_t safety_reserve_bytes{64ULL << 20};
    std::uint64_t serialized_plan_bytes{0};
    RuntimeDeviceMemorySnapshot pre_load_memory_snapshot;
    bool pre_load_memory_snapshot_available{false};
    void* stream{nullptr};
    std::shared_ptr<IRuntimeDeviceAllocator> allocator{make_cuda_runtime_device_allocator()};
    RuntimeKvDeviceCopy device_copy;
    RuntimeKvStreamSynchronize stream_synchronize;
    RuntimeDeviceMemoryQuery query_device_memory;
};

// Synchronize the current CUDA device and return an exact cudaMemGetInfo
// observation. Model loaders use this immediately before engine
// deserialization so the receipt can distinguish pre-load and post-load free
// memory without pretending that their delta is pure weight memory.
RuntimeDeviceMemorySnapshot query_runtime_device_memory_snapshot(const char* phase);

void set_runtime_device_memory_qualification_observer(
    RuntimeDeviceMemoryQualificationObserver observer);
void set_runtime_device_memory_qualification_pre_snapshot_action(
    RuntimeDeviceMemoryQualificationPreSnapshotAction action);

// Resolve the post-load policy, reserve one shared actual-shape context block,
// re-check free memory, and finally allocate exactly R rows of one contiguous
// K/V slab. No profile-MAX K/V allocation is made by this function.
std::unique_ptr<RuntimeKvStateCore> create_runtime_kv_state(const RuntimeKvSetupRequest& request);

} // namespace trtmc
