#pragma once

#include <memory>
#include <string>

namespace trtmc {

struct DistributedRuntimeGroup {
    int world_size{1};
    int rank{0};
    int local_rank{0};
    int tp_size{1};
    void* communicator{nullptr};
    std::shared_ptr<void> owner;
};

struct DistributedPlanRuntimeConfig {
    bool enabled{false};
    int world_size{1};
    int tp_size{1};
    int pp_size{1};
    int cp_size{1};
    int dp_size{1};
    int ep_size{1};
    std::string component;
    std::string rank_section_pattern;
};

// Parse the runtime-relevant part of distributed_plan.json for one component.
// The initial runtime supports TP-only plans; non-TP mesh axes must be 1.
DistributedPlanRuntimeConfig parse_distributed_plan_runtime_config(
    const std::string& plan_json, const std::string& component);

// Replace "{rank}" in a rank-local section pattern with the concrete rank.
std::string distributed_rank_section_name(const std::string& pattern, int rank);

// Initialize an NCCL communicator for TensorRT 11.0+ distributed collective layers.
//
// This intentionally avoids compile-time MPI/NCCL dependencies: ranks are
// discovered from common mpirun environment variables, and NCCL is loaded with
// dlopen at runtime. Rank 0 writes the NCCL unique ID to a small rendezvous
// file under /tmp unless TRTMC_NCCL_RENDEZVOUS points elsewhere.
DistributedRuntimeGroup initialize_tensor_parallel_group(int tp_size);

using MeshRuntimeGroup = DistributedRuntimeGroup;
using MeshRuntimeConfig = DistributedPlanRuntimeConfig;

// Initialize the runtime group described by a distributed plan. This is the
// plan-driven mesh-runtime entry point used by plugins.
MeshRuntimeGroup initialize_mesh_runtime_group(const MeshRuntimeConfig& config);

// Compatibility wrapper for existing plugin call sites.
DistributedRuntimeGroup initialize_distributed_group(
    const DistributedPlanRuntimeConfig& config);

} // namespace trtmc
