/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/distributed_runtime.h"

#include "runtime/core/cuda_common.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

namespace trtmc {

namespace {

struct NcclUniqueId {
    char internal[128];
};

using NcclComm = void*;
using NcclResult = int;
using NcclGetUniqueIdFn = NcclResult (*)(NcclUniqueId*);
using NcclCommInitRankFn = NcclResult (*)(NcclComm*, int, NcclUniqueId, int);
using NcclCommDestroyFn = NcclResult (*)(NcclComm);
using NcclGetErrorStringFn = const char* (*)(NcclResult);
using NcclAllGatherFn = NcclResult (*)(const void*, void*, std::size_t, int, NcclComm,
                                       cudaStream_t);

constexpr int kNcclFloat32 = 7;

int env_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
        return fallback;
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw)
        return fallback;
    return static_cast<int>(value);
}

std::string env_string(const char* name, const std::string& fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
        return fallback;
    return raw;
}

int detect_world_size() {
    int size = env_int("OMPI_COMM_WORLD_SIZE", -1);
    if (size > 0)
        return size;
    size = env_int("PMI_SIZE", -1);
    if (size > 0)
        return size;
    return env_int("WORLD_SIZE", 1);
}

int detect_rank() {
    int rank = env_int("OMPI_COMM_WORLD_RANK", -1);
    if (rank >= 0)
        return rank;
    rank = env_int("PMI_RANK", -1);
    if (rank >= 0)
        return rank;
    return env_int("RANK", 0);
}

std::filesystem::path rendezvous_path(const std::string& group_key = {}) {
    std::string base = env_string("TRTMC_NCCL_RENDEZVOUS", "");
    std::filesystem::path path;
    if (!base.empty()) {
        path = std::filesystem::path(base);
    } else {
        std::string job = env_string("OMPI_COMM_WORLD_JOBID", "");
        if (job.empty())
            job = env_string("PMIX_NAMESPACE", "");
        if (job.empty())
            job = std::to_string(getppid());
        path = std::filesystem::temp_directory_path() /
               ("trtmc_nccl_" + std::to_string(getuid()) + "_" + job + ".bin");
    }
    if (group_key.empty())
        return path;
    if (group_key.find_first_not_of(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-") !=
        std::string::npos) {
        throw std::invalid_argument("NCCL subgroup key contains unsafe characters");
    }
    return std::filesystem::path(path.string() + "." + group_key);
}

class NcclRuntime {
  public:
    NcclRuntime() {
        handle_ = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr)
            handle_ = dlopen("libnccl.so", RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            throw std::runtime_error(
                std::string("Failed to load NCCL for tensor parallel runtime: ") + dlerror());
        }
        get_unique_id_ = load<NcclGetUniqueIdFn>("ncclGetUniqueId");
        comm_init_rank_ = load<NcclCommInitRankFn>("ncclCommInitRank");
        comm_destroy_ = load<NcclCommDestroyFn>("ncclCommDestroy");
        get_error_string_ = load<NcclGetErrorStringFn>("ncclGetErrorString");
        all_gather_ = load<NcclAllGatherFn>("ncclAllGather");
    }

    ~NcclRuntime() {
        if (comm_ != nullptr) {
            if (env_int("TRTMC_NCCL_SKIP_DESTROY", 0) != 0) {
                // Escape hatch for process-exit hangs in external runtimes.
                // Normal teardown should destroy the communicator after TRT
                // contexts and engines have released it.
                comm_ = nullptr;
                handle_ = nullptr;
            } else {
                comm_destroy_(comm_);
                comm_ = nullptr;
            }
        }
        if (handle_ != nullptr)
            dlclose(handle_);
    }

    void init(int tp_size, int rank, const NcclUniqueId& id) {
        check(comm_init_rank_(&comm_, tp_size, id, rank), "ncclCommInitRank");
    }

    NcclUniqueId unique_id() {
        NcclUniqueId id{};
        check(get_unique_id_(&id), "ncclGetUniqueId");
        return id;
    }

    void* communicator() const { return comm_; }

    void all_gather_float(const void* send_buffer, void* receive_buffer, std::size_t element_count,
                          cudaStream_t stream) const {
        if (element_count > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
            throw std::invalid_argument("NCCL all-gather element count is too large");
        check(all_gather_(send_buffer, receive_buffer, element_count, kNcclFloat32, comm_, stream),
              "ncclAllGather");
    }

  private:
    template <typename T>
    T load(const char* symbol) {
        dlerror();
        void* raw = dlsym(handle_, symbol);
        const char* err = dlerror();
        if (err != nullptr || raw == nullptr)
            throw std::runtime_error(std::string("Failed to resolve NCCL symbol ") + symbol);
        return reinterpret_cast<T>(raw);
    }

    void check(NcclResult result, const char* op) const {
        if (result == 0)
            return;
        const char* msg = get_error_string_ ? get_error_string_(result) : "unknown NCCL error";
        throw std::runtime_error(std::string(op) + " failed: " + msg);
    }

    void* handle_{nullptr};
    NcclComm comm_{nullptr};
    NcclGetUniqueIdFn get_unique_id_{nullptr};
    NcclCommInitRankFn comm_init_rank_{nullptr};
    NcclCommDestroyFn comm_destroy_{nullptr};
    NcclGetErrorStringFn get_error_string_{nullptr};
    NcclAllGatherFn all_gather_{nullptr};
};

void write_unique_id(const std::filesystem::path& path, const NcclUniqueId& id) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("Failed to write NCCL rendezvous file: " + tmp);
        out.write(id.internal, sizeof(id.internal));
    }
    std::filesystem::rename(tmp, path);
}

NcclUniqueId read_unique_id(const std::filesystem::path& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!std::filesystem::exists(path)) {
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error("Timed out waiting for NCCL rendezvous file: " +
                                     path.string());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    NcclUniqueId id{};
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Failed to read NCCL rendezvous file: " + path.string());
    in.read(id.internal, sizeof(id.internal));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(id.internal)))
        throw std::runtime_error("Short NCCL rendezvous file: " + path.string());
    return id;
}

void bind_cuda_device_for_rank(int rank) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        return;
    if (rank >= count) {
        throw std::runtime_error("Tensor-parallel rank requires a visible CUDA device with the "
                                 "same ordinal for this single-node runtime");
    }
    auto status = cudaSetDevice(rank);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaSetDevice failed for tensor parallel rank: ") +
                                 cudaGetErrorString(status));
}

DistributedRuntimeGroup initialize_group(int group_size, int group_rank,
                                         const std::string& group_key, bool require_full_world) {
    DistributedRuntimeGroup group;
    group.tp_size = group_size;
    group.world_size = group_size;
    group.rank = group_rank;
    group.global_world_size = detect_world_size();
    group.global_rank = detect_rank();

    if (group_size <= 0)
        throw std::invalid_argument("Distributed group size must be positive");
    if (group_rank < 0 || group_rank >= group_size)
        throw std::invalid_argument("Distributed group rank is outside the group");
    if (group.global_rank < 0 || group.global_rank >= group.global_world_size)
        throw std::runtime_error("Distributed launcher rank is outside the global world");
    if (require_full_world && group.global_world_size != group_size) {
        throw std::runtime_error("Tensor-parallel runtime requires mpirun world size to equal "
                                 "tensor_parallel_size for this initial implementation");
    }
    if (group_size > group.global_world_size)
        throw std::runtime_error("Distributed subgroup is larger than the launcher world");
    bind_cuda_device_for_rank(group.global_rank);
    if (group_size == 1)
        return group;

    auto runtime = std::make_shared<NcclRuntime>();
    const auto path = rendezvous_path(group_key);
    NcclUniqueId id{};
    if (group.rank == 0) {
        id = runtime->unique_id();
        write_unique_id(path, id);
    } else {
        id = read_unique_id(path);
    }
    runtime->init(group_size, group.rank, id);
    group.communicator = runtime->communicator();
    group.owner = runtime;
    group.all_gather_float =
        [runtime = std::move(runtime)](const void* send_buffer, void* receive_buffer,
                                       std::size_t element_count, cudaStream_t stream) {
            runtime->all_gather_float(send_buffer, receive_buffer, element_count, stream);
        };
    return group;
}

} // namespace

DistributedRuntimeGroup initialize_tensor_parallel_group(int tp_size) {
    if (tp_size <= 1) {
        DistributedRuntimeGroup group;
        group.tp_size = tp_size;
        group.world_size = detect_world_size();
        group.rank = detect_rank();
        group.global_world_size = group.world_size;
        group.global_rank = group.rank;
        return group;
    }
    const int rank = detect_rank();
    return initialize_group(tp_size, rank, {}, true);
}

DistributedRuntimeGroup initialize_distributed_subgroup(int group_size, int group_rank,
                                                        const std::string& group_key) {
    if (group_key.empty())
        throw std::invalid_argument("Distributed subgroup key must not be empty");
    return initialize_group(group_size, group_rank, group_key, false);
}

void distributed_all_gather_float(const DistributedRuntimeGroup& group, const void* send_buffer,
                                  void* receive_buffer, std::size_t element_count,
                                  cudaStream_t stream) {
    if (group.world_size <= 1) {
        throw std::invalid_argument("Distributed all-gather requires a multi-rank group");
    }
    if (send_buffer == nullptr || receive_buffer == nullptr || stream == nullptr)
        throw std::invalid_argument("Distributed all-gather received a null argument");
    if (!group.all_gather_float)
        throw std::runtime_error("Distributed group does not provide NCCL all-gather");
    group.all_gather_float(send_buffer, receive_buffer, element_count, stream);
}

int distributed_process_world_size() {
    return detect_world_size();
}

int distributed_process_rank() {
    return detect_rank();
}

} // namespace trtmc
