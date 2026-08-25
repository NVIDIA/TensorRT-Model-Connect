/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/distributed_runtime.h"

#include "runtime/core/cuda_common.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace trtmc {

#if defined(_WIN32)

DistributedRuntimeGroup initialize_tensor_parallel_group(int tp_size) {
    DistributedRuntimeGroup group;
    group.tp_size = tp_size;
    if (tp_size > 1) {
        throw std::runtime_error(
            "Tensor-parallel NCCL runtime is not supported by the native Windows build");
    }
    return group;
}

#else

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

std::filesystem::path rendezvous_path() {
    std::string base = env_string("TRTMC_NCCL_RENDEZVOUS", "");
    if (!base.empty())
        return std::filesystem::path(base);
    std::string job = env_string("OMPI_COMM_WORLD_JOBID", "");
    if (job.empty())
        job = env_string("PMIX_NAMESPACE", "");
    if (job.empty())
        job = std::to_string(getppid());
    return std::filesystem::temp_directory_path() /
           ("trtmc_nccl_" + std::to_string(getuid()) + "_" + job + ".bin");
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

} // namespace

DistributedRuntimeGroup initialize_tensor_parallel_group(int tp_size) {
    DistributedRuntimeGroup group;
    group.tp_size = tp_size;
    group.world_size = detect_world_size();
    group.rank = detect_rank();

    if (tp_size <= 1)
        return group;
    if (group.world_size != tp_size) {
        throw std::runtime_error("Tensor-parallel runtime requires mpirun world size to equal "
                                 "tensor_parallel_size for this initial implementation");
    }
    if (group.rank < 0 || group.rank >= tp_size)
        throw std::runtime_error("Tensor-parallel rank is outside [0, tp_size)");

    bind_cuda_device_for_rank(group.rank);
    auto runtime = std::make_shared<NcclRuntime>();
    const auto path = rendezvous_path();
    NcclUniqueId id{};
    if (group.rank == 0) {
        id = runtime->unique_id();
        write_unique_id(path, id);
    } else {
        id = read_unique_id(path);
    }
    runtime->init(tp_size, group.rank, id);
    group.communicator = runtime->communicator();
    group.owner = std::move(runtime);
    return group;
}

#endif

} // namespace trtmc
