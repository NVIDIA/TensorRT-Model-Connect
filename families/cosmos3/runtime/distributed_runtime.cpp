/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/cosmos3/runtime/distributed_runtime.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace trtmc::cosmos3 {

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

int require_env_int(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
        throw std::runtime_error(std::string("cosmos3 context parallel requires ") + name);
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0')
        throw std::runtime_error(std::string("cosmos3 invalid integer in ") + name);
    return static_cast<int>(value);
}

int detect_world_size() {
    return require_env_int("OMPI_COMM_WORLD_SIZE");
}

int detect_rank() {
    return require_env_int("OMPI_COMM_WORLD_RANK");
}

std::filesystem::path rendezvous_path() {
    const char* path = std::getenv("TRTMC_NCCL_RENDEZVOUS");
    if (path == nullptr || *path == '\0')
        throw std::runtime_error("cosmos3 context parallel requires TRTMC_NCCL_RENDEZVOUS");
    return path;
}

class NcclRuntime {
  public:
    NcclRuntime() {
        handle_ = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            throw std::runtime_error(
                std::string("Failed to load NCCL for context parallel runtime: ") + dlerror());
        }
        get_unique_id_ = load<NcclGetUniqueIdFn>("ncclGetUniqueId");
        comm_init_rank_ = load<NcclCommInitRankFn>("ncclCommInitRank");
        comm_destroy_ = load<NcclCommDestroyFn>("ncclCommDestroy");
        get_error_string_ = load<NcclGetErrorStringFn>("ncclGetErrorString");
    }

    ~NcclRuntime() {
        if (comm_ != nullptr) {
            comm_destroy_(comm_);
            comm_ = nullptr;
        }
        if (handle_ != nullptr)
            dlclose(handle_);
    }

    void init(int cp_size, int rank, const NcclUniqueId& id) {
        check(comm_init_rank_(&comm_, cp_size, id, rank), "ncclCommInitRank");
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
        const char* msg = get_error_string_(result);
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

void bind_cuda_device_for_local_rank() {
    const int local_rank = require_env_int("OMPI_COMM_WORLD_LOCAL_RANK");
    int count = 0;
    const auto count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed for context parallel: ") +
                                 cudaGetErrorString(count_status));
    }
    if (local_rank < 0 || local_rank >= count)
        throw std::runtime_error("Context-parallel local rank has no matching visible CUDA device");
    auto status = cudaSetDevice(local_rank);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaSetDevice failed for context parallel rank: ") +
                                 cudaGetErrorString(status));
}

} // namespace

DistributedRuntimeGroup initialize_context_parallel_group(int cp_size) {
    DistributedRuntimeGroup group;
    group.cp_size = cp_size;
    if (cp_size == 1)
        return group;
    if (cp_size != 2)
        throw std::invalid_argument("Cosmos3 context parallel size must be 1 or 2");
    group.world_size = detect_world_size();
    group.rank = detect_rank();
    if (group.world_size != cp_size) {
        throw std::runtime_error("Context-parallel runtime requires mpirun world size to equal "
                                 "context_parallel_size for this initial implementation");
    }
    if (group.rank < 0 || group.rank >= cp_size)
        throw std::runtime_error("Context-parallel rank is outside [0, cp_size)");

    bind_cuda_device_for_local_rank();
    auto runtime = std::make_shared<NcclRuntime>();
    const auto path = rendezvous_path();
    NcclUniqueId id{};
    if (group.rank == 0) {
        id = runtime->unique_id();
        write_unique_id(path, id);
    } else {
        id = read_unique_id(path);
    }
    runtime->init(cp_size, group.rank, id);
    group.communicator = runtime->communicator();
    group.owner = std::move(runtime);
    return group;
}

} // namespace trtmc::cosmos3
