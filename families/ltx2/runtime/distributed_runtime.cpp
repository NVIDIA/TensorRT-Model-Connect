/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ltx2/runtime/distributed_runtime.h"

#include "trtmc/runtime/dynamic_library.h"

#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace trtmc::ltx2 {
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
using NcclGetVersionFn = NcclResult (*)(int*);

int require_env_int(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
        throw std::runtime_error(std::string("LTX-2.5 distributed runtime requires ") + name);
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0')
        throw std::runtime_error(std::string("LTX-2.5 distributed runtime has invalid ") + name);
    return static_cast<int>(value);
}

int detect_world_size() {
    return require_env_int("OMPI_COMM_WORLD_SIZE");
}

int detect_rank() {
    return require_env_int("OMPI_COMM_WORLD_RANK");
}

int detect_local_rank() {
    return require_env_int("OMPI_COMM_WORLD_LOCAL_RANK");
}

std::filesystem::path rendezvous_path() {
    const char* path = std::getenv("TRTMC_NCCL_RENDEZVOUS");
    if (path == nullptr || *path == '\0')
        throw std::runtime_error("LTX-2.5 distributed runtime requires TRTMC_NCCL_RENDEZVOUS");
    return path;
}

class NcclRuntime {
  public:
    NcclRuntime() {
        // NCCL is resolved at run time: TRTMC_NCCL_LIBRARY, else libnccl.so.2
        // (ELF) or nccl.dll (Windows) through the platform library search path.
        const std::string library = platform::nccl_library();
        try {
            library_ = std::make_unique<platform::DynamicLibrary>(library, "LTX-2.5 runtime: NCCL");
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string(error.what()) + ". Set " +
                                     platform::kNcclLibraryEnv +
                                     " to the NCCL shared library to use.");
        }
        get_unique_id_ = load<NcclGetUniqueIdFn>("ncclGetUniqueId");
        comm_init_rank_ = load<NcclCommInitRankFn>("ncclCommInitRank");
        comm_destroy_ = load<NcclCommDestroyFn>("ncclCommDestroy");
        get_error_string_ = load<NcclGetErrorStringFn>("ncclGetErrorString");
        int version = 0;
        const auto get_version =
            reinterpret_cast<NcclGetVersionFn>(library_->find_symbol("ncclGetVersion"));
        if (get_version != nullptr && get_version(&version) != 0)
            version = 0;
        std::cerr << "[ltx2] NCCL " << version << " loaded from " << library_->loaded_path()
                  << std::endl;
    }

    ~NcclRuntime() {
        if (comm_ != nullptr) {
            comm_destroy_(comm_);
            comm_ = nullptr;
        }
    }

    void init(int size, int rank, const NcclUniqueId& id) {
        check(comm_init_rank_(&comm_, size, id, rank), "ncclCommInitRank");
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
        return library_->require<T>(symbol);
    }

    void check(NcclResult result, const char* operation) const {
        if (result == 0)
            return;
        const char* message = get_error_string_(result);
        throw std::runtime_error(std::string(operation) + " failed: " + message);
    }

    std::unique_ptr<platform::DynamicLibrary> library_;
    NcclComm comm_{nullptr};
    NcclGetUniqueIdFn get_unique_id_{nullptr};
    NcclCommInitRankFn comm_init_rank_{nullptr};
    NcclCommDestroyFn comm_destroy_{nullptr};
    NcclGetErrorStringFn get_error_string_{nullptr};
};

void write_unique_id(const std::filesystem::path& path, const NcclUniqueId& id) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("Failed to write NCCL rendezvous file: " + temporary);
        output.write(id.internal, sizeof(id.internal));
    }
    std::filesystem::rename(temporary, path);
}

NcclUniqueId read_unique_id(const std::filesystem::path& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!std::filesystem::exists(path)) {
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("Timed out waiting for NCCL rendezvous file: " +
                                     path.string());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    NcclUniqueId id{};
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Failed to read NCCL rendezvous file: " + path.string());
    input.read(id.internal, sizeof(id.internal));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(id.internal)))
        throw std::runtime_error("Short NCCL rendezvous file: " + path.string());
    return id;
}

void bind_cuda_device_for_local_rank(int local_rank) {
    int count = 0;
    const auto count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed for LTX-2.5 runtime: ") +
                                 cudaGetErrorString(count_status));
    }
    if (local_rank < 0 || local_rank >= count) {
        throw std::runtime_error(
            "LTX-2.5 distributed local rank is outside the visible CUDA device range");
    }
    const auto status = cudaSetDevice(local_rank);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaSetDevice failed for LTX-2.5 distributed rank: ") +
            cudaGetErrorString(status));
    }
}

} // namespace

DistributedRuntimeGroup initialize_parallel_group(int parallel_size) {
    DistributedRuntimeGroup group;
    group.parallel_size = parallel_size;
    if (parallel_size <= 1)
        return group;

    group.world_size = detect_world_size();
    group.rank = detect_rank();
    if (group.world_size != parallel_size) {
        throw std::runtime_error(
            "LTX-2.5 distributed runtime requires launcher world size to equal parallel_size");
    }
    if (group.rank < 0 || group.rank >= parallel_size)
        throw std::runtime_error("LTX-2.5 distributed rank is outside parallel_size");

    bind_cuda_device_for_local_rank(detect_local_rank());
    auto runtime = std::make_shared<NcclRuntime>();
    const auto path = rendezvous_path();
    NcclUniqueId id{};
    if (group.rank == 0) {
        id = runtime->unique_id();
        write_unique_id(path, id);
    } else {
        id = read_unique_id(path);
    }
    runtime->init(parallel_size, group.rank, id);
    group.communicator = runtime->communicator();
    group.owner = std::move(runtime);
    return group;
}

} // namespace trtmc::ltx2
