/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/internvl/runtime/distributed_runtime.h"

#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace trtmc::internvl {
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

int requireEnvInt(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
        throw std::runtime_error(std::string("InternVL tensor parallel requires ") + name);
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0')
        throw std::runtime_error(std::string("InternVL invalid integer in ") + name);
    return static_cast<int>(value);
}

std::filesystem::path rendezvousPath() {
    const char* raw = std::getenv("TRTMC_NCCL_RENDEZVOUS");
    if (raw == nullptr || *raw == '\0')
        throw std::runtime_error("InternVL tensor parallel requires TRTMC_NCCL_RENDEZVOUS");
    return raw;
}

class NcclRuntime {
  public:
    NcclRuntime() {
        handle_ = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr)
            throw std::runtime_error(std::string("Failed to load NCCL for InternVL: ") + dlerror());
        getUniqueId_ = load<NcclGetUniqueIdFn>("ncclGetUniqueId");
        initRank_ = load<NcclCommInitRankFn>("ncclCommInitRank");
        destroy_ = load<NcclCommDestroyFn>("ncclCommDestroy");
        errorString_ = load<NcclGetErrorStringFn>("ncclGetErrorString");
    }

    ~NcclRuntime() {
        if (communicator_ != nullptr)
            destroy_(communicator_);
        if (handle_ != nullptr)
            dlclose(handle_);
    }

    NcclUniqueId uniqueId() {
        NcclUniqueId id{};
        check(getUniqueId_(&id), "ncclGetUniqueId");
        return id;
    }

    void initialize(int size, int rank, const NcclUniqueId& id) {
        check(initRank_(&communicator_, size, id, rank), "ncclCommInitRank");
    }

    void* communicator() const { return communicator_; }

  private:
    template <typename Function>
    Function load(const char* name) {
        dlerror();
        auto* symbol = dlsym(handle_, name);
        const char* error = dlerror();
        if (error != nullptr || symbol == nullptr)
            throw std::runtime_error(std::string("Failed to resolve NCCL symbol ") + name);
        return reinterpret_cast<Function>(symbol);
    }

    void check(NcclResult result, const char* operation) const {
        if (result == 0)
            return;
        throw std::runtime_error(std::string(operation) + " failed: " + errorString_(result));
    }

    void* handle_{nullptr};
    NcclComm communicator_{nullptr};
    NcclGetUniqueIdFn getUniqueId_{nullptr};
    NcclCommInitRankFn initRank_{nullptr};
    NcclCommDestroyFn destroy_{nullptr};
    NcclGetErrorStringFn errorString_{nullptr};
};

void writeUniqueId(const std::filesystem::path& path, const NcclUniqueId& id) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("Failed to write InternVL NCCL rendezvous");
        output.write(id.internal, sizeof(id.internal));
    }
    std::filesystem::rename(temporary, path);
}

NcclUniqueId readUniqueId(const std::filesystem::path& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!std::filesystem::exists(path)) {
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("Timed out waiting for InternVL NCCL rendezvous");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    NcclUniqueId id{};
    std::ifstream input(path, std::ios::binary);
    input.read(id.internal, sizeof(id.internal));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(id.internal)))
        throw std::runtime_error("InternVL NCCL rendezvous is truncated");
    return id;
}

void bindCudaDevice() {
    const int localRank = requireEnvInt("OMPI_COMM_WORLD_LOCAL_RANK");
    int count = 0;
    const auto countStatus = cudaGetDeviceCount(&count);
    if (countStatus != cudaSuccess)
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") +
                                 cudaGetErrorString(countStatus));
    if (localRank < 0 || localRank >= count)
        throw std::runtime_error("InternVL local rank has no visible CUDA device");
    const auto status = cudaSetDevice(localRank);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                                 cudaGetErrorString(status));
}

} // namespace

DistributedRuntimeGroup initialize_tensor_parallel_group(int tpSize) {
    DistributedRuntimeGroup group;
    group.tp_size = tpSize;
    if (tpSize <= 1)
        return group;
    const int worldSize = requireEnvInt("OMPI_COMM_WORLD_SIZE");
    group.rank = requireEnvInt("OMPI_COMM_WORLD_RANK");
    if (worldSize != tpSize)
        throw std::runtime_error("InternVL mpirun world size must equal tensor_parallel_size");
    if (group.rank < 0 || group.rank >= tpSize)
        throw std::runtime_error("InternVL rank is outside tensor_parallel_size");

    bindCudaDevice();
    auto runtime = std::make_shared<NcclRuntime>();
    const auto path = rendezvousPath();
    NcclUniqueId id{};
    if (group.rank == 0) {
        id = runtime->uniqueId();
        writeUniqueId(path, id);
    } else {
        id = readUniqueId(path);
    }
    runtime->initialize(tpSize, group.rank, id);
    group.communicator = runtime->communicator();
    group.owner = std::move(runtime);
    return group;
}

} // namespace trtmc::internvl
