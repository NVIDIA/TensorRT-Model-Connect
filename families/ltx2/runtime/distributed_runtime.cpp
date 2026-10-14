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
#include <system_error>
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
using NcclSendFn = NcclResult (*)(const void*, std::size_t, int, int, NcclComm, cudaStream_t);
using NcclRecvFn = NcclResult (*)(void*, std::size_t, int, int, NcclComm, cudaStream_t);
using NcclGroupFn = NcclResult (*)();
using NcclCommAbortFn = NcclResult (*)(NcclComm);
constexpr int kNcclUint8 = 1; // ncclUint8: transfers are byte copies

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

class NcclRuntime final : public PeerChannel {
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
        send_ = load<NcclSendFn>("ncclSend");
        recv_ = load<NcclRecvFn>("ncclRecv");
        group_start_ = load<NcclGroupFn>("ncclGroupStart");
        group_end_ = load<NcclGroupFn>("ncclGroupEnd");
        comm_abort_ = load<NcclCommAbortFn>("ncclCommAbort");
        int version = 0;
        const auto get_version =
            reinterpret_cast<NcclGetVersionFn>(library_->find_symbol("ncclGetVersion"));
        if (get_version != nullptr && get_version(&version) != 0)
            version = 0;
        std::cerr << "[ltx2] NCCL " << version << " loaded from " << library_->loaded_path()
                  << std::endl;
    }

    ~NcclRuntime() override {
        if (token_ != nullptr)
            cudaFree(token_);
        if (stream_ != nullptr)
            cudaStreamDestroy(stream_);
        if (comm_ != nullptr) {
            comm_destroy_(comm_);
            comm_ = nullptr;
        }
    }

    void run(const std::vector<PeerTransfer>& transfers,
             std::chrono::milliseconds timeout) override {
        if (comm_ == nullptr)
            throw std::runtime_error("LTX-2.5 peer transfer: the NCCL communicator was aborted");
        if (stream_ == nullptr &&
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("LTX-2.5 peer transfer: cudaStreamCreate failed");
        enqueue(transfers);
        // Poll instead of blocking, so a missing peer aborts the communicator rather than
        // leaving NCCL kernels spinning on the GPU.
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for (;;) {
            const auto status = cudaStreamQuery(stream_);
            if (status == cudaSuccess)
                return;
            if (status != cudaErrorNotReady)
                throw std::runtime_error(std::string("LTX-2.5 peer transfer failed: ") +
                                         cudaGetErrorString(status));
            if (std::chrono::steady_clock::now() > deadline) {
                comm_abort_(comm_);
                comm_ = nullptr;
                throw std::runtime_error(
                    "LTX-2.5 peer transfer timed out; the NCCL communicator was aborted");
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    // Rank 0 hears from every rank, then answers each: nobody leaves before all arrived. The
    // token is 64 KiB: with the Windows NCCL build used for the RTX PRO 6000 host, point-to-point
    // transfers below 32 KiB never complete (a 1-byte barrier hangs), larger ones do.
    void barrier(std::chrono::milliseconds timeout) override {
        constexpr std::size_t kToken = 64 * 1024;
        if (token_ == nullptr &&
            cudaMalloc(&token_, kToken * static_cast<std::size_t>(size_)) != cudaSuccess)
            throw std::runtime_error("LTX-2.5 rank barrier: cudaMalloc failed");
        auto* token = static_cast<char*>(token_);
        std::vector<PeerTransfer> gather;
        std::vector<PeerTransfer> release;
        for (int peer = 1; rank_ == 0 && peer < size_; ++peer) {
            gather.push_back({peer, token + kToken * peer, kToken, false});
            release.push_back({peer, token + kToken * peer, kToken, true});
        }
        if (rank_ != 0) {
            gather.push_back({0, token, kToken, true});
            release.push_back({0, token, kToken, false});
        }
        run(gather, timeout);
        run(release, timeout);
    }

    void init(int size, int rank, const NcclUniqueId& id) {
        check(comm_init_rank_(&comm_, size, id, rank), "ncclCommInitRank");
        size_ = size;
        rank_ = rank;
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

    void enqueue(const std::vector<PeerTransfer>& transfers) {
        check(group_start_(), "ncclGroupStart");
        for (const auto& t : transfers) {
            const auto status = t.send
                                    ? send_(t.device, t.bytes, kNcclUint8, t.peer, comm_, stream_)
                                    : recv_(t.device, t.bytes, kNcclUint8, t.peer, comm_, stream_);
            if (status != 0) {
                (void)group_end_();
                check(status, t.send ? "ncclSend" : "ncclRecv");
            }
        }
        check(group_end_(), "ncclGroupEnd");
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
    NcclSendFn send_{nullptr};
    NcclRecvFn recv_{nullptr};
    NcclGroupFn group_start_{nullptr};
    NcclGroupFn group_end_{nullptr};
    NcclCommAbortFn comm_abort_{nullptr};
    cudaStream_t stream_{nullptr};
    void* token_{nullptr};
    int size_{1};
    int rank_{0};
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
    if (group.rank == 0) {
        // ncclCommInitRank returns only after every rank joined, so every rank
        // has read the ID. Remove it so a reused path cannot hand a stale ID to
        // the next launch.
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
    group.communicator = runtime->communicator();
    group.channel = runtime;
    group.owner = std::move(runtime);
    return group;
}

} // namespace trtmc::ltx2
