/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// RtxBackend: IBackend implementation for TensorRT-RTX.
// Compiled into libtrtmc_backend_rtx.so. Links libtensorrt_rtx.so.
//
// Uses the RTX-specific NvInfer.h headers which declare IRuntimeCache,
// CudaGraphStrategy, and DynamicShapesKernelSpecializationStrategy.

#include "runtime/backend/prebound_backend.h"
#include "runtime/backend/trt_logger.h"
#include "runtime/core/cuda_common.h"
#include "trt_module_impl.h"
#include "trtmc/runtime/trt_backend.h"
#include "utils/sha256.h"

#include <NvInfer.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace trtmc {

namespace {

namespace fs = std::filesystem;

struct StreamSetup {
    cudaStream_t stream{nullptr};
    std::shared_ptr<void> owner;
};

#if defined(_WIN32)
// Some Windows CUDA configurations report memory-pool support while rejecting
// the asynchronous allocation path used by TensorRT-RTX. IGpuAsyncAllocator
// permits a synchronizing implementation, so retain the callback surface while
// backing it with cudaMalloc/cudaFree on Windows only.
class SynchronousGpuAllocator final : public nvinfer1::IGpuAsyncAllocator {
  public:
    void* allocateAsync(std::uint64_t size, std::uint64_t /*alignment*/,
                        nvinfer1::AllocatorFlags /*flags*/,
                        cudaStream_t /*stream*/) noexcept override {
        if (size == 0 || size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            return nullptr;
        void* memory = nullptr;
        return cudaMalloc(&memory, static_cast<std::size_t>(size)) == cudaSuccess ? memory
                                                                                  : nullptr;
    }

    bool deallocateAsync(void* memory, cudaStream_t /*stream*/) noexcept override {
        return memory == nullptr || cudaFree(memory) == cudaSuccess;
    }
};
#endif

StreamSetup resolve_stream(cudaStream_t requested_stream) {
    if (requested_stream) {
        return StreamSetup{requested_stream, {}};
    }

    auto owned = std::make_shared<CudaStream>();
    if (!owned->ok()) {
        throw std::runtime_error("[trtmc] Failed to create CUDA stream");
    }

    return StreamSetup{owned->get(), owned};
}

class PlanFileMutationGuard final {
  public:
    explicit PlanFileMutationGuard(const char* path) {
        if (path == nullptr || path[0] == '\0')
            throw std::invalid_argument("[trtmc] RTX plan file path must not be empty");
        try {
            path_ = fs::canonical(fs::path(path));
        } catch (const fs::filesystem_error& error) {
            throw std::runtime_error(std::string("[trtmc] Failed to resolve RTX plan file: ") +
                                     error.what());
        }
#if defined(_WIN32)
        // Deny write/delete sharing while the exact file is hashed and parsed.
        handle_ = CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            const std::error_code error(static_cast<int>(GetLastError()), std::system_category());
            throw std::runtime_error("[trtmc] Failed to lock RTX plan file: " + error.message());
        }
#else
        initial_size_ = fs::file_size(path_);
        initial_write_time_ = fs::last_write_time(path_);
#endif
    }

    ~PlanFileMutationGuard() {
#if defined(_WIN32)
        if (handle_ != INVALID_HANDLE_VALUE)
            (void)CloseHandle(handle_);
#endif
    }

    PlanFileMutationGuard(const PlanFileMutationGuard&) = delete;
    PlanFileMutationGuard& operator=(const PlanFileMutationGuard&) = delete;

    const fs::path& path() const noexcept { return path_; }

    void verify_unchanged() const {
#if !defined(_WIN32)
        if (fs::file_size(path_) != initial_size_ ||
            fs::last_write_time(path_) != initial_write_time_) {
            throw std::runtime_error("[trtmc] RTX plan file changed while it was parsed");
        }
#endif
    }

  private:
    fs::path path_;
#if defined(_WIN32)
    HANDLE handle_{INVALID_HANDLE_VALUE};
#else
    std::uintmax_t initial_size_{0};
    fs::file_time_type initial_write_time_{};
#endif
};

bool is_lower_hexadecimal(char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
}

void validate_plan_description(std::uint64_t size, const std::string& expected_sha256) {
    if (size == 0 || size > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument("[trtmc] RTX plan file range is invalid");
    }
    if (expected_sha256.size() != 64 ||
        !std::all_of(expected_sha256.begin(), expected_sha256.end(), is_lower_hexadecimal)) {
        throw std::invalid_argument(
            "[trtmc] Expected RTX plan SHA-256 must be lowercase hexadecimal");
    }
}

void validate_plan_file_range(std::uint64_t offset, std::uint64_t size, std::uint64_t file_size) {
    const auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max());
    if (offset > file_size || size > file_size - offset || offset > max_offset ||
        size > max_offset - offset) {
        throw std::invalid_argument("[trtmc] RTX plan section is outside its file");
    }
}

enum class ReaderDestination {
    host,
    device,
    invalid,
};

ReaderDestination classify_reader_destination(void* destination) noexcept {
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, destination);
    if (status == cudaErrorInvalidValue) {
        (void)cudaGetLastError();
        return ReaderDestination::host;
    }
    if (status != cudaSuccess)
        return ReaderDestination::invalid;
    if (attributes.type == cudaMemoryTypeUnregistered || attributes.type == cudaMemoryTypeHost)
        return ReaderDestination::host;
    if (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged)
        return ReaderDestination::device;
    return ReaderDestination::invalid;
}

void read_host_bytes(std::ifstream& file, void* destination, std::uint64_t requested) {
    file.read(static_cast<char*>(destination), static_cast<std::streamsize>(requested));
    if (file.gcount() != static_cast<std::streamsize>(requested))
        throw std::runtime_error("[trtmc] Failed to read RTX plan bytes into host memory");
}

void read_device_bytes(std::ifstream& file, std::vector<char>& staging, void* destination,
                       std::uint64_t requested) {
    std::uint64_t copied = 0;
    while (copied < requested) {
        const auto chunk = std::min<std::uint64_t>(staging.size(), requested - copied);
        file.read(staging.data(), static_cast<std::streamsize>(chunk));
        if (file.gcount() != static_cast<std::streamsize>(chunk))
            throw std::runtime_error("[trtmc] Failed to stage RTX plan bytes");
        if (cudaMemcpy(static_cast<unsigned char*>(destination) + copied, staging.data(),
                       static_cast<std::size_t>(chunk), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("[trtmc] Failed to copy RTX plan bytes to the device");
        }
        copied += chunk;
    }
}

void read_destination_bytes(std::ifstream& file, std::vector<char>& staging, void* destination,
                            std::uint64_t requested) {
    switch (classify_reader_destination(destination)) {
    case ReaderDestination::host:
        read_host_bytes(file, destination, requested);
        return;
    case ReaderDestination::device:
        read_device_bytes(file, staging, destination, requested);
        return;
    case ReaderDestination::invalid:
        throw std::runtime_error("[trtmc] Unsupported RTX plan reader destination");
    }
}

// TensorRT-RTX may request plan bytes into host or device memory and may seek
// within the stream. Restrict every operation to the validated bundle section.
class BoundedPlanStreamReader final : public nvinfer1::IStreamReaderV2 {
  public:
    BoundedPlanStreamReader(const char* path, std::uint64_t offset, std::uint64_t size,
                            const char* expected_sha256)
        : mutation_guard_(path), offset_(offset), size_(size), staging_(4U << 20),
          expected_sha256_(expected_sha256 == nullptr ? "" : expected_sha256) {
        validate_plan_description(size_, expected_sha256_);
        file_.open(mutation_guard_.path(), std::ios::binary | std::ios::ate);
        if (!file_)
            throw std::runtime_error("[trtmc] Failed to open RTX plan file");
        const auto end = file_.tellg();
        if (end < 0)
            throw std::runtime_error("[trtmc] Failed to determine RTX plan file size");
        const auto file_size = static_cast<std::uint64_t>(static_cast<std::streamoff>(end));
        validate_plan_file_range(offset_, size_, file_size);
    }

    void verify_sha256() {
        internal::Sha256 digest;
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(offset_), std::ios::beg);
        if (!file_)
            throw std::runtime_error("[trtmc] Failed to seek to RTX plan section");
        std::uint64_t remaining = size_;
        while (remaining != 0) {
            const auto chunk = std::min<std::uint64_t>(staging_.size(), remaining);
            file_.read(staging_.data(), static_cast<std::streamsize>(chunk));
            if (file_.gcount() != static_cast<std::streamsize>(chunk))
                throw std::runtime_error("[trtmc] Failed to hash RTX plan section");
            digest.update(staging_.data(), static_cast<std::size_t>(chunk));
            remaining -= chunk;
        }
        if (digest.hex_digest() != expected_sha256_)
            throw std::runtime_error("[trtmc] RTX plan SHA-256 mismatch");
        cursor_ = 0;
    }

    void verify_unchanged() const { mutation_guard_.verify_unchanged(); }

    int64_t read(void* destination, int64_t nb_bytes, cudaStream_t stream) noexcept override {
        (void)stream;
        if (destination == nullptr || nb_bytes < 0)
            return -1;
        if (nb_bytes == 0)
            return 0;
        try {
            return read_from_current_position(destination, nb_bytes);
        } catch (...) {
            return -1;
        }
    }

    bool seek(int64_t offset, nvinfer1::SeekPosition where) noexcept override {
        std::uint64_t base = 0;
        switch (where) {
        case nvinfer1::SeekPosition::kSET:
            break;
        case nvinfer1::SeekPosition::kCUR:
            base = cursor_;
            break;
        case nvinfer1::SeekPosition::kEND:
            base = size_;
            break;
        default:
            return false;
        }
        if (offset >= 0) {
            const auto delta = static_cast<std::uint64_t>(offset);
            if (base > size_ || delta > size_ - base)
                return false;
            cursor_ = base + delta;
            return true;
        }
        const auto magnitude = static_cast<std::uint64_t>(-(offset + 1)) + 1;
        if (magnitude > base)
            return false;
        cursor_ = base - magnitude;
        return true;
    }

  private:
    int64_t read_from_current_position(void* destination, int64_t nb_bytes) {
        const auto requested =
            std::min<std::uint64_t>(static_cast<std::uint64_t>(nb_bytes), size_ - cursor_);
        if (requested == 0)
            return 0;
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(offset_ + cursor_), std::ios::beg);
        if (!file_)
            throw std::runtime_error("[trtmc] Failed to seek in the RTX plan section");
        read_destination_bytes(file_, staging_, destination, requested);
        cursor_ += requested;
        return static_cast<int64_t>(requested);
    }

    PlanFileMutationGuard mutation_guard_;
    std::ifstream file_;
    std::uint64_t offset_{0};
    std::uint64_t size_{0};
    std::uint64_t cursor_{0};
    std::vector<char> staging_;
    std::string expected_sha256_;
};

void apply_weight_streaming_budget(nvinfer1::ICudaEngine& engine, std::int64_t budget,
                                   bool cuda_graphs) {
    if (budget < 0)
        return;
    if (cuda_graphs)
        throw std::invalid_argument(
            "[trtmc] RTX weight streaming is incompatible with CUDA graph capture");
    const std::int64_t streamable = engine.getStreamableWeightsSize();
    if (streamable <= 0)
        throw std::runtime_error("[trtmc] Engine was not built with TensorRT-RTX weight streaming");
    const std::int64_t applied = std::min(budget, streamable);
    if (!engine.setWeightStreamingBudgetV2(applied) ||
        engine.getWeightStreamingBudgetV2() != applied) {
        throw std::runtime_error("[trtmc] TensorRT-RTX rejected the weight streaming budget");
    }
    std::cerr << "[trtmc.rtx_weight_budget] requested_bytes=" << budget
              << " streamable_bytes=" << streamable << " applied_bytes=" << applied
              << " streaming_scratch_bytes=" << engine.getWeightStreamingScratchMemorySize()
              << '\n';
}

void validate_optimization_profile(const nvinfer1::ICudaEngine& engine, int32_t profile) {
    if (profile < 0 || profile >= engine.getNbOptimizationProfiles())
        throw std::invalid_argument("[trtmc] Invalid optimization profile index");
}

void configure_cuda_graphs(nvinfer1::IRuntimeConfig& config, bool enabled) {
    if (enabled &&
        !config.setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE)) {
        throw std::runtime_error("[trtmc] Failed to enable RTX CUDA graph capture");
    }
}

} // namespace

class RtxBackend final : public IBackend, public IPreboundBackend, public IFileBackedBackend {
  public:
    RtxBackend()
        : runtime_(create_trt_runtime())
#if defined(_WIN32)
          , staged_runtime_(create_trt_runtime())
#endif
    {
        if (!runtime_)
            throw std::runtime_error("[trtmc] Failed to create TRT-RTX runtime");
#if defined(_WIN32)
        if (!staged_runtime_)
            throw std::runtime_error("[trtmc] Failed to create staged TRT-RTX runtime");
        // This dedicated runtime serves only file-backed staged plans, so the
        // synchronizing allocator cannot change ordinary in-memory RTX paths.
        staged_runtime_->setGpuAllocator(&gpu_allocator_);
#endif
    }

    ~RtxBackend() override {
        flush_runtime_cache();
        delete runtime_cache_;
    }

    std::unique_ptr<ITrtModule> create_module(const void* plan_data, size_t plan_size,
                                              const ModuleCreateOptions& options) override {
        auto* engine = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (RTX)");
        return create_single_module(engine, options, {}, -1);
    }

    std::unique_ptr<ITrtModule>
    create_module_prebound(const void* plan_data, size_t plan_size,
                           const ModuleCreateOptions& options,
                           const std::vector<ModuleExternalBinding>& external_bindings) override {
        if (external_bindings.empty())
            throw std::invalid_argument("[trtmc] External I/O prebindings must not be empty");
        auto* engine = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (RTX)");
        return create_single_module(engine, options, external_bindings, -1);
    }

    std::unique_ptr<ITrtModule>
    create_module_from_file(const char* plan_path, std::uint64_t plan_offset,
                            std::uint64_t plan_size, const char* expected_sha256,
                            const ModuleCreateOptions& options,
                            const std::vector<ModuleExternalBinding>& external_bindings,
                            std::int64_t weight_streaming_budget_bytes,
                            bool retain_engine) override {
        if (weight_streaming_budget_bytes >= 0 && options.cuda_graphs) {
            throw std::invalid_argument(
                "[trtmc] RTX weight streaming is incompatible with CUDA graph capture");
        }
        const std::string cache_key = retain_engine
                                          ? retained_engine_key(expected_sha256,
                                                                weight_streaming_budget_bytes)
                                          : std::string{};
        std::unique_lock<std::mutex> retained_lock;
        if (retain_engine) {
            // Serialize retained-engine creation. Two concurrent deserializations
            // of the same multi-GiB plan can exceed the device-memory envelope
            // before the losing insertion is released.
            retained_lock = std::unique_lock<std::mutex>(retained_engines_mutex_);
            const auto hit = retained_engines_.find(cache_key);
            if (hit != retained_engines_.end()) {
                std::cerr << "[trtmc.rtx_engine_cache] hit=1\n";
                return create_single_module_from_engine(hit->second, options, external_bindings,
                                                        true);
            }
        }
        BoundedPlanStreamReader reader(plan_path, plan_offset, plan_size, expected_sha256);
        reader.verify_sha256();
        TrtUniquePtr<nvinfer1::ICudaEngine> engine(
            file_backed_runtime().deserializeCudaEngine(reader));
        reader.verify_unchanged();
        if (!engine)
            throw std::runtime_error("[trtmc] Failed to stream-deserialize engine (RTX)");
        if (retain_engine) {
            std::shared_ptr<nvinfer1::ICudaEngine> retained(
                engine.release(), [](nvinfer1::ICudaEngine* value) { delete value; });
            apply_weight_streaming_budget(*retained, weight_streaming_budget_bytes,
                                          options.cuda_graphs);
            auto module = create_single_module_from_engine(retained, options, external_bindings,
                                                           true);
            retained_engines_.emplace(cache_key, retained);
            std::cerr << "[trtmc.rtx_engine_cache] hit=0 retained=1\n";
            return module;
        }
        return create_single_module(engine.release(), options, external_bindings,
                                    weight_streaming_budget_bytes, true);
    }

    BackendDualProfileModules
    create_dual_profile_modules(const void* plan_data, size_t plan_size,
                                const ModuleCreateOptions& options) override {
        auto* engine_raw = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine_raw)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (RTX)");
        std::shared_ptr<nvinfer1::ICudaEngine> engine(engine_raw,
                                                      [](nvinfer1::ICudaEngine* p) { delete p; });

        StreamSetup stream_setup = resolve_stream(options.stream);

        const int32_t nprofiles = engine->getNbOptimizationProfiles();
        auto make_ctx_module = [&](int32_t profile_idx) -> std::unique_ptr<ITrtModule> {
            return create_profile_module(engine, stream_setup, options, profile_idx);
        };

        BackendDualProfileModules out;
        if (nprofiles < 2) {
            out.decode = make_ctx_module(0);
            return out;
        }
        out.prefill = make_ctx_module(0);
        out.decode = make_ctx_module(1);
        return out;
    }

    BackendProfileModules
    create_profile_modules(const void* plan_data, size_t plan_size,
                           const ModuleCreateOptions& options,
                           const std::vector<int32_t>& profile_indices) override {
        auto* engine_raw = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine_raw)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (RTX)");
        std::shared_ptr<nvinfer1::ICudaEngine> engine(engine_raw,
                                                      [](nvinfer1::ICudaEngine* p) { delete p; });

        StreamSetup stream_setup = resolve_stream(options.stream);
        const int32_t nprofiles = engine->getNbOptimizationProfiles();
        BackendProfileModules out;
        out.modules.reserve(profile_indices.size());
        for (int32_t profile_idx : profile_indices) {
            if (profile_idx < 0 || profile_idx >= nprofiles)
                continue;
            out.modules.push_back(BackendProfileModule{
                profile_idx, create_profile_module(engine, stream_setup, options, profile_idx)});
        }
        return out;
    }

    BackendContextModules
    create_context_modules(const void* plan_data, size_t plan_size,
                           const std::vector<ModuleCreateOptions>& options) override {
        if (options.empty())
            throw std::invalid_argument("[trtmc] Context module options must not be empty");
        auto* engine_raw = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine_raw)
            throw std::runtime_error("[trtmc] Failed to deserialize engine (RTX)");
        std::shared_ptr<nvinfer1::ICudaEngine> engine(engine_raw,
                                                      [](nvinfer1::ICudaEngine* p) { delete p; });

        BackendContextModules out;
        out.modules.reserve(options.size());
        for (const auto& lane_options : options) {
            const int32_t profile_idx = lane_options.optimization_profile;
            if (profile_idx < 0 || profile_idx >= engine->getNbOptimizationProfiles())
                throw std::invalid_argument("[trtmc] Invalid optimization profile index");
            StreamSetup stream_setup = resolve_stream(lane_options.stream);
            out.modules.push_back(
                create_profile_module(engine, stream_setup, lane_options, profile_idx));
        }
        return out;
    }

    const char* name() const override { return "trt_rtx"; }

  private:
#if defined(_WIN32)
    // Declared before runtime_ so it outlives the runtime and every engine
    // deserialized through it (members are destroyed in reverse order).
    SynchronousGpuAllocator gpu_allocator_;
#endif
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
#if defined(_WIN32)
    TrtUniquePtr<nvinfer1::IRuntime> staged_runtime_;
#endif
    nvinfer1::IRuntimeCache* runtime_cache_{nullptr};
    std::string cache_path_;
    std::mutex retained_engines_mutex_;
    std::unordered_map<std::string, std::shared_ptr<nvinfer1::ICudaEngine>> retained_engines_;

    nvinfer1::IRuntime& file_backed_runtime() {
#if defined(_WIN32)
        return *staged_runtime_;
#else
        return *runtime_;
#endif
    }

    static std::string retained_engine_key(const char* expected_sha256,
                                           std::int64_t weight_streaming_budget_bytes) {
        if (expected_sha256 == nullptr || expected_sha256[0] == '\0') {
            throw std::invalid_argument(
                "[trtmc] Retained RTX engines require a verified plan SHA-256");
        }
        int device = -1;
        const cudaError_t status = cudaGetDevice(&device);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("[trtmc] Failed to identify the CUDA device: ") +
                                     cudaGetErrorString(status));
        }
        return std::to_string(device) + ":" + expected_sha256 + ":" +
               std::to_string(weight_streaming_budget_bytes);
    }

    std::unique_ptr<ITrtModule> create_single_module_from_engine(
        const std::shared_ptr<nvinfer1::ICudaEngine>& engine,
        const ModuleCreateOptions& options,
        const std::vector<ModuleExternalBinding>& external_bindings,
        bool use_synchronous_allocator = false) {
        validate_optimization_profile(*engine, options.optimization_profile);
        const StreamSetup stream_setup = resolve_stream(options.stream);
        auto config = create_runtime_config(*engine, options);
        return create_execution_module(engine, config, stream_setup, options, external_bindings,
                                       use_synchronous_allocator);
    }

    std::unique_ptr<ITrtModule>
    create_single_module(nvinfer1::ICudaEngine* engine_raw, const ModuleCreateOptions& options,
                         const std::vector<ModuleExternalBinding>& external_bindings,
                         std::int64_t weight_streaming_budget_bytes,
                         bool use_synchronous_allocator = false) {
        std::shared_ptr<nvinfer1::ICudaEngine> engine(
            engine_raw, [](nvinfer1::ICudaEngine* value) { delete value; });
        apply_weight_streaming_budget(*engine, weight_streaming_budget_bytes, options.cuda_graphs);
        return create_single_module_from_engine(engine, options, external_bindings,
                                                use_synchronous_allocator);
    }

    std::shared_ptr<nvinfer1::IRuntimeConfig>
    create_runtime_config(nvinfer1::ICudaEngine& engine, const ModuleCreateOptions& options) {
        std::shared_ptr<nvinfer1::IRuntimeConfig> config(
            engine.createRuntimeConfig(), [](nvinfer1::IRuntimeConfig* value) { delete value; });
        if (!config)
            throw std::runtime_error("[trtmc] Failed to create RTX runtime config");
        if (options.runtime_cache_path && options.runtime_cache_path[0] != '\0')
            ensure_runtime_cache(config.get(), options.runtime_cache_path);
        configure_cuda_graphs(*config, options.cuda_graphs);
        return config;
    }

    std::unique_ptr<ITrtModule>
    create_execution_module(const std::shared_ptr<nvinfer1::ICudaEngine>& engine,
                            const std::shared_ptr<nvinfer1::IRuntimeConfig>& config,
                            const StreamSetup& stream_setup, const ModuleCreateOptions& options,
                            const std::vector<ModuleExternalBinding>& external_bindings,
                            bool use_synchronous_allocator) {
        std::unique_ptr<nvinfer1::IExecutionContext> context(
            engine->createExecutionContext(config.get()));
        if (!context)
            throw std::runtime_error("[trtmc] Failed to create RTX execution context");
#if defined(_WIN32)
        if (use_synchronous_allocator &&
            !context->setTemporaryStorageAllocator(&gpu_allocator_))
            throw std::runtime_error(
                "[trtmc] Failed to set synchronous RTX temporary-storage allocator");
#else
        (void)use_synchronous_allocator;
#endif
        auto module = std::make_unique<TrtModuleImpl>(
            engine.get(), context.get(), stream_setup.stream, options.optimization_profile, nullptr,
            external_bindings);
        // A completed TrtModuleImpl owns (or has already rejected and deleted)
        // the context. Before that point the local owner handles exceptions.
        (void)context.release();
        if (!module->ok())
            throw std::runtime_error("[trtmc] TrtModuleImpl creation failed (RTX)");
        module->keep_alive(engine);
        module->keep_alive(config);
        if (stream_setup.owner)
            module->keep_alive(stream_setup.owner);
        return module;
    }

    std::unique_ptr<ITrtModule>
    create_profile_module(const std::shared_ptr<nvinfer1::ICudaEngine>& engine,
                          const StreamSetup& stream_setup, const ModuleCreateOptions& options,
                          int32_t profile_idx) {
        auto* rt_config = engine->createRuntimeConfig();
        if (!rt_config)
            throw std::runtime_error("[trtmc] Failed to create RTX runtime config");
        if (options.runtime_cache_path && options.runtime_cache_path[0] != '\0')
            ensure_runtime_cache(rt_config, options.runtime_cache_path);
        if (options.cuda_graphs)
            rt_config->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);

        auto* ctx = engine->createExecutionContext(rt_config);
        delete rt_config;
        if (!ctx)
            throw std::runtime_error("[trtmc] Failed to create RTX execution context");
        auto mod =
            std::make_unique<TrtModuleImpl>(engine.get(), ctx, stream_setup.stream, profile_idx);
        if (!mod->ok())
            throw std::runtime_error("[trtmc] TrtModuleImpl creation failed (RTX)");
        mod->keep_alive(engine);
        if (stream_setup.owner)
            mod->keep_alive(stream_setup.owner);
        return mod;
    }

    void ensure_runtime_cache(nvinfer1::IRuntimeConfig* cfg, const char* path) {
        if (!runtime_cache_) {
            runtime_cache_ = cfg->createRuntimeCache();
            cache_path_ = path;
            std::ifstream ifs(path, std::ios::binary | std::ios::ate);
            if (ifs) {
                auto sz = ifs.tellg();
                if (sz > 0) {
                    std::vector<char> buf(static_cast<size_t>(sz));
                    ifs.seekg(0);
                    ifs.read(buf.data(), sz);
                    runtime_cache_->deserialize(buf.data(), buf.size());
                    std::cerr << "[trtmc] RTX runtime cache loaded: " << path << " (" << sz
                              << " bytes)\n";
                }
            }
        }
        cfg->setRuntimeCache(*runtime_cache_);
    }

    void flush_runtime_cache() {
        if (!runtime_cache_ || cache_path_.empty())
            return;
        auto* mem = runtime_cache_->serialize();
        if (mem && mem->size() > 0) {
            std::ofstream ofs(cache_path_, std::ios::binary | std::ios::trunc);
            if (ofs) {
                ofs.write(static_cast<const char*>(mem->data()),
                          static_cast<std::streamsize>(mem->size()));
                std::cerr << "[trtmc] RTX runtime cache saved: " << cache_path_ << " ("
                          << mem->size() << " bytes)\n";
            }
            delete mem;
        }
    }
};

} // namespace trtmc

extern "C" trtmc::IBackend* trtmc_create_backend() {
    try {
        return new trtmc::RtxBackend();
    } catch (const std::exception& e) {
        std::cerr << "[trtmc] RTX backend init failed: " << e.what() << std::endl;
        return nullptr;
    }
}

extern "C" void trtmc_destroy_backend(trtmc::IBackend* b) {
    delete b;
}

extern "C" const char* trtmc_backend_abi() {
    static const std::string abi =
        std::to_string(getInferLibMajorVersion()) + "." + std::to_string(getInferLibMinorVersion());
    return abi.c_str();
}

extern "C" const char* trtmc_backend_runtime_version() {
    static const std::string version = std::to_string(getInferLibMajorVersion()) + "." +
                                       std::to_string(getInferLibMinorVersion()) + "." +
                                       std::to_string(getInferLibPatchVersion()) + "." +
                                       std::to_string(getInferLibBuildVersion());
    return version.c_str();
}
