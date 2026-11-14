/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8/runtime/tensor_parallel.h"

#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"

#include <array>
#include <atomic>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace trtmc::qwen3_8 {
namespace {
namespace api = trtmc::internal;

void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Qwen native TP CUDA: ") + cudaGetErrorString(result));
}

class DeviceGuard {
  public:
    explicit DeviceGuard(int device) {
        check_cuda(cudaGetDevice(&previous_));
        check_cuda(cudaSetDevice(device));
    }
    ~DeviceGuard() { (void)cudaSetDevice(previous_); }

  private:
    int previous_{};
};

// NCCL performs only the collectives requested by native TensorRT graph layers.
// No family-owned CUDA kernels, MPI launcher or filesystem rendezvous is needed.
class Communicators {
  public:
    Communicators() {
        int count = 0;
        check_cuda(cudaGetDeviceCount(&count));
        if (count < 2)
            throw std::runtime_error("Qwen native TP2 requires two visible CUDA devices");
#if defined(_WIN32)
        library_ = LoadLibraryA("nccl.dll");
#else
        library_ = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
#endif
        if (!library_)
            throw std::runtime_error("Qwen native TP2 could not load NCCL");
        try {
            const auto init = symbol<int (*)(void**, int, const int*)>("ncclCommInitAll");
            destroy_ = symbol<int (*)(void*)>("ncclCommDestroy");
            abort_ = symbol<int (*)(void*)>("ncclCommAbort");
            const auto message = symbol<const char* (*)(int)>("ncclGetErrorString");
            const int devices[] = {0, 1};
            DeviceGuard guard(0);
            const int result = init(handles_.data(), 2, devices);
            if (result != 0)
                throw std::runtime_error(std::string("Qwen ncclCommInitAll: ") + message(result));
        } catch (...) {
            close();
            throw;
        }
    }
    ~Communicators() { close(); }
    void* get(int rank) const { return handles_.at(rank); }
    void abort() noexcept {
        if (aborted_.exchange(true))
            return;
        for (int rank = 0; rank < 2; ++rank) {
            int previous = 0;
            (void)cudaGetDevice(&previous);
            (void)cudaSetDevice(rank);
            if (handles_[rank])
                (void)abort_(handles_[rank]);
            (void)cudaSetDevice(previous);
        }
    }

  private:
    template <class Fn>
    Fn symbol(const char* name) {
#if defined(_WIN32)
        auto address = GetProcAddress(library_, name);
#else
        auto address = dlsym(library_, name);
#endif
        if (!address)
            throw std::runtime_error(std::string("Qwen native TP2 missing NCCL symbol: ") + name);
        return reinterpret_cast<Fn>(address);
    }
    void close() noexcept {
        if (!aborted_ && destroy_) {
            for (int rank = 0; rank < 2; ++rank)
                if (handles_[rank]) {
                    int previous = 0;
                    (void)cudaGetDevice(&previous);
                    (void)cudaSetDevice(rank);
                    (void)destroy_(handles_[rank]);
                    handles_[rank] = nullptr;
                    (void)cudaSetDevice(previous);
                }
        }
        if (library_) {
#if defined(_WIN32)
            FreeLibrary(library_);
#else
            dlclose(library_);
#endif
            library_ = nullptr;
        }
    }
#if defined(_WIN32)
    HMODULE library_{};
#else
    void* library_{};
#endif
    std::array<void*, 2> handles_{};
    int (*destroy_)(void*){};
    int (*abort_)(void*){};
    std::atomic<bool> aborted_{false};
};

class TensorParallelTask final : public ITextGeneration,
                                 public IVisionLanguageGeneration,
                                 public api::IModel,
                                 public api::ITextContinuation,
                                 public api::IImagesTextToText,
                                 public api::IVideoTextToText,
                                 public api::IImageVideoTextToText {
  public:
    TensorParallelTask(NativeRankFactory factory, bool media)
        : factory_(std::move(factory)), media_(media) {
        try {
            initialize();
            cache_task_metadata();
        } catch (...) {
            clear();
            throw;
        }
    }
    ~TensorParallelTask() { clear(); }
    const char* task() const noexcept override {
        return media_ ? "images_text_to_text" : "text_generation";
    }
    int32_t default_max_new_tokens() const override { return default_max_new_tokens_; }
    std::vector<api::TaskInstance> task_bindings() override { return bindings_; }
    TextResult generate(const std::string& text, const TextGenerationConfig& cfg) override {
        return invoke(
            [&](ITask& rank) { return dynamic_cast<ITextGeneration&>(rank).generate(text, cfg); });
    }
    TextResult generate(const std::string& text, const float* pixels, int32_t h, int32_t w,
                        const TextGenerationConfig& cfg) override {
        if (!media_)
            throw std::invalid_argument("Qwen native text TP bundle has no visual tower");
        return invoke([&](ITask& rank) {
            return dynamic_cast<IVisionLanguageGeneration&>(rank).generate(text, pixels, h, w, cfg);
        });
    }
    TextResult run(const api::TextContinuationRequest& req, api::ConfigView cfg) override {
        return typed<api::ITextContinuation>(req, cfg);
    }
    TextResult run(const api::ImagesTextToTextRequest& req, api::ConfigView cfg) override {
        return typed<api::IImagesTextToText>(req, cfg);
    }
    TextResult run(const api::VideoTextToTextRequest& req, api::ConfigView cfg) override {
        return typed<api::IVideoTextToText>(req, cfg);
    }
    TextResult run(const api::ImageVideoTextToTextRequest& req, api::ConfigView cfg) override {
        return typed<api::IImageVideoTextToText>(req, cfg);
    }

  private:
    void cache_task_metadata() {
        default_max_new_tokens_ =
            dynamic_cast<ITextGeneration&>(*ranks_[0]).default_max_new_tokens();
        if (!media_)
            return; // Core retains its established legacy text bridge.
        // NativeMediaTask supplies static field schemas. Retain them and bind to
        // this wrapper, not to ranks that may be replaced during recovery.
        auto fields = dynamic_cast<api::IModel&>(*ranks_[0]).task_bindings();
        for (auto& field : fields) {
            if (field.key.id == api::ITextContinuation::kTask)
                field.implementation = static_cast<api::ITextContinuation*>(this);
            else if (field.key.id == api::IImagesTextToText::kTask)
                field.implementation = static_cast<api::IImagesTextToText*>(this);
            else if (field.key.id == api::IVideoTextToText::kTask)
                field.implementation = static_cast<api::IVideoTextToText*>(this);
            else if (field.key.id == api::IImageVideoTextToText::kTask)
                field.implementation = static_cast<api::IImageVideoTextToText*>(this);
            else
                throw std::runtime_error("Unexpected Qwen native rank task contract");
        }
        bindings_ = std::move(fields);
    }
    void clear() noexcept {
        for (int rank = 0; rank < 2; ++rank) {
            int previous = 0;
            (void)cudaGetDevice(&previous);
            (void)cudaSetDevice(rank);
            ranks_[rank].reset();
            (void)cudaSetDevice(previous);
        }
        group_.reset();
    }
    void initialize() {
        clear();
        group_ = std::make_shared<Communicators>();
        // setCommunicator may split the parent communicator collectively.
        // Constructing rank 0 completely before rank 1 deadlocks that split.
        std::array<std::future<void>, 2> jobs;
        std::exception_ptr error;
        try {
            for (int rank = 0; rank < 2; ++rank) {
                jobs[rank] = std::async(std::launch::async, [&, rank] {
                    try {
                        DeviceGuard guard(rank);
                        ModuleCreateOptions options;
                        options.distributed_communicator = group_->get(rank);
                        options.distributed_owner = group_;
                        ranks_[rank].reset(factory_(rank, options));
                        if (!ranks_[rank])
                            throw std::runtime_error("Qwen native rank construction failed");
                    } catch (...) {
                        group_->abort();
                        throw;
                    }
                });
            }
        } catch (...) {
            group_->abort();
            error = std::current_exception();
        }
        for (auto& job : jobs)
            if (job.valid()) {
                try {
                    job.get();
                } catch (...) {
                    if (!error)
                        error = std::current_exception();
                }
            }
        if (error) {
            clear();
            std::rethrow_exception(error);
        }
        poisoned_ = false;
    }
    template <class Interface, class Request>
    TextResult typed(const Request& req, api::ConfigView cfg) {
        if (!media_)
            throw std::invalid_argument("Qwen native text TP bundle has no typed media task");
        return invoke([&](ITask& rank) { return dynamic_cast<Interface&>(rank).run(req, cfg); });
    }
    template <class Call>
    TextResult invoke(Call call) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (poisoned_)
            initialize();
        std::array<std::future<TextResult>, 2> jobs;
        try {
            for (int rank = 0; rank < 2; ++rank) {
                jobs[rank] = std::async(std::launch::async, [&, rank] {
                    try {
                        DeviceGuard guard(rank);
                        return call(*ranks_[rank]);
                    } catch (...) {
                        // Unblock the other rank if it is already waiting in a collective.
                        // Never retry the failed request. Recreate contexts on the next
                        // explicit request, including after input-validation errors.
                        poisoned_ = true;
                        group_->abort();
                        throw;
                    }
                });
            }
        } catch (...) {
            poisoned_ = true;
            group_->abort();
            for (auto& job : jobs)
                if (job.valid()) {
                    try {
                        (void)job.get();
                    } catch (...) {
                    }
                }
            throw;
        }
        std::array<TextResult, 2> results;
        std::exception_ptr error;
        for (int rank = 0; rank < 2; ++rank) {
            try {
                results[rank] = jobs[rank].get();
            } catch (...) {
                if (!error)
                    error = std::current_exception();
            }
        }
        if (error)
            std::rethrow_exception(error);
        if (results[0].token_ids != results[1].token_ids)
            throw std::runtime_error("Qwen native TP rank outputs disagree");
        return std::move(results[0]);
    }
    NativeRankFactory factory_;
    bool media_;
    int32_t default_max_new_tokens_{};
    std::vector<api::TaskInstance> bindings_;
    std::shared_ptr<Communicators> group_;
    std::array<std::unique_ptr<ITask>, 2> ranks_;
    std::atomic<bool> poisoned_{false};
    std::mutex mutex_;
};
// Preserve the existing core's legacy text adapter: only media publishes an
// explicit typed-task table. This wrapper must not advertise an empty IModel.
class TextTask final : public ITextGeneration {
  public:
    explicit TextTask(std::unique_ptr<TensorParallelTask> task) : task_(std::move(task)) {}
    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg) override {
        return task_->generate(prompt, cfg);
    }
    int32_t default_max_new_tokens() const override { return task_->default_max_new_tokens(); }

  private:
    std::unique_ptr<TensorParallelTask> task_;
};
} // namespace

ITask* create_native_tensor_parallel_task(NativeRankFactory factory, bool media) {
    auto task = std::make_unique<TensorParallelTask>(std::move(factory), media);
    if (media)
        return task.release();
    return new TextTask(std::move(task));
}
} // namespace trtmc::qwen3_8
