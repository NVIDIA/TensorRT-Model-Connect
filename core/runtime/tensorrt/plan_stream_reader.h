/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/plan_reader.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

// TensorRT may request either host or device destinations. File I/O remains on
// the host; one bounded pinned buffer supplies device copies on the given stream.
class TrtPlanStreamReader final : public nvinfer1::IStreamReaderV2 {
  public:
    static constexpr std::size_t transfer_bytes = 1U << 20;

    explicit TrtPlanStreamReader(std::unique_ptr<IPlanReader> source) : source_(std::move(source)) {
        if (!source_ || !source_->seek(0, PlanSeekOrigin::begin))
            throw std::runtime_error("Cannot rewind serialized plan stream");
    }

    ~TrtPlanStreamReader() override {
        if (transfer_ != nullptr) {
            // On a persistent CUDA failure, retain the pinned allocation until
            // process teardown rather than free storage a queued copy may use.
            if (pending_ && cudaStreamSynchronize(pending_stream_) != cudaSuccess)
                return;
            (void)cudaFreeHost(transfer_);
        }
    }

    TrtPlanStreamReader(const TrtPlanStreamReader&) = delete;
    TrtPlanStreamReader& operator=(const TrtPlanStreamReader&) = delete;

    std::int64_t read(void* destination, std::int64_t count,
                      cudaStream_t stream) noexcept override {
        std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
        try {
            lock.lock();
            if (error_)
                return -1;
            if (count < 0 ||
                static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument("Invalid serialized plan stream read size");
            if (count == 0)
                return 0;
            if (destination == nullptr)
                throw std::invalid_argument("Serialized plan stream requires a destination");

            cudaPointerAttributes attributes{};
            // Supported CUDA runtimes classify ordinary host pointers as
            // Unregistered. A query failure must not become an unchecked host write.
            require_cuda(cudaPointerGetAttributes(&attributes, destination));
            if (attributes.type == cudaMemoryTypeHost ||
                attributes.type == cudaMemoryTypeUnregistered) {
                const auto received = source_->read(destination, static_cast<std::size_t>(count));
                if (received > static_cast<std::size_t>(count))
                    throw std::runtime_error("Serialized plan source returned excess data");
                return static_cast<std::int64_t>(received);
            }

            if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged)
                throw std::runtime_error("Unsupported serialized plan destination memory type");

            if (transfer_ == nullptr)
                require_cuda(cudaMallocHost(&transfer_, transfer_bytes));
            std::size_t done = 0;
            const auto requested = static_cast<std::size_t>(count);
            while (done < requested) {
                const auto chunk =
                    source_->read(transfer_, std::min(transfer_bytes, requested - done));
                if (chunk == 0)
                    break;
                if (chunk > transfer_bytes || chunk > requested - done)
                    throw std::runtime_error("Serialized plan source returned excess data");
                pending_stream_ = stream;
                pending_ = true;
                require_cuda(cudaMemcpyAsync(static_cast<char*>(destination) + done, transfer_,
                                             chunk, cudaMemcpyDefault, stream));
                // Complete this copy before reusing or destroying the pinned
                // bytes. Synchronize only the stream supplied by TensorRT.
                require_cuda(cudaStreamSynchronize(stream));
                pending_ = false;
                done += chunk;
            }
            return static_cast<std::int64_t>(done);
        } catch (...) {
            if (lock.owns_lock())
                error_ = std::current_exception();
            return -1;
        }
    }

    bool seek(std::int64_t offset, nvinfer1::SeekPosition origin) noexcept override {
        std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
        try {
            lock.lock();
            if (error_)
                return false;
            switch (origin) {
            case nvinfer1::SeekPosition::kSET:
                return source_->seek(offset, PlanSeekOrigin::begin);
            case nvinfer1::SeekPosition::kCUR:
                return source_->seek(offset, PlanSeekOrigin::current);
            case nvinfer1::SeekPosition::kEND:
                return source_->seek(offset, PlanSeekOrigin::end);
            default:
                return false;
            }
        } catch (...) {
            if (lock.owns_lock())
                error_ = std::current_exception();
            return false;
        }
    }

    void rethrow_if_failed() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (error_)
            std::rethrow_exception(error_);
    }

  private:
    static void require_cuda(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Serialized plan CUDA transfer failed: ") +
                                     cudaGetErrorString(status));
    }

    std::unique_ptr<IPlanReader> source_;
    void* transfer_{nullptr};
    cudaStream_t pending_stream_{nullptr};
    bool pending_{false};
    mutable std::mutex mutex_;
    std::exception_ptr error_;
};

} // namespace trtmc
