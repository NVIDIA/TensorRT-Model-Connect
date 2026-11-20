/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The two cache tensors a cache-aware streaming encoder carries from one step
// to the next.
//
// A step reads cache_last_channel and cache_last_time and writes
// cache_last_channel_next and cache_last_time_next, which the following step
// reads. Where the pipeline lends device buffers, a stream keeps the caches in
// them and binds them as the encoder's inputs and outputs: two generations,
// exchanged after each step, so no cache byte crosses to the host. Elsewhere
// the encoder is given host tensors.
//
// Device buffers have exactly the configured sizes, so there an encoder is
// refused unless it reports its four cache tensors as static FP32 tensors of
// the configured shapes, runs on the CUDA stream the buffers are cleared on,
// and is enqueued eagerly: a CUDA graph would be reset by every step's
// bindings.

#include "trtmc/runtime/tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace rnnt {

// Owns one device allocation.
class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    ~DeviceBuffer() { cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    cudaError_t allocate(std::size_t bytes) {
        cudaFree(ptr_);
        ptr_ = nullptr;
        const cudaError_t error = cudaMalloc(&ptr_, bytes);
        if (error != cudaSuccess)
            ptr_ = nullptr;
        return error;
    }

    void* get() const { return ptr_; }

  private:
    void* ptr_{nullptr};
};

// Both generations of one stream's caches on the device.
struct StreamingCacheBuffers {
    StreamingCacheBuffers(std::size_t channel_bytes_in, std::size_t time_bytes_in)
        : channel_bytes(channel_bytes_in), time_bytes(time_bytes_in) {
        for (auto& buffer : channel) {
            if (buffer.allocate(channel_bytes) != cudaSuccess)
                throw std::runtime_error(
                    "RnntPipeline: unable to allocate streaming encoder cache buffers");
        }
        for (auto& buffer : time) {
            if (buffer.allocate(time_bytes) != cudaSuccess)
                throw std::runtime_error(
                    "RnntPipeline: unable to allocate streaming encoder cache buffers");
        }
    }

    const std::size_t channel_bytes;
    const std::size_t time_bytes;
    DeviceBuffer channel[2];
    DeviceBuffer time[2];
};

// Lends cache buffers to streams. A finished stream's buffers go to the next
// one, so a caller that makes a stream per request allocates only for the
// first. Not synchronized: a pipeline's streams share its modules and run on
// one thread.
class StreamingCacheBufferPool {
  public:
    std::unique_ptr<StreamingCacheBuffers> acquire(std::size_t channel_bytes,
                                                   std::size_t time_bytes) {
        for (auto it = idle_.begin(); it != idle_.end(); ++it) {
            if ((*it)->channel_bytes == channel_bytes && (*it)->time_bytes == time_bytes) {
                auto buffers = std::move(*it);
                idle_.erase(it);
                return buffers;
            }
        }
        return std::make_unique<StreamingCacheBuffers>(channel_bytes, time_bytes);
    }

    void release(std::unique_ptr<StreamingCacheBuffers> buffers) noexcept {
        if (!buffers)
            return;
        try {
            idle_.push_back(std::move(buffers));
        } catch (...) {
            // Not kept: `buffers` still owns the allocations and frees them.
        }
    }

  private:
    std::vector<std::unique_ptr<StreamingCacheBuffers>> idle_;
};

// One stream's encoder caches, on the device or in host vectors.
class StreamingEncoderCache {
  public:
    static inline const std::string kChannel{"cache_last_channel"};
    static inline const std::string kTime{"cache_last_time"};
    static inline const std::string kChannelNext{"cache_last_channel_next"};
    static inline const std::string kTimeNext{"cache_last_time_next"};

    // A new cache is all zero. `device_pool` lends the device buffers and takes
    // them back when the cache is destroyed; a null pool keeps the caches in
    // host vectors. `stream` is the CUDA stream the encoders run on, and
    // `first_encoder` the one the stream starts with: it is checked before any
    // buffer is taken.
    StreamingEncoderCache(std::vector<int64_t> channel_shape, std::vector<int64_t> time_shape,
                          std::shared_ptr<StreamingCacheBufferPool> device_pool,
                          cudaStream_t stream, const ITrtModule& first_encoder)
        : channel_shape_(std::move(channel_shape)), time_shape_(std::move(time_shape)),
          device_pool_(std::move(device_pool)), stream_(stream) {
        require(first_encoder);
        if (device_pool_) {
            device_ = device_pool_->acquire(element_count(channel_shape_) * sizeof(float),
                                            element_count(time_shape_) * sizeof(float));
        }
        reset();
    }

    ~StreamingEncoderCache() {
        if (device_pool_)
            device_pool_->release(std::move(device_));
    }

    StreamingEncoderCache(const StreamingEncoderCache&) = delete;
    StreamingEncoderCache& operator=(const StreamingEncoderCache&) = delete;

    bool on_device() const { return device_pool_ != nullptr; }

    // Back to the start of a stream: the caches the next step reads are zero.
    void reset() {
        if (!on_device()) {
            host_channel_[current_].assign(element_count(channel_shape_), 0.0F);
            host_time_[current_].assign(element_count(time_shape_), 0.0F);
            return;
        }
        // On the encoders' stream, so the fill is ordered before the next step.
        // Lent buffers hold an earlier stream's caches: until both fills are
        // queued no step may read them.
        cleared_ = false;
        if (cudaMemsetAsync(device_->channel[current_].get(), 0, device_->channel_bytes, stream_) !=
                cudaSuccess ||
            cudaMemsetAsync(device_->time[current_].get(), 0, device_->time_bytes, stream_) !=
                cudaSuccess)
            throw std::runtime_error(
                "RnntPipeline: unable to clear streaming encoder cache buffers");
        cleared_ = true;
    }

    // Refuses an encoder that device buffers cannot be bound to; see the top
    // of this file. Host tensors suit any encoder.
    void require(const ITrtModule& encoder) const {
        if (!on_device())
            return;
        if (encoder.stream() != stream_)
            throw std::runtime_error("RnntPipeline: a streaming encoder runs on another CUDA "
                                     "stream than its cache buffers are cleared on");
        if (encoder.cuda_graph_active())
            throw std::runtime_error("RnntPipeline: a streaming encoder under a CUDA graph "
                                     "cannot take new cache buffers in every step");
        require_tensor(encoder, kChannel, true, channel_shape_);
        require_tensor(encoder, kTime, true, time_shape_);
        require_tensor(encoder, kChannelNext, false, channel_shape_);
        require_tensor(encoder, kTimeNext, false, time_shape_);
    }

    // Gives `encoder` the caches of the step it is about to run: device buffers
    // bound as its cache inputs and outputs, or host tensors added to `inputs`.
    // The device buffers are bound in every step, because the first-step and
    // the steady encoder alternate and streams share a pipeline's encoders. So
    // the tensors' addresses change from step to step: a pipeline lends device
    // buffers only where its encoders are enqueued eagerly, since a captured
    // CUDA graph would have to follow every change.
    void attach(ITrtModule& encoder, TensorMap& inputs) {
        require(encoder);
        if (!on_device()) {
            inputs[kChannel] = host_tensor(host_channel_[current_], channel_shape_);
            inputs[kTime] = host_tensor(host_time_[current_], time_shape_);
            return;
        }
        if (!cleared_)
            throw std::runtime_error(
                "RnntPipeline: streaming encoder cache buffers were not cleared");
        const int next = current_ ^ 1;
        encoder.bind_external(kChannel, device_->channel[current_].get());
        encoder.bind_external(kTime, device_->time[current_].get());
        encoder.bind_external(kChannelNext, device_->channel[next].get());
        encoder.bind_external(kTimeNext, device_->time[next].get());
    }

    // Takes the step's next caches from `outputs` when they are host tensors;
    // on the device the encoder has already written them into the bound
    // buffers.
    void collect(const TensorMap& outputs) {
        if (on_device())
            return;
        const auto channel = outputs.find(kChannelNext);
        const auto time = outputs.find(kTimeNext);
        if (channel == outputs.end() || time == outputs.end())
            throw std::runtime_error("RnntPipeline: streaming encoder missing required outputs");
        const int next = current_ ^ 1;
        const auto* channel_src = static_cast<const float*>(channel->second.data);
        host_channel_[next].assign(channel_src, channel_src + channel->second.numel());
        const auto* time_src = static_cast<const float*>(time->second.data);
        host_time_[next].assign(time_src, time_src + time->second.numel());
    }

    // The step is complete: the caches it wrote are the ones the following
    // step reads. Until this call a failed step leaves the stream's caches as
    // they were.
    void advance() { current_ ^= 1; }

  private:
    static std::size_t element_count(const std::vector<int64_t>& shape) {
        std::size_t count = 1;
        for (const auto dim : shape)
            count *= static_cast<std::size_t>(dim);
        return count;
    }

    static Tensor host_tensor(std::vector<float>& values, const std::vector<int64_t>& shape) {
        Tensor tensor;
        tensor.data = values.data();
        tensor.shape = shape;
        tensor.dtype = DType::kFloat32;
        return tensor;
    }

    // A bound buffer has exactly the size the stream allocated, so the engine
    // must have the tensor, and of that size: static, FP32 and of the
    // configured shape.
    static void require_tensor(const ITrtModule& encoder, const std::string& name, bool is_input,
                               const std::vector<int64_t>& shape) {
        const bool reported = is_input ? encoder.has_input(name) : encoder.has_output(name);
        if (!reported || (is_input && encoder.input_is_dynamic(name)) ||
            encoder.tensor_dtype(name) != DType::kFloat32 || encoder.tensor_shape(name) != shape)
            throw std::runtime_error("RnntPipeline: streaming encoder tensor '" + name +
                                     "' is missing or does not match the runtime configuration");
    }

    std::vector<int64_t> channel_shape_;
    std::vector<int64_t> time_shape_;
    std::shared_ptr<StreamingCacheBufferPool> device_pool_;
    cudaStream_t stream_{nullptr};
    std::unique_ptr<StreamingCacheBuffers> device_;
    std::vector<float> host_channel_[2];
    std::vector<float> host_time_[2];
    int current_{0};
    bool cleared_{false};
};

} // namespace rnnt
} // namespace trtmc
