#pragma once

// qwen_vl-owned CUDA stream RAII helper.

#include <cuda_runtime_api.h>

namespace trtmc {

class QwenVlCudaStream final {
  public:
    QwenVlCudaStream() { status_ = cudaStreamCreate(&stream_); }
    ~QwenVlCudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    QwenVlCudaStream(const QwenVlCudaStream&) = delete;
    QwenVlCudaStream& operator=(const QwenVlCudaStream&) = delete;
    QwenVlCudaStream(QwenVlCudaStream&& other) noexcept
        : stream_(other.stream_), status_(other.status_) {
        other.stream_ = nullptr;
    }
    QwenVlCudaStream& operator=(QwenVlCudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            status_ = other.status_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    bool ok() const { return status_ == cudaSuccess; }
    cudaStream_t get() const { return stream_; }

  private:
    cudaStream_t stream_{nullptr};
    cudaError_t status_{cudaSuccess};
};

} // namespace trtmc
