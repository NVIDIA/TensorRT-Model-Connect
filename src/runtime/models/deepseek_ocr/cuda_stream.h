#pragma once

// deepseek_ocr-owned CUDA stream RAII helper.

#include <cuda_runtime_api.h>

namespace trtmc {

class DeepseekOcrCudaStream final {
  public:
    DeepseekOcrCudaStream() { status_ = cudaStreamCreate(&stream_); }
    ~DeepseekOcrCudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    DeepseekOcrCudaStream(const DeepseekOcrCudaStream&) = delete;
    DeepseekOcrCudaStream& operator=(const DeepseekOcrCudaStream&) = delete;
    DeepseekOcrCudaStream(DeepseekOcrCudaStream&& other) noexcept
        : stream_(other.stream_), status_(other.status_) {
        other.stream_ = nullptr;
    }
    DeepseekOcrCudaStream& operator=(DeepseekOcrCudaStream&& other) noexcept {
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
