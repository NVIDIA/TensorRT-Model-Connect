#pragma once

// lance-owned CUDA stream RAII helper.

#include <cuda_runtime_api.h>

namespace trtmc {

class LanceCudaStream final {
  public:
    LanceCudaStream() { status_ = cudaStreamCreate(&stream_); }
    ~LanceCudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    LanceCudaStream(const LanceCudaStream&) = delete;
    LanceCudaStream& operator=(const LanceCudaStream&) = delete;
    LanceCudaStream(LanceCudaStream&& other) noexcept
        : stream_(other.stream_), status_(other.status_) {
        other.stream_ = nullptr;
    }
    LanceCudaStream& operator=(LanceCudaStream&& other) noexcept {
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
