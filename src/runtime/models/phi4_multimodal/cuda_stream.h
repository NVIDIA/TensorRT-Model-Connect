#pragma once

// phi4_multimodal-owned CUDA stream RAII helper.

#include <cuda_runtime_api.h>

namespace trtmc {

class Phi4MultimodalCudaStream final {
  public:
    Phi4MultimodalCudaStream() { status_ = cudaStreamCreate(&stream_); }
    ~Phi4MultimodalCudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    Phi4MultimodalCudaStream(const Phi4MultimodalCudaStream&) = delete;
    Phi4MultimodalCudaStream& operator=(const Phi4MultimodalCudaStream&) = delete;
    Phi4MultimodalCudaStream(Phi4MultimodalCudaStream&& other) noexcept
        : stream_(other.stream_), status_(other.status_) {
        other.stream_ = nullptr;
    }
    Phi4MultimodalCudaStream& operator=(Phi4MultimodalCudaStream&& other) noexcept {
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
