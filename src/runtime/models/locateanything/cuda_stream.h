#pragma once

// locateanything-owned CUDA stream RAII helper.

#include <cuda_runtime_api.h>

namespace trtmc {

class LocateAnythingCudaStream final {
  public:
    LocateAnythingCudaStream() { status_ = cudaStreamCreate(&stream_); }
    ~LocateAnythingCudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    LocateAnythingCudaStream(const LocateAnythingCudaStream&) = delete;
    LocateAnythingCudaStream& operator=(const LocateAnythingCudaStream&) = delete;
    LocateAnythingCudaStream(LocateAnythingCudaStream&& other) noexcept
        : stream_(other.stream_), status_(other.status_) {
        other.stream_ = nullptr;
    }
    LocateAnythingCudaStream& operator=(LocateAnythingCudaStream&& other) noexcept {
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
