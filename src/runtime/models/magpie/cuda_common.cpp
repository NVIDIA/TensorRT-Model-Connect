#include "runtime/models/magpie/cuda_common.h"

namespace trtmc {

MagpieCudaStream::MagpieCudaStream() {
    status_ = cudaStreamCreate(&stream_);
}

MagpieCudaStream::~MagpieCudaStream() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

MagpieCudaStream::MagpieCudaStream(MagpieCudaStream&& other) noexcept
    : stream_(other.stream_), status_(other.status_) {
    other.stream_ = nullptr;
}

MagpieCudaStream& MagpieCudaStream::operator=(MagpieCudaStream&& other) noexcept {
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

bool MagpieCudaStream::ok() const {
    return status_ == cudaSuccess;
}

cudaStream_t MagpieCudaStream::get() const {
    return stream_;
}

MagpieCudaBuffer::MagpieCudaBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ == 0) {
        return;
    }
    status_ = cudaMalloc(&ptr_, bytes_);
}

MagpieCudaBuffer::~MagpieCudaBuffer() {
    if (ptr_ != nullptr) {
        cudaFree(ptr_);
    }
}

MagpieCudaBuffer::MagpieCudaBuffer(MagpieCudaBuffer&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_), status_(other.status_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
}

MagpieCudaBuffer& MagpieCudaBuffer::operator=(MagpieCudaBuffer&& other) noexcept {
    if (this != &other) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        status_ = other.status_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

bool MagpieCudaBuffer::ok() const {
    return status_ == cudaSuccess;
}

void* MagpieCudaBuffer::data() const {
    return ptr_;
}

std::size_t MagpieCudaBuffer::size() const {
    return bytes_;
}

} // namespace trtmc
