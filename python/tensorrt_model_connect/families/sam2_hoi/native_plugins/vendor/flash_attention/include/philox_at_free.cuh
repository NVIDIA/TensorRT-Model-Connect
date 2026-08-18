#pragma once

// ATen-free transcription of the PhiloxCudaState layout and inline unpack helper
// from PyTorch commit e2d141dbde55c2a4370fac5165b0561b6af4798b:
//   aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh
//   aten/src/ATen/cuda/detail/UnpackRaw.cuh
// Dropout remains compile-time off; retaining this dead-state path preserves the
// reviewed FlashAttention kernel's code generation without linking ATen or Torch.

#include <cstdint>
#include <tuple>

namespace at {

struct PhiloxCudaState {
    PhiloxCudaState() = default;
    PhiloxCudaState(uint64_t seed, uint64_t offset) {
        seed_.val = seed;
        offset_.val = offset;
    }
    PhiloxCudaState(int64_t* seed, int64_t* offset_extragraph,
                    uint32_t offset_intragraph) {
        seed_.ptr = seed;
        offset_.ptr = offset_extragraph;
        offset_intragraph_ = offset_intragraph;
        captured_ = true;
    }

    union Payload {
        uint64_t val;
        int64_t* ptr;
    };

    Payload seed_{};
    Payload offset_{};
    uint32_t offset_intragraph_ = 0;
    bool captured_ = false;
};

}  // namespace at

namespace at::cuda::philox {

__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(
    at::PhiloxCudaState arg) {
    if (arg.captured_) {
        return std::make_tuple(
            static_cast<uint64_t>(*arg.seed_.ptr),
            static_cast<uint64_t>(*arg.offset_.ptr + arg.offset_intragraph_));
    }
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
}

}  // namespace at::cuda::philox
