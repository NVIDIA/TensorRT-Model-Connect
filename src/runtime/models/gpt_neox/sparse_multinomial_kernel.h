#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc {

struct GptNeoxTorchMultinomialExecutionPolicy {
    int32_t total_threads{0};
    uint64_t counter_offset{0};
};

GptNeoxTorchMultinomialExecutionPolicy
gpt_neox_compute_torch_multinomial_execution_policy(int32_t numel);

void gpt_neox_gpu_sparse_torch_multinomial_exact(const int32_t* d_indices, const float* d_probs,
                                                 int32_t keep, uint64_t seed, uint64_t base_offset,
                                                 int32_t total_threads, int32_t* d_token_id,
                                                 cudaStream_t stream);

} // namespace trtmc
