#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc {

// GPU-side argmax over a float logit vector.
// Writes the index of the maximum element to d_token_id (device memory).
// Optionally writes the max logit value to d_logit_val (pass nullptr to skip).
// Runs asynchronously on the given stream.
void phi_moe_gpu_argmax(const float* d_logits, int32_t vocab_size, int32_t* d_token_id,
                        float* d_logit_val, cudaStream_t stream);

} // namespace trtmc
