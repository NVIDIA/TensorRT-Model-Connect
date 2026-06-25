#pragma once

#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc {

bool bloom_triattention_score_candidates_gpu(
    const void* d_cache, DType cache_dtype, int32_t kv_dim, int32_t head_dim, bool rope_interleaved,
    const int32_t* d_candidate_indices, int32_t candidate_count,
    const int32_t* d_positions_per_head, const float* d_inv_freq, const float* d_cos_phase,
    const float* d_sin_phase, int32_t num_offsets, const int32_t* d_head_offsets,
    const int32_t* d_head_cache_indices, const float* d_q_mean_real, const float* d_q_mean_imag,
    const float* d_q_abs_mean, const float* d_freq_scale_sq, int32_t kv_head_count,
    bool disable_mlr, bool disable_trig, bool aggregation_max, float* d_scores_out,
    cudaStream_t stream);

bool bloom_triattention_compact_rows_gpu(const void* d_src, void* d_scratch, DType cache_dtype,
                                         int32_t kv_dim, const int32_t* d_keep_indices,
                                         int32_t keep_count, int32_t head_dim, int32_t num_kv_heads,
                                         int32_t query_group_size, cudaStream_t stream);

} // namespace trtmc
