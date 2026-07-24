/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>

namespace trtmc::runtime_kv {

// The qualified implementation is intentionally bounded. cuDNN's selected
// plan workspace is checked against this limit at runtime; TensorRT reserves
// it once per execution context rather than once per transformer layer.
inline constexpr std::size_t kCudnnAttentionPlanWorkspaceLimit = 1U << 20;
inline constexpr std::size_t kCudnnAttentionControlScalarCount = 4;

struct CudnnAttentionConfig {
    int32_t num_query_heads{0};
    int32_t num_kv_heads{0};
    int32_t head_dim{0};
    int32_t chunk_limit{0};
};

// True only when the plugin was compiled against cuDNN Frontend and the
// loaded cuDNN ABI is new enough for the qualified SDPA graph.
bool native_cudnn_attention_available() noexcept;

// Upper bound used by IPluginV3::getWorkspaceSize. padded_query_rows is one
// for decode and C for the chunked prefill role.
std::size_t cudnn_attention_workspace_size(const CudnnAttentionConfig& config,
                                           int32_t padded_query_rows) noexcept;

class CudnnAttentionExecutor {
  public:
    explicit CudnnAttentionExecutor(CudnnAttentionConfig config);
    ~CudnnAttentionExecutor();

    CudnnAttentionExecutor(const CudnnAttentionExecutor&) = delete;
    CudnnAttentionExecutor& operator=(const CudnnAttentionExecutor&) = delete;

    // Graph plans are immutable and shared process-wide by
    // (device, role/C, P, Hq, Hkv, D). Each executor retains its own cuDNN
    // handle and variant pack, so separate TensorRT contexts do not share
    // mutable launch state.
    bool prepare(int32_t padded_kv_rows, int32_t padded_query_rows) noexcept;
    bool prepare_history(int32_t padded_kv_rows, int32_t padded_query_rows) noexcept;
    bool prepare_current(int32_t padded_query_rows) noexcept;

    bool execute_history(const void* query, const void* history_k, const void* history_v,
                         void* history_context, void* history_log_sum_exp,
                         const int32_t* sequence_length_q, const int32_t* sequence_length_history,
                         void* plan_workspace, std::size_t plan_workspace_capacity,
                         cudaStream_t stream) noexcept;

    bool execute_current(const void* query, const void* current_k, const void* current_v,
                         void* current_context, void* current_log_sum_exp,
                         const int32_t* sequence_length_q, const int32_t* sequence_length_current,
                         void* plan_workspace, std::size_t plan_workspace_capacity,
                         cudaStream_t stream) noexcept;

    // Execute two normalized SDPA segments. The history graph is noncausal
    // over token-major [T,Hkv*D] cache rows; the current graph is lower-right
    // causal over head-major [1,Hkv,C,D] rows. Both graphs emit their
    // normalized context plus log-sum-exp so the caller can combine them
    // without materializing logits or copying history.
    bool execute_segmented(const void* query, const void* history_k, const void* history_v,
                           const void* current_k, const void* current_v, void* history_context,
                           void* current_context, void* history_log_sum_exp,
                           void* current_log_sum_exp, const int32_t* sequence_length_q,
                           const int32_t* sequence_length_history,
                           const int32_t* sequence_length_current, void* plan_workspace,
                           std::size_t plan_workspace_capacity, cudaStream_t stream) noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<CudnnAttentionExecutor>
make_cudnn_attention_executor(CudnnAttentionConfig config) noexcept;

} // namespace trtmc::runtime_kv
