/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>

#define VSA_BLK128 false
#define VSA_BHSD true
#include "block_sparse_launch_sm100a.cuh"

namespace {

constexpr int kBatch = 1;
constexpr int kHeads = 56;
constexpr int kHeadDim = 128;
constexpr int kTileSize = 64;
constexpr int kPrefixTiles = 16;
constexpr int kVideoTiles = 630;
constexpr int kTiles = kPrefixTiles + kVideoTiles;
constexpr int kTopK = 63;
constexpr int kSparseKeys = kPrefixTiles + kTopK;
constexpr int kSequence = kTiles * kTileSize;
constexpr int64_t kIndexElements = static_cast<int64_t>(kBatch) * kHeads * kTiles * kTiles;
constexpr int64_t kCountElements = static_cast<int64_t>(kBatch) * kHeads * kTiles;
constexpr int64_t kWorkspaceBytes = (kIndexElements + kCountElements) * sizeof(int32_t);

void require_cuda_contiguous(const tvm::ffi::TensorView& tensor, const char* name) {
    if (tensor.device().device_type != kDLCUDA)
        TVM_FFI_THROW(ValueError) << name << " must be a CUDA tensor";
    if (!tensor.IsContiguous())
        TVM_FFI_THROW(ValueError) << name << " must be contiguous";
}

void require_dtype(const tvm::ffi::TensorView& tensor, uint8_t code, uint8_t bits,
                   const char* name) {
    const auto dtype = tensor.dtype();
    if (dtype.code != code || dtype.bits != bits || dtype.lanes != 1)
        TVM_FFI_THROW(ValueError) << name << " has the wrong dtype";
}

template <int Rank>
void require_shape(const tvm::ffi::TensorView& tensor, const int64_t (&shape)[Rank],
                   const char* name) {
    if (tensor.ndim() != Rank)
        TVM_FFI_THROW(ValueError) << name << " has the wrong rank";
    for (int index = 0; index < Rank; ++index) {
        if (tensor.shape()[index] != shape[index])
            TVM_FFI_THROW(ValueError) << name << " has the wrong shape";
    }
}

__global__ void build_block_map(const int32_t* topk, int32_t* q2k_idx, int32_t* q2k_num) {
    const int row = static_cast<int>(blockIdx.x);
    const int query_tile = row % kTiles;
    const int count = query_tile < kPrefixTiles ? kTiles : kSparseKeys;
    if (threadIdx.x == 0)
        q2k_num[row] = count;
    int32_t* output = q2k_idx + static_cast<int64_t>(row) * kTiles;
    if (query_tile < kPrefixTiles) {
        for (int key = static_cast<int>(threadIdx.x); key < kTiles; key += blockDim.x)
            output[key] = key;
        return;
    }
    for (int key = static_cast<int>(threadIdx.x); key < kPrefixTiles; key += blockDim.x)
        output[key] = key;
    const int32_t* selected = topk + static_cast<int64_t>(row) * kTopK;
    for (int key = static_cast<int>(threadIdx.x); key < kTopK; key += blockDim.x)
        output[kPrefixTiles + key] = kPrefixTiles + selected[key];
}

void run_vsa(tvm::ffi::TensorView query, tvm::ffi::TensorView key,
             tvm::ffi::TensorView value, tvm::ffi::TensorView topk,
             tvm::ffi::TensorView variable_block_sizes,
             tvm::ffi::TensorView workspace, tvm::ffi::TensorView output) {
    for (const auto& item : {std::pair{&query, "query"}, std::pair{&key, "key"},
                             std::pair{&value, "value"}, std::pair{&topk, "topk"},
                             std::pair{&variable_block_sizes, "variable_block_sizes"},
                             std::pair{&workspace, "workspace"}, std::pair{&output, "output"}})
        require_cuda_contiguous(*item.first, item.second);
    if (query.device().device_id != key.device().device_id ||
        query.device().device_id != value.device().device_id ||
        query.device().device_id != topk.device().device_id ||
        query.device().device_id != variable_block_sizes.device().device_id ||
        query.device().device_id != workspace.device().device_id ||
        query.device().device_id != output.device().device_id)
        TVM_FFI_THROW(ValueError) << "FastH3 VSA tensors must share one CUDA device";

    require_dtype(query, kDLBfloat, 16, "query");
    require_dtype(key, kDLBfloat, 16, "key");
    require_dtype(value, kDLBfloat, 16, "value");
    require_dtype(output, kDLBfloat, 16, "output");
    require_dtype(topk, kDLInt, 32, "topk");
    require_dtype(variable_block_sizes, kDLInt, 32, "variable_block_sizes");
    require_dtype(workspace, kDLUInt, 8, "workspace");

    const int64_t qkv_shape[] = {kBatch, kHeads, kSequence, kHeadDim};
    const int64_t topk_shape[] = {kBatch, kHeads, kTiles, kTopK};
    const int64_t sizes_shape[] = {kTiles};
    require_shape(query, qkv_shape, "query");
    require_shape(key, qkv_shape, "key");
    require_shape(value, qkv_shape, "value");
    require_shape(output, qkv_shape, "output");
    require_shape(topk, topk_shape, "topk");
    require_shape(variable_block_sizes, sizes_shape, "variable_block_sizes");
    if (workspace.numel() < kWorkspaceBytes)
        TVM_FFI_THROW(ValueError) << "FastH3 VSA workspace is too small";

    auto stream = reinterpret_cast<cudaStream_t>(
        TVMFFIEnvGetStream(kDLCUDA, query.device().device_id));
    auto* workspace_bytes = static_cast<uint8_t*>(workspace.data_ptr()) + workspace.byte_offset();
    auto* q2k_idx = reinterpret_cast<int32_t*>(workspace_bytes);
    auto* q2k_num = q2k_idx + kIndexElements;
    auto* topk_data = reinterpret_cast<const int32_t*>(
        static_cast<const uint8_t*>(topk.data_ptr()) + topk.byte_offset());
    build_block_map<<<kBatch * kHeads * kTiles, 256, 0, stream>>>(
        topk_data, q2k_idx, q2k_num);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << "FastH3 VSA block-map kernel failed: "
                                    << cudaGetErrorString(status);

    BlockSparseVsaArgs args{};
    args.q = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<const uint8_t*>(query.data_ptr()) + query.byte_offset());
    args.k = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<const uint8_t*>(key.data_ptr()) + key.byte_offset());
    args.v = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<const uint8_t*>(value.data_ptr()) + value.byte_offset());
    args.v_t = nullptr;
    args.o = reinterpret_cast<__nv_bfloat16*>(
        static_cast<uint8_t*>(output.data_ptr()) + output.byte_offset());
    args.lse = nullptr;
    args.q2k_idx = q2k_idx;
    args.q2k_num = q2k_num;
    args.variable_block_sizes = reinterpret_cast<const int32_t*>(
        static_cast<const uint8_t*>(variable_block_sizes.data_ptr()) +
        variable_block_sizes.byte_offset());
    args.batch = kBatch;
    args.num_heads = kHeads;
    args.seqlen = kSequence;
    args.head_dim = kHeadDim;
    args.num_blocks = kTiles;
    args.max_kv = kTiles;
    args.sm_scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    status = launch_block_sparse_sm100a(args, stream);
    if (status != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << "FastH3 VSA sm100a attention failed: "
                                    << cudaGetErrorString(status);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run_vsa);
