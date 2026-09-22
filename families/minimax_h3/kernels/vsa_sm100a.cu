/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <utility>

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
constexpr int kSortItems = 1024;
constexpr int64_t kPoolElements = static_cast<int64_t>(kHeads) * kTiles * kHeadDim;
constexpr int64_t kScoreElements = static_cast<int64_t>(kHeads) * kTiles * kTiles;
constexpr int64_t kCountElements = static_cast<int64_t>(kHeads) * kTiles;
constexpr int64_t kWorkspaceBytes =
    3 * kPoolElements * sizeof(float) + 2 * kScoreElements * sizeof(float) +
    kCountElements * sizeof(int32_t);

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

template <typename T>
T* tensor_data(tvm::ffi::TensorView& tensor) {
    return reinterpret_cast<T*>(static_cast<uint8_t*>(tensor.data_ptr()) + tensor.byte_offset());
}

template <typename T>
const T* tensor_data(const tvm::ffi::TensorView& tensor) {
    return reinterpret_cast<const T*>(static_cast<const uint8_t*>(tensor.data_ptr()) +
                                      tensor.byte_offset());
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << "FastH3 VSA " << operation
                                    << " failed: " << cudaGetErrorString(status);
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        TVM_FFI_THROW(RuntimeError) << "FastH3 VSA " << operation
                                    << " failed with cuBLAS status "
                                    << static_cast<int>(status);
}

struct CublasHandle {
    cublasHandle_t value{};
    int device{-1};

    ~CublasHandle() {
        if (value != nullptr)
            cublasDestroy(value);
    }
};

cublasHandle_t get_cublas_handle(int device, cudaStream_t stream) {
    thread_local CublasHandle holder;
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device), "device query");
    if (current_device != device)
        TVM_FFI_THROW(RuntimeError) << "FastH3 VSA current CUDA device " << current_device
                                    << " does not match tensor device " << device;
    if (holder.value == nullptr || holder.device != device) {
        if (holder.value != nullptr) {
            check_cublas(cublasDestroy(holder.value), "cuBLAS handle reset");
            holder.value = nullptr;
        }
        check_cublas(cublasCreate(&holder.value), "cuBLAS handle creation");
        holder.device = device;
    }
    check_cublas(cublasSetStream(holder.value, stream), "cuBLAS stream binding");
    return holder.value;
}

__global__ void mean_pool_tiles(const __nv_bfloat16* input, const int32_t* sizes,
                                float* output) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kPoolElements)
        return;
    const int dimension = static_cast<int>(index % kHeadDim);
    const int64_t head_tile = index / kHeadDim;
    const int tile = static_cast<int>(head_tile % kTiles);
    const int head = static_cast<int>(head_tile / kTiles);
    const int valid = sizes[tile];
    const int64_t base =
        (static_cast<int64_t>(head) * kSequence + tile * kTileSize) * kHeadDim + dimension;
    float sum = 0.0F;
    for (int row = 0; row < valid; ++row)
        sum += __bfloat162float(input[base + static_cast<int64_t>(row) * kHeadDim]);
    output[index] = sum / static_cast<float>(valid > 0 ? valid : 1);
}

__device__ void swap_if_needed(float* values, int32_t* indices, int left, int right,
                               bool ascending) {
    const float left_value = values[left];
    const float right_value = values[right];
    const bool swap = ascending ? left_value > right_value : left_value < right_value;
    if (swap) {
        values[left] = right_value;
        values[right] = left_value;
        const int32_t left_index = indices[left];
        indices[left] = indices[right];
        indices[right] = left_index;
    }
}

__global__ void build_block_map(const float* scores, int32_t* q2k_idx, int32_t* q2k_num) {
    const int row = static_cast<int>(blockIdx.x);
    const int query_tile = row % kTiles;
    int32_t* output = q2k_idx + static_cast<int64_t>(row) * kTiles;
    if (query_tile < kPrefixTiles) {
        if (threadIdx.x == 0)
            q2k_num[row] = kTiles;
        for (int key = static_cast<int>(threadIdx.x); key < kTiles; key += blockDim.x)
            output[key] = key;
        return;
    }

    __shared__ float values[kSortItems];
    __shared__ int32_t indices[kSortItems];
    const float* row_scores = scores + static_cast<int64_t>(row) * kTiles;
    for (int item = static_cast<int>(threadIdx.x); item < kSortItems; item += blockDim.x) {
        if (item < kVideoTiles) {
            values[item] = row_scores[kPrefixTiles + item];
            indices[item] = item;
        } else {
            values[item] = -FLT_MAX;
            indices[item] = item;
        }
    }
    __syncthreads();

    for (int width = 2; width <= kSortItems; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            for (int item = static_cast<int>(threadIdx.x); item < kSortItems;
                 item += blockDim.x) {
                const int partner = item ^ stride;
                if (partner > item)
                    swap_if_needed(values, indices, item, partner, (item & width) == 0);
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0)
        q2k_num[row] = kSparseKeys;
    for (int key = static_cast<int>(threadIdx.x); key < kPrefixTiles; key += blockDim.x)
        output[key] = key;
    for (int rank = static_cast<int>(threadIdx.x); rank < kTopK; rank += blockDim.x)
        output[kPrefixTiles + rank] = kPrefixTiles + indices[kSortItems - 1 - rank];
}

__global__ void softmax_scores(float* scores, const int32_t* sizes) {
    const int row = static_cast<int>(blockIdx.x);
    float* row_scores = scores + static_cast<int64_t>(row) * kTiles;
    __shared__ float reduction[256];

    float local_max = -FLT_MAX;
    for (int column = static_cast<int>(threadIdx.x); column < kTiles; column += blockDim.x) {
        if (sizes[column] > 0)
            local_max = fmaxf(local_max, row_scores[column]);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x],
                                            reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = reduction[0];

    float local_sum = 0.0F;
    for (int column = static_cast<int>(threadIdx.x); column < kTiles; column += blockDim.x) {
        const float probability = sizes[column] > 0 ? expf(row_scores[column] - maximum) : 0.0F;
        row_scores[column] = probability;
        local_sum += probability;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_sum = 1.0F / reduction[0];
    for (int column = static_cast<int>(threadIdx.x); column < kTiles; column += blockDim.x)
        row_scores[column] *= inverse_sum;
}

__global__ void add_gate_compression(__nv_bfloat16* output, const __nv_bfloat16* gate,
                                     const float* compression) {
    constexpr int64_t elements = static_cast<int64_t>(kHeads) * kSequence * kHeadDim;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements)
        return;
    const int dimension = static_cast<int>(index % kHeadDim);
    const int64_t head_row = index / kHeadDim;
    const int row = static_cast<int>(head_row % kSequence);
    const int head = static_cast<int>(head_row / kSequence);
    const int tile = row / kTileSize;
    const int64_t pooled_index =
        (static_cast<int64_t>(head) * kTiles + tile) * kHeadDim + dimension;

    const __nv_bfloat16 compressed_bf16 = __float2bfloat16_rn(compression[pooled_index]);
    const __nv_bfloat16 product_bf16 = __float2bfloat16_rn(
        __bfloat162float(compressed_bf16) * __bfloat162float(gate[index]));
    output[index] = __float2bfloat16_rn(__bfloat162float(output[index]) +
                                        __bfloat162float(product_bf16));
}

void run_vsa(tvm::ffi::TensorView query, tvm::ffi::TensorView key,
             tvm::ffi::TensorView value, tvm::ffi::TensorView gate,
             tvm::ffi::TensorView variable_block_sizes,
             tvm::ffi::TensorView workspace, tvm::ffi::TensorView output) {
    for (const auto& item : {std::pair{&query, "query"}, std::pair{&key, "key"},
                             std::pair{&value, "value"}, std::pair{&gate, "gate"},
                             std::pair{&variable_block_sizes, "variable_block_sizes"},
                             std::pair{&workspace, "workspace"}, std::pair{&output, "output"}})
        require_cuda_contiguous(*item.first, item.second);
    if (query.device().device_id != key.device().device_id ||
        query.device().device_id != value.device().device_id ||
        query.device().device_id != gate.device().device_id ||
        query.device().device_id != variable_block_sizes.device().device_id ||
        query.device().device_id != workspace.device().device_id ||
        query.device().device_id != output.device().device_id)
        TVM_FFI_THROW(ValueError) << "FastH3 VSA tensors must share one CUDA device";

    require_dtype(query, kDLBfloat, 16, "query");
    require_dtype(key, kDLBfloat, 16, "key");
    require_dtype(value, kDLBfloat, 16, "value");
    require_dtype(gate, kDLBfloat, 16, "gate");
    require_dtype(output, kDLBfloat, 16, "output");
    require_dtype(variable_block_sizes, kDLInt, 32, "variable_block_sizes");
    require_dtype(workspace, kDLUInt, 8, "workspace");

    const int64_t qkv_shape[] = {kBatch, kHeads, kSequence, kHeadDim};
    const int64_t sizes_shape[] = {kTiles};
    require_shape(query, qkv_shape, "query");
    require_shape(key, qkv_shape, "key");
    require_shape(value, qkv_shape, "value");
    require_shape(gate, qkv_shape, "gate");
    require_shape(output, qkv_shape, "output");
    require_shape(variable_block_sizes, sizes_shape, "variable_block_sizes");
    if (workspace.numel() < kWorkspaceBytes)
        TVM_FFI_THROW(ValueError) << "FastH3 VSA workspace is too small";

    auto stream = reinterpret_cast<cudaStream_t>(
        TVMFFIEnvGetStream(kDLCUDA, query.device().device_id));
    auto* cursor = tensor_data<uint8_t>(workspace);
    auto* q_pool = reinterpret_cast<float*>(cursor);
    cursor += kPoolElements * sizeof(float);
    auto* k_pool = reinterpret_cast<float*>(cursor);
    cursor += kPoolElements * sizeof(float);
    auto* v_pool = reinterpret_cast<float*>(cursor);
    cursor += kPoolElements * sizeof(float);
    auto* scores = reinterpret_cast<float*>(cursor);
    cursor += kScoreElements * sizeof(float);
    auto* q2k_idx = reinterpret_cast<int32_t*>(cursor);
    cursor += kScoreElements * sizeof(int32_t);
    auto* q2k_num = reinterpret_cast<int32_t*>(cursor);

    const auto* sizes = tensor_data<int32_t>(variable_block_sizes);
    constexpr int kThreads = 256;
    const int pool_blocks = static_cast<int>((kPoolElements + kThreads - 1) / kThreads);
    mean_pool_tiles<<<pool_blocks, kThreads, 0, stream>>>(tensor_data<__nv_bfloat16>(query), sizes,
                                                         q_pool);
    mean_pool_tiles<<<pool_blocks, kThreads, 0, stream>>>(tensor_data<__nv_bfloat16>(key), sizes,
                                                         k_pool);
    mean_pool_tiles<<<pool_blocks, kThreads, 0, stream>>>(tensor_data<__nv_bfloat16>(value), sizes,
                                                         v_pool);
    check_cuda(cudaGetLastError(), "mean pooling");

    cublasHandle_t handle = get_cublas_handle(query.device().device_id, stream);
    const float score_alpha = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    const float beta = 0.0F;
    check_cublas(
        cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, kTiles, kTiles, kHeadDim,
                                  &score_alpha, k_pool, kHeadDim, kPoolElements / kHeads, q_pool,
                                  kHeadDim, kPoolElements / kHeads, &beta, scores, kTiles,
                                  kScoreElements / kHeads, kHeads),
        "routing score GEMM");

    build_block_map<<<kHeads * kTiles, kThreads, 0, stream>>>(scores, q2k_idx, q2k_num);
    check_cuda(cudaGetLastError(), "Top-K block-map selection");

    BlockSparseVsaArgs args{};
    args.q = tensor_data<__nv_bfloat16>(query);
    args.k = tensor_data<__nv_bfloat16>(key);
    args.v = tensor_data<__nv_bfloat16>(value);
    args.v_t = nullptr;
    args.o = tensor_data<__nv_bfloat16>(output);
    args.lse = nullptr;
    args.q2k_idx = q2k_idx;
    args.q2k_num = q2k_num;
    args.variable_block_sizes = sizes;
    args.batch = kBatch;
    args.num_heads = kHeads;
    args.seqlen = kSequence;
    args.head_dim = kHeadDim;
    args.num_blocks = kTiles;
    args.max_kv = kTiles;
    args.sm_scale = score_alpha;
    check_cuda(launch_block_sparse_sm100a(args, stream), "sm100a sparse attention");

    softmax_scores<<<kHeads * kTiles, kThreads, 0, stream>>>(scores, sizes);
    check_cuda(cudaGetLastError(), "compression softmax");

    const float compression_alpha = 1.0F;
    check_cublas(
        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, kTiles, kTiles,
                                  &compression_alpha, v_pool, kHeadDim, kPoolElements / kHeads,
                                  scores, kTiles, kScoreElements / kHeads, &beta, q_pool, kHeadDim,
                                  kPoolElements / kHeads, kHeads),
        "compression GEMM");

    constexpr int64_t output_elements = static_cast<int64_t>(kHeads) * kSequence * kHeadDim;
    const int output_blocks = static_cast<int>((output_elements + kThreads - 1) / kThreads);
    add_gate_compression<<<output_blocks, kThreads, 0, stream>>>(
        tensor_data<__nv_bfloat16>(output), tensor_data<__nv_bfloat16>(gate), q_pool);
    check_cuda(cudaGetLastError(), "gate compression");
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run_vsa);
