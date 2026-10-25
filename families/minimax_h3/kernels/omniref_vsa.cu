/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vsa_backend.h"

#include <atomic>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <string>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <utility>

#if defined(TRTMC_MINIMAX_H3_VSA_HAS_BLACKWELL)
#define VSA_BLK128 true
#define VSA_BHSD true
#include "block_sparse_launch_sm100a.cuh"
#endif

namespace {

constexpr int kBatch = 1;
constexpr int kHeadDim = 128;
constexpr int kTileSize = 128;
constexpr int kSortItems = 1024;

struct VsaGeometry {
    int heads;
    int tiles;
    int sequence;
    int64_t pool_elements;
    int64_t score_elements;
    int64_t workspace_bytes;
};
constexpr int kWarpSize = 32;
constexpr int kGenericWarpsPerBlock = 8;

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
        TVM_FFI_THROW(RuntimeError)
            << "FastH3 VSA " << operation << " failed: " << cudaGetErrorString(status);
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        TVM_FFI_THROW(RuntimeError) << "FastH3 VSA " << operation << " failed with cuBLAS status "
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

__device__ float warp_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffU, value, offset);
    return value;
}

__global__ void block_sparse_attention_generic(const __nv_bfloat16* query, const __nv_bfloat16* key,
                                               const __nv_bfloat16* value, const int32_t* q2k_idx,
                                               const int32_t* q2k_num, const int32_t* sizes,
                                               __nv_bfloat16* output, float scale, VsaGeometry g) {
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int64_t query_index = static_cast<int64_t>(blockIdx.x) * kGenericWarpsPerBlock + warp;
    if (query_index >= static_cast<int64_t>(g.heads) * g.sequence)
        return;

    const int row = static_cast<int>(query_index % g.sequence);
    const int head = static_cast<int>(query_index / g.sequence);
    const int query_tile = row / kTileSize;
    const int query_offset = row % kTileSize;
    const int64_t output_base = query_index * kHeadDim;
    if (query_offset >= sizes[query_tile]) {
#pragma unroll
        for (int item = 0; item < kHeadDim / kWarpSize; ++item)
            output[output_base + lane + item * kWarpSize] = __float2bfloat16_rn(0.0F);
        return;
    }

    float query_values[kHeadDim / kWarpSize];
    float accumulator[kHeadDim / kWarpSize] = {};
#pragma unroll
    for (int item = 0; item < kHeadDim / kWarpSize; ++item)
        query_values[item] = __bfloat162float(query[output_base + lane + item * kWarpSize]);

    float maximum = -FLT_MAX;
    float denominator = 0.0F;
    const int map_row = head * g.tiles + query_tile;
    const int selected_tiles = q2k_num[map_row];
    for (int selected = 0; selected < selected_tiles; ++selected) {
        int key_tile = lane == 0 ? q2k_idx[static_cast<int64_t>(map_row) * g.tiles + selected] : 0;
        key_tile = __shfl_sync(0xffffffffU, key_tile, 0);
        int valid = lane == 0 ? sizes[key_tile] : 0;
        valid = __shfl_sync(0xffffffffU, valid, 0);
        for (int key_offset = 0; key_offset < valid; ++key_offset) {
            const int key_row = key_tile * kTileSize + key_offset;
            const int64_t key_base = (static_cast<int64_t>(head) * g.sequence + key_row) * kHeadDim;
            float partial = 0.0F;
#pragma unroll
            for (int item = 0; item < kHeadDim / kWarpSize; ++item) {
                const int dimension = lane + item * kWarpSize;
                partial += query_values[item] * __bfloat162float(key[key_base + dimension]);
            }
            const float score = __shfl_sync(0xffffffffU, warp_sum(partial), 0) * scale;
            const float next_maximum = fmaxf(maximum, score);
            const float old_weight = __expf(maximum - next_maximum);
            const float new_weight = __expf(score - next_maximum);
            denominator = denominator * old_weight + new_weight;
#pragma unroll
            for (int item = 0; item < kHeadDim / kWarpSize; ++item) {
                const int dimension = lane + item * kWarpSize;
                accumulator[item] = accumulator[item] * old_weight +
                                    new_weight * __bfloat162float(value[key_base + dimension]);
            }
            maximum = next_maximum;
        }
    }

    const float inverse_denominator = denominator > 0.0F ? 1.0F / denominator : 0.0F;
#pragma unroll
    for (int item = 0; item < kHeadDim / kWarpSize; ++item)
        output[output_base + lane + item * kWarpSize] =
            __float2bfloat16_rn(accumulator[item] * inverse_denominator);
}

trtmc::minimax_h3::VsaBackend selected_backend(int device) {
    static std::mutex mutex;
    static int cached_device = -1;
    static trtmc::minimax_h3::VsaBackend cached_backend = trtmc::minimax_h3::VsaBackend::kGeneric;
    const std::lock_guard<std::mutex> lock(mutex);
    if (cached_device == device)
        return cached_backend;

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "device capability query");
    const char* value = std::getenv("TRTMC_MINIMAX_H3_VSA_BACKEND");
    const std::string request = value != nullptr ? value : "auto";
#if defined(TRTMC_MINIMAX_H3_VSA_HAS_BLACKWELL)
    constexpr bool kHasBlackwellKernels = true;
#else
    constexpr bool kHasBlackwellKernels = false;
#endif
    try {
        cached_backend = trtmc::minimax_h3::select_vsa_backend(
            request, properties.major, properties.minor, kHasBlackwellKernels);
    } catch (const std::exception& error) {
        TVM_FFI_THROW(RuntimeError) << error.what();
    }
    cached_device = device;
    std::cerr << "[MiniMax-H3 VSA] backend=" << trtmc::minimax_h3::vsa_backend_name(cached_backend)
              << " device=" << properties.name << " sm_" << properties.major << properties.minor
              << '\n';
    return cached_backend;
}

__global__ void mean_pool_tiles(const __nv_bfloat16* input, const int32_t* sizes, float* output,
                                VsaGeometry g) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= g.pool_elements)
        return;
    const int dimension = static_cast<int>(index % kHeadDim);
    const int64_t head_tile = index / kHeadDim;
    const int tile = static_cast<int>(head_tile % g.tiles);
    const int head = static_cast<int>(head_tile / g.tiles);
    const int valid = sizes[tile];
    const int64_t base =
        (static_cast<int64_t>(head) * g.sequence + tile * kTileSize) * kHeadDim + dimension;
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

__global__ void build_block_map(const float* scores, const int32_t* sizes, const int32_t* regions,
                                int region_count, int32_t* q2k_idx, int32_t* q2k_num,
                                VsaGeometry g) {
    const int row = static_cast<int>(blockIdx.x);
    const int query_tile = row % g.tiles;
    const int prefix = regions[0];
    assert(prefix >= 0 && prefix < g.tiles);
    int32_t* output = q2k_idx + static_cast<int64_t>(row) * g.tiles;
    if (sizes[query_tile] == 0) {
        if (threadIdx.x == 0) {
            output[0] = 0;
            q2k_num[row] = 1;
        }
        return;
    }
    if (query_tile < prefix) {
        if (threadIdx.x == 0)
            q2k_num[row] = g.tiles;
        for (int key = static_cast<int>(threadIdx.x); key < g.tiles; key += blockDim.x)
            output[key] = key;
        return;
    }
    for (int key = static_cast<int>(threadIdx.x); key < prefix; key += blockDim.x)
        output[key] = key;
    __shared__ float values[kSortItems];
    __shared__ int32_t indices[kSortItems];
    const float* row_scores = scores + static_cast<int64_t>(row) * g.tiles;
    int selected = prefix;
    int previous_end = prefix;
    for (int region = 0; region < region_count; ++region) {
        const int begin = regions[region * 3];
        const int end = regions[region * 3 + 1];
        const int keep = regions[region * 3 + 2];
        const int length = end - begin;
        assert(begin == previous_end && end <= g.tiles && length > 0 && length <= kSortItems &&
               keep > 0 && keep <= length);
        assert(selected + keep <= g.tiles);
        for (int item = static_cast<int>(threadIdx.x); item < kSortItems; item += blockDim.x) {
            values[item] = item < length ? row_scores[begin + item] : -FLT_MAX;
            indices[item] = begin + item;
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
        for (int rank = static_cast<int>(threadIdx.x); rank < keep; rank += blockDim.x)
            output[selected + rank] = indices[kSortItems - 1 - rank];
        selected += keep;
        previous_end = end;
        __syncthreads();
    }
    if (threadIdx.x == 0)
        q2k_num[row] = selected;
}

__global__ void softmax_scores(float* scores, const int32_t* sizes, VsaGeometry g) {
    const int row = static_cast<int>(blockIdx.x);
    float* row_scores = scores + static_cast<int64_t>(row) * g.tiles;
    __shared__ float reduction[256];

    float local_max = -FLT_MAX;
    for (int column = static_cast<int>(threadIdx.x); column < g.tiles; column += blockDim.x) {
        if (sizes[column] > 0)
            local_max = fmaxf(local_max, row_scores[column]);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = reduction[0];

    float local_sum = 0.0F;
    for (int column = static_cast<int>(threadIdx.x); column < g.tiles; column += blockDim.x) {
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
    for (int column = static_cast<int>(threadIdx.x); column < g.tiles; column += blockDim.x)
        row_scores[column] *= inverse_sum;
}

__global__ void add_gate_compression(__nv_bfloat16* output, const __nv_bfloat16* gate,
                                     const float* compression, const int32_t* sizes,
                                     VsaGeometry g) {
    const int64_t elements = static_cast<int64_t>(g.heads) * g.sequence * kHeadDim;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements)
        return;
    const int dimension = static_cast<int>(index % kHeadDim);
    const int64_t head_row = index / kHeadDim;
    const int row = static_cast<int>(head_row % g.sequence);
    const int head = static_cast<int>(head_row / g.sequence);
    const int tile = row / kTileSize;
    if (row % kTileSize >= sizes[tile]) {
        output[index] = __float2bfloat16_rn(0.0F);
        return;
    }
    const int64_t pooled_index =
        (static_cast<int64_t>(head) * g.tiles + tile) * kHeadDim + dimension;
    const __nv_bfloat16 compressed = __float2bfloat16_rn(compression[pooled_index]);
    // Match the released BF16 addcmul: round compression first, then perform
    // the multiply/add together before the single final BF16 rounding.
    output[index] =
        __float2bfloat16_rn(__fmaf_rn(__bfloat162float(compressed), __bfloat162float(gate[index]),
                                      __bfloat162float(output[index])));
}

void run_vsa(tvm::ffi::TensorView query, tvm::ffi::TensorView key, tvm::ffi::TensorView value,
             tvm::ffi::TensorView gate, tvm::ffi::TensorView variable_block_sizes,
             tvm::ffi::TensorView regions, tvm::ffi::TensorView workspace,
             tvm::ffi::TensorView output) {
    for (const auto& item :
         {std::pair{&query, "query"}, std::pair{&key, "key"}, std::pair{&value, "value"},
          std::pair{&gate, "gate"}, std::pair{&variable_block_sizes, "block_sizes"},
          std::pair{&regions, "regions"}, std::pair{&workspace, "workspace"},
          std::pair{&output, "output"}}) {
        require_cuda_contiguous(*item.first, item.second);
        if (item.first->device().device_id != query.device().device_id)
            TVM_FFI_THROW(ValueError) << "OmniRef VSA tensors must share one CUDA device";
    }
    require_dtype(query, kDLBfloat, 16, "query");
    require_dtype(key, kDLBfloat, 16, "key");
    require_dtype(value, kDLBfloat, 16, "value");
    require_dtype(gate, kDLBfloat, 16, "gate");
    require_dtype(output, kDLBfloat, 16, "output");
    require_dtype(variable_block_sizes, kDLInt, 32, "block_sizes");
    require_dtype(regions, kDLInt, 32, "regions");
    require_dtype(workspace, kDLUInt, 8, "workspace");
    if (query.ndim() != 4 || query.shape()[0] != 1 || query.shape()[1] <= 0 ||
        query.shape()[1] > 256 || query.shape()[3] != kHeadDim ||
        variable_block_sizes.ndim() != 1 || variable_block_sizes.shape()[0] <= 0 ||
        variable_block_sizes.shape()[0] > 8192 || variable_block_sizes.shape()[0] % 2 != 0 ||
        query.shape()[2] != variable_block_sizes.shape()[0] * kTileSize || regions.ndim() != 2 ||
        regions.shape()[0] <= 0 || regions.shape()[0] > 64 || regions.shape()[1] != 3)
        TVM_FFI_THROW(ValueError) << "Invalid OmniRef VSA head/tile/region geometry";
    VsaGeometry g{};
    g.heads = static_cast<int>(query.shape()[1]);
    g.tiles = static_cast<int>(variable_block_sizes.shape()[0]);
    g.sequence = g.tiles * kTileSize;
    g.pool_elements = static_cast<int64_t>(g.heads) * g.tiles * kHeadDim;
    g.score_elements = static_cast<int64_t>(g.heads) * g.tiles * g.tiles;
    g.workspace_bytes = 3 * g.pool_elements * sizeof(float) + 2 * g.score_elements * sizeof(float) +
                        static_cast<int64_t>(g.heads) * g.tiles * sizeof(int32_t);
    const int64_t qkv_shape[] = {kBatch, g.heads, g.sequence, kHeadDim};
    require_shape(query, qkv_shape, "query");
    require_shape(key, qkv_shape, "key");
    require_shape(value, qkv_shape, "value");
    require_shape(gate, qkv_shape, "gate");
    require_shape(output, qkv_shape, "output");
    if (workspace.numel() < g.workspace_bytes)
        TVM_FFI_THROW(ValueError) << "OmniRef VSA workspace is too small";
    auto stream =
        reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, query.device().device_id));
    auto* cursor = tensor_data<uint8_t>(workspace);
    auto* q_pool = reinterpret_cast<float*>(cursor);
    cursor += g.pool_elements * sizeof(float);
    auto* k_pool = reinterpret_cast<float*>(cursor);
    cursor += g.pool_elements * sizeof(float);
    auto* v_pool = reinterpret_cast<float*>(cursor);
    cursor += g.pool_elements * sizeof(float);
    auto* scores = reinterpret_cast<float*>(cursor);
    cursor += g.score_elements * sizeof(float);
    auto* q2k_idx = reinterpret_cast<int32_t*>(cursor);
    cursor += g.score_elements * sizeof(int32_t);
    auto* q2k_num = reinterpret_cast<int32_t*>(cursor);
    const auto* sizes = tensor_data<int32_t>(variable_block_sizes);
    const char* trace_option = std::getenv("TRTMC_MINIMAX_H3_VSA_TRACE");
    const bool trace = trace_option && std::string(trace_option) == "1";
    static std::atomic<uint64_t> next_trace_id{0};
    const uint64_t trace_id = trace ? next_trace_id.fetch_add(1, std::memory_order_relaxed) : 0;
    const auto trace_start = trace ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    const auto trace_stage = [&](const char* stage) {
        if (!trace)
            return;
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - trace_start)
                                 .count();
        std::cerr << "[MiniMax-H3 VSA trace] call=" << trace_id << " stage=" << stage
                  << " heads=" << g.heads << " tiles=" << g.tiles << " rows=" << g.sequence
                  << " regions=" << regions.shape()[0] << " host_elapsed_us=" << elapsed
                  << std::endl;
    };
    constexpr int kThreads = 256;
    const int pool_blocks = static_cast<int>((g.pool_elements + kThreads - 1) / kThreads);
    trace_stage("pooling_enter");
    mean_pool_tiles<<<pool_blocks, kThreads, 0, stream>>>(tensor_data<__nv_bfloat16>(query), sizes,
                                                          q_pool, g);
    mean_pool_tiles<<<pool_blocks, kThreads, 0, stream>>>(tensor_data<__nv_bfloat16>(key), sizes,
                                                          k_pool, g);
    mean_pool_tiles<<<pool_blocks, kThreads, 0, stream>>>(tensor_data<__nv_bfloat16>(value), sizes,
                                                          v_pool, g);
    check_cuda(cudaGetLastError(), "OmniRef mean pooling");
    trace_stage("pooling_enqueued");
    trace_stage("routing_handle_enter");
    cublasHandle_t handle = get_cublas_handle(query.device().device_id, stream);
    check_cublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "IEEE routing math mode");
    const float alpha = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    const float beta = 0.0F;
    trace_stage("routing_gemm_enter");
    check_cublas(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, g.tiles, g.tiles,
                                           kHeadDim, &alpha, k_pool, kHeadDim,
                                           g.pool_elements / g.heads, q_pool, kHeadDim,
                                           g.pool_elements / g.heads, &beta, scores, g.tiles,
                                           g.score_elements / g.heads, g.heads),
                 "OmniRef routing GEMM");
    trace_stage("routing_gemm_enqueued");
    trace_stage("routing_map_enter");
    build_block_map<<<g.heads * g.tiles, kThreads, 0, stream>>>(
        scores, sizes, tensor_data<int32_t>(regions), static_cast<int>(regions.shape()[0]), q2k_idx,
        q2k_num, g);
    check_cuda(cudaGetLastError(), "OmniRef per-region Top-K");
    trace_stage("routing_map_enqueued");
    const auto backend = selected_backend(query.device().device_id);
    trace_stage("fine_attention_enter");
    if (backend == trtmc::minimax_h3::VsaBackend::kBlackwell) {
#if defined(TRTMC_MINIMAX_H3_VSA_HAS_BLACKWELL)
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
        args.num_heads = g.heads;
        args.seqlen = g.sequence;
        args.head_dim = kHeadDim;
        args.num_blocks = g.tiles;
        args.max_kv = g.tiles;
        args.sm_scale = alpha;
        check_cuda(launch_block_sparse_sm100a(args, stream), "OmniRef Blackwell attention");
#else
        TVM_FFI_THROW(RuntimeError) << "OmniRef VSA selected an unavailable Blackwell kernel";
#endif
    } else {
        constexpr int kGenericThreads = kGenericWarpsPerBlock * kWarpSize;
        const int64_t queries = static_cast<int64_t>(g.heads) * g.sequence;
        const int blocks =
            static_cast<int>((queries + kGenericWarpsPerBlock - 1) / kGenericWarpsPerBlock);
        block_sparse_attention_generic<<<blocks, kGenericThreads, 0, stream>>>(
            tensor_data<__nv_bfloat16>(query), tensor_data<__nv_bfloat16>(key),
            tensor_data<__nv_bfloat16>(value), q2k_idx, q2k_num, sizes,
            tensor_data<__nv_bfloat16>(output), alpha, g);
        check_cuda(cudaGetLastError(), "OmniRef generic attention");
    }
    trace_stage("fine_attention_enqueued");
    trace_stage("compression_softmax_enter");
    softmax_scores<<<g.heads * g.tiles, kThreads, 0, stream>>>(scores, sizes, g);
    check_cuda(cudaGetLastError(), "OmniRef compression softmax");
    trace_stage("compression_softmax_enqueued");
    const float compression_alpha = 1.0F;
    trace_stage("compression_gemm_enter");
    check_cublas(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, g.tiles,
                                           g.tiles, &compression_alpha, v_pool, kHeadDim,
                                           g.pool_elements / g.heads, scores, g.tiles,
                                           g.score_elements / g.heads, &beta, q_pool, kHeadDim,
                                           g.pool_elements / g.heads, g.heads),
                 "OmniRef compression GEMM");
    trace_stage("compression_gemm_enqueued");
    const int64_t elements = static_cast<int64_t>(g.heads) * g.sequence * kHeadDim;
    const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
    trace_stage("gated_add_enter");
    add_gate_compression<<<blocks, kThreads, 0, stream>>>(
        tensor_data<__nv_bfloat16>(output), tensor_data<__nv_bfloat16>(gate), q_pool, sizes, g);
    check_cuda(cudaGetLastError(), "OmniRef gated addcmul");
    trace_stage("return_enqueued");
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run_vsa);
