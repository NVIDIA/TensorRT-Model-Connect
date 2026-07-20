/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Isolated Wan2.2 TI2V-5B patch-embedding probe.  It deliberately mirrors the
 * official autocast Conv3d contract with the cuDNN backend API:
 *
 *   BF16 NCDHW x BF16 OIDHW -> FP32 accumulate -> BF16 NCDHW
 *
 * PyTorch 2.12/cuDNN 9.20 does not fuse bias into this operation.  Bias is
 * added to the materialized BF16 convolution output and the result is then
 * transposed to contiguous [27280, 3072] rows.  Bias mode 1 reproduces that
 * sequence.  Modes 0 and 2 are qualification controls (no bias and a fused
 * transpose+bias kernel respectively).
 *
 * This DSO has no torch/ATen dependency and is not wired into production.
 */

#include <NvInferRuntime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::wan22::patch_cudnn_probe {

constexpr int32_t kN = 1;
constexpr int32_t kC = 48;
constexpr int32_t kD = 31;
constexpr int32_t kH = 44;
constexpr int32_t kW = 80;
constexpr int32_t kK = 3072;
constexpr int32_t kOutD = 31;
constexpr int32_t kOutH = 22;
constexpr int32_t kOutW = 40;
constexpr int32_t kRows = kOutD * kOutH * kOutW;
constexpr int64_t kOutputElements = static_cast<int64_t>(kRows) * kK;
constexpr size_t kConvOutputBytes = static_cast<size_t>(kOutputElements) * sizeof(__nv_bfloat16);
constexpr int64_t kUidWeight = 119;
constexpr int64_t kUidInput = 120;
constexpr int64_t kUidOutput = 121;

// External linkage is intentional: ctypes consumes this exact POD.
struct CandidateInfo {
    int32_t heuristic_index{-1};
    int32_t plan_status{-1};
    int64_t engine_id{-1};
    uint64_t plan_workspace_bytes{0};
    uint64_t numerical_notes_mask{0};
    uint64_t behavior_notes_mask{0};
};

namespace {

struct ProbeConfig {
    int32_t heuristic_index{0};
    int32_t bias_mode{1};
};

static_assert(sizeof(ProbeConfig) == 2 * sizeof(int32_t));

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

const char* status_name(cudnnStatus_t status) {
    const char* text = cudnnGetErrorString(status);
    return text != nullptr ? text : "unknown cuDNN status";
}

void destroy_backend(cudnnBackendDescriptor_t& descriptor) {
    if (descriptor != nullptr)
        cudnnBackendDestroyDescriptor(descriptor);
    descriptor = nullptr;
}

template <typename T>
bool set_attribute(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name,
                   cudnnBackendAttributeType_t type, int64_t count, const T* value,
                   std::string& error) {
    const cudnnStatus_t status = cudnnBackendSetAttribute(descriptor, name, type, count, value);
    if (status == CUDNN_STATUS_SUCCESS)
        return true;
    error = "cudnnBackendSetAttribute(" + std::to_string(static_cast<int>(name)) +
            ") failed: " + status_name(status);
    return false;
}

bool finalize(cudnnBackendDescriptor_t descriptor, const char* name, std::string& error) {
    const cudnnStatus_t status = cudnnBackendFinalize(descriptor);
    if (status == CUDNN_STATUS_SUCCESS)
        return true;
    error = std::string("cudnnBackendFinalize(") + name + ") failed: " + status_name(status);
    return false;
}

template <typename T>
bool get_scalar(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name,
                cudnnBackendAttributeType_t type, T& value) {
    int64_t count = 0;
    return cudnnBackendGetAttribute(descriptor, name, type, 1, &count, &value) ==
               CUDNN_STATUS_SUCCESS &&
           count == 1;
}

template <typename T>
uint64_t enum_mask(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name,
                   cudnnBackendAttributeType_t type) {
    int64_t count = 0;
    if (cudnnBackendGetAttribute(descriptor, name, type, 0, &count, nullptr) !=
            CUDNN_STATUS_SUCCESS ||
        count <= 0)
        return 0;
    std::vector<T> values(static_cast<size_t>(count));
    if (cudnnBackendGetAttribute(descriptor, name, type, count, &count, values.data()) !=
        CUDNN_STATUS_SUCCESS)
        return 0;
    uint64_t mask = 0;
    for (const T value : values) {
        const auto index = static_cast<int>(value);
        if (index >= 0 && index < 64)
            mask |= uint64_t{1} << index;
    }
    return mask;
}

int64_t engine_id_from_json(const std::string& value) {
    constexpr const char* marker = "\"engineId\":";
    const size_t position = value.find(marker);
    if (position == std::string::npos)
        return -1;
    const char* begin = value.c_str() + position + std::strlen(marker);
    char* end = nullptr;
    const long long parsed = std::strtoll(begin, &end, 10);
    return end != begin ? static_cast<int64_t>(parsed) : -1;
}

__global__ void add_bias_ncdhw(__nv_bfloat16* tensor, const __nv_bfloat16* bias) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kOutputElements)
        return;
    const int64_t channel = index / kRows;
    const float value = __bfloat162float(tensor[index]) + __bfloat162float(bias[channel]);
    tensor[index] = __float2bfloat16_rn(value);
}

template <bool AddBias>
__global__ void transpose_ncdhw_to_rows(const __nv_bfloat16* input, const __nv_bfloat16* bias,
                                        __nv_bfloat16* output) {
    __shared__ __nv_bfloat16 tile[32][33];
    const int row_base = static_cast<int>(blockIdx.x) * 32;
    const int channel_base = static_cast<int>(blockIdx.y) * 32;

#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        const int row = row_base + static_cast<int>(threadIdx.x);
        const int channel = channel_base + static_cast<int>(threadIdx.y) + offset;
        if (row < kRows && channel < kK) {
            __nv_bfloat16 value = input[static_cast<int64_t>(channel) * kRows + row];
            if constexpr (AddBias) {
                value =
                    __float2bfloat16_rn(__bfloat162float(value) + __bfloat162float(bias[channel]));
            }
            tile[threadIdx.y + offset][threadIdx.x] = value;
        }
    }
    __syncthreads();

#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        const int row = row_base + static_cast<int>(threadIdx.y) + offset;
        const int channel = channel_base + static_cast<int>(threadIdx.x);
        if (row < kRows && channel < kK)
            output[static_cast<int64_t>(row) * kK + channel] =
                tile[threadIdx.x][threadIdx.y + offset];
    }
}

class PatchCudnnContext {
  public:
    explicit PatchCudnnContext(ProbeConfig config) : config_(config) {}
    PatchCudnnContext(const PatchCudnnContext&) = delete;
    PatchCudnnContext& operator=(const PatchCudnnContext&) = delete;
    ~PatchCudnnContext() { reset(); }

    int initialize() {
        reset();
        if (config_.heuristic_index < -1 || config_.bias_mode < 0 || config_.bias_mode > 2) {
            error_ = "invalid probe configuration";
            return 1;
        }
        if (!check(cudnnCreate(&handle_), "cudnnCreate"))
            return 1;
        if (!make_tensor(x_desc_, {kN, kC, kD, kH, kW},
                         {kC * kD * kH * kW, kD * kH * kW, kH * kW, kW, 1}, kUidInput, "x") ||
            !make_tensor(w_desc_, {kK, kC, 1, 2, 2}, {kC * 4, 4, 4, 2, 1}, kUidWeight, "w") ||
            !make_tensor(y_desc_, {kN, kK, kOutD, kOutH, kOutW},
                         {kK * kRows, kRows, kOutH * kOutW, kOutW, 1}, kUidOutput, "y"))
            return 1;
        if (!make_convolution() || !make_operation_graph() || !query_candidates())
            return 1;

        if (config_.heuristic_index >= 0) {
            if (config_.heuristic_index >= static_cast<int32_t>(configs_.size())) {
                error_ = "heuristic index is outside returned candidate count";
                return 1;
            }
            if (!make_plan(configs_[static_cast<size_t>(config_.heuristic_index)], plan_,
                           plan_workspace_bytes_))
                return 1;
            plan_json_ = read_plan_json(plan_);
        }
        initialized_ = true;
        return 0;
    }

    int run(const void* latent, const void* weight, const void* bias, void* output, void* workspace,
            size_t workspace_bytes, cudaStream_t stream) {
        if (!initialized_ || plan_ == nullptr || latent == nullptr || weight == nullptr ||
            bias == nullptr || output == nullptr || workspace == nullptr) {
            error_ = "run called with an uninitialized context or null tensor pointer";
            return 1;
        }
        if (workspace_bytes < total_workspace_bytes()) {
            error_ = "workspace is smaller than selected plan plus materialized BF16 output";
            return 1;
        }
        if (!check(cudnnSetStream(handle_, stream), "cudnnSetStream"))
            return 1;

        auto* workspace_bytes_ptr = static_cast<unsigned char*>(workspace);
        void* plan_workspace = plan_workspace_bytes_ == 0 ? nullptr : workspace_bytes_ptr;
        auto* conv_output = reinterpret_cast<__nv_bfloat16*>(workspace_bytes_ptr +
                                                             align_up(plan_workspace_bytes_, 256));
        if (!prepare_variant_pack(latent, weight, conv_output, plan_workspace))
            return 1;
        if (!check(cudnnBackendExecute(handle_, plan_, variant_pack_), "cudnnBackendExecute"))
            return 1;

        constexpr int threads = 256;
        const int blocks = static_cast<int>((kOutputElements + threads - 1) / threads);
        const dim3 transpose_block(32, 8);
        const dim3 transpose_grid((kRows + 31) / 32, (kK + 31) / 32);
        if (config_.bias_mode == 1) {
            add_bias_ncdhw<<<blocks, threads, 0, stream>>>(conv_output,
                                                           static_cast<const __nv_bfloat16*>(bias));
            transpose_ncdhw_to_rows<false><<<transpose_grid, transpose_block, 0, stream>>>(
                conv_output, static_cast<const __nv_bfloat16*>(bias),
                static_cast<__nv_bfloat16*>(output));
        } else if (config_.bias_mode == 2) {
            transpose_ncdhw_to_rows<true><<<transpose_grid, transpose_block, 0, stream>>>(
                conv_output, static_cast<const __nv_bfloat16*>(bias),
                static_cast<__nv_bfloat16*>(output));
        } else {
            transpose_ncdhw_to_rows<false><<<transpose_grid, transpose_block, 0, stream>>>(
                conv_output, static_cast<const __nv_bfloat16*>(bias),
                static_cast<__nv_bfloat16*>(output));
        }
        const cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            error_ = std::string("post-convolution CUDA kernel failed: ") +
                     cudaGetErrorString(cuda_status);
            return 1;
        }
        return 0;
    }

    const std::vector<CandidateInfo>& candidates() const { return candidates_; }
    const std::string& error() const { return error_; }
    const std::string& plan_json() const { return plan_json_; }
    uint64_t cudnn_version() const { return static_cast<uint64_t>(cudnnGetVersion()); }
    size_t total_workspace_bytes() const {
        return align_up(plan_workspace_bytes_, 256) + kConvOutputBytes;
    }
    size_t plan_workspace_bytes() const { return plan_workspace_bytes_; }

  private:
    bool check(cudnnStatus_t status, const char* operation) {
        if (status == CUDNN_STATUS_SUCCESS)
            return true;
        error_ = std::string(operation) + " failed: " + status_name(status);
        return false;
    }

    bool create_backend(cudnnBackendDescriptorType_t type, cudnnBackendDescriptor_t& output,
                        const char* name) {
        const cudnnStatus_t status = cudnnBackendCreateDescriptor(type, &output);
        if (status == CUDNN_STATUS_SUCCESS)
            return true;
        error_ = std::string("create ") + name + " failed: " + status_name(status);
        return false;
    }

    bool make_tensor(cudnnBackendDescriptor_t& output, std::array<int64_t, 5> dimensions,
                     std::array<int64_t, 5> strides, int64_t uid, const char* name) {
        if (!create_backend(CUDNN_BACKEND_TENSOR_DESCRIPTOR, output, name))
            return false;
        const cudnnDataType_t type = CUDNN_DATA_BFLOAT16;
        const int64_t alignment = 32;
        const bool is_virtual = false;
        return set_attribute(output, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type,
                             error_) &&
               set_attribute(output, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 5,
                             dimensions.data(), error_) &&
               set_attribute(output, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 5, strides.data(),
                             error_) &&
               set_attribute(output, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid,
                             error_) &&
               set_attribute(output, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1,
                             &alignment, error_) &&
               set_attribute(output, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1,
                             &is_virtual, error_) &&
               finalize(output, name, error_);
    }

    bool make_convolution() {
        if (!create_backend(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, conv_desc_, "convolution"))
            return false;
        const cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;
        const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
        const int64_t spatial_dims = 3;
        const std::array<int64_t, 3> pads{0, 0, 0};
        const std::array<int64_t, 3> strides{1, 2, 2};
        const std::array<int64_t, 3> dilations{1, 1, 1};
        return set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1,
                             &compute_type, error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                             CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode, error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1,
                             &spatial_dims, error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 3,
                             pads.data(), error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 3,
                             pads.data(), error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 3,
                             strides.data(), error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 3,
                             dilations.data(), error_) &&
               finalize(conv_desc_, "convolution", error_);
    }

    bool make_operation_graph() {
        if (!create_backend(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, operation_,
                            "convolution forward operation"))
            return false;
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        if (!set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x_desc_, error_) ||
            !set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w_desc_, error_) ||
            !set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y_desc_, error_) ||
            !set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv_desc_, error_) ||
            !set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                           CUDNN_TYPE_FLOAT, 1, &alpha, error_) ||
            !set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                           CUDNN_TYPE_FLOAT, 1, &beta, error_) ||
            !finalize(operation_, "convolution forward operation", error_))
            return false;

        if (!create_backend(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, operation_graph_,
                            "operation graph"))
            return false;
        const std::array<cudnnBackendDescriptor_t, 1> operations{operation_};
        return set_attribute(operation_graph_, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                             CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, operations.data(), error_) &&
               // cuDNN 9.20 still requires the deprecated handle attribute to
               // bind the target device before INSTANT heuristics are queried.
               set_attribute(operation_graph_, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE,
                             1, &handle_, error_) &&
               finalize(operation_graph_, "operation graph", error_);
    }

    bool query_candidates() {
        if (!create_backend(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, heuristic_, "heuristic"))
            return false;
        const cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_INSTANT;
        if (!set_attribute(heuristic_, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation_graph_, error_) ||
            !set_attribute(heuristic_, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode,
                           error_) ||
            !finalize(heuristic_, "heuristic", error_))
            return false;

        int64_t count = 0;
        if (!check(cudnnBackendGetAttribute(heuristic_, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                            CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr),
                   "query heuristic result count"))
            return false;
        if (count <= 0 || count > 1024) {
            error_ = "cuDNN returned an invalid heuristic result count";
            return false;
        }
        configs_.resize(static_cast<size_t>(count), nullptr);
        for (auto& config : configs_) {
            if (!create_backend(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, config, "engine config"))
                return false;
        }
        int64_t returned = 0;
        if (!check(cudnnBackendGetAttribute(heuristic_, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                            CUDNN_TYPE_BACKEND_DESCRIPTOR, count, &returned,
                                            configs_.data()),
                   "get heuristic results"))
            return false;
        configs_.resize(static_cast<size_t>(returned));
        candidates_.reserve(configs_.size());
        for (size_t index = 0; index < configs_.size(); ++index) {
            CandidateInfo info{};
            info.heuristic_index = static_cast<int32_t>(index);
            cudnnBackendDescriptor_t engine = nullptr;
            if (get_scalar(configs_[index], CUDNN_ATTR_ENGINECFG_ENGINE,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, engine)) {
                get_scalar(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64,
                           info.engine_id);
                info.numerical_notes_mask = enum_mask<cudnnBackendNumericalNote_t>(
                    engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE);
                info.behavior_notes_mask = enum_mask<cudnnBackendBehaviorNote_t>(
                    engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE);
            }
            // The query-only context qualifies every returned configuration.
            // A selected execution context skips this O(candidate_count) plan
            // construction and builds only its requested plan below.
            if (config_.heuristic_index < 0) {
                cudnnBackendDescriptor_t trial_plan = nullptr;
                size_t trial_workspace = 0;
                const cudnnStatus_t status =
                    make_plan_status(configs_[index], trial_plan, trial_workspace);
                info.plan_status = static_cast<int32_t>(status);
                info.plan_workspace_bytes = static_cast<uint64_t>(trial_workspace);
                if (status == CUDNN_STATUS_SUCCESS)
                    info.engine_id = engine_id_from_json(read_plan_json(trial_plan));
                destroy_backend(trial_plan);
            }
            candidates_.push_back(info);
        }
        return true;
    }

    cudnnStatus_t make_plan_status(cudnnBackendDescriptor_t config,
                                   cudnnBackendDescriptor_t& output, size_t& workspace) {
        workspace = 0;
        cudnnStatus_t status =
            cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &output);
        if (status != CUDNN_STATUS_SUCCESS)
            return status;
        status = cudnnBackendSetAttribute(output, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &config);
        if (status == CUDNN_STATUS_SUCCESS)
            status = cudnnBackendFinalize(output);
        if (status == CUDNN_STATUS_SUCCESS) {
            int64_t bytes = 0;
            int64_t count = 0;
            status = cudnnBackendGetAttribute(output, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                              CUDNN_TYPE_INT64, 1, &count, &bytes);
            if (status == CUDNN_STATUS_SUCCESS && count == 1 && bytes >= 0)
                workspace = static_cast<size_t>(bytes);
        }
        return status;
    }

    bool make_plan(cudnnBackendDescriptor_t config, cudnnBackendDescriptor_t& output,
                   size_t& workspace) {
        const cudnnStatus_t status = make_plan_status(config, output, workspace);
        if (status == CUDNN_STATUS_SUCCESS)
            return true;
        error_ = std::string("execution plan creation failed: ") + status_name(status);
        destroy_backend(output);
        return false;
    }

    std::string read_plan_json(cudnnBackendDescriptor_t plan) {
        int64_t count = 0;
        if (cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                     CUDNN_TYPE_CHAR, 0, &count, nullptr) != CUDNN_STATUS_SUCCESS ||
            count <= 0)
            return {};
        std::string result(static_cast<size_t>(count), '\0');
        if (cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                     CUDNN_TYPE_CHAR, count, &count,
                                     result.data()) != CUDNN_STATUS_SUCCESS)
            return {};
        while (!result.empty() && result.back() == '\0')
            result.pop_back();
        return result;
    }

    bool prepare_variant_pack(const void* latent, const void* weight, void* conv_output,
                              void* plan_workspace) {
        if (variant_pack_ != nullptr && latent == cached_latent_ && weight == cached_weight_ &&
            conv_output == cached_conv_output_ && plan_workspace == cached_plan_workspace_)
            return true;
        destroy_backend(variant_pack_);
        if (!create_backend(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, variant_pack_, "variant pack"))
            return false;
        const std::array<int64_t, 3> uids{kUidInput, kUidWeight, kUidOutput};
        const std::array<void*, 3> pointers{const_cast<void*>(latent), const_cast<void*>(weight),
                                            conv_output};
        if (!set_attribute(variant_pack_, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3,
                           uids.data(), error_) ||
            !set_attribute(variant_pack_, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                           CUDNN_TYPE_VOID_PTR, 3, pointers.data(), error_) ||
            !set_attribute(variant_pack_, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1,
                           &plan_workspace, error_) ||
            !finalize(variant_pack_, "variant pack", error_))
            return false;
        cached_latent_ = latent;
        cached_weight_ = weight;
        cached_conv_output_ = conv_output;
        cached_plan_workspace_ = plan_workspace;
        return true;
    }

    void reset() {
        initialized_ = false;
        cached_latent_ = nullptr;
        cached_weight_ = nullptr;
        cached_conv_output_ = nullptr;
        cached_plan_workspace_ = nullptr;
        destroy_backend(variant_pack_);
        destroy_backend(plan_);
        for (auto& config : configs_)
            destroy_backend(config);
        configs_.clear();
        candidates_.clear();
        destroy_backend(heuristic_);
        destroy_backend(operation_graph_);
        destroy_backend(operation_);
        destroy_backend(conv_desc_);
        destroy_backend(y_desc_);
        destroy_backend(w_desc_);
        destroy_backend(x_desc_);
        if (handle_ != nullptr)
            cudnnDestroy(handle_);
        handle_ = nullptr;
        plan_workspace_bytes_ = 0;
        plan_json_.clear();
    }

    ProbeConfig config_{};
    cudnnHandle_t handle_{nullptr};
    cudnnBackendDescriptor_t x_desc_{nullptr};
    cudnnBackendDescriptor_t w_desc_{nullptr};
    cudnnBackendDescriptor_t y_desc_{nullptr};
    cudnnBackendDescriptor_t conv_desc_{nullptr};
    cudnnBackendDescriptor_t operation_{nullptr};
    cudnnBackendDescriptor_t operation_graph_{nullptr};
    cudnnBackendDescriptor_t heuristic_{nullptr};
    std::vector<cudnnBackendDescriptor_t> configs_;
    std::vector<CandidateInfo> candidates_;
    cudnnBackendDescriptor_t plan_{nullptr};
    cudnnBackendDescriptor_t variant_pack_{nullptr};
    size_t plan_workspace_bytes_{0};
    std::string plan_json_;
    std::string error_;
    const void* cached_latent_{nullptr};
    const void* cached_weight_{nullptr};
    void* cached_conv_output_{nullptr};
    void* cached_plan_workspace_{nullptr};
    bool initialized_{false};
};

int32_t read_int32(const nvinfer1::PluginFieldCollection* fields, const char* name,
                   int32_t fallback) {
    if (fields == nullptr)
        return fallback;
    for (int32_t index = 0; index < fields->nbFields; ++index) {
        const auto& field = fields->fields[index];
        if (field.name != nullptr && std::strcmp(field.name, name) == 0 && field.data != nullptr &&
            field.type == nvinfer1::PluginFieldType::kINT32 && field.length == 1) {
            int32_t value = 0;
            std::memcpy(&value, field.data, sizeof(value));
            return value;
        }
    }
    return fallback;
}

} // namespace

class PatchCudnnProbePlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22PatchCudnnProbe";
    static constexpr const char* kVERSION = "1";

    explicit PatchCudnnProbePlugin(ProbeConfig config) : config_(config) {
        // TensorRT can ask for workspace before its legacy initialize hook.
        // Resolve the target-local cuDNN plan eagerly so that workspace sizing
        // includes both the plan scratch and materialized NCDHW output.
        initialize_context();
    }
    PatchCudnnProbePlugin(const void* data, size_t length) {
        if (data != nullptr && length == sizeof(config_))
            std::memcpy(&config_, data, sizeof(config_));
        initialize_context();
    }

    char const* getPluginType() const noexcept override { return kNAME; }
    char const* getPluginVersion() const noexcept override { return kVERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return initialize_context(); }
    void terminate() noexcept override { context_.reset(); }
    void attachToContext(cudnnContext*, cublasContext*,
                         nvinfer1::IGpuAllocator*) noexcept override {
        if (initialize_context() != 0)
            std::fprintf(stderr, "Wan22PatchCudnnProbe attachToContext failed\n");
    }
    void detachFromContext() noexcept override { context_.reset(); }
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override { return sizeof(config_); }
    void serialize(void* buffer) const noexcept override {
        std::memcpy(buffer, &config_, sizeof(config_));
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    nvinfer1::DataType getOutputDataType(int32_t, nvinfer1::DataType const*,
                                         int32_t) const noexcept override {
        return nvinfer1::DataType::kBF16;
    }
    PatchCudnnProbePlugin* clone() const noexcept override {
        auto* result = new PatchCudnnProbePlugin(config_);
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const*, int32_t,
                                            nvinfer1::IExprBuilder& builder) noexcept override {
        nvinfer1::DimsExprs output{};
        output.nbDims = 2;
        output.d[0] = builder.constant(kRows);
        output.d[1] = builder.constant(kK);
        return output;
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        return input_count == 3 && output_count == 1 && position >= 0 && position < 4 &&
               in_out[position].format == nvinfer1::TensorFormat::kLINEAR &&
               in_out[position].type == nvinfer1::DataType::kBF16;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return context_ != nullptr ? context_->total_workspace_bytes() : kConvOutputBytes;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (context_ == nullptr && initialize_context() != 0)
            return 1;
        if (context_ == nullptr || inputs == nullptr || outputs == nullptr)
            return 1;
        const auto& x = input_desc[0].dims;
        const auto& w = input_desc[1].dims;
        const auto& b = input_desc[2].dims;
        if (x.nbDims != 5 || x.d[0] != kN || x.d[1] != kC || x.d[2] != kD || x.d[3] != kH ||
            x.d[4] != kW || w.nbDims != 5 || w.d[0] != kK || w.d[1] != kC || w.d[2] != 1 ||
            w.d[3] != 2 || w.d[4] != 2 || b.nbDims != 1 || b.d[0] != kK) {
            std::fprintf(stderr, "Wan22PatchCudnnProbe input shape mismatch\n");
            return 1;
        }
        const int status = context_->run(inputs[0], inputs[1], inputs[2], outputs[0], workspace,
                                         getWorkspaceSize(nullptr, 0, nullptr, 0), stream);
        if (status != 0)
            std::fprintf(stderr, "Wan22PatchCudnnProbe enqueue: %s\n", context_->error().c_str());
        return status;
    }

  private:
    int32_t initialize_context() noexcept {
        context_ = std::make_unique<PatchCudnnContext>(config_);
        const int status = context_->initialize();
        if (status != 0) {
            std::fprintf(stderr, "Wan22PatchCudnnProbe initialize: %s\n",
                         context_->error().c_str());
            context_.reset();
        }
        return status;
    }
    ProbeConfig config_{};
    std::unique_ptr<PatchCudnnContext> context_;
    std::string namespace_;
};

class PatchCudnnProbeCreator final : public nvinfer1::IPluginCreator {
  public:
    PatchCudnnProbeCreator() {
        entries_[0] = {"heuristic_index", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        entries_[1] = {"bias_mode", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_ = {2, entries_};
    }
    char const* getPluginName() const noexcept override { return PatchCudnnProbePlugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return PatchCudnnProbePlugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2*
    createPlugin(char const*, nvinfer1::PluginFieldCollection const* fields) noexcept override {
        ProbeConfig config{};
        config.heuristic_index = read_int32(fields, "heuristic_index", 0);
        config.bias_mode = read_int32(fields, "bias_mode", 1);
        return config.heuristic_index >= 0 && config.bias_mode >= 0 && config.bias_mode <= 2
                   ? new PatchCudnnProbePlugin(config)
                   : nullptr;
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return length == sizeof(ProbeConfig) ? new PatchCudnnProbePlugin(data, length) : nullptr;
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginField entries_[2]{};
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::wan22::patch_cudnn_probe

static nvinfer1::PluginRegistrar<trtmc::wan22::patch_cudnn_probe::PatchCudnnProbeCreator>
    plugin_registrar_wan22_patch_cudnn_probe{};

extern "C" {

using Wan22PatchCudnnCandidateInfo = trtmc::wan22::patch_cudnn_probe::CandidateInfo;

int trtmc_wan22_patch_cudnn_query(Wan22PatchCudnnCandidateInfo* output, int32_t capacity) {
    trtmc::wan22::patch_cudnn_probe::ProbeConfig config{-1, 1};
    trtmc::wan22::patch_cudnn_probe::PatchCudnnContext context(config);
    if (context.initialize() != 0) {
        std::fprintf(stderr, "Wan22 patch cuDNN query: %s\n", context.error().c_str());
        return -1;
    }
    const auto& candidates = context.candidates();
    if (output != nullptr && capacity > 0) {
        const int32_t copied = std::min(capacity, static_cast<int32_t>(candidates.size()));
        std::memcpy(output, candidates.data(), static_cast<size_t>(copied) * sizeof(*output));
    }
    return static_cast<int32_t>(candidates.size());
}

void* trtmc_wan22_patch_cudnn_create(int32_t heuristic_index, int32_t bias_mode) {
    trtmc::wan22::patch_cudnn_probe::ProbeConfig config{heuristic_index, bias_mode};
    auto context = std::make_unique<trtmc::wan22::patch_cudnn_probe::PatchCudnnContext>(config);
    if (context->initialize() != 0) {
        std::fprintf(stderr, "Wan22 patch cuDNN create: %s\n", context->error().c_str());
        return nullptr;
    }
    return context.release();
}

void trtmc_wan22_patch_cudnn_destroy(void* opaque) {
    delete static_cast<trtmc::wan22::patch_cudnn_probe::PatchCudnnContext*>(opaque);
}

uint64_t trtmc_wan22_patch_cudnn_workspace_bytes(void* opaque) {
    if (opaque == nullptr)
        return 0;
    return static_cast<trtmc::wan22::patch_cudnn_probe::PatchCudnnContext*>(opaque)
        ->total_workspace_bytes();
}

uint64_t trtmc_wan22_patch_cudnn_plan_workspace_bytes(void* opaque) {
    if (opaque == nullptr)
        return 0;
    return static_cast<trtmc::wan22::patch_cudnn_probe::PatchCudnnContext*>(opaque)
        ->plan_workspace_bytes();
}

uint64_t trtmc_wan22_patch_cudnn_version(void* opaque) {
    if (opaque == nullptr)
        return 0;
    return static_cast<trtmc::wan22::patch_cudnn_probe::PatchCudnnContext*>(opaque)
        ->cudnn_version();
}

int trtmc_wan22_patch_cudnn_plan_json(void* opaque, char* output, int32_t capacity) {
    if (opaque == nullptr)
        return -1;
    const auto& value =
        static_cast<trtmc::wan22::patch_cudnn_probe::PatchCudnnContext*>(opaque)->plan_json();
    const int32_t required = static_cast<int32_t>(value.size() + 1);
    if (output != nullptr && capacity > 0) {
        const int32_t copied = std::min(capacity - 1, static_cast<int32_t>(value.size()));
        if (copied > 0)
            std::memcpy(output, value.data(), static_cast<size_t>(copied));
        output[std::max(copied, 0)] = '\0';
    }
    return required;
}

int trtmc_wan22_patch_cudnn_run(void* opaque, const void* latent, const void* weight,
                                const void* bias, void* output, void* workspace,
                                uint64_t workspace_bytes, void* stream) {
    if (opaque == nullptr)
        return 1;
    return static_cast<trtmc::wan22::patch_cudnn_probe::PatchCudnnContext*>(opaque)->run(
        latent, weight, bias, output, workspace, static_cast<size_t>(workspace_bytes),
        static_cast<cudaStream_t>(stream));
}

void trtmc_wan22_patch_cudnn_probe_force_link() {}

} // extern "C"
