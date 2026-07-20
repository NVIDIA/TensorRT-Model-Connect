/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Fixed-shape FP32 cuBLASLt tactic probe for the Wan2.2 final projection.
 * This qualification-only DSO has no PyTorch, ATen, or TensorRT dependency.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::wan22::final_projection_probe {

constexpr int32_t kM = 27'280;
constexpr int32_t kN = 192;
constexpr int32_t kK = 3'072;
constexpr int32_t kMaxHeuristics = 128;

struct AlgoInfo {
    int32_t heuristic_index{-1};
    int32_t algorithm_id{-1};
    int32_t tile_id{-1};
    int32_t stages_id{-1};
    int32_t split_k{-1};
    int32_t reduction_scheme{-1};
    int32_t cta_swizzle{-1};
    int32_t custom_option{-1};
    int32_t inner_shape_id{-1};
    int32_t cluster_shape_id{-1};
    uint64_t workspace_bytes{0};
    float waves_count{0.0F};
};

namespace {

template <typename T>
bool algorithm_attribute(const cublasLtMatmulAlgo_t& algorithm,
                         cublasLtMatmulAlgoConfigAttributes_t attribute, T* value) {
    size_t written = 0;
    return value != nullptr &&
           cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attribute, value, sizeof(T),
                                                &written) == CUBLAS_STATUS_SUCCESS &&
           written == sizeof(T);
}

AlgoInfo describe_algorithm(const cublasLtMatmulHeuristicResult_t& result, int32_t index) {
    AlgoInfo info{};
    info.heuristic_index = index;
    info.workspace_bytes = static_cast<uint64_t>(result.workspaceSize);
    info.waves_count = result.wavesCount;
    algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_ID, &info.algorithm_id);
    uint32_t unsigned_value = 0;
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &unsigned_value))
        info.tile_id = static_cast<int32_t>(unsigned_value);
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &unsigned_value))
        info.stages_id = static_cast<int32_t>(unsigned_value);
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &unsigned_value))
        info.reduction_scheme = static_cast<int32_t>(unsigned_value);
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &unsigned_value))
        info.cta_swizzle = static_cast<int32_t>(unsigned_value);
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &unsigned_value))
        info.custom_option = static_cast<int32_t>(unsigned_value);
    int32_t signed_value = 0;
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &signed_value))
        info.split_k = signed_value;
    uint16_t short_value = 0;
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &short_value))
        info.inner_shape_id = static_cast<int32_t>(short_value);
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &short_value))
        info.cluster_shape_id = static_cast<int32_t>(short_value);
    return info;
}

} // namespace

class Context {
  public:
    Context(int32_t heuristic_index, int32_t workspace_mib)
        : heuristic_index_(heuristic_index), workspace_mib_(workspace_mib) {}
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context() { reset(); }

    int initialize() {
        reset();
        if (heuristic_index_ < 0 || workspace_mib_ < 0) {
            error_ = "invalid heuristic index or workspace";
            return 1;
        }
        if (!check(cublasLtCreate(&handle_), "cublasLtCreate"))
            return 1;
        cublasOperation_t transpose_weight = CUBLAS_OP_T;
        cublasOperation_t transpose_input = CUBLAS_OP_N;
        if (!check(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                   "create strict FP32 matmul") ||
            !check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                                                  &transpose_weight, sizeof(transpose_weight)),
                   "set TRANSA") ||
            !check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB,
                                                  &transpose_input, sizeof(transpose_input)),
                   "set TRANSB"))
            return 1;
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        if (!check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epilogue, sizeof(epilogue)),
                   "set bias epilogue"))
            return 1;
        if (!create_layout(&weight_layout_, kK, kN, kK, "weight") ||
            !create_layout(&input_layout_, kK, kM, kK, "input") ||
            !create_layout(&output_layout_, kN, kM, kN, "output"))
            return 1;
        if (!check(cublasLtMatmulPreferenceCreate(&preference_), "create preference"))
            return 1;
        const size_t workspace_limit = workspace_limit_bytes();
        if (!check(cublasLtMatmulPreferenceSetAttribute(preference_,
                                                        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &workspace_limit, sizeof(workspace_limit)),
                   "set workspace limit"))
            return 1;
        candidates_.resize(kMaxHeuristics);
        int32_t returned = 0;
        if (!check(cublasLtMatmulAlgoGetHeuristic(
                       handle_, operation_, weight_layout_, input_layout_, output_layout_,
                       output_layout_, preference_, kMaxHeuristics, candidates_.data(), &returned),
                   "query heuristics"))
            return 1;
        candidates_.resize(static_cast<size_t>(std::max(returned, 0)));
        if (heuristic_index_ >= returned) {
            error_ = "heuristic index is outside returned candidate count";
            return 1;
        }
        const auto& selected = candidates_[static_cast<size_t>(heuristic_index_)];
        if (selected.state != CUBLAS_STATUS_SUCCESS) {
            error_ = "selected heuristic is not usable";
            return 1;
        }
        info_ = describe_algorithm(selected, heuristic_index_);
        initialized_ = true;
        return 0;
    }

    int run(const float* input, const float* weight, const float* bias, float* output,
            void* workspace, size_t workspace_bytes, cudaStream_t stream) {
        if (!initialized_ || input == nullptr || weight == nullptr || bias == nullptr ||
            output == nullptr) {
            error_ = "run called with an uninitialized context or null tensor";
            return 1;
        }
        if (workspace_bytes < static_cast<size_t>(info_.workspace_bytes)) {
            error_ = "workspace is smaller than selected tactic requirement";
            return 1;
        }
        if (!check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                  &bias, sizeof(bias)),
                   "set bias pointer"))
            return 1;
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const auto& selected = candidates_[static_cast<size_t>(heuristic_index_)];
        return check(cublasLtMatmul(handle_, operation_, &alpha, weight, weight_layout_, input,
                                    input_layout_, &beta, output, output_layout_, output,
                                    output_layout_, &selected.algo, workspace, workspace_bytes,
                                    stream),
                     "cublasLtMatmul")
                   ? 0
                   : 1;
    }

    const AlgoInfo& info() const { return info_; }
    const std::string& error() const { return error_; }
    size_t workspace_limit_bytes() const {
        return static_cast<size_t>(workspace_mib_) * 1024U * 1024U;
    }
    std::vector<AlgoInfo> candidate_info() const {
        std::vector<AlgoInfo> result;
        result.reserve(candidates_.size());
        for (size_t index = 0; index < candidates_.size(); ++index) {
            if (candidates_[index].state == CUBLAS_STATUS_SUCCESS)
                result.push_back(
                    describe_algorithm(candidates_[index], static_cast<int32_t>(index)));
        }
        return result;
    }

  private:
    bool check(cublasStatus_t status, const char* operation) {
        if (status == CUBLAS_STATUS_SUCCESS)
            return true;
        error_ = std::string(operation) + " failed with cuBLAS status " +
                 std::to_string(static_cast<int32_t>(status));
        return false;
    }
    bool create_layout(cublasLtMatrixLayout_t* layout, uint64_t rows, uint64_t columns,
                       int64_t leading_dimension, const char* name) {
        return check(
            cublasLtMatrixLayoutCreate(layout, CUDA_R_32F, rows, columns, leading_dimension),
            (std::string("create ") + name + " layout").c_str());
    }
    void reset() {
        initialized_ = false;
        candidates_.clear();
        if (preference_ != nullptr)
            cublasLtMatmulPreferenceDestroy(preference_);
        if (output_layout_ != nullptr)
            cublasLtMatrixLayoutDestroy(output_layout_);
        if (input_layout_ != nullptr)
            cublasLtMatrixLayoutDestroy(input_layout_);
        if (weight_layout_ != nullptr)
            cublasLtMatrixLayoutDestroy(weight_layout_);
        if (operation_ != nullptr)
            cublasLtMatmulDescDestroy(operation_);
        if (handle_ != nullptr)
            cublasLtDestroy(handle_);
        preference_ = nullptr;
        output_layout_ = nullptr;
        input_layout_ = nullptr;
        weight_layout_ = nullptr;
        operation_ = nullptr;
        handle_ = nullptr;
        info_ = {};
    }

    int32_t heuristic_index_{0};
    int32_t workspace_mib_{32};
    cublasLtHandle_t handle_{nullptr};
    cublasLtMatmulDesc_t operation_{nullptr};
    cublasLtMatrixLayout_t weight_layout_{nullptr};
    cublasLtMatrixLayout_t input_layout_{nullptr};
    cublasLtMatrixLayout_t output_layout_{nullptr};
    cublasLtMatmulPreference_t preference_{nullptr};
    std::vector<cublasLtMatmulHeuristicResult_t> candidates_;
    AlgoInfo info_{};
    std::string error_;
    bool initialized_{false};
};

} // namespace trtmc::wan22::final_projection_probe

extern "C" {

using Wan22FinalProjectionAlgoInfo = trtmc::wan22::final_projection_probe::AlgoInfo;

int trtmc_wan22_final_projection_probe_query(int32_t workspace_mib,
                                             Wan22FinalProjectionAlgoInfo* output,
                                             int32_t capacity) {
    trtmc::wan22::final_projection_probe::Context context(0, workspace_mib);
    if (context.initialize() != 0) {
        std::fprintf(stderr, "Wan22 final projection query: %s\n", context.error().c_str());
        return -1;
    }
    const auto candidates = context.candidate_info();
    if (output != nullptr && capacity > 0) {
        const int32_t copied = std::min(capacity, static_cast<int32_t>(candidates.size()));
        std::memcpy(output, candidates.data(), static_cast<size_t>(copied) * sizeof(*output));
    }
    return static_cast<int32_t>(candidates.size());
}

void* trtmc_wan22_final_projection_probe_create(int32_t heuristic_index, int32_t workspace_mib) {
    auto context = std::make_unique<trtmc::wan22::final_projection_probe::Context>(heuristic_index,
                                                                                   workspace_mib);
    if (context->initialize() != 0) {
        std::fprintf(stderr, "Wan22 final projection create: %s\n", context->error().c_str());
        return nullptr;
    }
    return context.release();
}

void trtmc_wan22_final_projection_probe_destroy(void* opaque) {
    delete static_cast<trtmc::wan22::final_projection_probe::Context*>(opaque);
}

uint64_t trtmc_wan22_final_projection_probe_workspace_bytes(void* opaque) {
    if (opaque == nullptr)
        return 0;
    return static_cast<trtmc::wan22::final_projection_probe::Context*>(opaque)
        ->info()
        .workspace_bytes;
}

int trtmc_wan22_final_projection_probe_run(void* opaque, const float* input, const float* weight,
                                           const float* bias, float* output, void* workspace,
                                           uint64_t workspace_bytes, void* stream) {
    if (opaque == nullptr)
        return 1;
    return static_cast<trtmc::wan22::final_projection_probe::Context*>(opaque)->run(
        input, weight, bias, output, workspace, static_cast<size_t>(workspace_bytes),
        static_cast<cudaStream_t>(stream));
}

} // extern "C"
