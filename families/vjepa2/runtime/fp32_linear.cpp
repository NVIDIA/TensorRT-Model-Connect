/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vjepa2/runtime/fp32_kernels.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cublasLt.h>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace trtmc::vjepa2 {
namespace {
std::filesystem::path math_library_path() {
    Dl_info information{};
    if (!dladdr(reinterpret_cast<const void*>(&create_linear_context), &information) ||
        !information.dli_fname)
        throw std::runtime_error("VJEPA2 could not locate its CUDA model library");
    const auto directory = std::filesystem::canonical(information.dli_fname).parent_path();
    if (directory.filename() == "bin" &&
        directory.parent_path().filename() == "tensorrt_model_connect") {
        const auto library = directory / "../../nvidia/cu13/lib/libcublasLt.so.13";
        if (std::filesystem::is_regular_file(library))
            return std::filesystem::canonical(library);
        return "libcublasLt.so.13";
    }
    const std::filesystem::path embedded(TRTMC_VJEPA2_CUBLASLT_BUILD_LIBRARY);
    return std::filesystem::is_regular_file(embedded) ? embedded
                                                      : std::filesystem::path("libcublasLt.so.13");
}

class MathApi {
  public:
    MathApi() {
        const auto path = math_library_path();
        library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library_)
            throw std::runtime_error(std::string("VJEPA2 could not load cuBLASLt: ") + dlerror());
        try {
            create = load<decltype(create)>("cublasLtCreate");
            destroy = load<decltype(destroy)>("cublasLtDestroy");
            create_operation = load<decltype(create_operation)>("cublasLtMatmulDescCreate");
            destroy_operation = load<decltype(destroy_operation)>("cublasLtMatmulDescDestroy");
            set_operation = load<decltype(set_operation)>("cublasLtMatmulDescSetAttribute");
            create_layout = load<decltype(create_layout)>("cublasLtMatrixLayoutCreate");
            destroy_layout = load<decltype(destroy_layout)>("cublasLtMatrixLayoutDestroy");
            create_preference = load<decltype(create_preference)>("cublasLtMatmulPreferenceCreate");
            destroy_preference =
                load<decltype(destroy_preference)>("cublasLtMatmulPreferenceDestroy");
            set_preference = load<decltype(set_preference)>("cublasLtMatmulPreferenceSetAttribute");
            heuristic = load<decltype(heuristic)>("cublasLtMatmulAlgoGetHeuristic");
            matmul = load<decltype(matmul)>("cublasLtMatmul");
        } catch (...) {
            dlclose(library_);
            throw;
        }
    }
    ~MathApi() { dlclose(library_); }
    MathApi(const MathApi&) = delete;
    MathApi& operator=(const MathApi&) = delete;

    decltype(&cublasLtCreate) create{};
    decltype(&cublasLtDestroy) destroy{};
    decltype(&cublasLtMatmulDescCreate) create_operation{};
    decltype(&cublasLtMatmulDescDestroy) destroy_operation{};
    decltype(&cublasLtMatmulDescSetAttribute) set_operation{};
    decltype(&cublasLtMatrixLayoutCreate) create_layout{};
    decltype(&cublasLtMatrixLayoutDestroy) destroy_layout{};
    decltype(&cublasLtMatmulPreferenceCreate) create_preference{};
    decltype(&cublasLtMatmulPreferenceDestroy) destroy_preference{};
    decltype(&cublasLtMatmulPreferenceSetAttribute) set_preference{};
    decltype(&cublasLtMatmulAlgoGetHeuristic) heuristic{};
    decltype(&cublasLtMatmul) matmul{};

  private:
    template <typename Function>
    Function load(const char* name) {
        auto* address = dlsym(library_, name);
        if (!address)
            throw std::runtime_error(std::string("VJEPA2 cuBLASLt symbol missing: ") + name);
        return reinterpret_cast<Function>(address);
    }
    void* library_{};
};

std::shared_ptr<MathApi> math_api() {
    // Share only immutable function pointers. Each execution context owns its
    // cuBLASLt handle, matrix descriptors, algorithm and TensorRT workspace.
    static const auto api = std::make_shared<MathApi>();
    return api;
}

std::uint32_t alignment(const void* pointer) {
    std::uint32_t result = 256;
    while (reinterpret_cast<std::uintptr_t>(pointer) % result != 0)
        result /= 2;
    return result;
}
} // namespace

class LinearContext {
  public:
    LinearContext() : api_(math_api()) {
        if (api_->create(&handle_) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("VJEPA2 could not create its cuBLASLt handle");
    }
    ~LinearContext() {
        clear_descriptors();
        api_->destroy(handle_);
    }
    LinearContext(const LinearContext&) = delete;
    LinearContext& operator=(const LinearContext&) = delete;

    int run(const float* input, const float* weight, const float* bias, float* output, int rows,
            int input_width, int output_width, void* workspace, cudaStream_t stream) noexcept {
        const std::array<int, 3> shape{rows, input_width, output_width};
        const std::array<std::uint32_t, 4> alignments{alignment(weight), alignment(input),
                                                      alignment(output),
                                                      std::min(alignment(output), alignment(bias))};
        if (!valid_ || shape != shape_ || alignments != alignments_) {
            const int status = configure(shape, alignments, bias);
            if (status != CUBLAS_STATUS_SUCCESS)
                return status;
        }
        const auto status =
            api_->set_operation(operation_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        if (status != CUBLAS_STATUS_SUCCESS)
            return status;
        const float alpha = 1.0F, beta = 0.0F;
        return api_->matmul(handle_, operation_, &alpha, weight, weight_layout_, input,
                            input_layout_, &beta, output, output_layout_, output, output_layout_,
                            &algorithm_, workspace, kLinearWorkspaceBytes, stream);
    }

  private:
    void clear_descriptors() noexcept {
        valid_ = false;
        if (preference_)
            api_->destroy_preference(preference_);
        if (operation_)
            api_->destroy_operation(operation_);
        for (auto layout : {weight_layout_, input_layout_, output_layout_})
            if (layout)
                api_->destroy_layout(layout);
        preference_ = nullptr;
        operation_ = nullptr;
        weight_layout_ = input_layout_ = output_layout_ = nullptr;
    }

    int configure(const std::array<int, 3>& shape, const std::array<std::uint32_t, 4>& alignments,
                  const float* bias) noexcept {
        clear_descriptors();
        const auto [rows, inputs, outputs] = shape;
#define VJEPA2_CUBLAS_CHECK(call)                                                                  \
    do {                                                                                           \
        const auto status = (call);                                                                \
        if (status != CUBLAS_STATUS_SUCCESS)                                                       \
            return status;                                                                         \
    } while (false)
        VJEPA2_CUBLAS_CHECK(api_->create_operation(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
        VJEPA2_CUBLAS_CHECK(api_->set_operation(operation_, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                                sizeof(transpose)));
        VJEPA2_CUBLAS_CHECK(
            api_->set_operation(operation_, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal)));
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        VJEPA2_CUBLAS_CHECK(api_->set_operation(operation_, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
        VJEPA2_CUBLAS_CHECK(api_->set_operation(operation_, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &bias, sizeof(bias)));
        VJEPA2_CUBLAS_CHECK(
            api_->create_layout(&weight_layout_, CUDA_R_32F, inputs, outputs, inputs));
        VJEPA2_CUBLAS_CHECK(api_->create_layout(&input_layout_, CUDA_R_32F, inputs, rows, inputs));
        VJEPA2_CUBLAS_CHECK(
            api_->create_layout(&output_layout_, CUDA_R_32F, outputs, rows, outputs));
        VJEPA2_CUBLAS_CHECK(api_->create_preference(&preference_));
        const std::size_t budget = kLinearWorkspaceBytes;
        VJEPA2_CUBLAS_CHECK(api_->set_preference(
            preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &budget, sizeof(budget)));
        constexpr std::array<cublasLtMatmulPreferenceAttributes_t, 4> attributes{
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
        for (std::size_t i = 0; i < attributes.size(); ++i)
            VJEPA2_CUBLAS_CHECK(api_->set_preference(preference_, attributes[i], &alignments[i],
                                                     sizeof(alignments[i])));
        cublasLtMatmulHeuristicResult_t result{};
        int count = 0;
        VJEPA2_CUBLAS_CHECK(api_->heuristic(handle_, operation_, weight_layout_, input_layout_,
                                            output_layout_, output_layout_, preference_, 1, &result,
                                            &count));
        if (count != 1)
            return CUBLAS_STATUS_NOT_SUPPORTED;
        if (result.state != CUBLAS_STATUS_SUCCESS)
            return result.state;
        algorithm_ = result.algo;
        shape_ = shape;
        alignments_ = alignments;
        valid_ = true;
        return CUBLAS_STATUS_SUCCESS;
#undef VJEPA2_CUBLAS_CHECK
    }

    std::shared_ptr<MathApi> api_;
    cublasLtHandle_t handle_{};
    cublasLtMatmulDesc_t operation_{};
    cublasLtMatrixLayout_t weight_layout_{}, input_layout_{}, output_layout_{};
    cublasLtMatmulPreference_t preference_{};
    cublasLtMatmulAlgo_t algorithm_{};
    std::array<int, 3> shape_{};
    std::array<std::uint32_t, 4> alignments_{};
    bool valid_{false};
};

std::shared_ptr<LinearContext> create_linear_context() {
    return std::make_shared<LinearContext>();
}

int linear_fp32(LinearContext& context, const float* input, const float* weight, const float* bias,
                float* output, int rows, int input_width, int output_width, void* workspace,
                std::size_t workspace_bytes, cudaStream_t stream) noexcept {
    if (!input || !weight || !bias || !output || !workspace || rows <= 0 || input_width <= 0 ||
        output_width <= 0 || rows > INT_MAX / input_width || rows > INT_MAX / output_width ||
        workspace_bytes < kLinearWorkspaceBytes || alignment(workspace) < 256 || output == input ||
        output == weight || output == bias)
        return CUBLAS_STATUS_INVALID_VALUE;
    return context.run(input, weight, bias, output, rows, input_width, output_width, workspace,
                       stream);
}
} // namespace trtmc::vjepa2
