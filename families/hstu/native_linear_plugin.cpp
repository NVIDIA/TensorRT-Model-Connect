/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-only UVQK linear adapter. Every GPU operation is supplied by cuBLASLt.
#include <NvInferRuntime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

namespace {
constexpr char kName[] = "HstuUvqk";
constexpr char kVersion[] = "1";
#ifndef HSTU_PLUGIN_NAMESPACE
#error "The model builder must supply a specialization namespace"
#endif
constexpr char kNamespace[] = HSTU_PLUGIN_NAMESPACE;
constexpr std::size_t kWorkspaceBytes = 8U << 20;
constexpr std::size_t kWorkspaceAlignment = 256;
constexpr std::uint32_t kDataAlignment = 16;
bool selection_diagnostics() {
    const auto* value = std::getenv("TRTMC_HSTU_CUBLASLT_DIAGNOSTICS");
    return value && std::strcmp(value, "1") == 0;
}
void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}
bool extent(std::int64_t value, bool dynamic) {
    return (value > 0 && value <= std::numeric_limits<std::int32_t>::max()) ||
           (dynamic && value == -1);
}
bool descriptor(const nvinfer1::PluginTensorDesc& desc, int index, bool dynamic) {
    if (desc.type != nvinfer1::DataType::kBF16 || desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (index == 2)
        return desc.dims.nbDims == 1 && desc.dims.d[0] == 1024;
    if (desc.dims.nbDims != 2)
        return false;
    if (index == 1)
        return desc.dims.d[0] == 256 && desc.dims.d[1] == 1024;
    return extent(desc.dims.d[0], dynamic) && desc.dims.d[1] == (index == 0 ? 256 : 1024);
}
bool aligned(const void* pointer, std::size_t alignment) {
    return pointer && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}
std::uint32_t pointer_alignment(const void* pointer) {
    if (!pointer)
        return 0;
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    std::uint32_t result = 1;
    while (result < 256 && address % (result * 2) == 0)
        result *= 2;
    return result;
}
struct State {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weight{}, input{}, output{};
    cublasLtMatmulPreference_t preference{};
    cublasLtMatmulAlgo_t algorithm{};
    std::size_t workspace_bytes{};
    std::int64_t tokens{}, input_width{}, output_width{};
    std::array<std::uint32_t, 4> alignments{};
    std::uint32_t bias_alignment{};
    const void* bias_pointer{};
    State() = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    ~State() {
        if (preference)
            (void)cublasLtMatmulPreferenceDestroy(preference);
        if (output)
            (void)cublasLtMatrixLayoutDestroy(output);
        if (input)
            (void)cublasLtMatrixLayoutDestroy(input);
        if (weight)
            (void)cublasLtMatrixLayoutDestroy(weight);
        if (operation)
            (void)cublasLtMatmulDescDestroy(operation);
        if (handle)
            (void)cublasLtDestroy(handle);
    }
    void prepare(std::int64_t t, std::int64_t in, std::int64_t out,
                 const std::array<std::uint32_t, 4>& actual_alignments, const void* actual_bias) {
        alignments = actual_alignments;
        bias_alignment = pointer_alignment(actual_bias);
        bias_pointer = actual_bias;
        tokens = t;
        input_width = in;
        output_width = out;
        check(cublasLtCreate(&handle), "cuBLASLt handle");
        check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
              "FP32 operation descriptor");
        const cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                             &pointer_mode, sizeof(pointer_mode)),
              "host coefficients");
        const cublasOperation_t transpose = CUBLAS_OP_N;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                             sizeof(transpose)),
              "A no transpose");
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
                                             sizeof(transpose)),
              "B no transpose");
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                             sizeof(epilogue)),
              "bias epilogue");
        const cudaDataType_t bias_dtype = CUDA_R_16BF;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                             &bias_dtype, sizeof(bias_dtype)),
              "BF16 bias");
        // Selection happens on the first real uncaptured call. The bias
        // address and independent matrix alignments are actual runtime values.
        const void* aligned_bias_hint = actual_bias;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                             &aligned_bias_hint, sizeof(aligned_bias_hint)),
              "bias alignment hint");
        // Y^T[Out,T] = W^T[Out,In] * X^T[In,T]. All physical inputs/output
        // remain row-major; these column-major descriptors provide the views.
        check(cublasLtMatrixLayoutCreate(&weight, CUDA_R_16BF, out, in, out), "weight layout");
        check(cublasLtMatrixLayoutCreate(&input, CUDA_R_16BF, in, t, in), "input layout");
        check(cublasLtMatrixLayoutCreate(&output, CUDA_R_16BF, out, t, out), "output layout");
        const cublasLtOrder_t column_major = CUBLASLT_ORDER_COL;
        for (auto layout : {weight, input, output})
            check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                   &column_major, sizeof(column_major)),
                  "column-major view");
        check(cublasLtMatmulPreferenceCreate(&preference), "heuristic preferences");
        const std::uint64_t workspace_limit = kWorkspaceBytes;
        check(cublasLtMatmulPreferenceSetAttribute(preference,
                                                   CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &workspace_limit, sizeof(workspace_limit)),
              "workspace limit");
        const std::uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        check(cublasLtMatmulPreferenceSetAttribute(preference,
                                                   CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                   &reduction_mask, sizeof(reduction_mask)),
              "FP32 reduction filter");
        // In the pinned toolkit, one admitted BF16 kernel reports INPUT_16F
        // in algorithm capabilities. Its executed MMA descriptor was independently
        // checked as BF16/BF16 -> FP32 and wide-range BF16 oracles pass exactly.
        // This does not qualify every algorithm reporting INPUT_16F. Keep
        // vendor versions pinned and qualify newly selected paths with the
        // original numerical gates and BF16 exponent-range controls.
        // All actual matrix and bias descriptors remain explicitly BF16.
        const std::uint64_t implementation_mask = CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK |
                                                  CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F |
                                                  CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF |
                                                  CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F |
                                                  CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F;
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_IMPL_MASK,
                                                   &implementation_mask,
                                                   sizeof(implementation_mask)),
              "BF16/FP32 numerical filter");
        const std::array<cublasLtMatmulPreferenceAttributes_t, 4> attributes{
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
        for (std::size_t index = 0; index < attributes.size(); ++index)
            check(cublasLtMatmulPreferenceSetAttribute(
                      preference, attributes[index], &alignments[index], sizeof(alignments[index])),
                  "actual data alignment preference");
        std::array<cublasLtMatmulHeuristicResult_t, 16> choices{};
        int count = 0;
        check(cublasLtMatmulAlgoGetHeuristic(handle, operation, weight, input, output, output,
                                             preference, choices.size(), choices.data(), &count),
              "actual-shape heuristic");
        for (int i = 0; i < count; ++i) {
            if (choices[i].state != CUBLAS_STATUS_SUCCESS ||
                choices[i].workspaceSize > kWorkspaceBytes)
                continue;
            std::uint32_t reduction = 0;
            std::uint64_t numerical = 0;
            std::size_t written = 0;
            check(cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo,
                                                       CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                       &reduction, sizeof(reduction), &written),
                  "selected reduction");
            check(cublasLtMatmulAlgoCapGetAttribute(&choices[i].algo,
                                                    CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                                    &numerical, sizeof(numerical), &written),
                  "selected numerical mode");
            const bool safe_reduction = reduction == CUBLASLT_REDUCTION_SCHEME_NONE ||
                                        reduction == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
            const bool safe_math = (numerical & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F) &&
                                   !(numerical & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F) &&
                                   !(numerical & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32);
            if (!safe_reduction || !safe_math)
                continue;
            algorithm = choices[i].algo;
            workspace_bytes = choices[i].workspaceSize;
            // This function runs only for a real uncaptured preparation. The
            // environment is observed once per selection, never during replay.
            if (selection_diagnostics()) {
                std::int32_t id = -1;
                check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, CUBLASLT_ALGO_CONFIG_ID, &id,
                                                           sizeof(id), &written),
                      "selected algorithm ID");
                auto config32 = [&](cublasLtMatmulAlgoConfigAttributes_t attribute) {
                    std::uint32_t value{};
                    std::size_t bytes{};
                    check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attribute, &value,
                                                               sizeof(value), &bytes),
                          "selected algorithm configuration");
                    if (bytes != sizeof(value))
                        throw std::runtime_error("Unexpected algorithm configuration size");
                    return value;
                };
                const auto tile = config32(CUBLASLT_ALGO_CONFIG_TILE_ID);
                const auto stages = config32(CUBLASLT_ALGO_CONFIG_STAGES_ID);
                const auto split_k = config32(CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
                const auto custom = config32(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
                const auto swizzle = config32(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
                std::uint16_t cluster{};
                check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm,
                                                           CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
                                                           &cluster, sizeof(cluster), &written),
                      "selected algorithm cluster");
                if (written != sizeof(cluster))
                    throw std::runtime_error("Unexpected cluster configuration size");
                // First-use diagnostic only; prepared enqueue and graph replay do
                // not query candidates or print. It is not performance evidence.
                std::fprintf(stderr,
                             "HSTU_LT_SELECTION T=%lld In=%lld Out=%lld id=%d numerical=%llu "
                             "reduction=%u workspace=%zu align=%u,%u,%u,%u bias_align_capped256=%u "
                             "bias_pointer=%p tile=%u stages=%u split_k=%u custom=%u swizzle=%u "
                             "cluster=%u impl_pref=%llu\n",
                             static_cast<long long>(tokens), static_cast<long long>(input_width),
                             static_cast<long long>(output_width), id,
                             static_cast<unsigned long long>(numerical), reduction, workspace_bytes,
                             alignments[0], alignments[1], alignments[2], alignments[3],
                             bias_alignment, bias_pointer, tile, stages, split_k, custom, swizzle,
                             unsigned(cluster),
                             static_cast<unsigned long long>(implementation_mask));
            }
            return;
        }
        throw std::runtime_error("No admitted BF16/FP32 cuBLASLt heuristic within8MiB workspace");
    }
};

class Plugin final : public nvinfer1::IPluginV3,
                     public nvinfer1::IPluginV3OneCore,
                     public nvinfer1::IPluginV3OneBuild,
                     public nvinfer1::IPluginV3OneRuntime {
  public:
    Plugin() = default;
    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    Plugin* clone() noexcept override { return new (std::nothrow) Plugin(); }
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(nvinfer1::DataType* out, int32_t outputs,
                               const nvinfer1::DataType* in,
                               int32_t inputs) const noexcept override {
        if (!out || !in || inputs != 3 || outputs != 1)
            return 1;
        for (int i = 0; i < 3; ++i)
            if (in[i] != nvinfer1::DataType::kBF16)
                return 1;
        out[0] = nvinfer1::DataType::kBF16;
        return 0;
    }
    int32_t getOutputShapes(const nvinfer1::DimsExprs* in, int32_t inputs,
                            const nvinfer1::DimsExprs*, int32_t shape_inputs,
                            nvinfer1::DimsExprs* out, int32_t outputs,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (!in || !out || inputs != 3 || outputs != 1 || shape_inputs || in[0].nbDims != 2 ||
            in[1].nbDims != 2)
            return 1;
        out[0].nbDims = 2;
        out[0].d[0] = in[0].d[0];
        out[0].d[1] = in[1].d[1];
        return 0;
    }
    bool supportsFormatCombination(int32_t position, const nvinfer1::DynamicPluginTensorDesc* desc,
                                   int32_t inputs, int32_t outputs) noexcept override {
        return desc && inputs == 3 && outputs == 1 && position >= 0 && position < 4 &&
               descriptor(desc[position].desc, position, true);
    }
    int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t inputs,
                            const nvinfer1::DynamicPluginTensorDesc* out,
                            int32_t outputs) noexcept override {
        if (!in || !out || inputs != 3 || outputs != 1)
            return 1;
        for (int i = 0; i < 3; ++i)
            if (!descriptor(in[i].desc, i, true))
                return 1;
        return descriptor(out[0].desc, 3, true) ? 0 : 1;
    }
    std::size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
                                 const nvinfer1::DynamicPluginTensorDesc*,
                                 int32_t) const noexcept override {
        return kWorkspaceBytes + kWorkspaceAlignment - 1;
    }
    const char* getTimingCacheID() noexcept override {
        return "hstu-uvqk-e256-bf16-fp32-pointer-8mib-v1";
    }
    const char* getMetadataString() noexcept override {
        return "X[T,In];WT[In,Out];bias[Out];Y[T,Out];BF16_IO_FP32_compute;no_TF32;no_reduced_"
               "reduction;actual_pointer_heuristic_before_capture";
    }
    int32_t onShapeChange(const nvinfer1::PluginTensorDesc* in, int32_t inputs,
                          const nvinfer1::PluginTensorDesc* out,
                          int32_t outputs) noexcept override {
        try {
            ready_ = false;
            if (!in || !out || inputs != 3 || outputs != 1)
                return 1;
            for (int i = 0; i < 3; ++i)
                if (!descriptor(in[i], i, false))
                    return 1;
            if (!descriptor(out[0], 3, false) || in[0].dims.d[1] != in[1].dims.d[0] ||
                in[2].dims.d[0] != in[1].dims.d[1] || out[0].dims.d[0] != in[0].dims.d[0] ||
                out[0].dims.d[1] != in[1].dims.d[1])
                return 1;
            tokens_ = in[0].dims.d[0];
            input_width_ = in[0].dims.d[1];
            output_width_ = in[1].dims.d[1];
            // Actual addresses are supplied only by enqueue. No selection is
            // attempted here using fictitious alignment hints.
            ready_ = true;
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s shape: %s\n", kName, e.what());
            return 1;
        }
    }
    int32_t enqueue(const nvinfer1::PluginTensorDesc* desc,
                    const nvinfer1::PluginTensorDesc* result, const void* const* in,
                    void* const* out, void* workspace, cudaStream_t stream) noexcept override {
        try {
            if (!ready_ || !in || !out || !desc || !result)
                return 1;
            for (int index = 0; index < 3; ++index)
                if (!descriptor(desc[index], index, false))
                    return 1;
            if (!descriptor(result[0], 3, false) || desc[0].dims.d[0] != tokens_ ||
                desc[0].dims.d[1] != input_width_ || desc[1].dims.d[0] != input_width_ ||
                desc[1].dims.d[1] != output_width_ || desc[2].dims.d[0] != output_width_ ||
                result[0].dims.d[0] != tokens_ || result[0].dims.d[1] != output_width_)
                return 1;
            if (!aligned(in[0], kDataAlignment) || !aligned(in[1], kDataAlignment) ||
                !aligned(in[2], kDataAlignment) || !aligned(out[0], kDataAlignment))
                return 1;
            const std::array<std::uint32_t, 4> actual_alignments{
                pointer_alignment(in[1]), pointer_alignment(in[0]), pointer_alignment(out[0]),
                pointer_alignment(out[0])};
            if (!state_valid_ || !state_ || state_->tokens != tokens_ ||
                state_->input_width != input_width_ || state_->output_width != output_width_ ||
                state_->alignments != actual_alignments || state_->bias_pointer != in[2]) {
                cudaStreamCaptureStatus capture{};
                if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
                    capture != cudaStreamCaptureStatusNone)
                    throw std::runtime_error("Real uncaptured preparation required before capture");
                auto next = std::make_unique<State>();
                next->prepare(tokens_, input_width_, output_width_, actual_alignments, in[2]);
                state_ = std::move(next);
                state_valid_ = true;
            }
            void* scratch = nullptr;
            if (state_->workspace_bytes) {
                if (!workspace)
                    return 1;
                const auto raw = reinterpret_cast<std::uintptr_t>(workspace);
                scratch = reinterpret_cast<void*>((raw + kWorkspaceAlignment - 1) &
                                                  ~(kWorkspaceAlignment - 1));
            }
            const void* bias = in[2];
            check(cublasLtMatmulDescSetAttribute(
                      state_->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)),
                  "runtime bias pointer");
            const float alpha = 1.0F, beta = 0.0F;
            check(cublasLtMatmul(state_->handle, state_->operation, &alpha, in[1], state_->weight,
                                 in[0], state_->input, &beta, out[0], state_->output, out[0],
                                 state_->output, &state_->algorithm, scratch,
                                 state_->workspace_bytes, stream),
                  "cuBLASLt UVQK");
            const auto cuda_status = cudaPeekAtLastError();
            if (cuda_status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(cuda_status));
            return 0;
        } catch (const std::exception& e) {
            ready_ = false;
            state_valid_ = false;
            std::fprintf(stderr, "%s enqueue: %s\n", kName, e.what());
            return 1;
        }
    }
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }
    const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept override {
        return &fields_;
    }

  private:
    bool ready_{false};
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
    std::unique_ptr<State> state_;
    bool state_valid_{false};
    std::int64_t tokens_{}, input_width_{}, output_width_{};
};
class Creator final : public nvinfer1::IPluginCreatorV3One {
  public:
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV3* createPlugin(const char*, const nvinfer1::PluginFieldCollection* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields && fields->nbFields)
            return nullptr;
        return new (std::nothrow) Plugin();
    }

  private:
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
};
Creator creator;
} // namespace

// Hidden by the model library's export map. The attention translation unit
// returns both creators through TensorRT's one scoped getCreators entry point.
nvinfer1::IPluginCreatorInterface* hstu_linear_creator() noexcept {
    return &creator;
}
