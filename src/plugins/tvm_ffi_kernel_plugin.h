#pragma once

// TVM-FFI kernel bridge plugin for TensorRT (IPluginV2DynamicExt).
// Wraps any TVM-FFI-registered function into a TRT engine graph.
// Uses the stable V2 plugin API for maximum platform compatibility.
// Guarded: requires both TRT and TVM-FFI at build time.

#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI

#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

// Parsed output specification from shape_spec JSON.
struct TvmFfiOutputSpec {
    std::vector<int32_t> dims;
    int32_t same_as_input_index{-1};
    int32_t dtype{0}; // 0 = float32, 1 = float16, 2 = bfloat16
};

// Extra scalar/pointer argument passed after tensors in TVMFFIFunctionCall.
struct TvmFfiExtraArg {
    int32_t type_index{0}; // kTVMFFINone=0, kTVMFFIInt=1, kTVMFFIFloat=3, kTVMFFIOpaquePtr=4
    int64_t v_int{0};
    double v_float{0.0};
};

class TvmFfiKernelPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    TvmFfiKernelPlugin() = default;
    TvmFfiKernelPlugin(const std::string& kernel_name, const std::string& shape_spec);
    TvmFfiKernelPlugin(const void* data, size_t length);

    // IPluginV2
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void setPluginNamespace(char const* ns) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt
    TvmFfiKernelPlugin* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
                                            int32_t nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
                         nvinfer1::DynamicPluginTensorDesc const* out,
                         int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
                            nvinfer1::PluginTensorDesc const* outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    static constexpr const char* kPLUGIN_NAME = "TvmFfiKernel";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    void parse_shape_spec();

    std::string kernel_name_;
    std::string shape_spec_;
    std::string namespace_;
    int32_t num_inputs_{0};
    int32_t num_outputs_{0};
    int64_t workspace_bytes_{0};
    std::vector<TvmFfiOutputSpec> output_specs_;
    std::vector<TvmFfiExtraArg> extra_args_;
    void* cached_fn_{nullptr};
};

} // namespace trtmc

#endif // TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI
