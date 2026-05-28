#if TRTMC_HAS_TRT

#include "plugins/sana_wm_t2i_modulate_plugin.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>

namespace trtmc {
namespace {

__device__ __forceinline__ float bf16_to_float(const uint16_t value) {
    union {
        uint32_t u32;
        float f32;
    } bits{};
    bits.u32 = static_cast<uint32_t>(value) << 16U;
    return bits.f32;
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(const float value) {
    union {
        float f32;
        uint32_t u32;
    } bits{};
    bits.f32 = value;
    bits.u32 += 0x7FFFU + ((bits.u32 >> 16U) & 1U);
    return static_cast<uint16_t>(bits.u32 >> 16U);
}

__global__ void modulate_kernel(const uint16_t* input, const uint16_t* shift,
                                const uint16_t* scale, uint16_t* output, int64_t total,
                                int64_t spatial, int64_t channels, int64_t shift_spatial) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int64_t channel = idx % channels;
    const int64_t spatial_index = (idx / channels) % spatial;
    const int64_t frame_row = idx / (spatial * channels);
    const int64_t shift_index = (frame_row * shift_spatial +
                                 (shift_spatial == 1 ? 0 : spatial_index)) *
                                    channels +
                                channel;
    const uint16_t one_plus_scale =
        float_to_bf16_bits(1.0F + bf16_to_float(scale[shift_index]));
    const uint16_t scaled = float_to_bf16_bits(bf16_to_float(input[idx]) *
                                               bf16_to_float(one_plus_scale));
    output[idx] = float_to_bf16_bits(bf16_to_float(scaled) + bf16_to_float(shift[shift_index]));
}

bool launch_ok() {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "SanaWmT2IModulate failed: %s\n", cudaGetErrorString(status));
    return false;
}

int64_t product(const nvinfer1::Dims& dims) {
    int64_t total = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
        total *= dims.d[i];
    return total;
}

} // namespace

SanaWmT2IModulatePlugin::SanaWmT2IModulatePlugin(const void* /*data*/, size_t /*length*/) {}

char const* SanaWmT2IModulatePlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmT2IModulatePlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmT2IModulatePlugin::getNbOutputs() const noexcept { return 1; }

int32_t SanaWmT2IModulatePlugin::initialize() noexcept { return 0; }

void SanaWmT2IModulatePlugin::terminate() noexcept {}

void SanaWmT2IModulatePlugin::destroy() noexcept { delete this; }

size_t SanaWmT2IModulatePlugin::getSerializationSize() const noexcept { return 0; }

void SanaWmT2IModulatePlugin::serialize(void* /*buffer*/) const noexcept {}

void SanaWmT2IModulatePlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmT2IModulatePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmT2IModulatePlugin::getOutputDataType(
    int32_t /*index*/, nvinfer1::DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const
    noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmT2IModulatePlugin* SanaWmT2IModulatePlugin::clone() const noexcept {
    auto* plugin = new SanaWmT2IModulatePlugin();
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmT2IModulatePlugin::getOutputDimensions(
    int32_t /*outputIndex*/, nvinfer1::DimsExprs const* inputs, int32_t /*nbInputs*/,
    nvinfer1::IExprBuilder& /*exprBuilder*/) noexcept {
    return inputs[0];
}

bool SanaWmT2IModulatePlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t /*nbInputs*/,
    int32_t /*nbOutputs*/) noexcept {
    const auto& desc = inOut[pos];
    return desc.format == nvinfer1::TensorFormat::kLINEAR &&
           desc.type == nvinfer1::DataType::kBF16;
}

void SanaWmT2IModulatePlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    nvinfer1::DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {}

size_t SanaWmT2IModulatePlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* /*inputs*/, int32_t /*nbInputs*/,
    nvinfer1::PluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    return 0;
}

int32_t SanaWmT2IModulatePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                         nvinfer1::PluginTensorDesc const* /*outputDesc*/,
                                         void const* const* inputs, void* const* outputs,
                                         void* /*workspace*/, cudaStream_t stream) noexcept {
    if (inputs == nullptr || outputs == nullptr)
        return 1;
    const auto& input_dims = inputDesc[0].dims;
    const auto& shift_dims = inputDesc[1].dims;
    if (input_dims.nbDims != 4 || shift_dims.nbDims != 4)
        return 1;
    const int64_t total = product(input_dims);
    const int64_t spatial = input_dims.d[2];
    const int64_t channels = input_dims.d[3];
    const int64_t shift_spatial = shift_dims.d[2];
    if (total <= 0 || spatial <= 0 || channels <= 0 || shift_spatial <= 0)
        return 1;
    constexpr int32_t kThreads = 256;
    modulate_kernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads), kThreads, 0,
                      stream>>>(static_cast<const uint16_t*>(inputs[0]),
                                static_cast<const uint16_t*>(inputs[1]),
                                static_cast<const uint16_t*>(inputs[2]),
                                static_cast<uint16_t*>(outputs[0]), total, spatial, channels,
                                shift_spatial);
    return launch_ok() ? 0 : 1;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT
