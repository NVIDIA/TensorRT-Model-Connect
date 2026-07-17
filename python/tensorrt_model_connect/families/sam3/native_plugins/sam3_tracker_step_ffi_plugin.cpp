/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam3_tracker_step_ffi_plugin.h"

#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <utility>
#include <vector>

namespace trtmc::sam3 {

namespace {

constexpr std::array<nvinfer1::DataType, TrackerStepFfiPlugin::kInputCount> kInputTypes{
    nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT,
    nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT,
    nvinfer1::DataType::kINT32, nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT32,
    nvinfer1::DataType::kINT32,
};

constexpr int32_t kSpatialTokens = 72 * 72;
constexpr int32_t kMaximumMemoryFrames = 10;
constexpr int32_t kMaximumPointers = 19;

bool is_lower_hex(std::string_view value) {
    return std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
    });
}

bool valid_content_addressed_name(std::string_view name, int32_t batch_size) {
    const std::string prefix = batch_size == 1 ? "trtmc.sam3.tracker_step.b1.split_aoti."
                                               : "trtmc.sam3.tracker_step.b2.split_aoti.";
    constexpr std::size_t kDigestCharacters = 20;
    return name.starts_with(prefix) && name.size() == prefix.size() + kDigestCharacters &&
           is_lower_hex(name.substr(prefix.size()));
}

bool exact_dimensions(const nvinfer1::Dims& dims, std::initializer_list<int64_t> expected) {
    if (dims.nbDims != static_cast<int32_t>(expected.size()))
        return false;
    std::size_t index = 0;
    for (const int64_t dimension : expected) {
        if (dims.d[index++] != dimension)
            return false;
    }
    return true;
}

bool ranged_dimensions(const nvinfer1::Dims& dims, std::initializer_list<int64_t> expected,
                       int32_t dynamic_index, int64_t maximum, bool allow_dynamic) {
    if (dims.nbDims != static_cast<int32_t>(expected.size()))
        return false;
    std::size_t index = 0;
    for (const int64_t dimension : expected) {
        const int64_t actual = dims.d[index];
        if (static_cast<int32_t>(index) == dynamic_index) {
            if (actual != -1 && (actual < 1 || actual > maximum))
                return false;
            if (!allow_dynamic && actual == -1)
                return false;
        } else if (actual != dimension) {
            return false;
        }
        ++index;
    }
    return true;
}

bool descriptor_shapes_valid(const nvinfer1::PluginTensorDesc* inputs,
                             const nvinfer1::PluginTensorDesc* outputs, int32_t batch_size,
                             bool allow_dynamic) {
    if (inputs == nullptr || outputs == nullptr)
        return false;
    if (!exact_dimensions(inputs[0].dims, {1, 32, 288, 288}) ||
        !exact_dimensions(inputs[1].dims, {1, 64, 144, 144}) ||
        !exact_dimensions(inputs[2].dims, {1, 256, 72, 72}) ||
        !exact_dimensions(inputs[3].dims, {1, 256, 72, 72}) ||
        !ranged_dimensions(inputs[4].dims, {batch_size, 1, kSpatialTokens, 64}, 1,
                           kMaximumMemoryFrames, allow_dynamic) ||
        !ranged_dimensions(inputs[5].dims, {batch_size, 1, kSpatialTokens, 64}, 1,
                           kMaximumMemoryFrames, allow_dynamic) ||
        !ranged_dimensions(inputs[6].dims, {batch_size, 1}, 1, kMaximumMemoryFrames,
                           allow_dynamic) ||
        !ranged_dimensions(inputs[7].dims, {batch_size, 1, 256}, 1, kMaximumPointers,
                           allow_dynamic) ||
        !ranged_dimensions(inputs[8].dims, {batch_size, 1}, 1, kMaximumPointers, allow_dynamic) ||
        !exact_dimensions(inputs[9].dims, {1}) ||
        !exact_dimensions(outputs[0].dims, {batch_size, TrackerStepFfiPlugin::kPackedWidth}))
        return false;
    const int32_t memory_frames = inputs[4].dims.d[1];
    const int32_t pointer_count = inputs[7].dims.d[1];
    return inputs[5].dims.d[1] == memory_frames && inputs[6].dims.d[1] == memory_frames &&
           inputs[8].dims.d[1] == pointer_count;
}

bool descriptor_types_valid(const nvinfer1::PluginTensorDesc* inputs,
                            const nvinfer1::PluginTensorDesc* outputs) {
    if (inputs == nullptr || outputs == nullptr)
        return false;
    for (int32_t index = 0; index < TrackerStepFfiPlugin::kInputCount; ++index) {
        if (inputs[index].type != kInputTypes[static_cast<std::size_t>(index)] ||
            inputs[index].format != nvinfer1::TensorFormat::kLINEAR)
            return false;
    }
    return outputs[0].type == nvinfer1::DataType::kFLOAT &&
           outputs[0].format == nvinfer1::TensorFormat::kLINEAR;
}

bool dynamic_descriptor_contract_valid(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                       int32_t input_count,
                                       const nvinfer1::DynamicPluginTensorDesc* outputs,
                                       int32_t output_count, int32_t batch_size) {
    if (inputs == nullptr || outputs == nullptr ||
        input_count != TrackerStepFfiPlugin::kInputCount ||
        output_count != TrackerStepFfiPlugin::kOutputCount)
        return false;

    std::array<nvinfer1::PluginTensorDesc, TrackerStepFfiPlugin::kInputCount> descriptors{};
    std::array<nvinfer1::PluginTensorDesc, TrackerStepFfiPlugin::kInputCount> minima{};
    std::array<nvinfer1::PluginTensorDesc, TrackerStepFfiPlugin::kInputCount> maxima{};
    for (int32_t index = 0; index < input_count; ++index) {
        descriptors[static_cast<std::size_t>(index)] = inputs[index].desc;
        minima[static_cast<std::size_t>(index)] = inputs[index].desc;
        maxima[static_cast<std::size_t>(index)] = inputs[index].desc;
        minima[static_cast<std::size_t>(index)].dims = inputs[index].min;
        maxima[static_cast<std::size_t>(index)].dims = inputs[index].max;
    }
    return descriptor_shapes_valid(descriptors.data(), &outputs[0].desc, batch_size, true) &&
           descriptor_types_valid(descriptors.data(), &outputs[0].desc) &&
           descriptor_shapes_valid(minima.data(), &outputs[0].desc, batch_size, false) &&
           descriptor_shapes_valid(maxima.data(), &outputs[0].desc, batch_size, false);
}

DLDataType to_dl_dtype(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT:
        return {kDLFloat, 32, 1};
    case nvinfer1::DataType::kHALF:
        return {kDLFloat, 16, 1};
    case nvinfer1::DataType::kBF16:
        return {kDLBfloat, 16, 1};
    case nvinfer1::DataType::kINT32:
        return {kDLInt, 32, 1};
    default:
        return {kDLUInt, 0, 0};
    }
}

using DlShapeStorage = std::array<int64_t, nvinfer1::Dims::MAX_DIMS>;

void fill_dl_tensor(DLTensor& tensor, DlShapeStorage& shape_storage, void* data,
                    const nvinfer1::PluginTensorDesc& desc, int32_t device_id) {
    for (int32_t index = 0; index < desc.dims.nbDims; ++index)
        shape_storage[static_cast<std::size_t>(index)] = desc.dims.d[index];
    tensor.data = data;
    tensor.device = {kDLCUDA, device_id};
    tensor.ndim = desc.dims.nbDims;
    tensor.dtype = to_dl_dtype(desc.type);
    tensor.shape = shape_storage.data();
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
}

void report_ffi_error(const std::string& kernel_name) {
    TVMFFIObjectHandle error = nullptr;
    TVMFFIErrorMoveFromRaised(&error);
    if (error == nullptr)
        return;
    auto* cell = reinterpret_cast<const TVMFFIErrorCell*>(
        static_cast<const char*>(static_cast<const void*>(error)) + sizeof(TVMFFIObject));
    if (cell->message.data != nullptr && cell->message.size > 0) {
        std::cerr << "[Sam3TrackerStepFfi] " << kernel_name << ": "
                  << std::string(cell->message.data, static_cast<std::size_t>(cell->message.size))
                  << '\n';
    }
    TVMFFIObjectDecRef(error);
}

class FfiStreamScope {
  public:
    FfiStreamScope(int32_t device_id, cudaStream_t stream) : device_id_(device_id) {
        installed_ =
            TVMFFIEnvSetStream(kDLCUDA, device_id, reinterpret_cast<TVMFFIStreamHandle>(stream),
                               &previous_) == 0;
    }

    FfiStreamScope(const FfiStreamScope&) = delete;
    FfiStreamScope& operator=(const FfiStreamScope&) = delete;

    ~FfiStreamScope() {
        if (installed_ && !restored_)
            (void)restore();
    }

    bool installed() const noexcept { return installed_; }

    bool restore() noexcept {
        if (!installed_ || restored_)
            return installed_;
        restored_ = TVMFFIEnvSetStream(kDLCUDA, device_id_, previous_, nullptr) == 0;
        return restored_;
    }

  private:
    int32_t device_id_{0};
    TVMFFIStreamHandle previous_{nullptr};
    bool installed_{false};
    bool restored_{false};
};

} // namespace

TrackerStepFfiPlugin::TrackerStepFfiPlugin(std::string kernel_name, int32_t batch_size)
    : kernel_name_(std::move(kernel_name)), batch_size_(batch_size) {}

TrackerStepFfiPlugin::TrackerStepFfiPlugin(const void* data, std::size_t length) {
    if (data == nullptr || length < sizeof(uint32_t) + sizeof(int32_t))
        return;
    auto* cursor = static_cast<const char*>(data);
    uint32_t name_length = 0;
    std::memcpy(&name_length, cursor, sizeof(name_length));
    cursor += sizeof(name_length);
    const auto required =
        sizeof(name_length) + static_cast<std::size_t>(name_length) + sizeof(batch_size_);
    if (required != length)
        return;
    kernel_name_.assign(cursor, name_length);
    cursor += name_length;
    std::memcpy(&batch_size_, cursor, sizeof(batch_size_));
}

char const* TrackerStepFfiPlugin::getPluginType() const noexcept {
    return kPluginName;
}

char const* TrackerStepFfiPlugin::getPluginVersion() const noexcept {
    return kPluginVersion;
}

int32_t TrackerStepFfiPlugin::getNbOutputs() const noexcept {
    return kOutputCount;
}

int32_t TrackerStepFfiPlugin::initialize() noexcept {
    if (!valid_contract() || (configuration_checked_.load(std::memory_order_acquire) &&
                              !configuration_valid_.load(std::memory_order_acquire)))
        return -1;
    return resolve_kernel() ? 0 : -1;
}

void TrackerStepFfiPlugin::terminate() noexcept {
    release_kernel();
}

void TrackerStepFfiPlugin::destroy() noexcept {
    release_kernel();
    delete this;
}

std::size_t TrackerStepFfiPlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) + kernel_name_.size() + sizeof(batch_size_);
}

void TrackerStepFfiPlugin::serialize(void* buffer) const noexcept {
    auto* cursor = static_cast<char*>(buffer);
    const auto name_length = static_cast<uint32_t>(kernel_name_.size());
    std::memcpy(cursor, &name_length, sizeof(name_length));
    cursor += sizeof(name_length);
    std::memcpy(cursor, kernel_name_.data(), kernel_name_.size());
    cursor += kernel_name_.size();
    std::memcpy(cursor, &batch_size_, sizeof(batch_size_));
}

void TrackerStepFfiPlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}

char const* TrackerStepFfiPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType TrackerStepFfiPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                           int32_t) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

TrackerStepFfiPlugin* TrackerStepFfiPlugin::clone() const noexcept {
    try {
        auto* plugin = new TrackerStepFfiPlugin(kernel_name_, batch_size_);
        plugin->namespace_ = namespace_;
        plugin->configuration_checked_.store(configuration_checked_.load(std::memory_order_acquire),
                                             std::memory_order_relaxed);
        plugin->configuration_valid_.store(configuration_valid_.load(std::memory_order_acquire),
                                           std::memory_order_relaxed);
        // TensorRT may enqueue a context clone without invoking initialize()
        // on that clone. Resolve an independent process-global TVM-FFI
        // function reference here so context creation cannot silently discard
        // the kernel resolved by the deserialized engine plugin.
        if (!plugin->resolve_kernel()) {
            delete plugin;
            return nullptr;
        }
        return plugin;
    } catch (...) {
        return nullptr;
    }
}

nvinfer1::DimsExprs
TrackerStepFfiPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const*, int32_t,
                                          nvinfer1::IExprBuilder& builder) noexcept {
    nvinfer1::DimsExprs output;
    output.nbDims = 2;
    output.d[0] = builder.constant(batch_size_);
    output.d[1] = builder.constant(kPackedWidth);
    return output;
}

bool TrackerStepFfiPlugin::supportsFormatCombination(int32_t position,
                                                     nvinfer1::PluginTensorDesc const* in_out,
                                                     int32_t input_count,
                                                     int32_t output_count) noexcept {
    if (in_out == nullptr || input_count != kInputCount || output_count != kOutputCount ||
        position < 0 || position >= kInputCount + kOutputCount)
        return false;
    if (in_out[position].format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (position < kInputCount)
        return in_out[position].type == kInputTypes[static_cast<std::size_t>(position)];
    return in_out[position].type == nvinfer1::DataType::kFLOAT;
}

void TrackerStepFfiPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                           int32_t input_count,
                                           nvinfer1::DynamicPluginTensorDesc const* outputs,
                                           int32_t output_count) noexcept {
    configuration_valid_.store(
        dynamic_descriptor_contract_valid(inputs, input_count, outputs, output_count, batch_size_),
        std::memory_order_release);
    configuration_checked_.store(true, std::memory_order_release);
}

std::size_t TrackerStepFfiPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                   nvinfer1::PluginTensorDesc const*,
                                                   int32_t) const noexcept {
    return 0;
}

bool TrackerStepFfiPlugin::valid_contract() const noexcept {
    return (batch_size_ == 1 || batch_size_ == 2) &&
           valid_content_addressed_name(kernel_name_, batch_size_);
}

bool TrackerStepFfiPlugin::resolve_kernel() noexcept {
    if (cached_function_ != nullptr)
        return true;
    TVMFFIByteArray name{kernel_name_.c_str(), kernel_name_.size()};
    TVMFFIObjectHandle resolved = nullptr;
    if (TVMFFIFunctionGetGlobal(&name, &resolved) != 0 || resolved == nullptr) {
        std::cerr << "[Sam3TrackerStepFfi] unresolved kernel " << kernel_name_ << '\n';
        return false;
    }
    cached_function_ = resolved;
    return true;
}

void TrackerStepFfiPlugin::release_kernel() noexcept {
    if (cached_function_ != nullptr) {
        TVMFFIObjectDecRef(cached_function_);
        cached_function_ = nullptr;
    }
}

int32_t TrackerStepFfiPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                      nvinfer1::PluginTensorDesc const* output_desc,
                                      void const* const* inputs, void* const* outputs, void*,
                                      cudaStream_t stream) noexcept {
    const auto fail = [this](const char* reason) {
        std::cerr << "[Sam3TrackerStepFfi] " << kernel_name_ << ": enqueue rejected: " << reason
                  << '\n';
        return -1;
    };
    if (!valid_contract() || input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr || outputs[0] == nullptr)
        return fail("invalid plugin contract or null TensorRT descriptor/storage");
    if (cached_function_ == nullptr)
        return fail("TVM-FFI global was not resolved during initialization");
    if (!descriptor_shapes_valid(input_desc, output_desc, batch_size_, false))
        return fail("runtime tensor shapes violate the tracker-step contract");
    if (!descriptor_types_valid(input_desc, output_desc))
        return fail("runtime tensor dtypes or formats violate the tracker-step contract");
    for (int32_t index = 0; index < kInputCount; ++index) {
        if (inputs[index] == nullptr)
            return fail("TensorRT supplied a null input buffer");
    }

    int32_t device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess)
        return fail("CUDA device lookup failed");
    std::array<DLTensor, kInputCount + kOutputCount> tensors{};
    std::array<DlShapeStorage, kInputCount + kOutputCount> shape_storage{};
    std::array<TVMFFIAny, kInputCount + kOutputCount + 1> arguments{};
    for (int32_t index = 0; index < kInputCount; ++index) {
        fill_dl_tensor(tensors[static_cast<std::size_t>(index)],
                       shape_storage[static_cast<std::size_t>(index)],
                       const_cast<void*>(inputs[index]), input_desc[index], device_id);
        arguments[static_cast<std::size_t>(index)].type_index = kTVMFFIDLTensorPtr;
        arguments[static_cast<std::size_t>(index)].v_ptr =
            &tensors[static_cast<std::size_t>(index)];
    }
    fill_dl_tensor(tensors.back(), shape_storage.back(), outputs[0], output_desc[0], device_id);
    arguments[kInputCount].type_index = kTVMFFIDLTensorPtr;
    arguments[kInputCount].v_ptr = &tensors.back();
    arguments.back().type_index = kTVMFFIOpaquePtr;
    arguments.back().v_ptr = reinterpret_cast<void*>(stream);

    FfiStreamScope stream_scope(device_id, stream);
    if (!stream_scope.installed())
        return fail("TVM-FFI rejected the TensorRT CUDA stream");
    TVMFFIAny result{};
    result.type_index = kTVMFFINone;
    const int call_status = TVMFFIFunctionCall(cached_function_, arguments.data(),
                                               static_cast<int32_t>(arguments.size()), &result);
    const bool restored = stream_scope.restore();
    if (call_status != 0)
        report_ffi_error(kernel_name_);
    if (call_status != 0)
        return -1;
    if (!restored)
        return fail("TVM-FFI failed to restore the previous CUDA stream");
    return 0;
}

namespace {

class TrackerStepFfiPluginCreator final : public nvinfer1::IPluginCreator {
  public:
    TrackerStepFfiPluginCreator() {
        fields_.push_back({"kernel_name", nullptr, nvinfer1::PluginFieldType::kCHAR, 0});
        fields_.push_back({"batch_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        field_collection_.nbFields = static_cast<int32_t>(fields_.size());
        field_collection_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return TrackerStepFfiPlugin::kPluginName;
    }
    char const* getPluginVersion() const noexcept override {
        return TrackerStepFfiPlugin::kPluginVersion;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        return &field_collection_;
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value != nullptr ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

    nvinfer1::IPluginV2*
    createPlugin(char const*, nvinfer1::PluginFieldCollection const* collection) noexcept override {
        try {
            std::string kernel_name;
            int32_t batch_size = 0;
            if (collection == nullptr || collection->nbFields < 0 ||
                (collection->nbFields > 0 && collection->fields == nullptr))
                return nullptr;
            for (int32_t index = 0; index < collection->nbFields; ++index) {
                const auto& field = collection->fields[index];
                if (field.name == nullptr)
                    return nullptr;
                if (std::strcmp(field.name, "kernel_name") == 0 && field.data != nullptr &&
                    field.type == nvinfer1::PluginFieldType::kCHAR) {
                    constexpr int32_t kMaximumKernelNameLength = 128;
                    if (field.length <= 0 || field.length > kMaximumKernelNameLength)
                        return nullptr;
                    kernel_name.assign(static_cast<const char*>(field.data),
                                       static_cast<std::size_t>(field.length));
                } else if (std::strcmp(field.name, "batch_size") == 0 && field.data != nullptr &&
                           field.type == nvinfer1::PluginFieldType::kINT32 && field.length == 1) {
                    std::memcpy(&batch_size, field.data, sizeof(batch_size));
                }
            }
            if (!valid_content_addressed_name(kernel_name, batch_size))
                return nullptr;
            auto* plugin = new TrackerStepFfiPlugin(std::move(kernel_name), batch_size);
            plugin->setPluginNamespace(namespace_.c_str());
            return plugin;
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           std::size_t length) noexcept override {
        try {
            constexpr std::size_t kMaximumSerializedPluginBytes = 256;
            if (data == nullptr || length < sizeof(uint32_t) + sizeof(int32_t) ||
                length > kMaximumSerializedPluginBytes)
                return nullptr;
            const auto* cursor = static_cast<const char*>(data);
            uint32_t name_length = 0;
            std::memcpy(&name_length, cursor, sizeof(name_length));
            cursor += sizeof(name_length);
            const auto required =
                sizeof(name_length) + static_cast<std::size_t>(name_length) + sizeof(int32_t);
            if (required != length)
                return nullptr;
            std::string kernel_name(cursor, static_cast<std::size_t>(name_length));
            cursor += name_length;
            int32_t batch_size = 0;
            std::memcpy(&batch_size, cursor, sizeof(batch_size));
            if (!valid_content_addressed_name(kernel_name, batch_size))
                return nullptr;
            auto* plugin = new TrackerStepFfiPlugin(std::move(kernel_name), batch_size);
            plugin->setPluginNamespace(namespace_.c_str());
            return plugin;
        } catch (...) {
            return nullptr;
        }
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection field_collection_{};
    std::string namespace_;
};

static nvinfer1::PluginRegistrar<TrackerStepFfiPluginCreator> tracker_step_plugin_registrar{};

} // namespace

} // namespace trtmc::sam3

extern "C" const char* trtmc_sam3_tracker_step_plugin_version() noexcept {
    return trtmc::sam3::TrackerStepFfiPlugin::kPluginVersion;
}
