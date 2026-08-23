/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// TVM-FFI kernel bridge plugin implementation (IPluginV2DynamicExt).

#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI

#include "plugins/tvm_ffi_kernel_plugin.h"

#include "plugins/tvm_ffi_runtime_bindings.h"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <utility>
#include <vector>

namespace trtmc {

// ---------------------------------------------------------------------------
// shape_spec parsing helpers (kept small for low CCN)
// ---------------------------------------------------------------------------

namespace {

} // namespace
// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TvmFfiKernelPlugin::TvmFfiKernelPlugin(const std::string& kernel_name,
                                       const std::string& shape_spec)
    : kernel_name_(kernel_name), shape_spec_(shape_spec) {
    parse_shape_spec();
}

TvmFfiKernelPlugin::TvmFfiKernelPlugin(const void* data, size_t length) {
    if (data == nullptr)
        throw std::runtime_error("TvmFfiKernelPlugin serialization data is null");
    auto* p = static_cast<const char*>(data);
    std::size_t remaining = length;
    auto read_str = [&]() -> std::string {
        if (remaining < sizeof(uint32_t))
            throw std::runtime_error("Truncated TvmFfiKernelPlugin serialization");
        uint32_t len = 0;
        std::memcpy(&len, p, sizeof(len));
        p += sizeof(len);
        remaining -= sizeof(len);
        if (static_cast<std::size_t>(len) > remaining)
            throw std::runtime_error("Truncated TvmFfiKernelPlugin string");
        std::string s(p, len);
        p += len;
        remaining -= len;
        return s;
    };
    kernel_name_ = read_str();
    shape_spec_ = read_str();
    if (remaining != 0)
        throw std::runtime_error("Trailing bytes in TvmFfiKernelPlugin serialization");
    parse_shape_spec();
    if (is_runtime_tvm_ffi_kernel_name(kernel_name_)) {
        bound_fn_ = active_tvm_ffi_binding(kernel_name_);
        if (bound_fn_ == nullptr) {
            throw std::runtime_error("No load-time TVM-FFI binding for slot '" + kernel_name_ +
                                     "'");
        }
    }
}

TvmFfiKernelPlugin::~TvmFfiKernelPlugin() = default;

void TvmFfiKernelPlugin::parse_shape_spec() {
    nlohmann::json j = nlohmann::json::parse(shape_spec_, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        j = nlohmann::json::object();
    }

    num_inputs_ = j.value("num_inputs", 1);
    num_outputs_ = j.value("num_outputs", 1);
    workspace_bytes_ = static_cast<int64_t>(j.value("workspace_bytes", 0));

    output_specs_.clear();
    if (j.contains("outputs") && j["outputs"].is_array()) {
        const auto& outputs_arr = j["outputs"];
        for (std::size_t i = 0; i < outputs_arr.size() && static_cast<int32_t>(i) < num_outputs_; ++i) {
            const auto& obj = outputs_arr[i];
            TvmFfiOutputSpec spec;
            if (obj.is_object()) {
                if (obj.contains("dims")) {
                    if (obj["dims"].is_string()) {
                        std::string dims_str = obj["dims"].get<std::string>();
                        if (dims_str.find("same_as_input_") == 0) {
                            try {
                                spec.same_as_input_index = static_cast<int32_t>(std::stoi(dims_str.substr(14)));
                            } catch (...) {
                                spec.same_as_input_index = -1;
                            }
                        } else {
                            spec.same_as_input_index = -1;
                        }
                    } else if (obj["dims"].is_array()) {
                        spec.same_as_input_index = -1;
                        for (std::size_t k = 0; k < obj["dims"].size() && k < 16; ++k) {
                            if (obj["dims"][k].is_number_integer()) {
                                spec.dims.push_back(obj["dims"][k].get<int32_t>());
                            }
                        }
                    } else {
                        spec.same_as_input_index = -1;
                    }
                } else {
                    spec.same_as_input_index = -1;
                }
                
                std::string dt = obj.value("dtype", "float32");
                if (dt == "bfloat16" || dt == "bf16") spec.dtype = 2;
                else if (dt == "float16" || dt == "half") spec.dtype = 1;
                else if (dt == "int32") spec.dtype = 3;
                else spec.dtype = 0;
            } else {
                spec.same_as_input_index = 0;
                spec.dtype = 0;
            }
            output_specs_.push_back(spec);
        }
    }
    
    while (static_cast<int32_t>(output_specs_.size()) < num_outputs_) {
        TvmFfiOutputSpec spec;
        spec.same_as_input_index = 0;
        spec.dtype = 0;
        output_specs_.push_back(spec);
    }
    
    extra_args_.clear();
    if (j.contains("extra_args") && j["extra_args"].is_array()) {
        for (const auto& obj : j["extra_args"]) {
            TvmFfiExtraArg arg;
            if (obj.is_object()) {
                std::string type_str = obj.value("type", "none");
                if (type_str == "int") {
                    arg.type_index = kTVMFFIInt;
                    if (obj.contains("value") && obj["value"].is_number()) {
                        arg.v_int = obj["value"].is_number_float() ? static_cast<int64_t>(obj["value"].get<double>()) : obj["value"].get<int64_t>();
                    } else {
                        arg.v_int = 0;
                    }
                } else if (type_str == "float") {
                    arg.type_index = kTVMFFIFloat;
                    if (obj.contains("value") && obj["value"].is_number()) {
                        arg.v_float = static_cast<double>(obj["value"].get<double>());
                    } else {
                        arg.v_float = 0.0;
                    }
                } else if (type_str == "ptr") {
                    arg.type_index = kTVMFFIOpaquePtr;
                } else {
                    arg.type_index = kTVMFFINone;
                }
            } else {
                arg.type_index = kTVMFFINone;
            }
            extra_args_.push_back(arg);
        }
    }
    if (num_inputs_ <= 0 || num_outputs_ <= 0 || workspace_bytes_ < 0 ||
        output_specs_.size() != static_cast<std::size_t>(num_outputs_)) {
        throw std::runtime_error("Invalid TvmFfiKernelPlugin shape specification");
    }
    for (const auto& output : output_specs_) {
        if (output.same_as_input_index < -1 || output.same_as_input_index >= num_inputs_) {
            throw std::runtime_error("TvmFfiKernelPlugin output input index is out of range");
        }
        for (int32_t dimension : output.dims) {
            if (dimension <= 0)
                throw std::runtime_error("TvmFfiKernelPlugin fixed dimensions must be positive");
        }
    }
}

// ---------------------------------------------------------------------------
// IPluginV2
// ---------------------------------------------------------------------------

char const* TvmFfiKernelPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* TvmFfiKernelPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t TvmFfiKernelPlugin::getNbOutputs() const noexcept {
    return num_outputs_;
}
int32_t TvmFfiKernelPlugin::initialize() noexcept {
    return 0;
}
void TvmFfiKernelPlugin::terminate() noexcept {}
void TvmFfiKernelPlugin::destroy() noexcept {
    delete this;
}
void TvmFfiKernelPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}
char const* TvmFfiKernelPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

size_t TvmFfiKernelPlugin::getSerializationSize() const noexcept {
    return 4 + kernel_name_.size() + 4 + shape_spec_.size();
}

void TvmFfiKernelPlugin::serialize(void* buffer) const noexcept {
    auto* p = static_cast<char*>(buffer);
    auto write_str = [&](const std::string& s) {
        uint32_t len = static_cast<uint32_t>(s.size());
        std::memcpy(p, &len, 4);
        p += 4;
        std::memcpy(p, s.data(), len);
        p += len;
    };
    write_str(kernel_name_);
    write_str(shape_spec_);
}

// ---------------------------------------------------------------------------
// IPluginV2Ext
// ---------------------------------------------------------------------------

nvinfer1::DataType TvmFfiKernelPlugin::getOutputDataType(int32_t index,
                                                         nvinfer1::DataType const* inputTypes,
                                                         int32_t) const noexcept {
    if (index < static_cast<int32_t>(output_specs_.size())) {
        const auto& spec = output_specs_[static_cast<std::size_t>(index)];
        if (spec.dtype == 2)
            return nvinfer1::DataType::kBF16;
        if (spec.dtype == 1)
            return nvinfer1::DataType::kHALF;
        if (spec.dtype == 3)
            return nvinfer1::DataType::kINT32;
        if (spec.same_as_input_index >= 0)
            return inputTypes[spec.same_as_input_index];
    }
    return nvinfer1::DataType::kFLOAT;
}

// ---------------------------------------------------------------------------
// IPluginV2DynamicExt
// ---------------------------------------------------------------------------

TvmFfiKernelPlugin* TvmFfiKernelPlugin::clone() const noexcept {
    try {
        auto p = std::make_unique<TvmFfiKernelPlugin>(kernel_name_, shape_spec_);
        p->namespace_ = namespace_;
        p->bound_fn_ = bound_fn_;
        return p.release();
    } catch (...) {
        return nullptr;
    }
}

nvinfer1::DimsExprs
TvmFfiKernelPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
                                        int32_t, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    if (outputIndex >= static_cast<int32_t>(output_specs_.size())) {
        return inputs[0];
    }
    const auto& spec = output_specs_[static_cast<std::size_t>(outputIndex)];
    if (spec.same_as_input_index >= 0) {
        return inputs[spec.same_as_input_index];
    }
    nvinfer1::DimsExprs out;
    out.nbDims = static_cast<int32_t>(spec.dims.size());
    for (int32_t d = 0; d < out.nbDims; ++d) {
        out.d[d] = exprBuilder.constant(spec.dims[static_cast<std::size_t>(d)]);
    }
    return out;
}

bool TvmFfiKernelPlugin::supportsFormatCombination(int32_t pos,
                                                   nvinfer1::PluginTensorDesc const* inOut, int32_t,
                                                   int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           (inOut[pos].type == nvinfer1::DataType::kBF16 ||
            inOut[pos].type == nvinfer1::DataType::kFLOAT ||
            inOut[pos].type == nvinfer1::DataType::kHALF ||
            inOut[pos].type == nvinfer1::DataType::kINT32);
}

void TvmFfiKernelPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                         nvinfer1::DynamicPluginTensorDesc const*,
                                         int32_t) noexcept {}

size_t TvmFfiKernelPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                            nvinfer1::PluginTensorDesc const*,
                                            int32_t) const noexcept {
    return static_cast<size_t>(workspace_bytes_);
}

namespace {

void report_tvm_ffi_error(const std::string& kernel_name) {
    TVMFFIObjectHandle err_obj = nullptr;
    TVMFFIErrorMoveFromRaised(&err_obj);
    if (err_obj != nullptr) {
        auto* cell = reinterpret_cast<const TVMFFIErrorCell*>(
            static_cast<const char*>(static_cast<const void*>(err_obj)) + sizeof(TVMFFIObject));
        if (cell->message.data != nullptr && cell->message.size > 0) {
            std::cerr << "[TvmFfiKernelPlugin] " << kernel_name << ": "
                      << std::string(cell->message.data,
                                     static_cast<std::size_t>(cell->message.size))
                      << '\n';
        }
        TVMFFIObjectDecRef(err_obj);
    }
}

constexpr int32_t kMaxTensorRank = 8;

// Keep explicit DLPack shape and contiguous-stride storage alive for the FFI
// call. CuTe's TVM-FFI wrapper reads both arrays.
bool fill_dl_tensor(DLTensor& t, int64_t* shape, int64_t* strides, void* data,
                    const nvinfer1::PluginTensorDesc& desc, int device_id) {
    if (desc.dims.nbDims < 0 || desc.dims.nbDims > kMaxTensorRank)
        return false;
    int64_t stride = 1;
    for (int32_t dim = desc.dims.nbDims - 1; dim >= 0; --dim) {
        if (desc.dims.d[dim] < 0)
            return false;
        shape[dim] = desc.dims.d[dim];
        strides[dim] = stride;
        stride *= shape[dim];
    }
    t.data = data;
    t.device = {kDLCUDA, device_id};
    t.ndim = desc.dims.nbDims;
    t.shape = shape;
    t.strides = strides;
    t.byte_offset = 0;
    if (desc.type == nvinfer1::DataType::kFLOAT)
        t.dtype = DLDataType{kDLFloat, 32, 1};
    else if (desc.type == nvinfer1::DataType::kBF16)
        t.dtype = DLDataType{kDLBfloat, 16, 1};
    else if (desc.type == nvinfer1::DataType::kINT32)
        t.dtype = DLDataType{kDLInt, 32, 1};
    else
        t.dtype = DLDataType{kDLFloat, 16, 1};
    return true;
}

// Bind tensor descriptors to DLTensor + TVMFFIAny argument arrays.
// Returns the next argument index after all bound tensors.
int32_t bind_tensors(DLTensor* dl_tensors, int64_t* shapes, int64_t* strides, TVMFFIAny* args,
                     nvinfer1::PluginTensorDesc const* descs, void const* const* buffers,
                     int32_t count, int32_t arg_idx, int device_id) {
    for (int32_t i = 0; i < count; ++i, ++arg_idx) {
        auto tidx = static_cast<std::size_t>(arg_idx);
        if (!fill_dl_tensor(dl_tensors[tidx], shapes + tidx * kMaxTensorRank,
                            strides + tidx * kMaxTensorRank, const_cast<void*>(buffers[i]),
                            descs[i], device_id))
            return -1;
        args[arg_idx].type_index = kTVMFFIDLTensorPtr;
        args[arg_idx].v_ptr = &dl_tensors[tidx];
    }
    return arg_idx;
}

// Bind the workspace tensor if present. Returns the next argument index.
int32_t bind_workspace_tensor(DLTensor* dl_tensors, TVMFFIAny* args, void* workspace,
                              int64_t* workspace_shape, int64_t* workspace_strides, int32_t arg_idx,
                              int device_id) {
    auto tidx = static_cast<std::size_t>(arg_idx);
    DLTensor& wt = dl_tensors[tidx];
    wt.data = workspace;
    wt.device = {kDLCUDA, device_id};
    wt.ndim = 1;
    wt.shape = workspace_shape;
    wt.strides = workspace_strides;
    wt.byte_offset = 0;
    wt.dtype = {kDLUInt, 8, 1};
    args[arg_idx].type_index = kTVMFFIDLTensorPtr;
    args[arg_idx].v_ptr = &wt;
    return arg_idx + 1;
}

// Append extra scalar/pointer args from the parsed shape_spec.
void append_extra_args(TVMFFIAny* args, int32_t base_idx,
                       const std::vector<TvmFfiExtraArg>& extra_args) {
    for (std::size_t i = 0; i < extra_args.size(); ++i) {
        auto idx = static_cast<std::size_t>(base_idx) + i;
        const auto& ea = extra_args[i];
        args[idx].type_index = ea.type_index;
        if (ea.type_index == kTVMFFIInt)
            args[idx].v_int64 = ea.v_int;
        else if (ea.type_index == kTVMFFIFloat)
            std::memcpy(&args[idx].v_int64, &ea.v_float, sizeof(double));
        else
            args[idx].v_ptr = nullptr;
    }
}

bool resolve_tvm_ffi_function(const std::string& kernel_name, TvmFfiBoundFunctionPtr* function) {
    if (*function != nullptr)
        return true;
    *function = resolve_global_tvm_ffi_function(kernel_name);
    if (*function != nullptr)
        return true;
    std::cerr << "[TvmFfiKernelPlugin] Failed to resolve kernel: " << kernel_name << '\n';
    return false;
}

int32_t invoke_tvm_ffi_function(void* function, TVMFFIAny* args, int32_t num_args, int device_id,
                                cudaStream_t stream, const std::string& kernel_name) {
    TVMFFIStreamHandle previous_stream = nullptr;
    if (TVMFFIEnvSetStream(kDLCUDA, device_id, reinterpret_cast<TVMFFIStreamHandle>(stream),
                           &previous_stream) != 0) {
        report_tvm_ffi_error(kernel_name);
        return -1;
    }

    TVMFFIAny result{};
    result.type_index = kTVMFFINone;
    const int call_status = TVMFFIFunctionCall(function, args, num_args, &result);
    if (call_status != 0)
        report_tvm_ffi_error(kernel_name);
    const int restore_status = TVMFFIEnvSetStream(kDLCUDA, device_id, previous_stream, nullptr);
    if (restore_status != 0)
        report_tvm_ffi_error(kernel_name);
    if (result.type_index >= kTVMFFIStaticObjectBegin && result.v_obj != nullptr)
        TVMFFIObjectDecRef(result.v_obj);
    return call_status == 0 && restore_status == 0 ? 0 : -1;
}

} // namespace

int32_t TvmFfiKernelPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                    nvinfer1::PluginTensorDesc const* outputDesc,
                                    void const* const* inputs, void* const* outputs,
                                    void* workspace, cudaStream_t stream) noexcept {
    if (!resolve_tvm_ffi_function(kernel_name_, &bound_fn_))
        return -1;

    // 2. Build argument array: [inputs..., workspace_tmp, outputs..., extra_args...]
    const bool has_workspace = workspace_bytes_ > 0;
    if (has_workspace && workspace == nullptr) {
        std::cerr << "[TvmFfiKernelPlugin] TensorRT did not provide configured workspace\n";
        return -1;
    }
    const int32_t total_tensors = num_inputs_ + (has_workspace ? 1 : 0) + num_outputs_;
    const int32_t total_args = total_tensors + static_cast<int32_t>(extra_args_.size());
    std::vector<DLTensor> dl_tensors(static_cast<std::size_t>(total_tensors));
    std::vector<int64_t> shapes(static_cast<std::size_t>(total_tensors) * kMaxTensorRank);
    std::vector<int64_t> strides(static_cast<std::size_t>(total_tensors) * kMaxTensorRank);
    std::vector<TVMFFIAny> args(static_cast<std::size_t>(total_args));

    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess)
        return -1;

    // Bind inputs, optional workspace, outputs, and extra args
    int32_t arg_idx = bind_tensors(dl_tensors.data(), shapes.data(), strides.data(), args.data(),
                                   inputDesc, inputs, num_inputs_, 0, device_id);
    if (arg_idx < 0)
        return -1;

    int64_t workspace_shape[1] = {workspace_bytes_};
    int64_t workspace_strides[1] = {1};
    if (has_workspace)
        arg_idx = bind_workspace_tensor(dl_tensors.data(), args.data(), workspace, workspace_shape,
                                        workspace_strides, arg_idx, device_id);

    auto* out_bufs = reinterpret_cast<void const* const*>(outputs);
    arg_idx = bind_tensors(dl_tensors.data(), shapes.data(), strides.data(), args.data(),
                           outputDesc, out_bufs, num_outputs_, arg_idx, device_id);
    if (arg_idx < 0)
        return -1;

    append_extra_args(args.data(), arg_idx, extra_args_);
    return invoke_tvm_ffi_function(bound_fn_->handle(), args.data(), total_args, device_id, stream,
                                   kernel_name_);
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI
