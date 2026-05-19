// TVM-FFI kernel bridge plugin implementation (IPluginV2DynamicExt).

#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI

#include "plugins/tvm_ffi_kernel_plugin.h"

#include "utils/json_helpers.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
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

TvmFfiOutputSpec parse_single_output_spec(const std::string& obj) {
    TvmFfiOutputSpec spec;
    std::string dims_str = extract_json_string(obj, "dims", "");
    if (dims_str.find("same_as_input_") == 0) {
        spec.same_as_input_index = static_cast<int32_t>(std::stoi(dims_str.substr(14)));
    } else {
        auto arr = extract_json_int_array(obj, "dims", 16);
        for (auto d : arr)
            spec.dims.push_back(d);
        spec.same_as_input_index = -1;
    }
    std::string dt = extract_json_string(obj, "dtype", "float32");
    spec.dtype = (dt == "float16" || dt == "half") ? 1 : 0;
    return spec;
}

std::pair<std::size_t, std::size_t> find_outputs_array_bounds(const std::string& s) {
    auto kp = s.find("\"outputs\"");
    if (kp == std::string::npos)
        return {std::string::npos, std::string::npos};
    auto a = s.find('[', kp);
    if (a == std::string::npos)
        return {std::string::npos, std::string::npos};
    return {a + 1, s.find(']', a)};
}

std::vector<TvmFfiOutputSpec> scan_output_objects(const std::string& s, std::size_t pos,
                                                  std::size_t end, int32_t max) {
    std::vector<TvmFfiOutputSpec> specs;
    while (pos < end && static_cast<int32_t>(specs.size()) < max) {
        auto os = s.find('{', pos);
        if (os == std::string::npos || os >= end)
            break;
        auto oe = s.find('}', os);
        if (oe == std::string::npos)
            break;
        specs.push_back(parse_single_output_spec(s.substr(os, oe - os + 1)));
        pos = oe + 1;
    }
    return specs;
}

std::vector<TvmFfiOutputSpec> parse_outputs_array(const std::string& s, int32_t n) {
    auto [pos, end] = find_outputs_array_bounds(s);
    if (pos == std::string::npos || end == std::string::npos) {
        std::vector<TvmFfiOutputSpec> defaults(static_cast<std::size_t>(n));
        for (auto& d : defaults) {
            d.same_as_input_index = 0;
            d.dtype = 0;
        }
        return defaults;
    }
    return scan_output_objects(s, pos, end, n);
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TvmFfiKernelPlugin::TvmFfiKernelPlugin(const std::string& kernel_name,
                                       const std::string& shape_spec)
    : kernel_name_(kernel_name), shape_spec_(shape_spec) {
    parse_shape_spec();
}

TvmFfiKernelPlugin::TvmFfiKernelPlugin(const void* data, size_t /*length*/) {
    auto* p = static_cast<const char*>(data);
    auto read_str = [&]() -> std::string {
        uint32_t len = 0;
        std::memcpy(&len, p, 4);
        p += 4;
        std::string s(p, len);
        p += len;
        return s;
    };
    kernel_name_ = read_str();
    shape_spec_ = read_str();
    parse_shape_spec();
}

namespace {

TvmFfiExtraArg parse_single_extra_arg(const std::string& obj) {
    TvmFfiExtraArg arg;
    std::string type_str = extract_json_string(obj, "type", "none");
    if (type_str == "int") {
        arg.type_index = kTVMFFIInt;
        arg.v_int = static_cast<int64_t>(extract_json_int(obj, "value", 0));
    } else if (type_str == "float") {
        arg.type_index = kTVMFFIFloat;
        arg.v_float = static_cast<double>(extract_json_float(obj, "value", 0.0f));
    } else if (type_str == "ptr") {
        arg.type_index = kTVMFFIOpaquePtr;
    } else {
        arg.type_index = kTVMFFINone;
    }
    return arg;
}

std::pair<std::size_t, std::size_t> find_extra_args_array_bounds(const std::string& spec) {
    auto key_pos = spec.find("\"extra_args\"");
    if (key_pos == std::string::npos)
        return {std::string::npos, std::string::npos};
    auto arr_start = spec.find('[', key_pos);
    if (arr_start == std::string::npos)
        return {std::string::npos, std::string::npos};
    auto arr_end = spec.find(']', arr_start);
    return {arr_start + 1, arr_end};
}

std::vector<TvmFfiExtraArg> parse_extra_args(const std::string& spec) {
    std::vector<TvmFfiExtraArg> args;
    auto [pos, arr_end] = find_extra_args_array_bounds(spec);
    if (pos == std::string::npos || arr_end == std::string::npos)
        return args;

    while (pos < arr_end) {
        auto os = spec.find('{', pos);
        if (os == std::string::npos || os >= arr_end)
            break;
        auto oe = spec.find('}', os);
        if (oe == std::string::npos)
            break;
        args.push_back(parse_single_extra_arg(spec.substr(os, oe - os + 1)));
        pos = oe + 1;
    }
    return args;
}

} // namespace

void TvmFfiKernelPlugin::parse_shape_spec() {
    num_inputs_ = extract_json_int(shape_spec_, "num_inputs", 1);
    num_outputs_ = extract_json_int(shape_spec_, "num_outputs", 1);
    workspace_bytes_ = static_cast<int64_t>(extract_json_int(shape_spec_, "workspace_bytes", 0));
    output_specs_ = parse_outputs_array(shape_spec_, num_outputs_);
    extra_args_ = parse_extra_args(shape_spec_);
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
        if (spec.dtype == 1)
            return nvinfer1::DataType::kHALF;
        if (spec.same_as_input_index >= 0)
            return inputTypes[spec.same_as_input_index];
    }
    return nvinfer1::DataType::kFLOAT;
}

// ---------------------------------------------------------------------------
// IPluginV2DynamicExt
// ---------------------------------------------------------------------------

TvmFfiKernelPlugin* TvmFfiKernelPlugin::clone() const noexcept {
    auto* p = new TvmFfiKernelPlugin(kernel_name_, shape_spec_);
    p->namespace_ = namespace_;
    return p;
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
           (inOut[pos].type == nvinfer1::DataType::kFLOAT ||
            inOut[pos].type == nvinfer1::DataType::kHALF);
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

// Fill a DLTensor from a TRT tensor descriptor.
void fill_dl_tensor(DLTensor& t, void* data, const nvinfer1::PluginTensorDesc& desc,
                    int device_id, int64_t* shape_storage) {
    for (int32_t i = 0; i < desc.dims.nbDims; ++i)
        shape_storage[i] = desc.dims.d[i];
    t.data = data;
    t.device = {kDLCUDA, device_id};
    t.ndim = desc.dims.nbDims;
    t.shape = shape_storage;
    t.strides = nullptr;
    t.byte_offset = 0;
    t.dtype = (desc.type == nvinfer1::DataType::kFLOAT) ? DLDataType{kDLFloat, 32, 1}
                                                        : DLDataType{kDLFloat, 16, 1};
}

// Bind tensor descriptors to DLTensor + TVMFFIAny argument arrays.
// Returns the next argument index after all bound tensors.
int32_t bind_tensors(DLTensor* dl_tensors, TVMFFIAny* args, nvinfer1::PluginTensorDesc const* descs,
                     void const* const* buffers, int32_t count, int32_t arg_idx, int device_id,
                     std::array<int64_t, 8>* shape_storage) {
    for (int32_t i = 0; i < count; ++i, ++arg_idx) {
        auto tidx = static_cast<std::size_t>(arg_idx);
        fill_dl_tensor(dl_tensors[tidx], const_cast<void*>(buffers[i]), descs[i], device_id,
                       shape_storage[tidx].data());
        args[arg_idx].type_index = kTVMFFIDLTensorPtr;
        args[arg_idx].v_ptr = &dl_tensors[tidx];
    }
    return arg_idx;
}

// Bind the workspace tensor if present. Returns the next argument index.
int32_t bind_workspace_tensor(DLTensor* dl_tensors, TVMFFIAny* args, void* workspace,
                              int64_t* workspace_shape, int32_t arg_idx, int device_id) {
    auto tidx = static_cast<std::size_t>(arg_idx);
    DLTensor& wt = dl_tensors[tidx];
    wt.data = workspace;
    wt.device = {kDLCUDA, device_id};
    wt.ndim = 1;
    wt.shape = workspace_shape;
    wt.strides = nullptr;
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
            args[idx].v_float64 = ea.v_float;
        else
            args[idx].v_ptr = nullptr;
    }
}

} // namespace

int32_t TvmFfiKernelPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                    nvinfer1::PluginTensorDesc const* outputDesc,
                                    void const* const* inputs, void* const* outputs,
                                    void* workspace, cudaStream_t stream) noexcept {
    // 1. Lazy-resolve the TVM-FFI function handle
    if (cached_fn_ == nullptr) {
        TVMFFIByteArray name_arr;
        name_arr.data = kernel_name_.c_str();
        name_arr.size = static_cast<int64_t>(kernel_name_.size());
        if (TVMFFIFunctionGetGlobal(&name_arr, &cached_fn_) != 0 || cached_fn_ == nullptr) {
            std::cerr << "[TvmFfiKernelPlugin] Failed to resolve kernel: " << kernel_name_ << '\n';
            return -1;
        }
    }

    // 2. Build argument array: [inputs..., workspace_tmp, outputs..., extra_args...]
    const bool has_workspace = (workspace_bytes_ > 0 && workspace != nullptr);
    const int32_t total_tensors = num_inputs_ + (has_workspace ? 1 : 0) + num_outputs_;
    const int32_t total_args = total_tensors + static_cast<int32_t>(extra_args_.size());
    std::vector<DLTensor> dl_tensors(static_cast<std::size_t>(total_tensors));
    std::vector<std::array<int64_t, 8>> shape_storage(static_cast<std::size_t>(total_tensors));
    std::vector<TVMFFIAny> args(static_cast<std::size_t>(total_args));

    int device_id = 0;
    cudaGetDevice(&device_id);

    // Bind inputs, optional workspace, outputs, and extra args
    int32_t arg_idx =
        bind_tensors(dl_tensors.data(), args.data(), inputDesc, inputs, num_inputs_, 0, device_id,
                     shape_storage.data());

    int64_t workspace_shape[1] = {workspace_bytes_};
    if (has_workspace)
        arg_idx = bind_workspace_tensor(dl_tensors.data(), args.data(), workspace, workspace_shape,
                                        arg_idx, device_id);

    auto* out_bufs = reinterpret_cast<void const* const*>(outputs);
    arg_idx = bind_tensors(dl_tensors.data(), args.data(), outputDesc, out_bufs, num_outputs_,
                           arg_idx, device_id, shape_storage.data());

    append_extra_args(args.data(), arg_idx, extra_args_);

    // 3. Call the TVM-FFI function on TensorRT's stream
    TVMFFIStreamHandle previous_stream = nullptr;
    if (TVMFFIEnvSetStream(kDLCUDA, device_id, stream, &previous_stream) != 0) {
        report_tvm_ffi_error(kernel_name_);
        return -1;
    }
    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    int ret = TVMFFIFunctionCall(cached_fn_, args.data(), total_args, &result);
    if (TVMFFIEnvSetStream(kDLCUDA, device_id, previous_stream, nullptr) != 0) {
        report_tvm_ffi_error(kernel_name_);
        return -1;
    }
    if (ret != 0) {
        report_tvm_ffi_error(kernel_name_);
        return -1;
    }
    return 0;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI
