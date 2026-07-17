/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/ATen.h>
#include <algorithm>
#include <array>
#include <c10/core/Device.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int32_t kInputCount = 4;
constexpr int32_t kTensorArgumentCount = kInputCount + 1;
constexpr int32_t kArgumentCount = kTensorArgumentCount + 1;
constexpr int32_t kSpatialTokens = 72 * 72;
constexpr int32_t kMemoryChannels = 64;
constexpr std::size_t kAotiRunnerCount = 2;

#ifndef SAM3_TORCH_VERSION
#define SAM3_TORCH_VERSION "unknown"
#endif
#ifndef SAM3_TVM_FFI_VERSION
#define SAM3_TVM_FFI_VERSION "unknown"
#endif
#ifndef SAM3_TENSORRT_VERSION
#define SAM3_TENSORRT_VERSION "unknown"
#endif
#ifndef SAM3_TORCH_CXX11_ABI
#define SAM3_TORCH_CXX11_ABI -1
#endif

using AotiLoader = torch::inductor::AOTIModelPackageLoader;

std::unique_ptr<AotiLoader> make_loader(const std::string& package_path, int32_t device_id) {
    return std::make_unique<AotiLoader>(package_path, "model", false, kAotiRunnerCount,
                                        static_cast<c10::DeviceIndex>(device_id));
}

struct Entry {
    Entry(std::string name, std::string package, std::string package_hash,
          std::string package_policy, int32_t batch)
        : global_name(std::move(name)), package_path(std::move(package)),
          package_sha256(std::move(package_hash)), policy(std::move(package_policy)),
          batch_size(batch) {}

    AotiLoader& loader_for_device(int32_t device_id) {
        std::lock_guard lock(loaders_mutex);
        const auto existing = loaders.find(device_id);
        if (existing != loaders.end())
            return *existing->second;
        auto loader = make_loader(package_path, device_id);
        auto* pointer = loader.get();
        loaders.emplace(device_id, std::move(loader));
        return *pointer;
    }

    std::string global_name;
    std::string package_path;
    std::string package_sha256;
    std::string policy;
    int32_t batch_size{0};
    std::mutex loaders_mutex;
    std::unordered_map<int32_t, std::unique_ptr<AotiLoader>> loaders;
};

class CudaDeviceScope {
  public:
    explicit CudaDeviceScope(int32_t target_device) {
        if (target_device < 0 || cudaGetDevice(&source_device_) != cudaSuccess)
            throw std::runtime_error("SAM3 tracker memory received an invalid CUDA device");
        if (source_device_ != target_device) {
            if (cudaSetDevice(target_device) != cudaSuccess)
                throw std::runtime_error("SAM3 tracker memory could not select its CUDA device");
            changed_ = true;
        }
    }

    CudaDeviceScope(const CudaDeviceScope&) = delete;
    CudaDeviceScope& operator=(const CudaDeviceScope&) = delete;

    ~CudaDeviceScope() {
        if (changed_)
            (void)cudaSetDevice(source_device_);
    }

    void restore() {
        if (changed_) {
            if (cudaSetDevice(source_device_) != cudaSuccess)
                throw std::runtime_error("SAM3 tracker memory could not restore its CUDA device");
            changed_ = false;
        }
    }

  private:
    int32_t source_device_{0};
    bool changed_{false};
};

bool valid_policy(std::string_view policy) {
    return policy == "soft" || policy == "hard";
}

bool valid_content_addressed_name(std::string_view name, std::string_view policy,
                                  int32_t batch_size) {
    if (!valid_policy(policy) || (batch_size != 1 && batch_size != 2))
        return false;
    const std::string prefix = "trtmc.sam3.tracker_memory." + std::string(policy) + ".b" +
                               std::to_string(batch_size) + ".fixed.";
    constexpr std::size_t kDigestCharacters = 20;
    return name.starts_with(prefix) && name.size() == prefix.size() + kDigestCharacters &&
           std::all_of(name.begin() + static_cast<std::ptrdiff_t>(prefix.size()), name.end(),
                       [](unsigned char character) {
                           return (character >= '0' && character <= '9') ||
                                  (character >= 'a' && character <= 'f');
                       });
}

bool valid_sha256(std::string_view digest) {
    return digest.size() == 64 &&
           std::all_of(digest.begin(), digest.end(), [](unsigned char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

DLTensor* dl_tensor(const TVMFFIAny& argument) {
    if (argument.type_index == kTVMFFITensor)
        return TVMFFITensorGetDLTensorPtr(argument.v_obj);
    if (argument.type_index == kTVMFFIDLTensorPtr)
        return static_cast<DLTensor*>(argument.v_ptr);
    return nullptr;
}

at::ScalarType at_dtype(const DLDataType& dtype) {
    if (dtype.lanes != 1)
        throw std::runtime_error("SAM3 tracker memory does not support vector DLPack dtypes");
    if (dtype.code == kDLFloat && dtype.bits == 32)
        return at::kFloat;
    if (dtype.code == kDLInt && dtype.bits == 32)
        return at::kInt;
    throw std::runtime_error("unsupported SAM3 tracker-memory DLPack dtype");
}

int64_t tensor_numel(const DLTensor& tensor) {
    int64_t result = 1;
    for (int32_t dimension = 0; dimension < tensor.ndim; ++dimension) {
        if (tensor.shape[dimension] <= 0)
            throw std::runtime_error("SAM3 tracker memory received a non-positive extent");
        result *= tensor.shape[dimension];
    }
    return result;
}

bool exact_shape(const DLTensor& tensor, std::initializer_list<int64_t> expected) {
    return tensor.ndim == static_cast<int32_t>(expected.size()) && tensor.shape != nullptr &&
           std::equal(expected.begin(), expected.end(), tensor.shape);
}

bool output_shape_matches(const DLTensor& tensor, int32_t batch_size) {
    if (batch_size == 1)
        return exact_shape(tensor, {2, kSpatialTokens, 1, kMemoryChannels});
    if (batch_size == 2)
        return exact_shape(tensor, {2, 2, kSpatialTokens, kMemoryChannels});
    return false;
}

bool dtype_is(const DLTensor& tensor, uint8_t code, uint8_t bits) {
    return tensor.dtype.code == code && tensor.dtype.bits == bits && tensor.dtype.lanes == 1;
}

bool has_contiguous_strides(const DLTensor& tensor) {
    if (tensor.strides == nullptr)
        return true;
    int64_t expected_stride = 1;
    for (int32_t index = tensor.ndim - 1; index >= 0; --index) {
        if (tensor.strides[index] != expected_stride)
            return false;
        expected_stride *= tensor.shape[index];
    }
    return true;
}

void validate_tensor_contract(const std::array<DLTensor*, kTensorArgumentCount>& tensors,
                              int32_t batch_size) {
    for (int32_t index = 0; index < kTensorArgumentCount; ++index) {
        if (tensors[index]->data == nullptr || tensors[index]->ndim <= 0 ||
            tensors[index]->shape == nullptr || !has_contiguous_strides(*tensors[index]))
            throw std::runtime_error("SAM3 tracker-memory tensor storage violates its contract");
        const bool integer_input = index == 3;
        const bool valid_dtype = integer_input ? dtype_is(*tensors[index], kDLInt, 32)
                                               : dtype_is(*tensors[index], kDLFloat, 32);
        if (!valid_dtype)
            throw std::runtime_error("SAM3 tracker-memory tensor dtype violates its contract");
    }
    if (!exact_shape(*tensors[0], {1, 256, 72, 72}) ||
        !exact_shape(*tensors[1], {batch_size, 1, 288, 288}) ||
        !exact_shape(*tensors[2], {batch_size, 1}) || !exact_shape(*tensors[3], {batch_size, 1}) ||
        !output_shape_matches(*tensors[4], batch_size))
        throw std::runtime_error("SAM3 tracker-memory tensor shape violates its contract");
}

at::Tensor wrap_tensor(DLTensor& tensor) {
    if (tensor.device.device_type != kDLCUDA)
        throw std::runtime_error("SAM3 tracker-memory tensor is not on CUDA");
    if (tensor.ndim < 0 || (tensor.ndim > 0 && tensor.shape == nullptr))
        throw std::runtime_error("SAM3 tracker-memory tensor has invalid dimensions");
    auto* data = static_cast<std::byte*>(tensor.data) + tensor.byte_offset;
    auto options = at::TensorOptions()
                       .dtype(at_dtype(tensor.dtype))
                       .device(at::Device(at::kCUDA, tensor.device.device_id));
    const at::IntArrayRef sizes(tensor.shape, tensor.ndim);
    if (tensor.strides != nullptr)
        return at::from_blob(
            data, sizes, at::IntArrayRef(tensor.strides, tensor.ndim), [](void*) {}, options);
    return at::from_blob(data, sizes, [](void*) {}, options);
}

void copy_packed_output(const at::Tensor& source, DLTensor& destination, int32_t batch_size,
                        cudaStream_t stream) {
    if (destination.device.device_type != kDLCUDA || destination.dtype.code != kDLFloat ||
        destination.dtype.bits != 32 || destination.dtype.lanes != 1)
        throw std::runtime_error("SAM3 tracker-memory destination must be CUDA FP32");
    const int64_t expected_numel =
        static_cast<int64_t>(2) * batch_size * kSpatialTokens * kMemoryChannels;
    if (!source.is_cuda() || source.scalar_type() != at::kFloat || !source.is_contiguous() ||
        source.get_device() != destination.device.device_id || source.numel() != expected_numel ||
        tensor_numel(destination) != expected_numel)
        throw std::runtime_error("SAM3 memory AOTI packed output violates the TensorRT contract");
    const std::vector<int64_t> expected_shape =
        batch_size == 1 ? std::vector<int64_t>{2, kSpatialTokens, 1, kMemoryChannels}
                        : std::vector<int64_t>{2, 2, kSpatialTokens, kMemoryChannels};
    if (source.sizes().vec() != expected_shape)
        throw std::runtime_error("SAM3 memory AOTI packed output has the wrong shape");
    auto* destination_data = static_cast<std::byte*>(destination.data) + destination.byte_offset;
    const auto bytes = static_cast<std::size_t>(expected_numel) * sizeof(float);
    if (cudaMemcpyAsync(destination_data, source.const_data_ptr(), bytes, cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess)
        throw std::runtime_error("SAM3 memory AOTI packed output copy failed");
}

cudaStream_t validate_stream_argument(const TVMFFIAny& argument, int32_t device_id) {
    if (argument.type_index != kTVMFFIOpaquePtr)
        throw std::runtime_error("SAM3 tracker memory is missing its TensorRT stream argument");
    auto stream = reinterpret_cast<cudaStream_t>(argument.v_ptr);
    auto environment_stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device_id));
    if (environment_stream != stream)
        throw std::runtime_error("SAM3 tracker-memory TVM-FFI stream handoff is inconsistent");
    return stream;
}

int tracker_memory_callback(void* self, const TVMFFIAny* arguments, int32_t argument_count,
                            TVMFFIAny* result) {
    try {
        if (arguments == nullptr || result == nullptr || argument_count != kArgumentCount)
            throw std::runtime_error("SAM3 tracker-memory callback received an invalid ABI");
        std::array<DLTensor*, kTensorArgumentCount> tensors{};
        for (int32_t index = 0; index < kTensorArgumentCount; ++index) {
            tensors[static_cast<std::size_t>(index)] = dl_tensor(arguments[index]);
            if (tensors[static_cast<std::size_t>(index)] == nullptr)
                throw std::runtime_error("SAM3 tracker-memory argument is not a DLTensor");
        }
        const int32_t device_id = tensors.front()->device.device_id;
        for (const auto* tensor : tensors) {
            if (tensor->device.device_type != kDLCUDA || tensor->device.device_id != device_id)
                throw std::runtime_error("SAM3 tracker-memory tensors must share one CUDA device");
        }
        auto& entry = *static_cast<Entry*>(self);
        validate_tensor_contract(tensors, entry.batch_size);
        CudaDeviceScope device_scope(device_id);
        const auto stream = validate_stream_argument(arguments[kTensorArgumentCount], device_id);

        std::vector<at::Tensor> wrapped_inputs;
        wrapped_inputs.reserve(kInputCount);
        for (int32_t index = 0; index < kInputCount; ++index)
            wrapped_inputs.push_back(wrap_tensor(*tensors[static_cast<std::size_t>(index)]));
        auto outputs =
            entry.loader_for_device(device_id).run(wrapped_inputs, reinterpret_cast<void*>(stream));
        if (outputs.size() != 1)
            throw std::runtime_error("SAM3 tracker-memory AOTI package returned the wrong arity");
        copy_packed_output(outputs.front(), *tensors.back(), entry.batch_size, stream);

        device_scope.restore();
        result->type_index = kTVMFFINone;
        result->v_int64 = 0;
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[sam3-tracker-memory-aoti] %s\n", error.what());
        TVMFFIErrorSetRaisedFromCStr("RuntimeError", error.what());
        return -1;
    } catch (...) {
        constexpr const char* message = "unknown SAM3 tracker-memory AOTI failure";
        std::fprintf(stderr, "[sam3-tracker-memory-aoti] %s\n", message);
        TVMFFIErrorSetRaisedFromCStr("RuntimeError", message);
        return -1;
    }
}

std::mutex registry_mutex;
// Process-global TVM functions retain raw Entry pointers. Keep every
// generation alive so a newly materialized bundle can replace the global
// lookup without invalidating already-created TensorRT contexts. The registry
// intentionally has process lifetime: destroying AOTI CUDA events during
// static teardown can run after the CUDA driver has already shut down.
std::vector<std::unique_ptr<Entry>>& retained_entries() {
    static auto* value = new std::vector<std::unique_ptr<Entry>>;
    return *value;
}

} // namespace

extern "C" int trtmc_sam3_tracker_memory_register_package(const char* global_name,
                                                          const char* package_path,
                                                          const char* package_sha256,
                                                          const char* policy,
                                                          int32_t batch_size) noexcept {
    try {
        if (global_name == nullptr || global_name[0] == '\0' || package_path == nullptr ||
            package_path[0] == '\0' || package_sha256 == nullptr || policy == nullptr ||
            !valid_content_addressed_name(global_name, policy, batch_size) ||
            !valid_sha256(package_sha256))
            return -1;
        std::lock_guard lock(registry_mutex);
        auto& entries = retained_entries();
        for (const auto& existing : entries) {
            if (existing->global_name == global_name &&
                (existing->package_sha256 != package_sha256 || existing->policy != policy ||
                 existing->batch_size != batch_size))
                return -2;
        }

        auto entry =
            std::make_unique<Entry>(global_name, package_path, package_sha256, policy, batch_size);
        auto* entry_pointer = entry.get();
        TVMFFIObjectHandle function = nullptr;
        if (TVMFFIFunctionCreate(entry_pointer, tracker_memory_callback, nullptr, &function) != 0)
            return -3;
        const TVMFFIByteArray name{global_name, std::strlen(global_name)};
        const int set_status = TVMFFIFunctionSetGlobal(&name, function, 1);
        TVMFFIObjectDecRef(function);
        if (set_status != 0)
            return -4;
        entries.push_back(std::move(entry));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[sam3-tracker-memory-aoti] registration failed: %s\n", error.what());
        return -5;
    }
}

extern "C" uint64_t trtmc_sam3_tracker_memory_aoti_abi_version() noexcept {
    return aoti_torch_abi_version();
}

extern "C" const char* trtmc_sam3_tracker_memory_torch_version() noexcept {
    return SAM3_TORCH_VERSION;
}

extern "C" const char* trtmc_sam3_tracker_memory_tvm_ffi_version() noexcept {
    return SAM3_TVM_FFI_VERSION;
}

extern "C" const char* trtmc_sam3_tracker_memory_tensorrt_version() noexcept {
    return SAM3_TENSORRT_VERSION;
}

extern "C" int32_t trtmc_sam3_tracker_memory_torch_cxx11_abi() noexcept {
    return SAM3_TORCH_CXX11_ABI;
}
