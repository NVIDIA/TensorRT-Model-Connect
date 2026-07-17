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

// Product boundary: each registered pipeline is one recurrent step containing
// the exact dynamic memory-conditioning encoder followed by the fixed-shape
// mask decoder. Neither package is independently callable through TVM-FFI.
constexpr int32_t kInputCount = 10;
constexpr int32_t kPackedWidth = 288 * 288 + 256 + 1 + 1;
constexpr int32_t kTensorArgumentCount = kInputCount + 1;
constexpr int32_t kArgumentCount = kTensorArgumentCount + 1;
constexpr int32_t kSpatialTokens = 72 * 72;
constexpr int32_t kMaximumMemoryFrames = 10;
constexpr int32_t kMaximumPointers = 19;
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
    // The threaded loader records a completion event before returning a runner
    // to its pool.  Two runners permit independent SAM3 sessions to overlap on
    // distinct TensorRT streams without reusing AOTI workspace that is still in
    // flight.
    return std::make_unique<AotiLoader>(package_path, "model", false, kAotiRunnerCount,
                                        static_cast<c10::DeviceIndex>(device_id));
}

struct DevicePipeline {
    DevicePipeline(const std::string& encoder_path, const std::string& decoder_path,
                   int32_t device_id)
        : encoder(make_loader(encoder_path, device_id)),
          decoder(make_loader(decoder_path, device_id)) {}

    std::unique_ptr<AotiLoader> encoder;
    std::unique_ptr<AotiLoader> decoder;
};

struct Entry {
    Entry(std::string name, std::string encoder, std::string decoder, std::string encoder_hash,
          std::string decoder_hash, int32_t batch)
        : global_name(std::move(name)), encoder_path(std::move(encoder)),
          decoder_path(std::move(decoder)), encoder_sha256(std::move(encoder_hash)),
          decoder_sha256(std::move(decoder_hash)), batch_size(batch) {}

    DevicePipeline& pipeline_for_device(int32_t device_id) {
        std::lock_guard lock(loaders_mutex);
        const auto existing = loaders.find(device_id);
        if (existing != loaders.end())
            return *existing->second;
        auto pipeline = std::make_unique<DevicePipeline>(encoder_path, decoder_path, device_id);
        auto* pointer = pipeline.get();
        loaders.emplace(device_id, std::move(pipeline));
        return *pointer;
    }

    std::string global_name;
    std::string encoder_path;
    std::string decoder_path;
    std::string encoder_sha256;
    std::string decoder_sha256;
    int32_t batch_size{0};
    std::mutex loaders_mutex;
    std::unordered_map<int32_t, std::unique_ptr<DevicePipeline>> loaders;
};

class CudaDeviceScope {
  public:
    explicit CudaDeviceScope(int32_t target_device) {
        if (target_device < 0 || cudaGetDevice(&source_device_) != cudaSuccess)
            throw std::runtime_error("SAM3 tracker step received an invalid CUDA device");
        if (source_device_ != target_device) {
            if (cudaSetDevice(target_device) != cudaSuccess)
                throw std::runtime_error("SAM3 tracker step could not select its CUDA device");
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
                throw std::runtime_error("SAM3 tracker step could not restore its CUDA device");
            changed_ = false;
        }
    }

  private:
    int32_t source_device_{0};
    bool changed_{false};
};

bool valid_content_addressed_name(std::string_view name, int32_t batch_size) {
    const std::string prefix = batch_size == 1 ? "trtmc.sam3.tracker_step.b1.split_aoti."
                                               : "trtmc.sam3.tracker_step.b2.split_aoti.";
    constexpr std::size_t kDigestCharacters = 20;
    if (!name.starts_with(prefix) || name.size() != prefix.size() + kDigestCharacters)
        return false;
    return std::all_of(name.begin() + static_cast<std::ptrdiff_t>(prefix.size()), name.end(),
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
        throw std::runtime_error("SAM3 tracker step does not support vector DLPack dtypes");
    if (dtype.code == kDLFloat && dtype.bits == 32)
        return at::kFloat;
    if (dtype.code == kDLFloat && dtype.bits == 16)
        return at::kHalf;
    if (dtype.code == kDLBfloat && dtype.bits == 16)
        return at::kBFloat16;
    if (dtype.code == kDLInt && dtype.bits == 32)
        return at::kInt;
    throw std::runtime_error("unsupported SAM3 tracker step DLPack dtype");
}

int64_t tensor_numel(const DLTensor& tensor) {
    int64_t result = 1;
    for (int32_t dimension = 0; dimension < tensor.ndim; ++dimension) {
        if (tensor.shape[dimension] <= 0)
            throw std::runtime_error("SAM3 tracker step received a non-positive extent");
        result *= tensor.shape[dimension];
    }
    return result;
}

bool exact_shape(const DLTensor& tensor, std::initializer_list<int64_t> expected) {
    if (tensor.ndim != static_cast<int32_t>(expected.size()) || tensor.shape == nullptr)
        return false;
    return std::equal(expected.begin(), expected.end(), tensor.shape);
}

bool ranged_shape(const DLTensor& tensor, std::initializer_list<int64_t> expected,
                  int32_t dynamic_index, int64_t maximum) {
    if (tensor.ndim != static_cast<int32_t>(expected.size()) || tensor.shape == nullptr)
        return false;
    std::size_t index = 0;
    for (const int64_t dimension : expected) {
        const int64_t actual = tensor.shape[index];
        if (static_cast<int32_t>(index) == dynamic_index) {
            if (actual < 1 || actual > maximum)
                return false;
        } else if (actual != dimension) {
            return false;
        }
        ++index;
    }
    return true;
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
            throw std::runtime_error("SAM3 tracker step tensor storage violates its contract");
        const bool integer_input = index == 6 || index == 8 || index == 9;
        const bool valid_dtype = integer_input ? dtype_is(*tensors[index], kDLInt, 32)
                                               : dtype_is(*tensors[index], kDLFloat, 32);
        if (!valid_dtype)
            throw std::runtime_error("SAM3 tracker step tensor dtype violates its contract");
    }
    if (!exact_shape(*tensors[0], {1, 32, 288, 288}) ||
        !exact_shape(*tensors[1], {1, 64, 144, 144}) ||
        !exact_shape(*tensors[2], {1, 256, 72, 72}) ||
        !exact_shape(*tensors[3], {1, 256, 72, 72}) ||
        !ranged_shape(*tensors[4], {batch_size, 1, kSpatialTokens, 64}, 1, kMaximumMemoryFrames) ||
        !ranged_shape(*tensors[5], {batch_size, 1, kSpatialTokens, 64}, 1, kMaximumMemoryFrames) ||
        !ranged_shape(*tensors[6], {batch_size, 1}, 1, kMaximumMemoryFrames) ||
        !ranged_shape(*tensors[7], {batch_size, 1, 256}, 1, kMaximumPointers) ||
        !ranged_shape(*tensors[8], {batch_size, 1}, 1, kMaximumPointers) ||
        !exact_shape(*tensors[9], {1}) || !exact_shape(*tensors[10], {batch_size, kPackedWidth}))
        throw std::runtime_error("SAM3 tracker step tensor shape violates its contract");
    const int64_t memory_frames = tensors[4]->shape[1];
    const int64_t pointer_count = tensors[7]->shape[1];
    if (tensors[5]->shape[1] != memory_frames || tensors[6]->shape[1] != memory_frames ||
        tensors[8]->shape[1] != pointer_count)
        throw std::runtime_error("SAM3 tracker step coupled M/P dimensions disagree");
}

at::Tensor wrap_tensor(DLTensor& tensor) {
    if (tensor.device.device_type != kDLCUDA)
        throw std::runtime_error("SAM3 tracker step tensor is not on CUDA");
    if (tensor.ndim < 0 || (tensor.ndim > 0 && tensor.shape == nullptr))
        throw std::runtime_error("SAM3 tracker step tensor has invalid dimensions");
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
        throw std::runtime_error("SAM3 tracker step destination must be CUDA FP32");
    const int64_t expected_numel = static_cast<int64_t>(batch_size) * kPackedWidth;
    if (!source.is_cuda() || source.scalar_type() != at::kFloat || !source.is_contiguous() ||
        source.get_device() != destination.device.device_id || source.numel() != expected_numel ||
        tensor_numel(destination) != expected_numel)
        throw std::runtime_error("SAM3 AOTI packed output violates the TensorRT contract");
    auto* destination_data = static_cast<std::byte*>(destination.data) + destination.byte_offset;
    const auto bytes = static_cast<std::size_t>(expected_numel) * sizeof(float);
    if (cudaMemcpyAsync(destination_data, source.const_data_ptr(), bytes, cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess)
        throw std::runtime_error("SAM3 AOTI packed output copy failed");
}

cudaStream_t validate_stream_argument(const TVMFFIAny& argument, int32_t device_id) {
    if (argument.type_index != kTVMFFIOpaquePtr)
        throw std::runtime_error("SAM3 tracker step is missing its TensorRT stream argument");
    auto stream = reinterpret_cast<cudaStream_t>(argument.v_ptr);
    auto environment_stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device_id));
    if (environment_stream != stream)
        throw std::runtime_error("SAM3 tracker step TVM-FFI stream handoff is inconsistent");
    return stream;
}

int tracker_step_callback(void* self, const TVMFFIAny* arguments, int32_t argument_count,
                          TVMFFIAny* result) {
    try {
        if (arguments == nullptr || result == nullptr || argument_count != kArgumentCount)
            throw std::runtime_error("SAM3 tracker step callback received an invalid ABI");
        std::array<DLTensor*, kTensorArgumentCount> tensors{};
        for (int32_t index = 0; index < kTensorArgumentCount; ++index) {
            tensors[static_cast<std::size_t>(index)] = dl_tensor(arguments[index]);
            if (tensors[static_cast<std::size_t>(index)] == nullptr)
                throw std::runtime_error("SAM3 tracker step argument is not a DLTensor");
        }
        const int32_t device_id = tensors.front()->device.device_id;
        for (const auto* tensor : tensors) {
            if (tensor->device.device_type != kDLCUDA || tensor->device.device_id != device_id)
                throw std::runtime_error("SAM3 tracker step tensors must share one CUDA device");
        }
        auto& entry = *static_cast<Entry*>(self);
        validate_tensor_contract(tensors, entry.batch_size);
        CudaDeviceScope device_scope(device_id);
        const auto stream = validate_stream_argument(arguments[kTensorArgumentCount], device_id);

        {
            auto& pipeline = entry.pipeline_for_device(device_id);
            std::vector<at::Tensor> wrapped_inputs;
            wrapped_inputs.reserve(kInputCount);
            for (int32_t index = 0; index < kInputCount; ++index)
                wrapped_inputs.push_back(wrap_tensor(*tensors[static_cast<std::size_t>(index)]));

            // Encoder ABI: feature_2, position_2, memory features/positions,
            // temporal offsets, pointers, pointer offsets, maximum pointers.
            std::vector<at::Tensor> encoder_inputs(wrapped_inputs.begin() + 2,
                                                   wrapped_inputs.end());
            auto conditioned =
                pipeline.encoder->run(encoder_inputs, reinterpret_cast<void*>(stream));
            if (conditioned.size() != 1)
                throw std::runtime_error("SAM3 tracker encoder returned the wrong arity");

            // Decoder ABI: high-resolution feature levels followed by the
            // conditioned feature. Both AOTI calls execute on TensorRT's stream.
            std::vector<at::Tensor> decoder_inputs{wrapped_inputs[0], wrapped_inputs[1],
                                                   conditioned.front()};
            auto outputs = pipeline.decoder->run(decoder_inputs, reinterpret_cast<void*>(stream));
            if (outputs.size() != 1)
                throw std::runtime_error("SAM3 tracker decoder returned the wrong arity");
            copy_packed_output(outputs.front(), *tensors.back(), entry.batch_size, stream);
        }
        device_scope.restore();
        result->type_index = kTVMFFINone;
        result->v_int64 = 0;
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[sam3-tracker-step-aoti] %s\n", error.what());
        TVMFFIErrorSetRaisedFromCStr("RuntimeError", error.what());
        return -1;
    } catch (...) {
        constexpr const char* message = "unknown SAM3 tracker-step AOTI failure";
        std::fprintf(stderr, "[sam3-tracker-step-aoti] %s\n", message);
        TVMFFIErrorSetRaisedFromCStr("RuntimeError", message);
        return -1;
    }
}

std::mutex registry_mutex;
// Every published TVM function retains a raw pointer to one entry. Keep all
// generations alive: replacing a process-global callback must not invalidate
// an execution context that already retained the previous function handle. The
// registry intentionally has process lifetime: destroying AOTI CUDA events
// during static teardown can run after the CUDA driver has already shut down.
std::vector<std::unique_ptr<Entry>>& retained_entries() {
    static auto* value = new std::vector<std::unique_ptr<Entry>>;
    return *value;
}

} // namespace

extern "C" int
trtmc_sam3_tracker_step_register_pipeline(const char* global_name, const char* encoder_path,
                                          const char* decoder_path, const char* encoder_sha256,
                                          const char* decoder_sha256, int32_t batch_size) noexcept {
    try {
        if (global_name == nullptr || global_name[0] == '\0' || encoder_path == nullptr ||
            encoder_path[0] == '\0' || decoder_path == nullptr || decoder_path[0] == '\0' ||
            encoder_sha256 == nullptr || decoder_sha256 == nullptr ||
            (batch_size != 1 && batch_size != 2) ||
            !valid_content_addressed_name(global_name, batch_size) ||
            !valid_sha256(encoder_sha256) || !valid_sha256(decoder_sha256))
            return -1;
        std::lock_guard lock(registry_mutex);
        auto& entries = retained_entries();
        for (const auto& existing : entries) {
            if (existing->global_name == global_name &&
                (existing->encoder_sha256 != encoder_sha256 ||
                 existing->decoder_sha256 != decoder_sha256 ||
                 existing->batch_size != batch_size)) {
                return -2;
            }
        }

        auto entry = std::make_unique<Entry>(global_name, encoder_path, decoder_path,
                                             encoder_sha256, decoder_sha256, batch_size);
        auto* entry_pointer = entry.get();
        TVMFFIObjectHandle function = nullptr;
        if (TVMFFIFunctionCreate(entry_pointer, tracker_step_callback, nullptr, &function) != 0)
            return -3;
        const TVMFFIByteArray name{global_name, std::strlen(global_name)};
        // A build and a subsequent same-process bundle load intentionally
        // materialize identical packages at different paths. Replace the
        // process-global lookup so newly deserialized TensorRT contexts bind
        // the bundle-owned paths. Existing contexts retain their old function
        // references and entry generations safely.
        const int set_status = TVMFFIFunctionSetGlobal(&name, function, 1);
        TVMFFIObjectDecRef(function);
        if (set_status != 0)
            return -4;
        entries.push_back(std::move(entry));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[sam3-tracker-step-aoti] registration failed: %s\n", error.what());
        return -5;
    }
}

extern "C" uint64_t trtmc_sam3_tracker_step_aoti_abi_version() noexcept {
    return aoti_torch_abi_version();
}

extern "C" const char* trtmc_sam3_tracker_step_torch_version() noexcept {
    return SAM3_TORCH_VERSION;
}

extern "C" const char* trtmc_sam3_tracker_step_tvm_ffi_version() noexcept {
    return SAM3_TVM_FFI_VERSION;
}

extern "C" const char* trtmc_sam3_tracker_step_tensorrt_version() noexcept {
    return SAM3_TENSORRT_VERSION;
}

extern "C" int32_t trtmc_sam3_tracker_step_torch_cxx11_abi() noexcept {
    return SAM3_TORCH_CXX11_ABI;
}
