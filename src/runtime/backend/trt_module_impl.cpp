/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trt_module_impl.h"

#include "runtime/core/trt_common.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

bool engine_has_io_tensor(nvinfer1::ICudaEngine* engine, const std::string& expected_name) {
    for (int32_t index = 0; index < engine->getNbIOTensors(); ++index) {
        const char* name = engine->getIOTensorName(index);
        if (name != nullptr && expected_name == name)
            return true;
    }
    return false;
}

bool runtime_memory_dtype_supported(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kINT8:
        return true;
    default:
        return false;
    }
}

void allocate_host_output_staging(
    std::unordered_map<std::string, std::vector<uint8_t>>& host_output_staging,
    const std::string& name, std::size_t nbytes, bool is_external) {
    if (nbytes > 0 && !is_external)
        host_output_staging[name].resize(nbytes);
}

void replace_host_output_staging(
    std::unordered_map<std::string, std::vector<uint8_t>>& host_output_staging,
    const std::string& name, std::size_t nbytes) {
    const auto found = host_output_staging.find(name);
    if (found != host_output_staging.end() && found->second.size() == nbytes)
        return;
    if (nbytes == 0) {
        host_output_staging.erase(name);
        return;
    }
    std::vector<uint8_t> exact_staging(nbytes);
    host_output_staging[name].swap(exact_staging);
}

bool is_power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

void validate_versioned_struct(uint32_t struct_size, uint32_t expected_size, uint32_t api_version,
                               const char* type_name) {
    if (api_version != kRuntimeMemoryBackendApiVersionCurrent) {
        throw std::invalid_argument(std::string(type_name) + " has unsupported api_version " +
                                    std::to_string(api_version));
    }
    if (struct_size < expected_size) {
        throw std::invalid_argument(std::string(type_name) + " has struct_size " +
                                    std::to_string(struct_size) + "; expected at least " +
                                    std::to_string(expected_size));
    }
}

int32_t current_cuda_device() {
    int device = -1;
    const auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDevice failed: ") +
                                 cudaGetErrorString(status));
    }
    return static_cast<int32_t>(device);
}

int32_t validate_cuda_allocation(void* pointer, std::size_t alignment, int32_t expected_device,
                                 const std::string& description) {
    if (pointer == nullptr)
        throw std::invalid_argument(description + " pointer is null");
    if (alignment < kRuntimeMemoryCudaAlignmentV1 || !is_power_of_two(alignment)) {
        throw std::invalid_argument(description + " alignment must be a power of two >= " +
                                    std::to_string(kRuntimeMemoryCudaAlignmentV1));
    }
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0) {
        throw std::invalid_argument(description + " pointer does not satisfy " +
                                    std::to_string(alignment) + "-byte alignment");
    }
    if (expected_device < 0)
        throw std::invalid_argument(description + " device must be specified");

    cudaPointerAttributes attributes{};
    const auto status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        // Clear the sticky error produced by querying a non-CUDA pointer.
        (void)cudaGetLastError();
        throw std::invalid_argument(description + " is not a CUDA allocation");
    }
#if CUDART_VERSION >= 10000
    if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged)
#else
    if (attributes.memoryType != cudaMemoryTypeDevice)
#endif
        throw std::invalid_argument(description + " is not device-accessible memory");
    if (attributes.device != expected_device) {
        throw std::invalid_argument(description + " belongs to CUDA device " +
                                    std::to_string(attributes.device) + ", not " +
                                    std::to_string(expected_device));
    }
    return attributes.device;
}

RuntimeMemoryBindingV1 shape_as_binding(const RuntimeMemoryShapeV1& shape) {
    RuntimeMemoryBindingV1 binding;
    binding.name = shape.name;
    binding.shape = shape.shape;
    binding.dtype = shape.dtype;
    binding.valid_tokens = shape.valid_tokens;
    binding.bound_tokens = shape.bound_tokens;
    binding.capacity_tokens = shape.capacity_tokens;
    binding.sequence_axis = shape.sequence_axis;
    return binding;
}

bool same_runtime_shape(const RuntimeMemoryBindingV1& left, const RuntimeMemoryBindingV1& right) {
    return left.shape == right.shape && left.dtype == right.dtype &&
           left.valid_tokens == right.valid_tokens && left.bound_tokens == right.bound_tokens &&
           left.capacity_tokens == right.capacity_tokens &&
           left.sequence_axis == right.sequence_axis;
}

bool same_runtime_shape(const RuntimeMemoryShapeV1& left, const RuntimeMemoryShapeV1& right) {
    return left.shape == right.shape && left.dtype == right.dtype &&
           left.valid_tokens == right.valid_tokens && left.bound_tokens == right.bound_tokens &&
           left.capacity_tokens == right.capacity_tokens &&
           left.sequence_axis == right.sequence_axis;
}

} // namespace

// --- DType conversion ---

DType TrtModuleImpl::from_trt_dtype(nvinfer1::DataType dt) {
    switch (dt) {
    case nvinfer1::DataType::kFLOAT:
        return DType::kFloat32;
    case nvinfer1::DataType::kHALF:
        return DType::kFloat16;
    case nvinfer1::DataType::kBF16:
        return DType::kBFloat16;
    case nvinfer1::DataType::kINT32:
        return DType::kInt32;
    case nvinfer1::DataType::kINT8:
        return DType::kInt8;
    default:
        return DType::kFloat32;
    }
}

// --- Construction ---

TrtModuleImpl::TrtModuleImpl(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* ctx,
                             cudaStream_t stream, int32_t profile_idx,
                             void* distributed_communicator,
                             const std::vector<ModuleExternalBinding>& external_bindings,
                             bool runtime_managed_context,
                             std::vector<std::string> deferred_runtime_tensors,
                             std::vector<RuntimeMemoryAliasPairV1> runtime_alias_pairs)
    : engine_(engine), ctx_(ctx), stream_(stream), profile_idx_(profile_idx),
      distributed_communicator_(distributed_communicator),
      runtime_managed_context_(runtime_managed_context),
      deferred_runtime_tensors_(deferred_runtime_tensors.begin(), deferred_runtime_tensors.end()),
      runtime_alias_pairs_(std::move(runtime_alias_pairs)),
      cuda_graph_(std::make_unique<CudaGraphExec>()) {
    if (!ctx_)
        return;
    try {
        device_ = current_cuda_device();
    } catch (const std::exception& error) {
        std::cerr << "[trt_module] Failed to resolve CUDA device: " << error.what() << '\n';
        delete ctx_;
        ctx_ = nullptr;
        return;
    }
    for (const auto& pair : runtime_alias_pairs_) {
        deferred_runtime_tensors_.insert(pair.input_name);
        deferred_runtime_tensors_.insert(pair.output_name);
    }
    try {
        validate_initial_external_bindings(engine, external_bindings);
        validate_deferred_runtime_tensors(engine);
    } catch (const std::exception& error) {
        std::cerr << "[trt_module] Invalid construction-time binding: " << error.what() << '\n';
        delete ctx_;
        ctx_ = nullptr;
        return;
    }
    if (!attach_distributed_communicator()) {
        delete ctx_;
        ctx_ = nullptr;
        return;
    }
    if (runtime_managed_context_ || profile_idx_ > 0) {
        if (!ctx_->setOptimizationProfileAsync(profile_idx_, stream_)) {
            std::cerr << "[trt_module] Failed to set optimization profile " << profile_idx_ << "\n";
            delete ctx_;
            ctx_ = nullptr;
            return;
        }
        const auto profile_sync_status = cudaStreamSynchronize(stream_);
        if (profile_sync_status != cudaSuccess) {
            std::cerr << "[trt_module] Failed to synchronize optimization profile " << profile_idx_
                      << ": " << cudaGetErrorString(profile_sync_status) << "\n";
            delete ctx_;
            ctx_ = nullptr;
            return;
        }
    }
    allocate_buffers(engine);
}

void TrtModuleImpl::validate_initial_external_bindings(
    nvinfer1::ICudaEngine* engine, const std::vector<ModuleExternalBinding>& external_bindings) {
    for (const auto& binding : external_bindings) {
        if (binding.tensor_name.empty())
            throw std::invalid_argument("tensor name must not be empty");
        if (binding.device_ptr == nullptr)
            throw std::invalid_argument("buffer for '" + binding.tensor_name + "' is null");
        if (initial_external_bindings_.count(binding.tensor_name) != 0)
            throw std::invalid_argument("duplicate tensor '" + binding.tensor_name + "'");

        if (!engine_has_io_tensor(engine, binding.tensor_name))
            throw std::invalid_argument("unknown tensor '" + binding.tensor_name + "'");

        const auto dims = engine->getTensorShape(binding.tensor_name.c_str());
        if (dims_are_dynamic(dims)) {
            throw std::invalid_argument("tensor '" + binding.tensor_name +
                                        "' is dynamic; prebinding requires a static shape");
        }
        std::vector<int64_t> shape;
        const auto required_bytes = compute_alloc_bytes(
            dims, from_trt_dtype(engine->getTensorDataType(binding.tensor_name.c_str())), shape);
        if (binding.capacity_bytes < required_bytes) {
            throw std::invalid_argument("buffer for '" + binding.tensor_name + "' has " +
                                        std::to_string(binding.capacity_bytes) +
                                        " bytes; expected at least " +
                                        std::to_string(required_bytes));
        }
        initial_external_bindings_.emplace(binding.tensor_name, binding.device_ptr);
    }
}

void TrtModuleImpl::validate_deferred_runtime_tensors(nvinfer1::ICudaEngine* engine) {
    if (!runtime_managed_context_ && !deferred_runtime_tensors_.empty()) {
        throw std::invalid_argument(
            "deferred runtime tensors require a USER_MANAGED execution context");
    }
    for (const auto& pair : runtime_alias_pairs_) {
        validate_versioned_struct(pair.struct_size, sizeof(RuntimeMemoryAliasPairV1),
                                  pair.api_version, "RuntimeMemoryAliasPairV1");
        if (pair.input_name.empty() || pair.output_name.empty() ||
            pair.input_name == pair.output_name) {
            throw std::invalid_argument(
                "runtime-memory alias pair requires distinct input/output names");
        }
        if (!engine_has_io_tensor(engine, pair.input_name) ||
            engine->getTensorIOMode(pair.input_name.c_str()) != nvinfer1::TensorIOMode::kINPUT) {
            throw std::invalid_argument("runtime-memory alias input '" + pair.input_name +
                                        "' is not an engine input");
        }
        if (!engine_has_io_tensor(engine, pair.output_name) ||
            engine->getTensorIOMode(pair.output_name.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) {
            throw std::invalid_argument("runtime-memory alias output '" + pair.output_name +
                                        "' is not an engine output");
        }
        if (!runtime_alias_input_to_output_.emplace(pair.input_name, pair.output_name).second ||
            !runtime_alias_output_to_input_.emplace(pair.output_name, pair.input_name).second) {
            throw std::invalid_argument(
                "runtime-memory alias endpoints must form unique one-to-one pairs");
        }
#if NV_TENSORRT_MAJOR >= 11
        const char* aliased_input = engine->getAliasedInputTensor(pair.output_name.c_str());
        if (aliased_input == nullptr || pair.input_name != aliased_input) {
            throw std::invalid_argument("engine alias mismatch: output '" + pair.output_name +
                                        "' does not alias input '" + pair.input_name + "'");
        }
#else
        throw std::invalid_argument(
            "runtime-memory alias verification requires TensorRT 11 or newer");
#endif
    }
    for (const auto& name : deferred_runtime_tensors_) {
        if (name.empty())
            throw std::invalid_argument("deferred runtime tensor name must not be empty");
        if (!engine_has_io_tensor(engine, name))
            throw std::invalid_argument("unknown deferred runtime tensor '" + name + "'");
        if (!runtime_memory_dtype_supported(engine->getTensorDataType(name.c_str()))) {
            throw std::invalid_argument("deferred runtime tensor '" + name +
                                        "' uses an unsupported dtype in API v1");
        }
        if (initial_external_bindings_.count(name) != 0) {
            throw std::invalid_argument(
                "tensor '" + name + "' cannot be both statically prebound and runtime deferred");
        }
        if (engine->getTensorLocation(name.c_str()) != nvinfer1::TensorLocation::kDEVICE) {
            throw std::invalid_argument("deferred runtime tensor '" + name +
                                        "' is not device-resident");
        }
        if (engine->isShapeInferenceIO(name.c_str())) {
            throw std::invalid_argument("deferred runtime tensor '" + name +
                                        "' cannot be shape-inference I/O in API v1");
        }
        if (engine->getTensorFormat(name.c_str(), profile_idx_) !=
            nvinfer1::TensorFormat::kLINEAR) {
            throw std::invalid_argument("deferred runtime tensor '" + name +
                                        "' must use LINEAR format in API v1");
        }
    }
}

void TrtModuleImpl::bind_external(const std::string& name, void* ptr,
                                  const std::vector<int64_t>& shape) {
    bind_external(name, ptr);
    if (shape.empty())
        return;
    auto it = buffers_.find(name);
    if (it == buffers_.end())
        return;
    update_dynamic_shape(name, it->second, shape);
}

std::size_t TrtModuleImpl::compute_capacity_bytes(const RuntimeMemoryBindingV1& binding) {
    std::size_t elements = 1;
    for (std::size_t axis = 0; axis < binding.shape.size(); ++axis) {
        uint64_t extent = static_cast<uint64_t>(binding.shape[axis]);
        if (static_cast<int32_t>(axis) == binding.sequence_axis)
            extent = binding.capacity_tokens;
        if (extent > std::numeric_limits<std::size_t>::max() ||
            elements > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(extent)) {
            throw std::overflow_error("runtime-memory tensor element count overflows size_t");
        }
        elements *= static_cast<std::size_t>(extent);
    }
    const auto element_bytes = dtype_size(binding.dtype);
    if (element_bytes == 0 || elements > std::numeric_limits<std::size_t>::max() / element_bytes) {
        throw std::overflow_error("runtime-memory tensor byte size overflows size_t");
    }
    return elements * element_bytes;
}

void TrtModuleImpl::validate_runtime_tensor_shape(const RuntimeMemoryBindingV1& binding,
                                                  BufferEntry& entry) {
    const auto engine_dims = engine_->getTensorShape(binding.name.c_str());
    if (engine_dims.nbDims < 0 ||
        binding.shape.size() != static_cast<std::size_t>(engine_dims.nbDims)) {
        throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                    "' has the wrong rank");
    }
    for (std::size_t axis = 0; axis < binding.shape.size(); ++axis) {
        if (binding.shape[axis] <= 0) {
            throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                        "' has a non-positive extent");
        }
        if (engine_dims.d[axis] >= 0 && engine_dims.d[axis] != binding.shape[axis]) {
            throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                        "' changes a static TensorRT dimension");
        }
    }

    if (binding.sequence_axis >= 0) {
        if (binding.sequence_axis >= static_cast<int32_t>(binding.shape.size())) {
            throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                        "' has an invalid sequence_axis");
        }
        if (binding.valid_tokens > binding.bound_tokens || binding.bound_tokens == 0 ||
            binding.bound_tokens > binding.capacity_tokens) {
            throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                        "' violates 0 <= valid <= bound <= capacity");
        }
        if (entry.is_input) {
            const bool valid_history =
                binding.valid_tokens == 0
                    ? binding.bound_tokens == 1
                    : binding.bound_tokens >= std::max<std::uint64_t>(binding.valid_tokens, 2);
            if (!valid_history) {
                throw std::invalid_argument("runtime-memory history input '" + binding.name +
                                            "' violates the cold-sentinel H/T contract");
            }
        } else if (binding.valid_tokens == 0 || binding.valid_tokens != binding.bound_tokens) {
            throw std::invalid_argument("runtime-memory current-row output '" + binding.name +
                                        "' must bind exact Sq");
        }
        if (static_cast<uint64_t>(binding.shape[binding.sequence_axis]) != binding.bound_tokens) {
            throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                        "' shape[sequence_axis] does not equal T");
        }
    } else if (binding.valid_tokens != 0 || binding.bound_tokens != 0 ||
               binding.capacity_tokens != 0) {
        throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                    "' supplies valid/bound/capacity without a sequence_axis");
    }

    if (entry.is_input) {
        if (entry.is_dynamic) {
            nvinfer1::Dims dims;
            dims.nbDims = static_cast<int32_t>(binding.shape.size());
            for (int32_t axis = 0; axis < dims.nbDims; ++axis)
                dims.d[axis] = binding.shape[axis];
            if (!ctx_->setInputShape(binding.name.c_str(), dims)) {
                throw std::invalid_argument("TensorRT rejected runtime shape for input '" +
                                            binding.name + "'");
            }
        }
    } else {
        const int32_t missing_shapes = ctx_->inferShapes(0, nullptr);
        if (missing_shapes != 0) {
            throw std::invalid_argument("cannot bind runtime output '" + binding.name +
                                        "' before all input shapes and shape-I/O addresses");
        }
        const auto actual_dims = ctx_->getTensorShape(binding.name.c_str());
        if (actual_dims.nbDims != static_cast<int32_t>(binding.shape.size()) ||
            dims_are_dynamic(actual_dims)) {
            throw std::invalid_argument("TensorRT did not infer a concrete shape for output '" +
                                        binding.name + "'");
        }
        for (int32_t axis = 0; axis < actual_dims.nbDims; ++axis) {
            if (actual_dims.d[axis] != binding.shape[axis]) {
                throw std::invalid_argument("runtime-memory output '" + binding.name +
                                            "' shape does not match TensorRT inference");
            }
        }
    }
}

bool TrtModuleImpl::restore_input_shape(const std::string& name, const BufferEntry& entry) {
    if (!entry.is_input || !entry.is_dynamic || entry.shape.empty())
        return true;
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(entry.shape.size());
    for (int32_t axis = 0; axis < dims.nbDims; ++axis)
        dims.d[axis] = entry.shape[axis];
    if (!ctx_->setInputShape(name.c_str(), dims)) {
        std::cerr << "[trt_module] Failed to restore input shape for '" << name << "'\n";
        return false;
    }
    return true;
}

void TrtModuleImpl::commit_runtime_shape(const RuntimeMemoryShapeV1& shape, BufferEntry& entry) {
    const bool tensor_shape_changed = entry.shape != shape.shape;
    note_runtime_input_shape_change(entry, shape.shape);
    entry.shape = shape.shape;
    entry.valid_tokens = shape.valid_tokens;
    entry.bound_tokens = shape.bound_tokens;
    entry.capacity_tokens = shape.capacity_tokens;
    entry.sequence_axis = shape.sequence_axis;
    entry.runtime_shape_generation = entry.is_input ? 0 : runtime_input_shape_generation_;
    entry.runtime_shape_declared = true;
    entry.runtime_descriptor_bound = false;
    if (tensor_shape_changed && use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
    if (tensor_shape_changed)
        invalidate_context_memory();
}

void TrtModuleImpl::set_runtime_binding_shape(const RuntimeMemoryShapeV1& shape) {
    validate_versioned_struct(shape.struct_size, sizeof(RuntimeMemoryShapeV1), shape.api_version,
                              "RuntimeMemoryShapeV1");
    if (!runtime_managed_context_)
        throw std::logic_error("runtime-memory shape planning requires a USER_MANAGED context");
    synchronize_runtime_reconfiguration("runtime-memory shape planning");
    const auto found = buffers_.find(shape.name);
    if (found == buffers_.end())
        throw std::invalid_argument("unknown runtime-memory tensor '" + shape.name + "'");
    auto& entry = found->second;
    if (!entry.is_runtime_deferred) {
        throw std::invalid_argument("tensor '" + shape.name +
                                    "' was not declared runtime-deferred");
    }
    if (runtime_alias_input_to_output_.count(shape.name) != 0 ||
        runtime_alias_output_to_input_.count(shape.name) != 0) {
        throw std::invalid_argument("alias endpoint '" + shape.name +
                                    "' requires set_runtime_alias_pair_shape()");
    }
    if (shape.dtype != entry.dtype)
        throw std::invalid_argument("runtime-memory shape dtype does not match the engine");

    auto shape_binding = shape_as_binding(shape);
    validate_runtime_tensor_shape(shape_binding, entry);
    commit_runtime_shape(shape, entry);
}

void TrtModuleImpl::set_runtime_alias_pair_shape(const RuntimeMemoryAliasShapeV1& shape) {
    validate_versioned_struct(shape.struct_size, sizeof(RuntimeMemoryAliasShapeV1),
                              shape.api_version, "RuntimeMemoryAliasShapeV1");
    validate_versioned_struct(shape.input.struct_size, sizeof(RuntimeMemoryShapeV1),
                              shape.input.api_version, "RuntimeMemoryShapeV1(input)");
    validate_versioned_struct(shape.output.struct_size, sizeof(RuntimeMemoryShapeV1),
                              shape.output.api_version, "RuntimeMemoryShapeV1(output)");
    if (!runtime_managed_context_)
        throw std::logic_error("runtime-memory shape planning requires a USER_MANAGED context");
    synchronize_runtime_reconfiguration("runtime-memory alias shape planning");

    const auto declared = runtime_alias_input_to_output_.find(shape.input.name);
    if (declared == runtime_alias_input_to_output_.end() || declared->second != shape.output.name) {
        throw std::invalid_argument("runtime-memory alias shape does not match a declared pair");
    }
    if (!same_runtime_shape(shape.input, shape.output)) {
        throw std::invalid_argument("runtime-memory alias endpoints must have identical "
                                    "shape/dtype/valid/bound/capacity");
    }

    auto input_found = buffers_.find(shape.input.name);
    auto output_found = buffers_.find(shape.output.name);
    if (input_found == buffers_.end() || output_found == buffers_.end())
        throw std::logic_error("declared runtime-memory alias buffers are missing");
    auto& input_entry = input_found->second;
    auto& output_entry = output_found->second;
    if (shape.input.dtype != input_entry.dtype || shape.output.dtype != output_entry.dtype)
        throw std::invalid_argument("runtime-memory alias dtype does not match the engine");

    const BufferEntry previous_input = input_entry;
    try {
        auto input_binding = shape_as_binding(shape.input);
        auto output_binding = shape_as_binding(shape.output);
        validate_runtime_tensor_shape(input_binding, input_entry);
        validate_runtime_tensor_shape(output_binding, output_entry);
    } catch (...) {
        if (!restore_input_shape(shape.input.name, previous_input))
            runtime_binding_poisoned_ = true;
        throw;
    }

    commit_runtime_shape(shape.input, input_entry);
    commit_runtime_shape(shape.output, output_entry);
    bound_runtime_alias_outputs_.erase(shape.output.name);
}

void TrtModuleImpl::set_runtime_input_shape(const RuntimeInputShapeV1& shape) {
    validate_versioned_struct(shape.struct_size, sizeof(RuntimeInputShapeV1), shape.api_version,
                              "RuntimeInputShapeV1");
    if (!runtime_managed_context_)
        throw std::logic_error("runtime input shape planning requires a USER_MANAGED context");
    synchronize_runtime_reconfiguration("runtime input shape planning");
    const auto found = buffers_.find(shape.name);
    if (found == buffers_.end() || !found->second.is_input)
        throw std::invalid_argument("unknown runtime input '" + shape.name + "'");
    auto& entry = found->second;
    if (entry.is_runtime_deferred) {
        throw std::invalid_argument("runtime-deferred input '" + shape.name +
                                    "' requires RuntimeMemoryShapeV1");
    }
    if (!entry.is_dynamic)
        throw std::invalid_argument("runtime input '" + shape.name + "' is not dynamic");

    const auto engine_dims = engine_->getTensorShape(shape.name.c_str());
    if (engine_dims.nbDims < 0 ||
        shape.shape.size() != static_cast<std::size_t>(engine_dims.nbDims)) {
        throw std::invalid_argument("runtime input '" + shape.name + "' has the wrong rank");
    }
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(shape.shape.size());
    for (int32_t axis = 0; axis < dims.nbDims; ++axis) {
        if (shape.shape[axis] <= 0)
            throw std::invalid_argument("runtime input '" + shape.name +
                                        "' has a non-positive extent");
        if (engine_dims.d[axis] >= 0 && engine_dims.d[axis] != shape.shape[axis]) {
            throw std::invalid_argument("runtime input '" + shape.name +
                                        "' changes a static TensorRT dimension");
        }
        dims.d[axis] = shape.shape[axis];
    }
    if (!ctx_->setInputShape(shape.name.c_str(), dims))
        throw std::invalid_argument("TensorRT rejected runtime input shape for '" + shape.name +
                                    "'");

    const bool tensor_shape_changed = entry.shape != shape.shape;
    note_runtime_input_shape_change(entry, shape.shape);
    if (tensor_shape_changed && use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
    entry.shape = shape.shape;
    entry.runtime_input_shape_explicit = true;
    if (tensor_shape_changed)
        invalidate_context_memory();
}

void TrtModuleImpl::validate_runtime_binding_descriptor(const RuntimeMemoryBindingV1& binding,
                                                        BufferEntry& entry) {
    validate_versioned_struct(binding.struct_size, sizeof(RuntimeMemoryBindingV1),
                              binding.api_version, "RuntimeMemoryBindingV1");
    if (!runtime_managed_context_)
        throw std::logic_error("runtime-memory binding requires a USER_MANAGED context");
    if (binding.name.empty())
        throw std::invalid_argument("runtime-memory tensor name must not be empty");
    if (!entry.is_runtime_deferred) {
        throw std::invalid_argument("tensor '" + binding.name +
                                    "' was not declared runtime-deferred");
    }
    if (!entry.runtime_shape_declared) {
        throw std::logic_error("runtime-memory shape for '" + binding.name +
                               "' must be planned before allocation");
    }
    if (!entry.is_input && runtime_alias_output_to_input_.count(binding.name) == 0 &&
        entry.runtime_shape_generation != runtime_input_shape_generation_) {
        throw std::logic_error("runtime-memory output shape for '" + binding.name +
                               "' is stale after an input-shape change");
    }
    if (!binding.lifetime)
        throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                    "' must retain a lifetime owner");
    if (binding.capacity_bytes == 0) {
        throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                    "' has zero capacity");
    }
    if (binding.dtype != entry.dtype) {
        throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                    "' dtype does not match the engine");
    }
    if (current_cuda_device() != device_) {
        throw std::logic_error("runtime-memory binding requires the module's CUDA device to be "
                               "current");
    }
    if (binding.device != device_) {
        throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                    "' targets a different CUDA device");
    }
    (void)validate_cuda_allocation(binding.pointer, binding.alignment, binding.device,
                                   "runtime-memory binding '" + binding.name + "'");

    if (binding.shape != entry.shape || binding.valid_tokens != entry.valid_tokens ||
        binding.bound_tokens != entry.bound_tokens ||
        binding.capacity_tokens != entry.capacity_tokens ||
        binding.sequence_axis != entry.sequence_axis) {
        throw std::invalid_argument("runtime-memory binding '" + binding.name +
                                    "' does not match its planned "
                                    "shape/valid/bound/capacity");
    }
    const auto required_bytes = compute_capacity_bytes(binding);
    if (binding.capacity_bytes < required_bytes) {
        throw std::invalid_argument("runtime-memory binding '" + binding.name + "' has " +
                                    std::to_string(binding.capacity_bytes) +
                                    " bytes; R requires at least " +
                                    std::to_string(required_bytes));
    }
}

void TrtModuleImpl::commit_runtime_binding(const RuntimeMemoryBindingV1& binding,
                                           BufferEntry& entry) {
    entry.d_ptr = binding.pointer;
    entry.nbytes = binding.capacity_bytes;
    entry.is_external = true;
    entry.runtime_descriptor_bound = true;
    entry.lifetime = binding.lifetime;
}

void TrtModuleImpl::invalidate_context_memory() {
    context_memory_queried_ = false;
    context_memory_bound_ = false;
    context_memory_generation_ = 0;
    context_memory_requirement_bytes_ = 0;
    context_memory_pointer_ = nullptr;
    context_memory_capacity_bytes_ = 0;
    context_memory_lifetime_.reset();
    if (use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
}

void TrtModuleImpl::synchronize_runtime_reconfiguration(const char* operation) {
    if (!runtime_managed_context_ || !runtime_execution_in_flight_)
        return;
    const auto status = cudaStreamSynchronize(stream_);
    if (status == cudaSuccess) {
        runtime_execution_in_flight_ = false;
        return;
    }
    runtime_binding_poisoned_ = true;
    throw std::runtime_error(
        std::string(operation) +
        " could not wait for in-flight TensorRT work: " + cudaGetErrorString(status));
}

void TrtModuleImpl::note_runtime_input_shape_change(const BufferEntry& entry,
                                                    const std::vector<int64_t>& new_shape) {
    if (!entry.is_input || entry.shape == new_shape)
        return;
    if (runtime_input_shape_generation_ == std::numeric_limits<uint64_t>::max()) {
        runtime_binding_poisoned_ = true;
        throw std::overflow_error("runtime input-shape generation overflow");
    }
    ++runtime_input_shape_generation_;
}

void TrtModuleImpl::bind_runtime_memory(const RuntimeMemoryBindingV1& binding) {
    auto found = buffers_.find(binding.name);
    if (found == buffers_.end())
        throw std::invalid_argument("unknown runtime-memory tensor '" + binding.name + "'");

    auto& entry = found->second;
    if (runtime_alias_input_to_output_.count(binding.name) != 0 ||
        runtime_alias_output_to_input_.count(binding.name) != 0) {
        throw std::invalid_argument("alias endpoint '" + binding.name +
                                    "' requires bind_runtime_memory_alias_pair()");
    }
    validate_runtime_binding_descriptor(binding, entry);
    synchronize_runtime_reconfiguration("runtime-memory binding");

    BufferEntry candidate = entry;
    candidate.d_ptr = binding.pointer;
    if (!bind_tensor_address(binding.name, candidate)) {
        throw std::runtime_error("TensorRT rejected runtime-memory address for '" + binding.name +
                                 "'");
    }
    if (entry.d_ptr != binding.pointer && use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
    commit_runtime_binding(binding, entry);
}

void TrtModuleImpl::bind_runtime_memory_alias_pair(const RuntimeMemoryAliasBindingV1& binding) {
    if (runtime_binding_poisoned_)
        throw std::logic_error("runtime-memory module is poisoned after a failed rollback");
    validate_versioned_struct(binding.struct_size, sizeof(RuntimeMemoryAliasBindingV1),
                              binding.api_version, "RuntimeMemoryAliasBindingV1");
    const auto declared = runtime_alias_input_to_output_.find(binding.input.name);
    if (declared == runtime_alias_input_to_output_.end() ||
        declared->second != binding.output.name) {
        throw std::invalid_argument("runtime-memory alias binding does not match a declared pair");
    }
    if (!same_runtime_shape(binding.input, binding.output) ||
        binding.input.pointer != binding.output.pointer ||
        binding.input.capacity_bytes != binding.output.capacity_bytes ||
        binding.input.alignment != binding.output.alignment ||
        binding.input.device != binding.output.device) {
        throw std::invalid_argument(
            "runtime-memory alias endpoints must describe one identical allocation");
    }

    auto input_found = buffers_.find(binding.input.name);
    auto output_found = buffers_.find(binding.output.name);
    if (input_found == buffers_.end() || output_found == buffers_.end())
        throw std::logic_error("declared runtime-memory alias buffers are missing");
    auto& input_entry = input_found->second;
    auto& output_entry = output_found->second;
    validate_runtime_binding_descriptor(binding.input, input_entry);
    validate_runtime_binding_descriptor(binding.output, output_entry);
    synchronize_runtime_reconfiguration("runtime-memory alias binding");

    void* const previous_input = input_entry.d_ptr;
    void* const previous_output = output_entry.d_ptr;
    const bool input_bound =
        ctx_->setTensorAddress(binding.input.name.c_str(), binding.input.pointer);
    const bool output_bound =
        input_bound && ctx_->setTensorAddress(binding.output.name.c_str(), binding.output.pointer);
    if (!input_bound || !output_bound) {
        const bool input_restored =
            ctx_->setTensorAddress(binding.input.name.c_str(), previous_input);
        const bool output_restored =
            ctx_->setTensorAddress(binding.output.name.c_str(), previous_output);
        if (!input_restored || !output_restored) {
            runtime_binding_poisoned_ = true;
            std::cerr << "[trt_module] Failed to roll back alias binding; module poisoned\n";
        }
        throw std::runtime_error("TensorRT rejected atomic runtime-memory alias binding '" +
                                 binding.input.name + "' -> '" + binding.output.name + "'");
    }

    if ((previous_input != binding.input.pointer || previous_output != binding.output.pointer) &&
        use_cuda_graph_ && cuda_graph_) {
        cuda_graph_->reset();
    }
    commit_runtime_binding(binding.input, input_entry);
    commit_runtime_binding(binding.output, output_entry);
    bound_runtime_alias_outputs_.insert(binding.output.name);
}

RuntimeMemoryContextRequirementV1 TrtModuleImpl::context_memory_requirement() {
    if (!runtime_managed_context_)
        throw std::logic_error("context-memory query requires a USER_MANAGED context");
    if (current_cuda_device() != device_) {
        throw std::logic_error(
            "context-memory query requires the module's CUDA device to be current");
    }
    for (const auto& [name, entry] : buffers_) {
        if (entry.is_runtime_deferred && !entry.runtime_shape_declared) {
            throw std::logic_error("runtime-memory tensor '" + name + "' has no planned shape");
        }
        if (entry.is_runtime_deferred && !entry.is_input &&
            runtime_alias_output_to_input_.count(name) == 0 &&
            entry.runtime_shape_generation != runtime_input_shape_generation_) {
            throw std::logic_error("runtime-memory output shape for '" + name +
                                   "' is stale after an input-shape change");
        }
        if (entry.is_input && entry.is_dynamic && !entry.is_runtime_deferred &&
            !entry.runtime_input_shape_explicit) {
            throw std::logic_error("dynamic runtime input '" + name +
                                   "' has no explicit candidate shape");
        }
        if (engine_->isShapeInferenceIO(name.c_str()) && entry.is_input) {
            throw std::logic_error(
                "ordinary shape-inference input '" + name +
                "' requires value-aware planning, which the current runtime-memory API does not "
                "provide");
        }
        if (engine_->isShapeInferenceIO(name.c_str()) && entry.d_ptr == nullptr) {
            throw std::logic_error("shape-inference I/O tensor '" + name +
                                   "' has no address before inferShapes");
        }
    }

    synchronize_runtime_reconfiguration("context-memory query");
    materialize_runtime_internal_inputs();
    int32_t missing_shapes = ctx_->inferShapes(0, nullptr);
    if (missing_shapes < 0)
        throw std::runtime_error("TensorRT inferShapes failed");
    if (missing_shapes > 0) {
        std::vector<const char*> names(static_cast<std::size_t>(missing_shapes), nullptr);
        const int32_t repeated = ctx_->inferShapes(missing_shapes, names.data());
        std::ostringstream message;
        message << "TensorRT has " << missing_shapes << " insufficiently specified input(s)";
        if (repeated > 0) {
            message << ":";
            for (int32_t index = 0; index < std::min(missing_shapes, repeated); ++index) {
                if (names[index])
                    message << " " << names[index];
            }
        }
        throw std::logic_error(message.str());
    }

    // Alias pairs are planned atomically, so planning a later pair may advance
    // the global input generation. Revalidate every deferred output against
    // TensorRT's final inference before accepting the generation.
    for (auto& [name, entry] : buffers_) {
        if (!entry.is_runtime_deferred || entry.is_input)
            continue;
        const auto actual_dims = ctx_->getTensorShape(name.c_str());
        if (actual_dims.nbDims != static_cast<int32_t>(entry.shape.size()) ||
            dims_are_dynamic(actual_dims)) {
            throw std::logic_error("runtime-memory output '" + name +
                                   "' has no concrete generation-matched shape");
        }
        for (int32_t axis = 0; axis < actual_dims.nbDims; ++axis) {
            if (actual_dims.d[axis] != entry.shape[static_cast<std::size_t>(axis)]) {
                throw std::logic_error("runtime-memory output shape for '" + name +
                                       "' is stale after an input-shape change");
            }
        }
        entry.runtime_shape_generation = runtime_input_shape_generation_;
    }

    materialize_runtime_internal_outputs();
    const auto new_requirement = ctx_->updateDeviceMemorySizeForShapes();
    const bool same_generation =
        context_memory_queried_ && context_memory_generation_ == runtime_input_shape_generation_;
    if (!same_generation || new_requirement != context_memory_requirement_bytes_) {
        context_memory_lifetime_.reset();
        context_memory_bound_ = false;
        context_memory_pointer_ = nullptr;
        context_memory_capacity_bytes_ = 0;
        if (use_cuda_graph_ && cuda_graph_)
            cuda_graph_->reset();
    }
    context_memory_requirement_bytes_ = new_requirement;
    context_memory_generation_ = runtime_input_shape_generation_;
    context_memory_queried_ = true;

    RuntimeMemoryContextRequirementV1 requirement;
    requirement.capacity_bytes = context_memory_requirement_bytes_;
    requirement.device = device_;
    return requirement;
}

void TrtModuleImpl::bind_context_memory(const RuntimeMemoryContextBlockV1& block) {
    validate_versioned_struct(block.struct_size, sizeof(RuntimeMemoryContextBlockV1),
                              block.api_version, "RuntimeMemoryContextBlockV1");
    if (!runtime_managed_context_)
        throw std::logic_error("context-memory binding requires a USER_MANAGED context");
    if (!context_memory_queried_) {
        throw std::logic_error(
            "context_memory_requirement() must be called after binding actual shapes");
    }
    if (context_memory_generation_ != runtime_input_shape_generation_) {
        throw std::logic_error("context-memory requirement is stale after an input-shape change");
    }
    if (current_cuda_device() != device_) {
        throw std::logic_error(
            "context-memory binding requires the module's CUDA device to be current");
    }
    if (block.device != device_)
        throw std::invalid_argument("context-memory block targets a different CUDA device");
    if (block.capacity_bytes < context_memory_requirement_bytes_) {
        throw std::invalid_argument("context-memory block has " +
                                    std::to_string(block.capacity_bytes) +
                                    " bytes; actual shapes require at least " +
                                    std::to_string(context_memory_requirement_bytes_));
    }
    if (block.capacity_bytes > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error("context-memory block exceeds TensorRT's int64 size");
    }
    if (block.capacity_bytes > 0) {
        if (!block.lifetime)
            throw std::invalid_argument("context-memory block must retain a lifetime owner");
        (void)validate_cuda_allocation(block.pointer, block.alignment, block.device,
                                       "context-memory block");
    } else if (block.pointer != nullptr || block.lifetime) {
        throw std::invalid_argument(
            "zero-sized context-memory block must not carry a pointer or lifetime");
    }

    synchronize_runtime_reconfiguration("context-memory binding");
    const bool block_changed = !context_memory_bound_ || context_memory_pointer_ != block.pointer ||
                               context_memory_capacity_bytes_ != block.capacity_bytes ||
                               context_memory_generation_ != runtime_input_shape_generation_;
    if (block_changed && use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
    ctx_->setDeviceMemoryV2(block.pointer, static_cast<int64_t>(block.capacity_bytes));
    context_memory_pointer_ = block.pointer;
    context_memory_capacity_bytes_ = block.capacity_bytes;
    context_memory_generation_ = runtime_input_shape_generation_;
    context_memory_lifetime_ = block.lifetime;
    context_memory_bound_ = true;
}

bool TrtModuleImpl::runtime_memory_ready() const noexcept {
    if (!runtime_managed_context_ || runtime_binding_poisoned_ || !context_memory_queried_ ||
        !context_memory_bound_ || context_memory_generation_ != runtime_input_shape_generation_)
        return false;
    for (const auto& name : deferred_runtime_tensors_) {
        const auto found = buffers_.find(name);
        if (found == buffers_.end() || !found->second.runtime_descriptor_bound)
            return false;
        if (!found->second.is_input && runtime_alias_output_to_input_.count(name) == 0 &&
            found->second.runtime_shape_generation != runtime_input_shape_generation_)
            return false;
    }
    for (const auto& [name, entry] : buffers_) {
        (void)name;
        if (entry.is_runtime_internal_dynamic &&
            (entry.d_ptr == nullptr ||
             entry.runtime_buffer_generation != runtime_input_shape_generation_)) {
            return false;
        }
    }
    for (const auto& [output_name, input_name] : runtime_alias_output_to_input_) {
        (void)input_name;
        if (bound_runtime_alias_outputs_.count(output_name) == 0)
            return false;
    }
    return true;
}

RuntimeMemoryEngineStatsV1 TrtModuleImpl::runtime_memory_engine_stats() const noexcept {
    RuntimeMemoryEngineStatsV1 stats;
    stats.engine_identity = reinterpret_cast<std::uintptr_t>(engine_);
    stats.cuda_graph_active = use_cuda_graph_;
    if (engine_ == nullptr)
        return stats;

    const auto total_weight_bytes =
        engine_->getEngineStat(nvinfer1::EngineStat::kTOTAL_WEIGHTS_SIZE);
    if (total_weight_bytes >= 0) {
        stats.total_weight_bytes = static_cast<std::uint64_t>(total_weight_bytes);
        stats.total_weight_bytes_available = true;
    }

    const auto streamable_weight_bytes = engine_->getStreamableWeightsSize();
    if (streamable_weight_bytes > 0) {
        stats.streamable_weight_bytes = static_cast<std::uint64_t>(streamable_weight_bytes);
        const auto budget = engine_->getWeightStreamingBudgetV2();
        if (budget >= 0) {
            stats.weight_streaming_budget_bytes = static_cast<std::uint64_t>(budget);
            stats.weight_streaming_budget_available = true;
        }
    } else {
        // TensorRT returns zero when weight streaming was not enabled. In
        // either that case or an engine with no streamable weights, all
        // logical engine weights are resident.
        stats.weight_streaming_budget_available = true;
    }

    for (const auto& [name, entry] : buffers_) {
        (void)name;
        if (entry.is_input && entry.is_runtime_internal_dynamic && !entry.is_external)
            stats.ordinary_device_input_bytes += static_cast<std::uint64_t>(entry.nbytes);
        if (!entry.is_input && entry.is_runtime_internal_dynamic && !entry.is_external)
            stats.ordinary_device_output_bytes += static_cast<std::uint64_t>(entry.nbytes);
        if (!entry.is_input && !entry.is_external)
            stats.device_output_bytes += static_cast<std::uint64_t>(entry.nbytes);
    }
    for (const auto& [name, staging] : host_output_staging_) {
        (void)name;
        stats.host_output_staging_bytes += static_cast<std::uint64_t>(staging.size());
    }
    return stats;
}

RuntimeMemoryTransferSnapshotV1 TrtModuleImpl::runtime_memory_transfer_snapshot() const {
    RuntimeMemoryTransferSnapshotV1 snapshot;
    snapshot.event_sequence = transfer_event_sequence_;
    snapshot.execution_attempt_events = execution_attempt_events_;
    snapshot.counters.reserve(transfer_counters_.size());
    for (const auto& [name, counter] : transfer_counters_) {
        (void)name;
        snapshot.counters.push_back(counter);
    }
    std::sort(snapshot.counters.begin(), snapshot.counters.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.tensor_name < rhs.tensor_name; });
    return snapshot;
}

void TrtModuleImpl::note_transfer(const std::string& name, const BufferEntry& entry,
                                  cudaMemcpyKind kind, std::size_t bytes) {
    if (!runtime_managed_context_ || bytes == 0)
        return;
    if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDeviceToDevice)
        return;
    auto& counter = transfer_counters_[name];
    counter.tensor_name = name;
    counter.runtime_kv_binding = counter.runtime_kv_binding || entry.is_runtime_deferred ||
                                 deferred_runtime_tensors_.count(name) != 0 ||
                                 runtime_alias_input_to_output_.count(name) != 0 ||
                                 runtime_alias_output_to_input_.count(name) != 0;
    const auto amount = static_cast<std::uint64_t>(bytes);
    auto add = [](std::uint64_t& target, std::uint64_t increment, const char* what) {
        if (increment > std::numeric_limits<std::uint64_t>::max() - target)
            throw std::overflow_error(std::string(what) + " counter overflow");
        target += increment;
    };
    if (kind == cudaMemcpyDeviceToHost) {
        add(counter.device_to_host_bytes, amount, "D2H transfer byte");
        add(counter.device_to_host_events, 1, "D2H transfer event");
    } else {
        add(counter.device_to_device_bytes, amount, "D2D transfer byte");
        add(counter.device_to_device_events, 1, "D2D transfer event");
    }
    add(transfer_event_sequence_, 1, "transfer sequence");
}

void TrtModuleImpl::ensure_runtime_memory_ready() const {
    if (runtime_managed_context_ && !runtime_memory_ready()) {
        throw std::logic_error(
            "USER_MANAGED module is not ready: plan all shapes, query/bind actual-shape context "
            "memory, and bind all deferred tensors before enqueue");
    }
}

int32_t TrtModuleImpl::input_rank(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end() || !it->second.is_input)
        return 0;
    return static_cast<int32_t>(it->second.shape.size());
}

bool TrtModuleImpl::input_is_dynamic(const std::string& name) const {
    auto it = buffers_.find(name);
    return it != buffers_.end() && it->second.is_input && it->second.is_dynamic;
}

bool TrtModuleImpl::attach_distributed_communicator() {
    if (distributed_communicator_ == nullptr || ctx_ == nullptr)
        return true;
#if NV_TENSORRT_MAJOR >= 11
    if (ctx_->setCommunicator(distributed_communicator_))
        return true;
    std::cerr << "[trt_module] Failed to set TRT distributed communicator\n";
    return false;
#else
    std::cerr << "[trt_module] TensorRT distributed communicator requires TRT 11.0+\n";
    return false;
#endif
}

bool TrtModuleImpl::bind_tensor_address(const std::string& name, const BufferEntry& entry) {
    if (!ctx_ || !entry.d_ptr)
        return false;
    const bool ok = entry.is_input ? ctx_->setInputTensorAddress(name.c_str(), entry.d_ptr)
                                   : ctx_->setOutputTensorAddress(name.c_str(), entry.d_ptr);
    if (!ok) {
        std::cerr << "[trt_module] Failed to bind " << (entry.is_input ? "input" : "output")
                  << " tensor address for '" << name << "'\n";
    }
    return ok;
}

void TrtModuleImpl::reset_execution_context() {
    // TensorRT execution contexts contain engine/profile/binding state, not
    // sequence-local KV or sampler state. Replacing the context here made a
    // warm request pay context creation, profile selection, synchronization,
    // dynamic-shape replay, and CUDA-graph recapture even when none changed.
    // Shape changes already invalidate CUDA graphs in update_dynamic_shape(),
    // and bind_external() updates the live context when a binding changes.
}

TrtModuleImpl::~TrtModuleImpl() {
    // Runtime-managed owners may release with cudaFreeAsync on stream_. Keep
    // the stream owner in keep_alive_ until every retained I/O/context owner
    // has been destroyed and its release has completed.
    if (runtime_managed_context_ && stream_)
        (void)cudaStreamSynchronize(stream_);
    flush_timing_events();
    // CUDA Graphs may contain TensorRT collective launches that retain the
    // distributed communicator. Destroy the captured graph before member
    // teardown releases distributed_owner from keep_alive_.
    if (cuda_graph_)
        cuda_graph_->reset();
    free_buffers();
    if (runtime_managed_context_ && stream_)
        (void)cudaStreamSynchronize(stream_);
    delete ctx_;
    ctx_ = nullptr;
    context_memory_lifetime_.reset();
    if (runtime_managed_context_ && stream_)
        (void)cudaStreamSynchronize(stream_);
}

void TrtModuleImpl::keep_alive(std::shared_ptr<void> resource) {
    keep_alive_.push_back(std::move(resource));
}

void TrtModuleImpl::set_timing_label(std::string label) {
    timing_label_ = label.empty() ? std::string("engine") : std::move(label);
}

// --- Buffer allocation helpers ---

bool TrtModuleImpl::dims_are_dynamic(const nvinfer1::Dims& dims) {
    for (int32_t d = 0; d < dims.nbDims; ++d)
        if (dims.d[d] == -1)
            return true;
    return false;
}

std::vector<int64_t> TrtModuleImpl::dims_to_shape(const nvinfer1::Dims& dims) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int32_t d = 0; d < dims.nbDims; ++d)
        shape.push_back(dims.d[d]);
    return shape;
}

void TrtModuleImpl::update_dynamic_shape(const std::string& name, BufferEntry& entry,
                                         const std::vector<int64_t>& new_shape) {
    // Skip static inputs: TRT rejects setInputShape on them even when the
    // engine as a whole advertises dynamic shapes via optimization profiles.
    if (!has_dynamic_shapes_ || !entry.is_dynamic || new_shape == entry.shape)
        return;
    if (runtime_managed_context_ && entry.is_runtime_deferred) {
        throw std::logic_error("runtime-deferred tensor '" + name +
                               "' shape must be changed through the planning API");
    }
    if (runtime_managed_context_)
        synchronize_runtime_reconfiguration("runtime input shape update");
    // Any captured CUDA graph was baked against the OLD shape; force a
    // re-capture on the next enqueue so the new shape actually takes.
    if (use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
    if (runtime_managed_context_)
        invalidate_context_memory();
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(new_shape.size());
    for (int32_t d = 0; d < dims.nbDims; ++d)
        dims.d[d] = new_shape[d];
    if (!ctx_->setInputShape(name.c_str(), dims))
        throw std::invalid_argument("TensorRT rejected runtime shape for input '" + name + "'");
    note_runtime_input_shape_change(entry, new_shape);
    entry.shape = new_shape;
    if (runtime_managed_context_)
        entry.runtime_input_shape_explicit = true;
}

std::size_t TrtModuleImpl::compute_alloc_bytes(const nvinfer1::Dims& dims, DType dtype,
                                               std::vector<int64_t>& shape_out) {
    shape_out.clear();
    std::size_t n = 1;
    for (int32_t d = 0; d < dims.nbDims; ++d) {
        int64_t dim = std::max(static_cast<int64_t>(dims.d[d]), int64_t{1});
        shape_out.push_back(dim);
        n *= static_cast<std::size_t>(dim);
    }
    return n * dtype_size(dtype);
}

std::size_t TrtModuleImpl::compute_shape_bytes(const std::vector<int64_t>& shape, DType dtype,
                                               const std::string& tensor_name) {
    std::size_t elements = 1;
    for (const auto extent : shape) {
        if (extent <= 0) {
            throw std::logic_error("ordinary dynamic tensor '" + tensor_name +
                                   "' has no concrete shape after inferShapes; "
                                   "IOutputAllocator is required");
        }
        const auto positive_extent = static_cast<std::uint64_t>(extent);
        if (positive_extent > std::numeric_limits<std::size_t>::max() ||
            elements > std::numeric_limits<std::size_t>::max() /
                           static_cast<std::size_t>(positive_extent)) {
            throw std::overflow_error("ordinary dynamic tensor '" + tensor_name +
                                      "' element count overflows size_t");
        }
        elements *= static_cast<std::size_t>(positive_extent);
    }
    const auto element_bytes = dtype_size(dtype);
    if (element_bytes == 0 || elements > std::numeric_limits<std::size_t>::max() / element_bytes) {
        throw std::overflow_error("ordinary dynamic tensor '" + tensor_name +
                                  "' byte size overflows size_t");
    }
    return elements * element_bytes;
}

std::size_t TrtModuleImpl::compute_concrete_bytes(const nvinfer1::Dims& dims, DType dtype,
                                                  std::vector<int64_t>& shape_out,
                                                  const std::string& tensor_name) {
    if (dims.nbDims < 0) {
        throw std::logic_error("ordinary dynamic tensor '" + tensor_name +
                               "' has no concrete shape after inferShapes; "
                               "IOutputAllocator is required");
    }
    shape_out = dims_to_shape(dims);
    return compute_shape_bytes(shape_out, dtype, tensor_name);
}

void TrtModuleImpl::detect_dynamic_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io) {
    has_dynamic_shapes_ = false;
    for (int32_t i = 0; i < num_io && !has_dynamic_shapes_; ++i) {
        const char* raw_name = engine->getIOTensorName(i);
        if (raw_name == nullptr)
            continue;
        const std::string name(raw_name);
        if (engine->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        if (dims_are_dynamic(engine->getTensorShape(name.c_str())))
            has_dynamic_shapes_ = true;
    }
}

void TrtModuleImpl::set_dynamic_input_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                             nvinfer1::OptProfileSelector selector) {
    for (int32_t i = 0; i < num_io; ++i) {
        const char* raw_name = engine->getIOTensorName(i);
        if (raw_name == nullptr)
            continue;
        const std::string name(raw_name);
        if (engine->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        if (dims_are_dynamic(engine->getTensorShape(name.c_str()))) {
            auto dims = engine->getProfileShape(name.c_str(), profile_idx_, selector);
            ctx_->setInputShape(name.c_str(), dims);
        }
    }
}

void TrtModuleImpl::allocate_single_input(nvinfer1::ICudaEngine* engine, const std::string& name,
                                          int32_t num_profiles) {
    auto trt_shape = engine->getTensorShape(name.c_str());
    auto dtype = from_trt_dtype(engine->getTensorDataType(name.c_str()));

    // Determine allocation shape (max) and initial runtime shape (opt).
    nvinfer1::Dims alloc_dims = trt_shape;
    nvinfer1::Dims init_dims = trt_shape;
    bool is_dynamic = has_dynamic_shapes_ && num_profiles > 0 && dims_are_dynamic(trt_shape);

    if (is_dynamic) {
        init_dims =
            engine->getProfileShape(name.c_str(), profile_idx_, nvinfer1::OptProfileSelector::kOPT);
        alloc_dims = runtime_managed_context_
                         ? init_dims
                         : engine->getProfileShape(name.c_str(), profile_idx_,
                                                   nvinfer1::OptProfileSelector::kMAX);
    }

    std::vector<int64_t> shape;
    std::size_t nbytes = compute_alloc_bytes(alloc_dims, dtype, shape);

    BufferEntry entry;
    entry.dtype = dtype;
    entry.nbytes = nbytes;
    entry.is_input = true;
    entry.is_dynamic = is_dynamic;
    entry.shape = is_dynamic ? dims_to_shape(init_dims) : shape;

    const auto external = initial_external_bindings_.find(name);
    if (external != initial_external_bindings_.end()) {
        entry.d_ptr = external->second;
        entry.is_external = true;
    } else if (deferred_runtime_tensors_.find(name) != deferred_runtime_tensors_.end()) {
        // The caller will provide this buffer after the runtime memory budget
        // is resolved. Mark it external immediately so an unbound deferred
        // entry can never be mistaken for runtime-owned storage.
        entry.is_external = true;
        entry.is_runtime_deferred = true;
    } else if (runtime_managed_context_ && is_dynamic) {
        // Runtime-memory modules materialize ordinary dynamic inputs only
        // after the planner supplies a concrete shape. Do not reserve MAX.
        entry.d_ptr = nullptr;
        entry.nbytes = 0;
        entry.is_runtime_internal_dynamic = true;
    } else if (nbytes > 0) {
        auto err = cudaMalloc(&entry.d_ptr, nbytes);
        if (err != cudaSuccess) {
            entry.d_ptr = nullptr;
            allocation_failed_ = true;
            std::cerr << "[trt_module] Failed to allocate " << nbytes << " bytes for input '"
                      << name << "': " << cudaGetErrorString(err) << '\n';
        } else {
            const auto memset_status = cudaMemsetAsync(entry.d_ptr, 0, nbytes, stream_);
            if (memset_status != cudaSuccess) {
                allocation_failed_ = true;
                std::cerr << "[trt_module] Failed to initialize input '" << name
                          << "': " << cudaGetErrorString(memset_status) << '\n';
            }
        }
    }

    if (entry.d_ptr && !bind_tensor_address(name, entry))
        allocation_failed_ = true;

    if (is_dynamic)
        ctx_->setInputShape(name.c_str(), init_dims);

    buffers_[name] = std::move(entry);
}

void TrtModuleImpl::allocate_input_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                           int32_t num_profiles) {
    for (int32_t i = 0; i < num_io; ++i) {
        const char* raw_name = engine->getIOTensorName(i);
        if (raw_name == nullptr)
            continue;
        const std::string name(raw_name);
        if (engine->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        allocate_single_input(engine, name, num_profiles);
    }
}

void TrtModuleImpl::allocate_output_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io) {
    for (int32_t i = 0; i < num_io; ++i) {
        const char* raw_name = engine->getIOTensorName(i);
        if (raw_name == nullptr)
            continue;
        const std::string name(raw_name);
        if (engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT)
            continue;

        auto dtype = from_trt_dtype(engine->getTensorDataType(name.c_str()));

        const auto engine_dims = engine->getTensorShape(name.c_str());
        const bool is_dynamic = engine_dims.nbDims < 0 || dims_are_dynamic(engine_dims);
        // Legacy modules query against profile MAX. Runtime-memory modules
        // retain the unresolved declaration until inferShapes() runs for the
        // planner's concrete inputs.
        nvinfer1::Dims out_dims = has_dynamic_shapes_ && !runtime_managed_context_
                                      ? ctx_->getTensorShape(name.c_str())
                                      : engine_dims;

        std::vector<int64_t> shape;
        std::size_t nbytes = compute_alloc_bytes(out_dims, dtype, shape);

        BufferEntry entry;
        entry.shape = shape;
        entry.dtype = dtype;
        entry.nbytes = nbytes;
        entry.is_input = false;
        entry.is_dynamic = is_dynamic;

        const auto external = initial_external_bindings_.find(name);
        if (external != initial_external_bindings_.end()) {
            entry.d_ptr = external->second;
            entry.is_external = true;
        } else if (deferred_runtime_tensors_.find(name) != deferred_runtime_tensors_.end()) {
            entry.is_external = true;
            entry.is_runtime_deferred = true;
        } else if (runtime_managed_context_ && is_dynamic) {
            entry.d_ptr = nullptr;
            entry.nbytes = 0;
            entry.is_runtime_internal_dynamic = true;
        } else if (nbytes > 0) {
            auto err = cudaMalloc(&entry.d_ptr, nbytes);
            if (err != cudaSuccess) {
                entry.d_ptr = nullptr;
                allocation_failed_ = true;
                std::cerr << "[trt_module] Failed to allocate " << nbytes << " bytes for output '"
                          << name << "': " << cudaGetErrorString(err) << '\n';
            } else {
                const auto memset_status = cudaMemsetAsync(entry.d_ptr, 0, nbytes, stream_);
                if (memset_status != cudaSuccess) {
                    allocation_failed_ = true;
                    std::cerr << "[trt_module] Failed to initialize output '" << name
                              << "': " << cudaGetErrorString(memset_status) << '\n';
                }
            }
        }

        if (entry.d_ptr && !bind_tensor_address(name, entry))
            allocation_failed_ = true;

        if (!entry.is_runtime_internal_dynamic)
            allocate_host_output_staging(host_output_staging_, name, nbytes, entry.is_external);

        buffers_[name] = std::move(entry);
    }
}

void TrtModuleImpl::materialize_runtime_internal_buffer(const std::string& name, BufferEntry& entry,
                                                        const nvinfer1::Dims& concrete_dims) {
    if (!runtime_managed_context_ || !entry.is_runtime_internal_dynamic || entry.is_external)
        return;

    std::vector<int64_t> concrete_shape;
    const auto required_bytes =
        compute_concrete_bytes(concrete_dims, entry.dtype, concrete_shape, name);

    // Host materialization describes the current logical output, not the
    // high-water capacity retained by the reusable device allocation.
    if (!entry.is_input)
        replace_host_output_staging(host_output_staging_, name, required_bytes);

    if (entry.d_ptr != nullptr && required_bytes <= entry.nbytes) {
        entry.shape = std::move(concrete_shape);
        entry.runtime_buffer_generation = runtime_input_shape_generation_;
        if (!bind_tensor_address(name, entry)) {
            throw std::runtime_error("TensorRT rejected reused ordinary dynamic buffer for '" +
                                     name + "'");
        }
        return;
    }

    synchronize_runtime_reconfiguration("ordinary dynamic I/O growth");
    void* replacement = nullptr;
    const auto allocation_status = cudaMalloc(&replacement, required_bytes);
    if (allocation_status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate " + std::to_string(required_bytes) +
                                 " concrete bytes for ordinary dynamic tensor '" + name +
                                 "': " + cudaGetErrorString(allocation_status));
    }

    BufferEntry candidate = entry;
    candidate.d_ptr = replacement;
    candidate.shape = concrete_shape;
    candidate.nbytes = required_bytes;
    if (!bind_tensor_address(name, candidate)) {
        const bool restored = ctx_->setTensorAddress(name.c_str(), entry.d_ptr);
        const auto release_status = cudaFree(replacement);
        if (!restored || release_status != cudaSuccess) {
            runtime_binding_poisoned_ = true;
            std::cerr << "[trt_module] Failed to roll back ordinary dynamic buffer growth for '"
                      << name << "'\n";
        }
        throw std::runtime_error("TensorRT rejected concrete ordinary dynamic buffer for '" + name +
                                 "'");
    }

    void* const previous = entry.d_ptr;
    entry.d_ptr = replacement;
    entry.shape = std::move(concrete_shape);
    entry.nbytes = required_bytes;
    entry.runtime_buffer_generation = runtime_input_shape_generation_;
    if (use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
    if (previous != nullptr) {
        const auto release_status = cudaFree(previous);
        if (release_status != cudaSuccess) {
            runtime_binding_poisoned_ = true;
            throw std::runtime_error("Failed to release superseded ordinary dynamic buffer for '" +
                                     name + "': " + cudaGetErrorString(release_status));
        }
    }
}

void TrtModuleImpl::materialize_runtime_internal_inputs() {
    for (auto& [name, entry] : buffers_) {
        if (!entry.is_input || !entry.is_runtime_internal_dynamic)
            continue;
        materialize_runtime_internal_buffer(name, entry, ctx_->getTensorShape(name.c_str()));
    }
}

void TrtModuleImpl::materialize_runtime_internal_outputs() {
    for (auto& [name, entry] : buffers_) {
        if (entry.is_input || !entry.is_runtime_internal_dynamic)
            continue;
        materialize_runtime_internal_buffer(name, entry, ctx_->getTensorShape(name.c_str()));
    }
}

// --- Buffer allocation ---

void TrtModuleImpl::allocate_buffers(nvinfer1::ICudaEngine* engine) {
    const int32_t num_io = engine->getNbIOTensors();
    const int32_t num_profiles = engine->getNbOptimizationProfiles();

    detect_dynamic_shapes(engine, num_io);

    // Pass 1: legacy modules allocate dynamic inputs at profile MAX. Opted-in
    // runtime-memory modules defer ordinary dynamic I/O to shape planning.
    allocate_input_buffers(engine, num_io, num_profiles);

    // Pass 2: only the legacy path temporarily infers outputs at profile MAX.
    if (!runtime_managed_context_ && has_dynamic_shapes_ && num_profiles > 0)
        set_dynamic_input_shapes(engine, num_io, nvinfer1::OptProfileSelector::kMAX);

    allocate_output_buffers(engine, num_io);

    if (!runtime_managed_context_ && has_dynamic_shapes_ && num_profiles > 0)
        set_dynamic_input_shapes(engine, num_io, nvinfer1::OptProfileSelector::kOPT);

    initial_external_bindings_.clear();
    const auto sync_status = cudaStreamSynchronize(stream_);
    if (sync_status != cudaSuccess) {
        allocation_failed_ = true;
        std::cerr << "[trt_module] Buffer initialization failed: "
                  << cudaGetErrorString(sync_status) << '\n';
    }
}

void TrtModuleImpl::free_buffers() {
    for (auto& [name, entry] : buffers_) {
        if (entry.d_ptr && !entry.is_external) {
            cudaFree(entry.d_ptr);
        }
        entry.d_ptr = nullptr;
    }
    buffers_.clear();
    host_output_staging_.clear();
    output_device_tensors_.clear();
}

// --- Forward pass (CPU → GPU → CPU) ---

TensorMap TrtModuleImpl::forward(const TensorMap& inputs) {
    forward_async(inputs);
    sync();
    return download_host_outputs(nullptr);
}

TensorMap TrtModuleImpl::forward_selected(const TensorMap& inputs,
                                          const std::vector<std::string>& host_output_names) {
    forward_async(inputs);
    sync();
    const std::unordered_set<std::string> selected(host_output_names.begin(),
                                                   host_output_names.end());
    return download_host_outputs(&selected);
}

TensorMap
TrtModuleImpl::download_host_outputs(const std::unordered_set<std::string>* selected_output_names) {
    // Download requested outputs only. Externally-bound outputs stay on device.
    TensorMap outputs;
    for (auto& [name, entry] : buffers_) {
        if (entry.is_input)
            continue;
        if (entry.is_external)
            continue;
        if (selected_output_names != nullptr &&
            selected_output_names->find(name) == selected_output_names->end())
            continue;

        std::vector<int64_t> runtime_shape = entry.shape;
        std::size_t runtime_nbytes = entry.nbytes;
        if (runtime_managed_context_ && entry.is_runtime_internal_dynamic) {
            runtime_nbytes = compute_shape_bytes(entry.shape, entry.dtype, name);
            const auto staging = host_output_staging_.find(name);
            if (entry.d_ptr == nullptr || runtime_nbytes > entry.nbytes ||
                staging == host_output_staging_.end() || staging->second.size() != runtime_nbytes) {
                throw std::logic_error("ordinary dynamic output '" + name +
                                       "' is not materialized at its concrete shape");
            }
        } else if (has_dynamic_shapes_ && ctx_ != nullptr) {
            std::vector<int64_t> inferred_shape;
            runtime_nbytes = compute_alloc_bytes(ctx_->getTensorShape(name.c_str()), entry.dtype,
                                                 inferred_shape);
            if (runtime_nbytes <= entry.nbytes) {
                runtime_shape = std::move(inferred_shape);
            } else {
                runtime_nbytes = entry.nbytes;
            }
        }

        auto& staging = host_output_staging_[name];
        const auto copy_status =
            cudaMemcpy(staging.data(), entry.d_ptr, runtime_nbytes, cudaMemcpyDeviceToHost);
        if (copy_status == cudaSuccess)
            note_transfer(name, entry, cudaMemcpyDeviceToHost, runtime_nbytes);

        Tensor t;
        t.data = staging.data();
        t.shape = std::move(runtime_shape);
        t.dtype = entry.dtype;
        outputs[name] = t;
    }
    return outputs;
}

// --- Forward async ---

void TrtModuleImpl::enable_cuda_graph() {
    use_cuda_graph_ = true;
    cuda_graph_->reset();
}

void TrtModuleImpl::forward_async(const TensorMap& inputs) {
    if (runtime_managed_context_)
        ensure_runtime_memory_ready();

    // Upload inputs H2D, updating shapes for dynamic engines
    for (const auto& [name, tensor] : inputs) {
        auto it = buffers_.find(name);
        if (it == buffers_.end())
            continue;
        auto& entry = it->second;
        if (!entry.is_input || !entry.d_ptr)
            continue;

        if (runtime_managed_context_ && entry.is_dynamic && tensor.shape != entry.shape) {
            throw std::invalid_argument("runtime input '" + name +
                                        "' does not match its planned concrete shape");
        }
        update_dynamic_shape(name, entry, tensor.shape);

        auto copy_bytes = tensor.nbytes();
        if (runtime_managed_context_) {
            if (tensor.dtype != entry.dtype)
                throw std::invalid_argument("runtime input '" + name + "' dtype mismatch");
            if (entry.is_dynamic) {
                const auto expected_bytes = compute_shape_bytes(entry.shape, entry.dtype, name);
                if (copy_bytes != expected_bytes) {
                    throw std::invalid_argument("runtime input '" + name +
                                                "' byte size does not match its planned shape");
                }
            }
            if (copy_bytes > entry.nbytes) {
                throw std::invalid_argument(
                    "runtime input '" + name + "' requires " + std::to_string(copy_bytes) +
                    " bytes but its materialized capacity is " + std::to_string(entry.nbytes));
            }
        } else {
            copy_bytes = std::min(copy_bytes, entry.nbytes);
        }
        if (copy_bytes > 0 && tensor.data) {
            cudaMemcpyAsync(entry.d_ptr, tensor.data, copy_bytes, cudaMemcpyHostToDevice, stream_);
        }
    }

    execute_enqueue();
}

void TrtModuleImpl::execute_enqueue() {
    if (execution_failed_) {
        throw std::runtime_error("TensorRT module is poisoned after an execution failure: " +
                                 execution_failure_);
    }
    if (ctx_ == nullptr || allocation_failed_ || runtime_binding_poisoned_) {
        throw std::runtime_error("TensorRT module is not in a valid state for execution");
    }
    ensure_runtime_memory_ready();
    if (execution_attempt_events_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("TensorRT execution-attempt counter overflow");
    ++execution_attempt_events_;
    record_timed_enqueue();
}

void TrtModuleImpl::require_execution_success(bool success, const std::string& operation) {
    if (success)
        return;
    execution_failed_ = true;
    execution_failure_ = operation;
    use_cuda_graph_ = false;
    if (cuda_graph_)
        cuda_graph_->reset();
    throw std::runtime_error(operation + " failed; TensorRT module has been poisoned");
}

bool TrtModuleImpl::begin_timing_event(TimingEvent& event) {
    if (cudaEventCreate(&event.start) != cudaSuccess)
        return false;
    if (cudaEventCreate(&event.stop) != cudaSuccess) {
        cudaEventDestroy(event.start);
        event.start = nullptr;
        return false;
    }
    if (cudaEventRecord(event.start, stream_) == cudaSuccess)
        return true;
    cudaEventDestroy(event.start);
    cudaEventDestroy(event.stop);
    event.start = nullptr;
    event.stop = nullptr;
    return false;
}

void TrtModuleImpl::finish_timing_event(TimingEvent event) noexcept {
    if (event.start && event.stop && cudaEventRecord(event.stop, stream_) == cudaSuccess) {
        try {
            timing_events_.push_back(event);
            return;
        } catch (...) {
            // Timing is observability only. The enqueue already succeeded;
            // discard these events instead of failing inference or leaking.
        }
    }
    if (event.start)
        cudaEventDestroy(event.start);
    if (event.stop)
        cudaEventDestroy(event.stop);
}

void TrtModuleImpl::record_timed_enqueue() {
    TimingEvent timing_event;
    const bool timing_ok = begin_timing_event(timing_event);
    const auto discard_timing_event = [&] {
        if (timing_event.start)
            cudaEventDestroy(timing_event.start);
        if (timing_event.stop)
            cudaEventDestroy(timing_event.stop);
        timing_event.start = nullptr;
        timing_event.stop = nullptr;
    };
    const auto require_enqueue_success = [&](bool success, const char* operation) {
        if (success) {
            // Once an enqueue succeeds, reconfiguration must synchronize
            // even if later timing-event retention throws.
            runtime_execution_in_flight_ = true;
            return;
        }
        discard_timing_event();
        require_execution_success(false, operation);
    };
    const auto finish_timing = [&] {
        if (timing_ok)
            finish_timing_event(timing_event);
    };

    if (use_cuda_graph_ && cuda_graph_->ready()) {
        require_enqueue_success(cuda_graph_->launch(stream_), "CUDA graph launch");
        finish_timing();
        return;
    }
    if (use_cuda_graph_) {
        if (!cuda_graph_->begin_capture(stream_)) {
            std::cerr << "[cuda_graph] Capture start failed, disabling CUDA Graphs\n";
            use_cuda_graph_ = false;
            require_enqueue_success(ctx_->enqueueV3(stream_),
                                    "TensorRT enqueueV3 after CUDA graph capture start failure");
            finish_timing();
            return;
        }

        const bool capture_enqueue_ok = ctx_->enqueueV3(stream_);
        if (!capture_enqueue_ok) {
            // Terminate capture before poisoning the module. end_capture()
            // destroys any successfully instantiated partial graph on reset.
            (void)cuda_graph_->end_capture(stream_);
            cuda_graph_->reset();
            require_enqueue_success(false, "TensorRT enqueueV3 during CUDA graph capture");
        }
        if (!cuda_graph_->end_capture(stream_)) {
            std::cerr << "[cuda_graph] Capture failed, disabling CUDA Graphs\n";
            use_cuda_graph_ = false;
            require_enqueue_success(ctx_->enqueueV3(stream_),
                                    "TensorRT enqueueV3 after CUDA graph capture failure");
        } else {
            require_enqueue_success(cuda_graph_->launch(stream_),
                                    "CUDA graph launch after capture");
        }
        finish_timing();
        return;
    }
    require_enqueue_success(ctx_->enqueueV3(stream_), "TensorRT enqueueV3");
    finish_timing();
}

void TrtModuleImpl::flush_timing_events() {
    if (timing_events_.empty())
        return;
    double total_ms = 0.0;
    int32_t launches = 0;
    for (auto& event : timing_events_) {
        if (event.stop && cudaEventSynchronize(event.stop) == cudaSuccess) {
            float elapsed_ms = 0.0F;
            if (cudaEventElapsedTime(&elapsed_ms, event.start, event.stop) == cudaSuccess) {
                total_ms += static_cast<double>(elapsed_ms);
                ++launches;
            }
        }
        if (event.start)
            cudaEventDestroy(event.start);
        if (event.stop)
            cudaEventDestroy(event.stop);
    }
    timing_events_.clear();
    if (launches <= 0)
        return;
    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << "[trtmc.engine_timing] label=\"" << timing_label_
         << "\" execute_ms=" << total_ms << " launches=" << launches;
    std::cerr << line.str() << '\n';
}

void TrtModuleImpl::sync() {
    const auto status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) {
        require_execution_success(false, std::string("CUDA stream synchronization: ") +
                                             cudaGetErrorString(status));
    }
    runtime_execution_in_flight_ = false;
    if (execution_failed_) {
        throw std::runtime_error("TensorRT module is poisoned after an execution failure: " +
                                 execution_failure_);
    }
}

// --- Forward device async (GPU → GPU, no sync) ---

void TrtModuleImpl::forward_device_async(const DeviceTensorMap& inputs) {
    if (runtime_managed_context_)
        ensure_runtime_memory_ready();

    // D2D copy input DeviceTensors into our buffers
    for (const auto& [name, dt_ptr] : inputs) {
        auto it = buffers_.find(name);
        if (it == buffers_.end() || !dt_ptr)
            continue;
        auto& entry = it->second;
        if (!entry.is_input || !entry.d_ptr)
            continue;

        if (runtime_managed_context_ && entry.is_dynamic && dt_ptr->shape() != entry.shape) {
            throw std::invalid_argument("runtime input '" + name +
                                        "' does not match its planned concrete shape");
        }
        update_dynamic_shape(name, entry, dt_ptr->shape());

        auto copy_bytes = dt_ptr->nbytes();
        if (runtime_managed_context_) {
            if (dt_ptr->dtype() != entry.dtype)
                throw std::invalid_argument("runtime input '" + name + "' dtype mismatch");
            if (entry.is_dynamic) {
                const auto expected_bytes = compute_shape_bytes(entry.shape, entry.dtype, name);
                if (copy_bytes != expected_bytes) {
                    throw std::invalid_argument("runtime input '" + name +
                                                "' byte size does not match its planned shape");
                }
            }
            if (copy_bytes > entry.nbytes) {
                throw std::invalid_argument(
                    "runtime input '" + name + "' requires " + std::to_string(copy_bytes) +
                    " bytes but its materialized capacity is " + std::to_string(entry.nbytes));
            }
        } else {
            copy_bytes = std::min(copy_bytes, entry.nbytes);
        }
        if (dt_ptr->data() != entry.d_ptr) {
            if (copy_bytes > 0) {
                const auto copy_status = cudaMemcpyAsync(entry.d_ptr, dt_ptr->data(), copy_bytes,
                                                         cudaMemcpyDeviceToDevice, stream_);
                if (copy_status == cudaSuccess)
                    note_transfer(name, entry, cudaMemcpyDeviceToDevice, copy_bytes);
            }
        }
    }

    // Execute (no sync — caller will sync or run more kernels on same stream)
    execute_enqueue();
}

// --- Forward device (GPU → GPU, synchronous) ---

DeviceTensorMap TrtModuleImpl::forward_device(const DeviceTensorMap& inputs) {
    forward_device_async(inputs);
    sync();

    // Return non-owning DeviceTensor* pointers to our internal output buffers.
    // The output_device_tensors_ map is lazily populated on first call.
    DeviceTensorMap out;
    for (auto& [name, entry] : buffers_) {
        if (entry.is_input)
            continue;

        auto it = output_device_tensors_.find(name);
        if (it == output_device_tensors_.end()) {
            // Create a non-owning view. DeviceTensor constructor allocates memory,
            // so we create a placeholder and overwrite its pointer below.
            // Instead, just map the name to nullptr for now — callers use device_ptr().
        }
        out[name] = nullptr; // callers access via device_ptr(name)
    }
    return out;
}

// --- Introspection ---

std::vector<TensorInfo> TrtModuleImpl::input_info() const {
    std::vector<TensorInfo> result;
    for (const auto& [name, entry] : buffers_) {
        if (!entry.is_input)
            continue;
        TensorInfo ti;
        ti.name = name;
        ti.shape = entry.shape;
        ti.dtype = entry.dtype;
        ti.is_input = true;
        result.push_back(ti);
    }
    return result;
}

std::vector<TensorInfo> TrtModuleImpl::output_info() const {
    std::vector<TensorInfo> result;
    for (const auto& [name, entry] : buffers_) {
        if (entry.is_input)
            continue;
        TensorInfo ti;
        ti.name = name;
        ti.shape = entry.shape;
        ti.dtype = entry.dtype;
        ti.is_input = false;
        result.push_back(ti);
    }
    return result;
}

bool TrtModuleImpl::has_input(const std::string& name) const {
    auto it = buffers_.find(name);
    return it != buffers_.end() && it->second.is_input;
}

bool TrtModuleImpl::has_output(const std::string& name) const {
    auto it = buffers_.find(name);
    return it != buffers_.end() && !it->second.is_input;
}

DType TrtModuleImpl::tensor_dtype(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end())
        return DType::kFloat32;
    return it->second.dtype;
}

std::vector<int64_t> TrtModuleImpl::tensor_shape(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end())
        return {};
    return it->second.shape;
}

namespace {

nvinfer1::OptProfileSelector to_trt_selector(ProfileShapeSelector selector) {
    switch (selector) {
    case ProfileShapeSelector::kMin:
        return nvinfer1::OptProfileSelector::kMIN;
    case ProfileShapeSelector::kOpt:
        return nvinfer1::OptProfileSelector::kOPT;
    case ProfileShapeSelector::kMax:
        return nvinfer1::OptProfileSelector::kMAX;
    }
    return nvinfer1::OptProfileSelector::kOPT;
}

} // namespace

std::vector<int64_t> TrtModuleImpl::input_profile_shape(const std::string& name,
                                                        int32_t profile_idx,
                                                        ProfileShapeSelector selector) const {
    if (engine_ == nullptr || !has_input(name))
        return {};
    if (profile_idx < 0 || profile_idx >= engine_->getNbOptimizationProfiles())
        return {};
    const auto dims =
        engine_->getProfileShape(name.c_str(), profile_idx, to_trt_selector(selector));
    if (dims.nbDims < 0)
        return {};
    return dims_to_shape(dims);
}

int32_t TrtModuleImpl::optimization_profile_count() const {
    return engine_ != nullptr ? engine_->getNbOptimizationProfiles() : 0;
}

// --- Direct buffer access ---

void* TrtModuleImpl::device_ptr(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end())
        return nullptr;
    return it->second.d_ptr;
}

void TrtModuleImpl::bind_external(const std::string& name, void* external_device_ptr) {
    auto it = buffers_.find(name);
    if (it == buffers_.end())
        return;

    auto& entry = it->second;
    if (entry.is_runtime_deferred) {
        throw std::logic_error("runtime-deferred tensor '" + name +
                               "' requires RuntimeMemoryBindingV1");
    }
    if (runtime_managed_context_ && entry.is_runtime_internal_dynamic) {
        throw std::logic_error("ordinary dynamic tensor '" + name +
                               "' is owned by runtime-memory shape planning");
    }
    if (runtime_managed_context_)
        synchronize_runtime_reconfiguration("external binding");

    // Free our own buffer if we allocated it
    if (entry.d_ptr && !entry.is_external) {
        cudaFree(entry.d_ptr);
    }

    if (entry.d_ptr != external_device_ptr && use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
    if (runtime_managed_context_)
        invalidate_context_memory();
    entry.d_ptr = external_device_ptr;
    entry.is_external = true;
    entry.lifetime.reset();

    // Update execution context binding
    if (ctx_ && external_device_ptr)
        bind_tensor_address(name, entry);
}

} // namespace trtmc
