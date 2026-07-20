/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trt_module_impl.h"

#include "runtime/core/trt_common.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

[[noreturn]] void throw_cuda_failure(cudaError_t status, const std::string& operation) {
    throw std::runtime_error("[trt_module] " + operation +
                             " failed: " + cudaGetErrorString(status));
}

void require_cuda_success(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess)
        throw_cuda_failure(status, operation);
}

cudaPointerAttributes require_external_pointer_attributes(const ModuleExternalBinding& binding) {
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, binding.device_ptr);
    if (status != cudaSuccess) {
        // cudaPointerGetAttributes records invalid host/unregistered pointers in
        // CUDA's per-thread last-error slot on runtimes that return an error.
        (void)cudaGetLastError();
        throw std::invalid_argument("[trt_module] External buffer for '" + binding.tensor_name +
                                    "' is not CUDA device-accessible");
    }
    return attributes;
}

#if CUDART_VERSION >= 10000
void require_integrated_mapped_host_pointer(const ModuleExternalBinding& binding,
                                            const cudaPointerAttributes& attributes,
                                            int current_device) {
    int integrated = 0;
    int can_map_host_memory = 0;
    require_cuda_success(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, current_device),
                         "querying integrated-device capability");
    require_cuda_success(
        cudaDeviceGetAttribute(&can_map_host_memory, cudaDevAttrCanMapHostMemory, current_device),
        "querying mapped-host capability");
    // A registered/pinned CPU address is not enough. TensorRT receives the
    // exact CUDA alias returned for a mapped allocation, and host-backed
    // bindings are accepted only on an integrated GPU where this avoids
    // consuming the cudaMalloc pool rather than introducing PCIe I/O.
    if (integrated != 0 && can_map_host_memory != 0 && attributes.devicePointer != nullptr &&
        attributes.devicePointer == binding.device_ptr) {
        return;
    }
    throw std::invalid_argument("[trt_module] External buffer for '" + binding.tensor_name +
                                "' is not mapped host memory on the current integrated CUDA "
                                "device");
}
#endif

void validate_external_device_pointer(const ModuleExternalBinding& binding) {
    int current_device = 0;
    require_cuda_success(cudaGetDevice(&current_device), "querying the current CUDA device");

    const cudaPointerAttributes attributes = require_external_pointer_attributes(binding);

#if CUDART_VERSION >= 10000
    if (attributes.type == cudaMemoryTypeManaged)
        return;
    if (attributes.type == cudaMemoryTypeHost) {
        require_integrated_mapped_host_pointer(binding, attributes, current_device);
        return;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != current_device) {
#else
    if (attributes.isManaged)
        return;
    if (attributes.memoryType != cudaMemoryTypeDevice || attributes.device != current_device) {
#endif
        throw std::invalid_argument("[trt_module] External buffer for '" + binding.tensor_name +
                                    "' is not device memory on the current CUDA device");
    }
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
        return DType::kUnsupported;
    }
}

// --- Construction ---

TrtModuleImpl::TrtModuleImpl(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* ctx,
                             cudaStream_t stream, int32_t profile_idx,
                             void* distributed_communicator,
                             const std::vector<ModuleExternalBinding>& external_bindings)
    : engine_(engine), ctx_(ctx), stream_(stream), profile_idx_(profile_idx),
      distributed_communicator_(distributed_communicator),
      cuda_graph_(std::make_unique<CudaGraphExec>()) {
    if (!engine_ || !ctx_) {
        if (ctx_) {
            delete ctx_;
            ctx_ = nullptr;
        }
        return;
    }

    try {
        validate_initial_external_bindings(engine, external_bindings);
        if (!attach_distributed_communicator())
            throw std::runtime_error("[trt_module] Failed to attach distributed communicator");
        if (profile_idx_ > 0) {
            if (!ctx_->setOptimizationProfileAsync(profile_idx_, stream_)) {
                throw std::runtime_error("[trt_module] Failed to set optimization profile " +
                                         std::to_string(profile_idx_));
            }
            require_cuda_success(cudaStreamSynchronize(stream_),
                                 "synchronizing optimization profile selection");
        }
        allocate_buffers(engine);
    } catch (const std::exception& error) {
        std::cerr << "[trt_module] Module initialization failed: " << error.what() << '\n';
        free_buffers();
        delete ctx_;
        ctx_ = nullptr;
    } catch (...) {
        std::cerr << "[trt_module] Module initialization failed with an unknown error\n";
        free_buffers();
        delete ctx_;
        ctx_ = nullptr;
    }
}

bool TrtModuleImpl::engine_has_io_tensor(nvinfer1::ICudaEngine* engine, const std::string& name) {
    for (int32_t index = 0; index < engine->getNbIOTensors(); ++index) {
        const char* candidate = engine->getIOTensorName(index);
        if (candidate != nullptr && name == candidate)
            return true;
    }
    return false;
}

std::size_t TrtModuleImpl::validate_initial_external_binding(nvinfer1::ICudaEngine* engine,
                                                             const ModuleExternalBinding& binding) {
    if (binding.tensor_name.empty())
        throw std::invalid_argument("[trt_module] Cannot prebind an unnamed tensor");
    if (!binding.device_ptr) {
        throw std::invalid_argument("[trt_module] Cannot prebind null external buffer for '" +
                                    binding.tensor_name + "'");
    }
    if (!engine_has_io_tensor(engine, binding.tensor_name)) {
        throw std::invalid_argument("[trt_module] Cannot prebind unknown tensor '" +
                                    binding.tensor_name + "'");
    }

    const auto tensor_shape = engine->getTensorShape(binding.tensor_name.c_str());
    if (dims_are_dynamic(tensor_shape)) {
        throw std::invalid_argument("[trt_module] Cannot prebind dynamic tensor '" +
                                    binding.tensor_name +
                                    "'; construction-time prebinding is static-only");
    }
    std::vector<int64_t> static_shape;
    const auto required_bytes = compute_alloc_bytes(
        tensor_shape, from_trt_dtype(engine->getTensorDataType(binding.tensor_name.c_str())),
        static_shape);
    if (binding.capacity_bytes < required_bytes) {
        throw std::invalid_argument("[trt_module] External buffer for '" + binding.tensor_name +
                                    "' provides " + std::to_string(binding.capacity_bytes) +
                                    " bytes, but the static tensor requires " +
                                    std::to_string(required_bytes));
    }
    validate_external_device_pointer(binding);
    return required_bytes;
}

void TrtModuleImpl::validate_initial_external_bindings(
    nvinfer1::ICudaEngine* engine, const std::vector<ModuleExternalBinding>& external_bindings) {
    initial_external_bindings_.clear();
    initial_external_bindings_.reserve(external_bindings.size());
    std::size_t prebound_bytes = 0;
    for (const auto& binding : external_bindings) {
        const auto required_bytes = validate_initial_external_binding(engine, binding);
        const bool inserted =
            initial_external_bindings_.emplace(binding.tensor_name, binding.device_ptr).second;
        if (!inserted) {
            throw std::invalid_argument("[trt_module] Duplicate prebinding for tensor '" +
                                        binding.tensor_name + "'");
        }
        if (prebound_bytes <= std::numeric_limits<std::size_t>::max() - required_bytes)
            prebound_bytes += required_bytes;
    }
    if (!external_bindings.empty()) {
        std::cerr << "[trt_module] Prebound " << external_bindings.size() << " static I/O buffers ("
                  << prebound_bytes
                  << " bytes); backend-owned I/O allocation is skipped, engine/context memory is "
                     "unchanged\n";
    }
}

TrtModuleImpl::BufferEntry& TrtModuleImpl::require_shaped_external_target(const std::string& name,
                                                                          void* ptr) {
    require_ready("binding shaped external buffer");
    auto it = buffers_.find(name);
    if (it == buffers_.end())
        throw std::invalid_argument("[trt_module] Cannot bind unknown tensor '" + name + "'");
    if (!ptr)
        throw std::invalid_argument("[trt_module] Cannot bind null external buffer for '" + name +
                                    "'");
    validate_external_device_pointer(ModuleExternalBinding{name, ptr, 0});
    return it->second;
}

void TrtModuleImpl::update_shape_with_binding_rollback(const std::string& name, BufferEntry& entry,
                                                       const std::vector<int64_t>& shape) {
    try {
        update_dynamic_shape(name, entry, shape);
    } catch (...) {
        const auto failure = std::current_exception();
        try {
            rebind_buffer_to_context(name, entry);
        } catch (...) {
            delete ctx_;
            ctx_ = nullptr;
        }
        std::rethrow_exception(failure);
    }
}

void TrtModuleImpl::bind_external(const std::string& name, void* ptr,
                                  const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        bind_external(name, ptr);
        return;
    }

    auto& entry = require_shaped_external_target(name, ptr);
    if (!entry.is_input)
        throw std::invalid_argument("[trt_module] Cannot set a shape for output '" + name + "'");
    if (entry.d_ptr == ptr) {
        if (!entry.is_external) {
            throw std::invalid_argument(
                "[trt_module] Cannot reclassify an owned buffer as external for '" + name + "'");
        }
        update_shape_with_binding_rollback(name, entry, shape);
        return;
    }

    // Bind the prospective address before changing either the tracked buffer
    // or its shape. If TensorRT rejects the shape, restore both context
    // bindings from the unchanged entry and leave ownership untouched.
    BufferEntry replacement = entry;
    replacement.d_ptr = ptr;
    replacement.is_external = true;
    bind_tensor_address(name, replacement);
    update_shape_with_binding_rollback(name, entry, shape);

    void* const old_owned_ptr = entry.is_external ? nullptr : entry.d_ptr;
    entry.d_ptr = ptr;
    entry.is_external = true;
    if (old_owned_ptr) {
        require_cuda_success(cudaFree(old_owned_ptr),
                             "releasing replaced buffer for '" + name + "'");
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

void TrtModuleImpl::recreate_context_with_profile() {
    require_ready("resetting the execution context");
    delete ctx_;
    ctx_ = engine_->createExecutionContext();
    if (!ctx_)
        throw std::runtime_error("[trt_module] Failed to recreate execution context");
    if (!attach_distributed_communicator()) {
        delete ctx_;
        ctx_ = nullptr;
        throw std::runtime_error("[trt_module] Failed to attach distributed communicator");
    }
    if (engine_->getNbOptimizationProfiles() > 0) {
        if (!ctx_->setOptimizationProfileAsync(profile_idx_, stream_)) {
            delete ctx_;
            ctx_ = nullptr;
            throw std::runtime_error("[trt_module] Failed to reset optimization profile");
        }
        require_cuda_success(cudaStreamSynchronize(stream_),
                             "synchronizing optimization profile reset");
    }
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

void TrtModuleImpl::bind_tensor_address(const std::string& name, const BufferEntry& entry) {
    require_ready("binding tensor address");
    if (!entry.d_ptr)
        throw std::invalid_argument("[trt_module] Cannot bind null tensor address for '" + name +
                                    "'");
    const bool ok = entry.is_input ? ctx_->setInputTensorAddress(name.c_str(), entry.d_ptr)
                                   : ctx_->setOutputTensorAddress(name.c_str(), entry.d_ptr);
    if (!ok)
        throw std::runtime_error("[trt_module] Failed to bind " +
                                 std::string(entry.is_input ? "input" : "output") +
                                 " tensor address for '" + name + "'");
}

void TrtModuleImpl::rebind_buffer_to_context(const std::string& name, const BufferEntry& entry) {
    // Fresh IExecutionContexts have neither tensor addresses nor dynamic
    // input shapes set; replay both from our cached BufferEntry so the next
    // enqueueV3 doesn't fail with "Not all shapes are specified".
    if (entry.d_ptr)
        bind_tensor_address(name, entry);
    if (!entry.is_dynamic || entry.shape.empty())
        return;
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(entry.shape.size());
    for (int32_t d = 0; d < dims.nbDims; ++d)
        dims.d[d] = entry.shape[d];
    if (!ctx_->setInputShape(name.c_str(), dims))
        throw std::runtime_error("[trt_module] Failed to restore input shape for '" + name + "'");
}

void TrtModuleImpl::reset_execution_context() {
    if (engine_ == nullptr)
        return;
    try {
        recreate_context_with_profile();
        if (use_cuda_graph_ && cuda_graph_)
            cuda_graph_->reset();
        for (auto& [name, entry] : buffers_)
            rebind_buffer_to_context(name, entry);
    } catch (...) {
        delete ctx_;
        ctx_ = nullptr;
        throw;
    }
}

TrtModuleImpl::~TrtModuleImpl() {
    flush_timing_events();
    free_buffers();
    delete ctx_;
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

void TrtModuleImpl::require_valid_dynamic_input_shape(const std::string& name,
                                                      const std::vector<int64_t>& shape) {
    nvinfer1::Dims dims;
    const auto max_rank = sizeof(dims.d) / sizeof(dims.d[0]);
    if (shape.empty() || shape.size() > max_rank)
        throw std::invalid_argument("[trt_module] Invalid input rank for '" + name + "'");
    for (const int64_t dimension : shape) {
        if (dimension <= 0 || dimension > std::numeric_limits<int32_t>::max())
            throw std::invalid_argument("[trt_module] Invalid input dimension for '" + name + "'");
    }
}

nvinfer1::Dims TrtModuleImpl::make_trt_dims(const std::vector<int64_t>& shape) {
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (int32_t d = 0; d < dims.nbDims; ++d)
        dims.d[d] = shape[d];
    return dims;
}

void TrtModuleImpl::update_dynamic_shape(const std::string& name, BufferEntry& entry,
                                         const std::vector<int64_t>& new_shape) {
    if (!entry.is_input)
        throw std::invalid_argument("[trt_module] Cannot set a shape for output '" + name + "'");
    if (!entry.is_dynamic) {
        if (new_shape != entry.shape) {
            throw std::invalid_argument("[trt_module] Static input shape mismatch for '" + name +
                                        "'");
        }
        return;
    }
    require_valid_dynamic_input_shape(name, new_shape);
    if (new_shape == entry.shape)
        return;
    // Any captured CUDA graph was baked against the OLD shape; force a
    // re-capture on the next enqueue so the new shape actually takes.
    if (use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
    const auto dims = make_trt_dims(new_shape);
    if (!ctx_->setInputShape(name.c_str(), dims))
        throw std::invalid_argument("[trt_module] TensorRT rejected input shape for '" + name +
                                    "'");
    entry.shape = new_shape;
}

std::size_t TrtModuleImpl::compute_alloc_bytes(const nvinfer1::Dims& dims, DType dtype,
                                               std::vector<int64_t>& shape_out) {
    if (dims.nbDims < 0)
        throw std::runtime_error("[trt_module] TensorRT returned an invalid tensor rank");
    shape_out.clear();
    std::size_t n = 1;
    for (int32_t d = 0; d < dims.nbDims; ++d) {
        const int64_t dim = std::max(static_cast<int64_t>(dims.d[d]), int64_t{1});
        if (n > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim))
            throw std::overflow_error("[trt_module] Tensor allocation size overflow");
        shape_out.push_back(dim);
        n *= static_cast<std::size_t>(dim);
    }
    const auto element_size = dtype_size(dtype);
    if (element_size == 0 || n > std::numeric_limits<std::size_t>::max() / element_size)
        throw std::overflow_error("[trt_module] Tensor allocation byte size overflow");
    return n * element_size;
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
            if (!ctx_->setInputShape(name.c_str(), dims)) {
                throw std::runtime_error("[trt_module] Failed to set profile shape for input '" +
                                         name + "'");
            }
        }
    }
}

TrtModuleImpl::BufferEntry
TrtModuleImpl::make_input_buffer_entry(nvinfer1::ICudaEngine* engine, const std::string& name,
                                       int32_t num_profiles, nvinfer1::Dims& initial_dims) const {
    const auto trt_shape = engine->getTensorShape(name.c_str());
    const auto dtype = from_trt_dtype(engine->getTensorDataType(name.c_str()));
    // Determine allocation shape (max) and initial runtime shape (opt).
    nvinfer1::Dims alloc_dims = trt_shape;
    initial_dims = trt_shape;
    const bool is_dynamic = has_dynamic_shapes_ && num_profiles > 0 && dims_are_dynamic(trt_shape);

    if (is_dynamic) {
        alloc_dims =
            engine->getProfileShape(name.c_str(), profile_idx_, nvinfer1::OptProfileSelector::kMAX);
        initial_dims =
            engine->getProfileShape(name.c_str(), profile_idx_, nvinfer1::OptProfileSelector::kOPT);
    }

    std::vector<int64_t> shape;
    const std::size_t nbytes = compute_alloc_bytes(alloc_dims, dtype, shape);

    BufferEntry entry;
    entry.dtype = dtype;
    entry.nbytes = nbytes;
    entry.is_input = true;
    entry.is_dynamic = is_dynamic;
    entry.shape = is_dynamic ? dims_to_shape(initial_dims) : shape;
    return entry;
}

TrtModuleImpl::BufferEntry TrtModuleImpl::make_output_buffer_entry(nvinfer1::ICudaEngine* engine,
                                                                   const std::string& name) const {
    const auto dtype = from_trt_dtype(engine->getTensorDataType(name.c_str()));
    // For dynamic engines, query the context for inferred output shape
    // (based on the max input shapes set by the caller).
    // For static engines, use the engine shape directly.
    const nvinfer1::Dims out_dims = has_dynamic_shapes_ ? ctx_->getTensorShape(name.c_str())
                                                        : engine->getTensorShape(name.c_str());
    std::vector<int64_t> shape;
    const std::size_t nbytes = compute_alloc_bytes(out_dims, dtype, shape);

    BufferEntry entry;
    entry.shape = std::move(shape);
    entry.dtype = dtype;
    entry.nbytes = nbytes;
    entry.is_input = false;
    return entry;
}

void TrtModuleImpl::attach_or_allocate_buffer(const std::string& name, const char* tensor_kind,
                                              BufferEntry& entry) {
    const auto external = initial_external_bindings_.find(name);
    if (external != initial_external_bindings_.end()) {
        entry.d_ptr = external->second;
        entry.is_external = true;
    } else if (entry.nbytes > 0) {
        require_cuda_success(cudaMalloc(&entry.d_ptr, entry.nbytes),
                             "allocating " + std::string(tensor_kind) + " '" + name + "'");
        const auto memset_status = cudaMemsetAsync(entry.d_ptr, 0, entry.nbytes, stream_);
        if (memset_status != cudaSuccess) {
            (void)cudaFree(entry.d_ptr);
            entry.d_ptr = nullptr;
            throw_cuda_failure(memset_status,
                               "initializing " + std::string(tensor_kind) + " '" + name + "'");
        }
    }
}

void TrtModuleImpl::release_owned_buffer(BufferEntry& entry) noexcept {
    if (entry.d_ptr && !entry.is_external)
        (void)cudaFree(entry.d_ptr);
    entry.d_ptr = nullptr;
}

void TrtModuleImpl::bind_buffer_or_release(const std::string& name, BufferEntry& entry) {
    if (!entry.d_ptr)
        return;
    try {
        bind_tensor_address(name, entry);
    } catch (...) {
        release_owned_buffer(entry);
        throw;
    }
}

void TrtModuleImpl::initialize_dynamic_input_shape_or_release(const std::string& name,
                                                              BufferEntry& entry,
                                                              const nvinfer1::Dims& initial_dims) {
    if (!entry.is_dynamic)
        return;
    if (!ctx_->setInputShape(name.c_str(), initial_dims)) {
        release_owned_buffer(entry);
        throw std::runtime_error("[trt_module] Failed to initialize dynamic input shape for '" +
                                 name + "'");
    }
}

void TrtModuleImpl::store_buffer_or_release(const std::string& name, BufferEntry& entry) {
    try {
        buffers_[name] = std::move(entry);
    } catch (...) {
        release_owned_buffer(entry);
        throw;
    }
}

void TrtModuleImpl::allocate_single_input(nvinfer1::ICudaEngine* engine, const std::string& name,
                                          int32_t num_profiles) {
    nvinfer1::Dims initial_dims;
    auto entry = make_input_buffer_entry(engine, name, num_profiles, initial_dims);
    attach_or_allocate_buffer(name, "input", entry);
    bind_buffer_or_release(name, entry);
    initialize_dynamic_input_shape_or_release(name, entry, initial_dims);
    store_buffer_or_release(name, entry);
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

void TrtModuleImpl::allocate_single_output(nvinfer1::ICudaEngine* engine, const std::string& name) {
    auto entry = make_output_buffer_entry(engine, name);
    attach_or_allocate_buffer(name, "output", entry);
    bind_buffer_or_release(name, entry);
    const std::size_t nbytes = entry.nbytes;
    const bool is_external = entry.is_external;
    store_buffer_or_release(name, entry);
    if (nbytes > 0 && !is_external)
        host_output_staging_[name].resize(nbytes);
}

void TrtModuleImpl::allocate_output_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io) {
    for (int32_t i = 0; i < num_io; ++i) {
        const char* raw_name = engine->getIOTensorName(i);
        if (raw_name == nullptr)
            continue;
        const std::string name(raw_name);
        if (engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT)
            continue;
        allocate_single_output(engine, name);
    }
}

// --- Buffer allocation ---

void TrtModuleImpl::allocate_buffers(nvinfer1::ICudaEngine* engine) {
    const int32_t num_io = engine->getNbIOTensors();
    const int32_t num_profiles = engine->getNbOptimizationProfiles();

    detect_dynamic_shapes(engine, num_io);

    // Pass 1: allocate input buffers (use profile-0 max shape for dynamic inputs).
    allocate_input_buffers(engine, num_io, num_profiles);

    // Pass 2: allocate output buffers. For dynamic shapes, temporarily set
    // inputs to max shapes, query inferred output shapes, then restore opt.
    if (has_dynamic_shapes_ && num_profiles > 0)
        set_dynamic_input_shapes(engine, num_io, nvinfer1::OptProfileSelector::kMAX);

    allocate_output_buffers(engine, num_io);

    if (has_dynamic_shapes_ && num_profiles > 0)
        set_dynamic_input_shapes(engine, num_io, nvinfer1::OptProfileSelector::kOPT);

    require_cuda_success(cudaStreamSynchronize(stream_), "synchronizing buffer initialization");
    initial_external_bindings_.clear();
}

void TrtModuleImpl::free_buffers() {
    for (auto& [name, entry] : buffers_) {
        if (entry.d_ptr && !entry.is_external) {
            cudaFree(entry.d_ptr);
        }
        entry.d_ptr = nullptr;
    }
    buffers_.clear();
    initial_external_bindings_.clear();
    host_output_staging_.clear();
    output_device_tensors_.clear();
}

void TrtModuleImpl::require_ready(const char* operation) const {
    if (ctx_ == nullptr)
        throw std::runtime_error(std::string("[trt_module] Cannot continue while ") + operation +
                                 ": module is not initialized");
}

void TrtModuleImpl::require_all_host_inputs(const TensorMap& inputs) const {
    require_ready("validating host inputs");
    for (const auto& [name, entry] : buffers_) {
        if (entry.is_input && !entry.is_external && inputs.find(name) == inputs.end())
            throw std::invalid_argument("[trt_module] Missing required input '" + name + "'");
    }
}

// --- Forward pass (CPU → GPU → CPU) ---

TensorMap TrtModuleImpl::forward(const TensorMap& inputs) {
    forward_async(inputs);
    sync();

    // Download outputs — skip externally-bound buffers (they stay on device)
    TensorMap outputs;
    for (auto& [name, entry] : buffers_) {
        if (entry.is_input)
            continue;
        if (entry.is_external)
            continue;

        std::vector<int64_t> runtime_shape = entry.shape;
        std::size_t runtime_nbytes = entry.nbytes;
        if (has_dynamic_shapes_) {
            std::vector<int64_t> inferred_shape;
            runtime_nbytes = compute_alloc_bytes(ctx_->getTensorShape(name.c_str()), entry.dtype,
                                                 inferred_shape);
            if (runtime_nbytes > entry.nbytes)
                throw std::runtime_error("[trt_module] Runtime output exceeds allocation for '" +
                                         name + "'");
            runtime_shape = std::move(inferred_shape);
        }

        if (!entry.d_ptr)
            throw std::runtime_error("[trt_module] Output '" + name + "' has no device buffer");
        auto staging = host_output_staging_.find(name);
        if (staging == host_output_staging_.end() || staging->second.size() < runtime_nbytes)
            throw std::runtime_error("[trt_module] Output staging is unavailable for '" + name +
                                     "'");
        require_cuda_success(
            cudaMemcpy(staging->second.data(), entry.d_ptr, runtime_nbytes, cudaMemcpyDeviceToHost),
            "copying output '" + name + "' from device");

        Tensor t;
        t.data = staging->second.data();
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
    require_all_host_inputs(inputs);
    // Upload inputs H2D, updating shapes for dynamic engines
    for (const auto& [name, tensor] : inputs) {
        auto it = buffers_.find(name);
        if (it == buffers_.end())
            throw std::invalid_argument("[trt_module] Unknown input '" + name + "'");
        auto& entry = it->second;
        if (!entry.is_input)
            throw std::invalid_argument("[trt_module] Tensor '" + name + "' is not an input");
        if (!entry.d_ptr)
            throw std::runtime_error("[trt_module] Input '" + name + "' has no device buffer");
        if (!tensor.data)
            throw std::invalid_argument("[trt_module] Input '" + name + "' has null host data");
        if (tensor.dtype != entry.dtype)
            throw std::invalid_argument("[trt_module] Input dtype mismatch for '" + name + "'");

        update_dynamic_shape(name, entry, tensor.shape);

        const auto copy_bytes = tensor.nbytes();
        if (copy_bytes == 0 || copy_bytes > entry.nbytes)
            throw std::invalid_argument("[trt_module] Input byte size mismatch for '" + name + "'");
        require_cuda_success(
            cudaMemcpyAsync(entry.d_ptr, tensor.data, copy_bytes, cudaMemcpyHostToDevice, stream_),
            "copying input '" + name + "' to device");
    }

    execute_enqueue();
}

void TrtModuleImpl::execute_enqueue() {
    record_timed_enqueue();
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

void TrtModuleImpl::finish_timing_event(TimingEvent event) {
    if (event.start && event.stop && cudaEventRecord(event.stop, stream_) == cudaSuccess) {
        timing_events_.push_back(event);
        return;
    }
    if (event.start)
        cudaEventDestroy(event.start);
    if (event.stop)
        cudaEventDestroy(event.stop);
}

void TrtModuleImpl::discard_timing_event(TimingEvent& event) noexcept {
    if (event.start)
        cudaEventDestroy(event.start);
    if (event.stop)
        cudaEventDestroy(event.stop);
    event.start = nullptr;
    event.stop = nullptr;
}

void TrtModuleImpl::enqueue_or_throw() {
    if (!ctx_->enqueueV3(stream_))
        throw std::runtime_error("[trt_module] TensorRT enqueueV3 failed");
}

void TrtModuleImpl::launch_cuda_graph_or_throw() {
    if (!cuda_graph_->launch(stream_))
        throw std::runtime_error("[trt_module] CUDA graph launch failed");
}

void TrtModuleImpl::capture_cuda_graph_or_fallback() {
    if (!cuda_graph_->begin_capture(stream_)) {
        std::cerr << "[cuda_graph] Capture start failed, disabling CUDA Graphs\n";
        use_cuda_graph_ = false;
        enqueue_or_throw();
        return;
    }
    if (!ctx_->enqueueV3(stream_)) {
        (void)cuda_graph_->end_capture(stream_);
        cuda_graph_->reset();
        throw std::runtime_error(
            "[trt_module] TensorRT enqueueV3 failed during CUDA graph capture");
    }
    if (!cuda_graph_->end_capture(stream_)) {
        std::cerr << "[cuda_graph] Capture failed, disabling CUDA Graphs\n";
        use_cuda_graph_ = false;
        enqueue_or_throw();
        return;
    }
    launch_cuda_graph_or_throw();
}

void TrtModuleImpl::enqueue_with_optional_cuda_graph() {
    if (!use_cuda_graph_) {
        enqueue_or_throw();
        return;
    }
    if (cuda_graph_->ready()) {
        launch_cuda_graph_or_throw();
        return;
    }
    capture_cuda_graph_or_fallback();
}

void TrtModuleImpl::record_timed_enqueue() {
    require_ready("enqueueing inference");
    TimingEvent timing_event;
    const bool timing_ok = begin_timing_event(timing_event);

    try {
        enqueue_with_optional_cuda_graph();
        if (timing_ok)
            finish_timing_event(timing_event);
    } catch (...) {
        discard_timing_event(timing_event);
        throw;
    }
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
    require_ready("synchronizing inference");
    require_cuda_success(cudaStreamSynchronize(stream_), "synchronizing inference stream");
}

// --- Forward device async (GPU → GPU, no sync) ---

TrtModuleImpl::BufferEntry& TrtModuleImpl::require_device_input(const std::string& name,
                                                                const DeviceTensor* tensor) {
    auto it = buffers_.find(name);
    if (it == buffers_.end())
        throw std::invalid_argument("[trt_module] Unknown device input '" + name + "'");
    if (!tensor)
        throw std::invalid_argument("[trt_module] Device input '" + name + "' is null");

    auto& entry = it->second;
    if (!entry.is_input)
        throw std::invalid_argument("[trt_module] Tensor '" + name + "' is not an input");
    if (!entry.d_ptr)
        throw std::runtime_error("[trt_module] Device input '" + name +
                                 "' has no destination buffer");
    if (!tensor->data())
        throw std::invalid_argument("[trt_module] Device input '" + name + "' has null data");
    if (tensor->dtype() != entry.dtype)
        throw std::invalid_argument("[trt_module] Device input dtype mismatch for '" + name + "'");
    return entry;
}

void TrtModuleImpl::forward_device_async(const DeviceTensorMap& inputs) {
    require_ready("validating device inputs");
    // D2D copy input DeviceTensors into our buffers
    //
    // An omitted entry is intentional in this API: performance-sensitive
    // callers may populate an internal input through device_ptr() or bind it
    // externally, then enqueue with an empty map. Supplied entries are still
    // validated strictly below.
    for (const auto& [name, dt_ptr] : inputs) {
        auto& entry = require_device_input(name, dt_ptr);

        update_dynamic_shape(name, entry, dt_ptr->shape());

        const auto copy_bytes = dt_ptr->nbytes();
        if (copy_bytes == 0 || copy_bytes > entry.nbytes)
            throw std::invalid_argument("[trt_module] Device input byte size mismatch for '" +
                                        name + "'");
        if (dt_ptr->data() != entry.d_ptr) {
            require_cuda_success(cudaMemcpyAsync(entry.d_ptr, dt_ptr->data(), copy_bytes,
                                                 cudaMemcpyDeviceToDevice, stream_),
                                 "copying device input '" + name + "'");
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
        return DType::kUnsupported;
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

bool TrtModuleImpl::owns_device_buffer(const std::string& name) const {
    const auto it = buffers_.find(name);
    return it != buffers_.end() && it->second.d_ptr != nullptr && !it->second.is_external;
}

bool TrtModuleImpl::has_host_output_staging(const std::string& name) const {
    return host_output_staging_.find(name) != host_output_staging_.end();
}

void TrtModuleImpl::bind_external(const std::string& name, void* external_device_ptr) {
    require_ready("binding external buffer");
    auto it = buffers_.find(name);
    if (it == buffers_.end())
        throw std::invalid_argument("[trt_module] Cannot bind unknown tensor '" + name + "'");
    if (!external_device_ptr)
        throw std::invalid_argument("[trt_module] Cannot bind null external buffer for '" + name +
                                    "'");
    validate_external_device_pointer(ModuleExternalBinding{name, external_device_ptr, 0});

    auto& entry = it->second;
    if (entry.d_ptr == external_device_ptr) {
        if (entry.is_external)
            return;
        throw std::invalid_argument(
            "[trt_module] Cannot reclassify an owned buffer as external for '" + name + "'");
    }

    BufferEntry replacement = entry;
    replacement.d_ptr = external_device_ptr;
    replacement.is_external = true;
    bind_tensor_address(name, replacement);

    void* const old_owned_ptr = entry.is_external ? nullptr : entry.d_ptr;
    entry.d_ptr = external_device_ptr;
    entry.is_external = true;

    // Externally-bound outputs intentionally remain on device (KV caches and
    // recurrent diffusion state are the common cases).  Their host staging
    // allocation is therefore dead memory and can be very large; release it
    // when the binding becomes external instead of retaining a second copy
    // for the lifetime of the module.
    if (!entry.is_input)
        host_output_staging_.erase(name);

    if (old_owned_ptr)
        require_cuda_success(cudaFree(old_owned_ptr),
                             "releasing replaced buffer for '" + name + "'");
}

} // namespace trtmc
