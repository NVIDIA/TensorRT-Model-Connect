/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trt_module_impl.h"

#include "runtime/core/trt_common.h"
#include "trtmc/runtime/measurement.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
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

void allocate_host_output_staging(
    std::unordered_map<std::string, std::vector<uint8_t>>& host_output_staging,
    const std::string& name, std::size_t nbytes, bool is_external) {
    if (nbytes > 0 && !is_external)
        host_output_staging[name].resize(nbytes);
}

bool engine_timing_enabled_from_environment() {
    const char* value = std::getenv("TRTMC_ENGINE_TIMING");
    if (value == nullptr || std::strcmp(value, "1") == 0)
        return true;
    if (std::strcmp(value, "0") == 0) {
        static std::once_flag logged;
        std::call_once(logged, [] {
            std::cerr << "[trtmc.engine_timing] collection=disabled "
                         "source=TRTMC_ENGINE_TIMING=0\n";
        });
        return false;
    }
    throw std::invalid_argument("TRTMC_ENGINE_TIMING must be exactly 0 or 1");
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
                             const std::vector<ModuleExternalBinding>& external_bindings)
    : engine_(engine), ctx_(ctx), stream_(stream), profile_idx_(profile_idx),
      distributed_communicator_(distributed_communicator),
      cuda_graph_(std::make_unique<CudaGraphExec>()) {
    if (!ctx_)
        return;
    try {
        // Read the process policy exactly once for this module. Later
        // environment changes cannot alter an already-created execution path.
        timing_enabled_ = engine_timing_enabled_from_environment();
    } catch (const std::exception& error) {
        std::cerr << "[trt_module] Invalid engine timing configuration: " << error.what() << '\n';
        delete ctx_;
        ctx_ = nullptr;
        return;
    }
    try {
        discover_tensor_aliases(engine);
        validate_initial_external_bindings(engine, external_bindings);
    } catch (const std::exception& error) {
        std::cerr << "[trt_module] Invalid external binding: " << error.what() << '\n';
        delete ctx_;
        ctx_ = nullptr;
        return;
    }
    if (!attach_distributed_communicator()) {
        delete ctx_;
        ctx_ = nullptr;
        return;
    }
    if (profile_idx_ > 0) {
        if (!ctx_->setOptimizationProfileAsync(profile_idx_, stream_)) {
            std::cerr << "[trt_module] Failed to set optimization profile " << profile_idx_ << "\n";
            delete ctx_;
            ctx_ = nullptr;
            return;
        }
        cudaStreamSynchronize(stream_);
    }
    allocate_buffers(engine);
}

void TrtModuleImpl::discover_tensor_aliases(nvinfer1::ICudaEngine* engine) {
#if NV_TENSORRT_MAJOR >= 11
    for (int32_t index = 0; index < engine->getNbIOTensors(); ++index) {
        const char* raw_output_name = engine->getIOTensorName(index);
        if (raw_output_name == nullptr ||
            engine->getTensorIOMode(raw_output_name) != nvinfer1::TensorIOMode::kOUTPUT) {
            continue;
        }
        const char* raw_input_name = engine->getAliasedInputTensor(raw_output_name);
        if (raw_input_name == nullptr)
            continue;
        if (!engine_has_io_tensor(engine, raw_input_name) ||
            engine->getTensorIOMode(raw_input_name) != nvinfer1::TensorIOMode::kINPUT) {
            throw std::invalid_argument("Aliased output '" + std::string(raw_output_name) +
                                        "' refers to invalid input '" +
                                        std::string(raw_input_name) + "'");
        }
        const std::string output_name(raw_output_name);
        const std::string input_name(raw_input_name);
        alias_input_by_output_.emplace(output_name, input_name);
        alias_outputs_by_input_[input_name].push_back(output_name);
        alias_groups_ready_ = false;
    }
#else
    (void)engine;
#endif
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
        if (alias_input_by_output_.count(binding.tensor_name) != 0 ||
            alias_outputs_by_input_.count(binding.tensor_name) != 0) {
            throw std::invalid_argument("TensorRT alias tensor '" + binding.tensor_name +
                                        "' must be bound after module creation");
        }

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
    flush_timing_events();
    // CUDA Graphs may contain TensorRT collective launches that retain the
    // distributed communicator. Destroy the captured graph before member
    // teardown releases distributed_owner from keep_alive_.
    if (cuda_graph_)
        cuda_graph_->reset();
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

void TrtModuleImpl::update_dynamic_shape(const std::string& name, BufferEntry& entry,
                                         const std::vector<int64_t>& new_shape) {
    // Skip static inputs: TRT rejects setInputShape on them even when the
    // engine as a whole advertises dynamic shapes via optimization profiles.
    if (!has_dynamic_shapes_ || !entry.is_dynamic || new_shape == entry.shape)
        return;
    // Any captured CUDA graph was baked against the OLD shape; force a
    // re-capture on the next enqueue so the new shape actually takes.
    if (use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(new_shape.size());
    for (int32_t d = 0; d < dims.nbDims; ++d)
        dims.d[d] = new_shape[d];
    ctx_->setInputShape(name.c_str(), dims);
    entry.shape = new_shape;
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
        alloc_dims =
            engine->getProfileShape(name.c_str(), profile_idx_, nvinfer1::OptProfileSelector::kMAX);
        init_dims =
            engine->getProfileShape(name.c_str(), profile_idx_, nvinfer1::OptProfileSelector::kOPT);
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
    } else if (should_allocate_input(name, nbytes)) {
        auto err = cudaMalloc(&entry.d_ptr, nbytes);
        if (err != cudaSuccess)
            entry.d_ptr = nullptr;
        else
            cudaMemsetAsync(entry.d_ptr, 0, nbytes, stream_);
    }

    if (entry.d_ptr)
        bind_tensor_address(name, entry);

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

        // For dynamic engines, query the context for inferred output shape
        // (based on the max input shapes set by the caller).
        // For static engines, use the engine shape directly.
        nvinfer1::Dims out_dims = has_dynamic_shapes_ ? ctx_->getTensorShape(name.c_str())
                                                      : engine->getTensorShape(name.c_str());

        std::vector<int64_t> shape;
        std::size_t nbytes = compute_alloc_bytes(out_dims, dtype, shape);

        BufferEntry entry;
        entry.shape = shape;
        entry.dtype = dtype;
        entry.nbytes = nbytes;
        entry.is_input = false;

        if (alias_input_by_output_.count(name) != 0) {
            buffers_[name] = std::move(entry);
            continue;
        }

        const auto external = initial_external_bindings_.find(name);
        if (external != initial_external_bindings_.end()) {
            entry.d_ptr = external->second;
            entry.is_external = true;
        } else if (nbytes > 0) {
            auto err = cudaMalloc(&entry.d_ptr, nbytes);
            if (err != cudaSuccess)
                entry.d_ptr = nullptr;
            else
                cudaMemsetAsync(entry.d_ptr, 0, nbytes, stream_);
        }

        if (entry.d_ptr)
            bind_tensor_address(name, entry);

        allocate_host_output_staging(host_output_staging_, name, nbytes, entry.is_external);

        buffers_[name] = std::move(entry);
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

    initial_external_bindings_.clear();
    cudaStreamSynchronize(stream_);
}

bool TrtModuleImpl::should_allocate_input(const std::string& name, std::size_t nbytes) const {
    return nbytes > 0 && alias_outputs_by_input_.count(name) == 0;
}

void TrtModuleImpl::validate_alias_outputs_exist(
    const std::vector<std::string>& output_names) const {
    for (const auto& output_name : output_names) {
        if (buffers_.count(output_name) == 0) {
            throw std::invalid_argument("Incomplete TensorRT alias group for output '" +
                                        output_name + "'");
        }
    }
}

void TrtModuleImpl::bind_alias_outputs_or_invalidate(const std::vector<std::string>& output_names,
                                                     void* ptr) {
    for (const auto& output_name : output_names) {
        auto output = buffers_.find(output_name);
        BufferEntry candidate_output = output->second;
        candidate_output.d_ptr = ptr;
        candidate_output.is_external = true;
        if (!bind_tensor_address(output_name, candidate_output)) {
            // TensorRT address updates are not transactional. Once part of a
            // group has changed, discard the context rather than risk enqueue
            // with mixed state addresses.
            if (cuda_graph_)
                cuda_graph_->reset();
            delete ctx_;
            ctx_ = nullptr;
            throw std::runtime_error("TensorRT rejected external alias output '" + output_name +
                                     "'");
        }
    }
}

void TrtModuleImpl::reset_cuda_graph_if_rebound(void* previous_ptr, void* ptr) {
    if (previous_ptr != ptr && use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
}

void TrtModuleImpl::bind_alias_group(const std::string& input_name, void* ptr) {
    if (ptr == nullptr)
        throw std::invalid_argument("External alias buffer for '" + input_name + "' is null");

    auto input = buffers_.find(input_name);
    const auto outputs = alias_outputs_by_input_.find(input_name);
    if (input == buffers_.end() || outputs == alias_outputs_by_input_.end()) {
        throw std::invalid_argument("Incomplete TensorRT alias group for input '" + input_name +
                                    "'");
    }
    validate_alias_outputs_exist(outputs->second);

    BufferEntry candidate_input = input->second;
    candidate_input.d_ptr = ptr;
    candidate_input.is_external = true;
    if (!bind_tensor_address(input_name, candidate_input)) {
        throw std::runtime_error("TensorRT rejected external alias input '" + input_name + "'");
    }

    bind_alias_outputs_or_invalidate(outputs->second, ptr);

    reset_cuda_graph_if_rebound(input->second.d_ptr, ptr);
    input->second.d_ptr = ptr;
    input->second.is_external = true;
    for (const auto& output_name : outputs->second) {
        auto& output = buffers_.at(output_name);
        output.d_ptr = ptr;
        output.is_external = true;
    }
    alias_groups_ready_ = false;
}

void TrtModuleImpl::validate_alias_groups_bound() const {
    for (const auto& [input_name, output_names] : alias_outputs_by_input_) {
        const auto input = buffers_.find(input_name);
        if (input == buffers_.end() || input->second.d_ptr == nullptr) {
            throw std::runtime_error("TensorRT alias input '" + input_name +
                                     "' has no external buffer");
        }
        for (const auto& output_name : output_names) {
            const auto output = buffers_.find(output_name);
            if (output == buffers_.end() || output->second.d_ptr == nullptr ||
                output->second.d_ptr != input->second.d_ptr) {
                throw std::runtime_error("TensorRT alias output '" + output_name +
                                         "' is not bound to input '" + input_name + "'");
            }
        }
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

    // Download outputs — skip externally-bound buffers (they stay on device)
    TensorMap outputs;
    for (auto& [name, entry] : buffers_) {
        if (entry.is_input)
            continue;
        if (entry.is_external)
            continue;

        std::vector<int64_t> runtime_shape = entry.shape;
        std::size_t runtime_nbytes = entry.nbytes;
        if (has_dynamic_shapes_ && ctx_ != nullptr) {
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
        cudaMemcpy(staging.data(), entry.d_ptr, runtime_nbytes, cudaMemcpyDeviceToHost);

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
    runtime_measurement::record_model_call_start();
    // Upload inputs H2D, updating shapes for dynamic engines
    for (const auto& [name, tensor] : inputs) {
        auto it = buffers_.find(name);
        if (it == buffers_.end())
            continue;
        auto& entry = it->second;
        if (!entry.is_input || !entry.d_ptr)
            continue;

        update_dynamic_shape(name, entry, tensor.shape);

        auto copy_bytes = std::min(tensor.nbytes(), entry.nbytes);
        if (copy_bytes > 0 && tensor.data) {
            cudaMemcpyAsync(entry.d_ptr, tensor.data, copy_bytes, cudaMemcpyHostToDevice, stream_);
        }
    }

    execute_enqueue();
}

void TrtModuleImpl::execute_enqueue() {
    if (!ctx_)
        throw std::runtime_error("TensorRT execution context is unavailable");
    if (!alias_groups_ready_) {
        validate_alias_groups_bound();
        alias_groups_ready_ = true;
    }
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

void TrtModuleImpl::record_timed_enqueue() {
    TimingEvent timing_event;
    const bool timing_ok = timing_enabled_ && begin_timing_event(timing_event);
    if (use_cuda_graph_ && cuda_graph_->ready()) {
        cuda_graph_->launch(stream_);
        if (timing_ok)
            finish_timing_event(timing_event);
        return;
    }
    if (use_cuda_graph_) {
        cuda_graph_->begin_capture(stream_);
        ctx_->enqueueV3(stream_);
        if (!cuda_graph_->end_capture(stream_)) {
            std::cerr << "[cuda_graph] Capture failed, disabling CUDA Graphs\n";
            use_cuda_graph_ = false;
            ctx_->enqueueV3(stream_);
        } else {
            cuda_graph_->launch(stream_);
        }
        if (timing_ok)
            finish_timing_event(timing_event);
        return;
    }
    ctx_->enqueueV3(stream_);
    if (timing_ok)
        finish_timing_event(timing_event);
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
    cudaStreamSynchronize(stream_);
}

// --- Forward device async (GPU → GPU, no sync) ---

void TrtModuleImpl::forward_device_async(const DeviceTensorMap& inputs) {
    runtime_measurement::record_model_call_start();
    // D2D copy input DeviceTensors into our buffers
    for (const auto& [name, dt_ptr] : inputs) {
        auto it = buffers_.find(name);
        if (it == buffers_.end() || !dt_ptr)
            continue;
        auto& entry = it->second;
        if (!entry.is_input || !entry.d_ptr)
            continue;

        update_dynamic_shape(name, entry, dt_ptr->shape());

        if (dt_ptr->data() != entry.d_ptr) {
            auto copy_bytes = std::min(dt_ptr->nbytes(), entry.nbytes);
            if (copy_bytes > 0) {
                cudaMemcpyAsync(entry.d_ptr, dt_ptr->data(), copy_bytes, cudaMemcpyDeviceToDevice,
                                stream_);
            }
        }
    }

    // Execute (no sync — caller will sync or run more kernels on same stream)
    execute_enqueue();
}

// --- Forward device (GPU → GPU, synchronous) ---

DeviceTensorMap TrtModuleImpl::forward_device(const DeviceTensorMap& inputs) {
    forward_device_async(inputs);
    cudaStreamSynchronize(stream_);

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

    if (alias_input_by_output_.count(name) != 0) {
        throw std::invalid_argument("Bind TensorRT alias input '" +
                                    alias_input_by_output_.at(name) + "', not output '" + name +
                                    "'");
    }
    if (alias_outputs_by_input_.count(name) != 0) {
        bind_alias_group(name, external_device_ptr);
        return;
    }

    auto& entry = it->second;
    if (external_device_ptr == nullptr)
        throw std::invalid_argument("External buffer for '" + name + "' is null");
    if (external_device_ptr == entry.d_ptr)
        return;

    // Bind the candidate before changing ownership. If TensorRT rejects the
    // address, the module must retain both its previous context binding and its
    // still-live owned buffer.
    BufferEntry candidate = entry;
    candidate.d_ptr = external_device_ptr;
    candidate.is_external = true;
    if (!bind_tensor_address(name, candidate))
        throw std::runtime_error("TensorRT rejected external buffer for '" + name + "'");

    void* const previous_ptr = entry.d_ptr;
    const bool free_previous =
        previous_ptr != nullptr && !entry.is_external && previous_ptr != external_device_ptr;
    reset_cuda_graph_if_rebound(previous_ptr, external_device_ptr);
    entry = std::move(candidate);
    if (free_previous)
        cudaFree(previous_ptr);
}

} // namespace trtmc
