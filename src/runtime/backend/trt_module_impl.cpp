#include "trt_module_impl.h"

#include "runtime/core/trt_common.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trtmc {

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
                             void* distributed_communicator)
    : engine_(engine), ctx_(ctx), stream_(stream), profile_idx_(profile_idx),
      distributed_communicator_(distributed_communicator),
      cuda_graph_(std::make_unique<CudaGraphExec>()) {
    if (!ctx_)
        return;
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

void TrtModuleImpl::recreate_context_with_profile() {
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
        cudaStreamSynchronize(stream_);
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

void TrtModuleImpl::rebind_buffer_to_context(const std::string& name, const BufferEntry& entry) {
    // Fresh IExecutionContexts have neither tensor addresses nor dynamic
    // input shapes set; replay both from our cached BufferEntry so the next
    // enqueueV3 doesn't fail with "Not all shapes are specified".
    if (entry.d_ptr)
        ctx_->setTensorAddress(name.c_str(), entry.d_ptr);
    if (!entry.is_dynamic || entry.shape.empty())
        return;
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(entry.shape.size());
    for (int32_t d = 0; d < dims.nbDims; ++d)
        dims.d[d] = entry.shape[d];
    ctx_->setInputShape(name.c_str(), dims);
}

void TrtModuleImpl::reset_execution_context() {
    if (engine_ == nullptr)
        return;
    recreate_context_with_profile();
    if (use_cuda_graph_ && cuda_graph_)
        cuda_graph_->reset();
    for (auto& [name, entry] : buffers_)
        rebind_buffer_to_context(name, entry);
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
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        if (dims_are_dynamic(engine->getTensorShape(name)))
            has_dynamic_shapes_ = true;
    }
}

void TrtModuleImpl::set_dynamic_input_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                             nvinfer1::OptProfileSelector selector) {
    for (int32_t i = 0; i < num_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        if (dims_are_dynamic(engine->getTensorShape(name))) {
            auto dims = engine->getProfileShape(name, profile_idx_, selector);
            ctx_->setInputShape(name, dims);
        }
    }
}

void TrtModuleImpl::allocate_single_input(nvinfer1::ICudaEngine* engine, const char* name,
                                          int32_t num_profiles) {
    auto trt_shape = engine->getTensorShape(name);
    auto dtype = from_trt_dtype(engine->getTensorDataType(name));

    // Determine allocation shape (max) and initial runtime shape (opt).
    nvinfer1::Dims alloc_dims = trt_shape;
    nvinfer1::Dims init_dims = trt_shape;
    bool is_dynamic = has_dynamic_shapes_ && num_profiles > 0 && dims_are_dynamic(trt_shape);

    if (is_dynamic) {
        alloc_dims =
            engine->getProfileShape(name, profile_idx_, nvinfer1::OptProfileSelector::kMAX);
        init_dims = engine->getProfileShape(name, profile_idx_, nvinfer1::OptProfileSelector::kOPT);
    }

    std::vector<int64_t> shape;
    std::size_t nbytes = compute_alloc_bytes(alloc_dims, dtype, shape);

    BufferEntry entry;
    entry.dtype = dtype;
    entry.nbytes = nbytes;
    entry.is_input = true;
    entry.is_dynamic = is_dynamic;
    entry.shape = is_dynamic ? dims_to_shape(init_dims) : shape;

    if (nbytes > 0) {
        auto err = cudaMalloc(&entry.d_ptr, nbytes);
        if (err != cudaSuccess)
            entry.d_ptr = nullptr;
        else
            cudaMemsetAsync(entry.d_ptr, 0, nbytes, stream_);
    }

    if (entry.d_ptr)
        ctx_->setTensorAddress(name, entry.d_ptr);

    if (is_dynamic)
        ctx_->setInputShape(name, init_dims);

    buffers_[name] = std::move(entry);
}

void TrtModuleImpl::allocate_input_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                           int32_t num_profiles) {
    for (int32_t i = 0; i < num_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        allocate_single_input(engine, name, num_profiles);
    }
}

void TrtModuleImpl::allocate_output_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io) {
    for (int32_t i = 0; i < num_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            continue;

        auto dtype = from_trt_dtype(engine->getTensorDataType(name));

        // For dynamic engines, query the context for inferred output shape
        // (based on the max input shapes set by the caller).
        // For static engines, use the engine shape directly.
        nvinfer1::Dims out_dims =
            has_dynamic_shapes_ ? ctx_->getTensorShape(name) : engine->getTensorShape(name);

        std::vector<int64_t> shape;
        std::size_t nbytes = compute_alloc_bytes(out_dims, dtype, shape);

        BufferEntry entry;
        entry.shape = shape;
        entry.dtype = dtype;
        entry.nbytes = nbytes;
        entry.is_input = false;

        if (nbytes > 0) {
            auto err = cudaMalloc(&entry.d_ptr, nbytes);
            if (err != cudaSuccess)
                entry.d_ptr = nullptr;
            else
                cudaMemsetAsync(entry.d_ptr, 0, nbytes, stream_);
        }

        if (entry.d_ptr)
            ctx_->setTensorAddress(name, entry.d_ptr);

        if (nbytes > 0)
            host_output_staging_[name].resize(nbytes);

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

    cudaStreamSynchronize(stream_);
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

        auto& staging = host_output_staging_[name];
        cudaMemcpy(staging.data(), entry.d_ptr, entry.nbytes, cudaMemcpyDeviceToHost);

        Tensor t;
        t.data = staging.data();
        t.shape = entry.shape;
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
    const bool timing_ok = begin_timing_event(timing_event);
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

    auto& entry = it->second;

    // Free our own buffer if we allocated it
    if (entry.d_ptr && !entry.is_external) {
        cudaFree(entry.d_ptr);
    }

    entry.d_ptr = external_device_ptr;
    entry.is_external = true;

    // Update execution context binding
    if (ctx_ && external_device_ptr) {
        ctx_->setTensorAddress(name.c_str(), external_device_ptr);
    }
}

} // namespace trtmc
