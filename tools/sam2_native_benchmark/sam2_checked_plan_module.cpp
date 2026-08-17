/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_checked_plan_module.h"

#include "utils/sha256.h"

#include <NvInfer.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc::sam2::benchmark {
namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("SAM2 checked benchmark module: " + message);
}

void requireCuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess)
        fail(operation + " failed: " + cudaGetErrorString(status));
}

class ScopedDevice final {
  public:
    explicit ScopedDevice(std::int32_t desired) : desired_(desired) {
        requireCuda(cudaGetDevice(&previous_), "CUDA device query");
        if (previous_ != desired_)
            requireCuda(cudaSetDevice(desired_), "CUDA device selection");
    }

    ~ScopedDevice() {
        if (previous_ != desired_)
            (void)cudaSetDevice(previous_);
    }

    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

  private:
    std::int32_t desired_{-1};
    std::int32_t previous_{-1};
};

template <typename T>
struct TrtDelete {
    void operator()(T* object) const noexcept {
        if (object != nullptr)
            delete object;
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

class CheckedLogger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) {
            // Do not write from TensorRT's noexcept callback. The operation
            // that fails is checked by its caller and supplies the diagnostic.
            (void)message;
        }
    }
};

DType runtimeDtype(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT:
        return DType::kFloat32;
    case nvinfer1::DataType::kBF16:
        return DType::kBFloat16;
    default:
        fail("engine contains a data type outside the FP32/BF16 SAM2 ABI");
    }
}

std::vector<std::int64_t> staticShape(const nvinfer1::Dims& dimensions) {
    if (dimensions.nbDims <= 0)
        fail("engine tensor has an invalid rank");
    std::vector<std::int64_t> result;
    result.reserve(static_cast<std::size_t>(dimensions.nbDims));
    for (std::int32_t index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] <= 0)
            fail("engine tensor is not fixed-shape");
        result.push_back(dimensions.d[index]);
    }
    return result;
}

std::size_t byteCount(const std::vector<std::int64_t>& shape, DType dtype) {
    std::size_t elements = 1U;
    for (const auto dimension : shape) {
        const auto extent = static_cast<std::size_t>(dimension);
        if (extent > std::numeric_limits<std::size_t>::max() / elements)
            fail("engine tensor element count overflowed");
        elements *= extent;
    }
    const auto element_bytes = dtype_size(dtype);
    if (element_bytes > std::numeric_limits<std::size_t>::max() / elements)
        fail("engine tensor byte count overflowed");
    return elements * element_bytes;
}

class DeviceAllocation final {
  public:
    DeviceAllocation(std::size_t bytes, std::int32_t device) : bytes_(bytes), device_(device) {
        if (bytes_ == 0U)
            fail("refusing to allocate an empty tensor");
        ScopedDevice selected(device_);
        requireCuda(cudaMalloc(&data_, bytes_), "CUDA tensor allocation");
    }

    ~DeviceAllocation() {
        if (data_ == nullptr)
            return;
        try {
            ScopedDevice selected(device_);
            (void)cudaFree(data_);
        } catch (...) {
        }
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    void* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }

  private:
    void* data_{nullptr};
    std::size_t bytes_{0U};
    std::int32_t device_{-1};
};

struct Binding final {
    TensorInfo info;
    std::unique_ptr<DeviceAllocation> owned;
    void* address{nullptr};
    bool external{false};
    std::vector<std::uint8_t> host_output;
};

} // namespace

struct CheckedPlanModuleFactory::SharedState final {
    explicit SharedState(cudaStream_t requested_stream) : stream(requested_stream) {
        if (stream == nullptr)
            fail("factory requires a non-null CUDA stream");
        unsigned int flags = 0U;
        requireCuda(cudaStreamGetFlags(stream, &flags), "CUDA stream flags query");
        if ((flags & cudaStreamNonBlocking) == 0U)
            fail("factory requires cudaStreamNonBlocking");
        requireCuda(cudaStreamGetDevice(stream, &device), "CUDA stream-device query");
        ScopedDevice selected(device);
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime)
            fail("TensorRT runtime creation failed");
    }

    CheckedLogger logger;
    cudaStream_t stream{nullptr};
    std::int32_t device{-1};
    TrtPtr<nvinfer1::IRuntime> runtime;
    mutable std::mutex loaded_plan_mutex;
    std::vector<std::pair<std::string, std::string>> loaded_plan_sha256;
};

namespace {

class CheckedPlanModule final : public ITrtModule {
  public:
    CheckedPlanModule(std::shared_ptr<CheckedPlanModuleFactory::SharedState> shared,
                      std::string section, const void* plan_data, std::size_t plan_size)
        : shared_(std::move(shared)), section_(std::move(section)) {
        if (shared_ == nullptr || plan_data == nullptr || plan_size == 0U)
            fail("cannot deserialize an empty plan for " + section_);
        ScopedDevice selected(shared_->device);
        engine_.reset(shared_->runtime->deserializeCudaEngine(plan_data, plan_size));
        if (!engine_)
            fail("TensorRT plan deserialization failed for " + section_);
        if (engine_->getProfilingVerbosity() != nvinfer1::ProfilingVerbosity::kDETAILED)
            fail("engine was not built with detailed profiling verbosity for " + section_);
        if (engine_->getNbOptimizationProfiles() != 1)
            fail("engine does not have exactly one optimization profile for " + section_);
        context_.reset(engine_->createExecutionContext());
        if (!context_)
            fail("TensorRT execution-context creation failed for " + section_);
        if (!context_->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE) ||
            context_->getNvtxVerbosity() != nvinfer1::ProfilingVerbosity::kNONE) {
            fail("execution context did not retain disabled NVTX verbosity for " + section_);
        }
        initializeBindings();
        requireCuda(cudaStreamSynchronize(shared_->stream),
                    "initial tensor allocation synchronization for " + section_);
    }

    ~CheckedPlanModule() override {
        try {
            ScopedDevice selected(shared_->device);
            (void)cudaStreamSynchronize(shared_->stream);
            context_.reset();
            engine_.reset();
            bindings_.clear();
        } catch (...) {
            context_.reset();
            engine_.reset();
            bindings_.clear();
        }
    }

    TensorMap forward(const TensorMap& inputs) override {
        forward_async(inputs);
        TensorMap result;
        for (auto& [name, binding] : bindings_) {
            if (binding.info.is_input)
                continue;
            binding.host_output.resize(binding.owned->bytes());
            requireCuda(cudaMemcpyAsync(binding.host_output.data(), binding.address,
                                        binding.owned->bytes(), cudaMemcpyDeviceToHost,
                                        shared_->stream),
                        "output download for " + section_ + ":" + name);
        }
        sync();
        for (auto& [name, binding] : bindings_) {
            if (binding.info.is_input)
                continue;
            result.emplace(
                name, Tensor{binding.host_output.data(), binding.info.shape, binding.info.dtype});
        }
        return result;
    }

    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        fail("device-copy forward is outside the fixed SAM2 benchmark path");
    }

    void forward_device_async(const DeviceTensorMap&) override {
        fail("asynchronous device-copy forward is outside the fixed SAM2 benchmark path");
    }

    void forward_async(const TensorMap& inputs) override {
        if (poisoned_ || context_ == nullptr)
            fail("execution context is unavailable for " + section_);
        ScopedDevice selected(shared_->device);
        for (const auto& [name, tensor] : inputs) {
            auto found = bindings_.find(name);
            if (found == bindings_.end() || !found->second.info.is_input)
                fail("received an unknown input for " + section_ + ":" + name);
            auto& binding = found->second;
            if (tensor.data == nullptr || tensor.shape != binding.info.shape ||
                tensor.dtype != binding.info.dtype || tensor.nbytes() != binding.owned->bytes()) {
                fail("input contract drifted for " + section_ + ":" + name);
            }
            requireCuda(cudaMemcpyAsync(binding.address, tensor.data, binding.owned->bytes(),
                                        cudaMemcpyHostToDevice, shared_->stream),
                        "input upload for " + section_ + ":" + name);
        }
        for (const auto& [name, binding] : bindings_) {
            if (binding.info.is_input && inputs.count(name) == 0U && !binding.external)
                fail("missing non-external input for " + section_ + ":" + name);
        }
        if (!context_->enqueueV3(shared_->stream)) {
            poisoned_ = true;
            fail("TensorRT enqueueV3 returned false for " + section_);
        }
    }

    void sync() override {
        ScopedDevice selected(shared_->device);
        requireCuda(cudaStreamSynchronize(shared_->stream),
                    "CUDA stream synchronization for " + section_);
    }

    cudaStream_t stream() const override { return shared_->stream; }

    void enable_cuda_graph() override {
        fail("CUDA graphs are disabled for the qualified SAM2 benchmark path");
    }

    bool cuda_graph_active() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }

    std::vector<TensorInfo> input_info() const override { return metadata(true); }
    std::vector<TensorInfo> output_info() const override { return metadata(false); }

    bool has_input(const std::string& name) const override {
        const auto found = bindings_.find(name);
        return found != bindings_.end() && found->second.info.is_input;
    }

    bool has_output(const std::string& name) const override {
        const auto found = bindings_.find(name);
        return found != bindings_.end() && !found->second.info.is_input;
    }

    DType tensor_dtype(const std::string& name) const override { return required(name).info.dtype; }

    std::vector<std::int64_t> tensor_shape(const std::string& name) const override {
        return required(name).info.shape;
    }

    std::vector<std::int64_t> input_profile_shape(const std::string& name, std::int32_t profile,
                                                  ProfileShapeSelector) const override {
        if (profile != 0 || !has_input(name))
            fail("invalid profile-shape query for " + section_ + ":" + name);
        return required(name).info.shape;
    }

    std::int32_t optimization_profile_count() const override { return 1; }

    void* device_ptr(const std::string& name) const override { return required(name).address; }

    void bind_external(const std::string& name, void* pointer) override {
        if (poisoned_ || pointer == nullptr)
            fail("invalid external binding for " + section_ + ":" + name);
        auto& binding = required(name);
        validateDevicePointer(pointer, binding.owned->bytes(), name);
        ScopedDevice selected(shared_->device);
        const bool accepted = binding.info.is_input
                                  ? context_->setInputTensorAddress(name.c_str(), pointer)
                                  : context_->setOutputTensorAddress(name.c_str(), pointer);
        if (!accepted) {
            poisoned_ = true;
            fail("TensorRT rejected external binding for " + section_ + ":" + name);
        }
        binding.address = pointer;
        binding.external = true;
    }

    void bind_external(const std::string& name, void* pointer,
                       const std::vector<std::int64_t>& shape) override {
        if (!shape.empty() && shape != required(name).info.shape)
            fail("external binding shape drifted for " + section_ + ":" + name);
        bind_external(name, pointer);
    }

    std::int32_t input_rank(const std::string& name) const override {
        return has_input(name) ? static_cast<std::int32_t>(required(name).info.shape.size()) : 0;
    }

    bool input_is_dynamic(const std::string& name) const override {
        if (!has_input(name))
            fail("dynamic-shape query used an unknown input for " + section_ + ":" + name);
        return false;
    }

    void reset_execution_context() override {
        if (poisoned_)
            fail("cannot reset a context after a failed checked enqueue for " + section_);
    }

    void set_timing_label(std::string) override {}
    bool ok() const override { return context_ != nullptr && !poisoned_; }

    void keep_alive(std::shared_ptr<void> resource) override {
        if (resource == nullptr)
            fail("cannot retain a null resource for " + section_);
        keep_alive_.push_back(std::move(resource));
    }

  private:
    void initializeBindings() {
        const auto count = engine_->getNbIOTensors();
        if (count <= 0)
            fail("engine has no I/O tensors for " + section_);
        for (std::int32_t index = 0; index < count; ++index) {
            const char* raw_name = engine_->getIOTensorName(index);
            if (raw_name == nullptr || raw_name[0] == '\0')
                fail("engine contains an unnamed tensor for " + section_);
            const std::string name(raw_name);
            const auto mode = engine_->getTensorIOMode(raw_name);
            if (mode != nvinfer1::TensorIOMode::kINPUT && mode != nvinfer1::TensorIOMode::kOUTPUT) {
                fail("engine tensor has an invalid I/O mode for " + section_ + ":" + name);
            }
            if (engine_->getTensorLocation(raw_name) != nvinfer1::TensorLocation::kDEVICE)
                fail("engine tensor is not device-resident for " + section_ + ":" + name);
            TensorInfo info{name, staticShape(engine_->getTensorShape(raw_name)),
                            runtimeDtype(engine_->getTensorDataType(raw_name)),
                            mode == nvinfer1::TensorIOMode::kINPUT};
            Binding binding;
            binding.info = std::move(info);
            binding.owned = std::make_unique<DeviceAllocation>(
                byteCount(binding.info.shape, binding.info.dtype), shared_->device);
            binding.address = binding.owned->data();
            requireCuda(
                cudaMemsetAsync(binding.address, 0, binding.owned->bytes(), shared_->stream),
                "tensor initialization for " + section_ + ":" + name);
            const bool accepted =
                binding.info.is_input
                    ? context_->setInputTensorAddress(name.c_str(), binding.address)
                    : context_->setOutputTensorAddress(name.c_str(), binding.address);
            if (!accepted)
                fail("TensorRT rejected initial binding for " + section_ + ":" + name);
            if (!bindings_.emplace(name, std::move(binding)).second)
                fail("engine contains duplicate tensor names for " + section_);
        }
    }

    std::vector<TensorInfo> metadata(bool input) const {
        std::vector<TensorInfo> result;
        for (const auto& [name, binding] : bindings_) {
            (void)name;
            if (binding.info.is_input == input)
                result.push_back(binding.info);
        }
        return result;
    }

    Binding& required(const std::string& name) {
        const auto found = bindings_.find(name);
        if (found == bindings_.end())
            fail("unknown tensor for " + section_ + ":" + name);
        return found->second;
    }

    const Binding& required(const std::string& name) const {
        const auto found = bindings_.find(name);
        if (found == bindings_.end())
            fail("unknown tensor for " + section_ + ":" + name);
        return found->second;
    }

    void validateDevicePointer(void* pointer, std::size_t bytes, const std::string& name) const {
        cudaPointerAttributes attributes{};
        const auto status = cudaPointerGetAttributes(&attributes, pointer);
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            fail("external binding is not CUDA storage for " + section_ + ":" + name);
        }
#if CUDART_VERSION >= 10000
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != shared_->device)
#else
        if (attributes.memoryType != cudaMemoryTypeDevice || attributes.device != shared_->device)
#endif
            fail("external binding is on the wrong CUDA device for " + section_ + ":" + name);
        const auto base = reinterpret_cast<std::uintptr_t>(attributes.devicePointer);
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        if (base == 0U || address < base ||
            bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
            fail("external binding span is invalid for " + section_ + ":" + name);
        }
    }

    std::shared_ptr<CheckedPlanModuleFactory::SharedState> shared_;
    std::string section_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    std::unordered_map<std::string, Binding> bindings_;
    std::vector<std::shared_ptr<void>> keep_alive_;
    bool poisoned_{false};
};

} // namespace

CheckedPlanModuleFactory::CheckedPlanModuleFactory(cudaStream_t stream)
    : state_(std::make_shared<SharedState>(stream)) {}

CheckedPlanModuleFactory::~CheckedPlanModuleFactory() = default;

std::string planSha256(const void* plan_data, std::size_t plan_size) {
    internal::Sha256 hash;
    hash.update(plan_data, plan_size);
    return hash.hex_digest();
}

std::unique_ptr<ITrtModule>
makeCheckedModule(const std::shared_ptr<CheckedPlanModuleFactory::SharedState>& shared,
                  std::string_view section, const void* plan_data, std::size_t plan_size) {
    if (section.empty())
        fail("plan section name must not be empty");
    const std::string owned_section(section);
    const std::string digest = planSha256(plan_data, plan_size);
    auto result = std::make_unique<CheckedPlanModule>(shared, owned_section, plan_data, plan_size);
    {
        std::lock_guard<std::mutex> lock(shared->loaded_plan_mutex);
        const auto duplicate =
            std::find_if(shared->loaded_plan_sha256.begin(), shared->loaded_plan_sha256.end(),
                         [&](const auto& value) { return value.first == owned_section; });
        if (duplicate != shared->loaded_plan_sha256.end())
            fail("plan section was deserialized more than once: " + owned_section);
        shared->loaded_plan_sha256.emplace_back(owned_section, digest);
    }
    return result;
}

std::unique_ptr<ITrtModule> CheckedPlanModuleFactory::create(std::string_view section,
                                                             const void* plan_data,
                                                             std::size_t plan_size) const {
    return makeCheckedModule(state_, section, plan_data, plan_size);
}

NativePlanModuleFactory CheckedPlanModuleFactory::callback() const {
    const auto retained = state_;
    return [retained](std::string_view section, const void* plan_data,
                      std::size_t plan_size) -> std::unique_ptr<ITrtModule> {
        return makeCheckedModule(retained, section, plan_data, plan_size);
    };
}

std::vector<std::pair<std::string, std::string>>
CheckedPlanModuleFactory::loadedPlanSha256() const {
    std::lock_guard<std::mutex> lock(state_->loaded_plan_mutex);
    return state_->loaded_plan_sha256;
}

} // namespace trtmc::sam2::benchmark
