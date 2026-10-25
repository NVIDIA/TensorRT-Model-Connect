/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(TRTMC_TEST_PLAN_STREAM_READER)

#include "runtime/tensorrt/trt_module_impl.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

// CPU-only transfer simulation deliberately keeps source pointers queued until
// synchronization. Reusing or freeing staging bytes early corrupts the result.
namespace {
struct Transfer {
    void* destination;
    const void* source;
    std::size_t count;
    cudaStream_t stream;
};
std::unordered_map<void*, std::size_t> pinned;
std::vector<Transfer> queued;
std::vector<char>* device = nullptr;
cudaMemoryType destination_type = cudaMemoryTypeDevice;
bool attribute_error = false;
bool copy_error = false;
int sync_errors = 0;
bool unsafe_free = false;
std::size_t largest_pinned = 0;
std::size_t copy_count = 0;
cudaStream_t expected_stream = reinterpret_cast<cudaStream_t>(0x1234);

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

class MemoryReader final : public trtmc::IPlanReader {
  public:
    explicit MemoryReader(std::size_t size) : bytes(size) {
        for (std::size_t i = 0; i < size; ++i)
            bytes[i] = static_cast<char>((i / 65537 + i % 251) % 251);
    }
    ~MemoryReader() override {
        if (alive != nullptr)
            *alive = false;
    }
    std::uint64_t size() const noexcept override { return bytes.size(); }
    std::size_t read(void* destination, std::size_t count) override {
        if (cursor >= throw_at)
            throw std::runtime_error("Injected source I/O failure");
        maximum_read = std::max(maximum_read, count);
        count = std::min(count, bytes.size() - cursor);
        std::memcpy(destination, bytes.data() + cursor, count);
        cursor += count;
        return count;
    }
    bool seek(std::int64_t offset, trtmc::PlanSeekOrigin origin) override {
        const auto base = origin == trtmc::PlanSeekOrigin::begin ? 0
                          : origin == trtmc::PlanSeekOrigin::end ? bytes.size()
                                                                 : cursor;
        const auto next = static_cast<std::int64_t>(base) + offset;
        if (next < 0 || static_cast<std::uint64_t>(next) > bytes.size())
            return false;
        cursor = static_cast<std::size_t>(next);
        return true;
    }
    std::vector<char> bytes;
    bool* alive{nullptr};
    std::size_t cursor{0};
    std::size_t maximum_read{0};
    std::size_t throw_at{std::numeric_limits<std::size_t>::max()};
};

void expect_failure(const trtmc::TrtPlanStreamReader& reader) {
    bool failed = false;
    try {
        reader.rethrow_if_failed();
    } catch (const std::exception&) {
        failed = true;
    }
    check(failed, "Callback swallowed its source or transfer error");
}
} // namespace

extern "C" cudaError_t CUDARTAPI cudaPointerGetAttributes(cudaPointerAttributes* attributes,
                                                          const void* pointer) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto first = device == nullptr ? 0 : reinterpret_cast<std::uintptr_t>(device->data());
    if (device != nullptr && address >= first && address - first < device->size()) {
        attributes->type = destination_type;
        return cudaSuccess;
    }
    attributes->type = cudaMemoryTypeUnregistered;
    return attribute_error ? cudaErrorInvalidValue : cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
    return cudaSuccess;
}
extern "C" const char* CUDARTAPI cudaGetErrorString(cudaError_t) {
    return "injected CUDA error";
}
extern "C" cudaError_t CUDARTAPI cudaMallocHost(void** pointer, std::size_t size) {
    *pointer = std::malloc(size);
    if (!*pointer)
        return cudaErrorMemoryAllocation;
    pinned[*pointer] = size;
    largest_pinned = std::max(largest_pinned, size);
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* pointer) {
    for (const auto& transfer : queued) {
        if (transfer.source == pointer) {
            unsafe_free = true;
            return cudaErrorNotReady;
        }
    }
    pinned.erase(pointer);
    std::free(pointer);
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void* destination, const void* source,
                                                 std::size_t count, cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
    if (stream != expected_stream || kind != cudaMemcpyDefault ||
        pinned.find(const_cast<void*>(source)) == pinned.end())
        return cudaErrorInvalidValue;
    if (copy_error)
        return cudaErrorLaunchFailure;
    queued.push_back({destination, source, count, stream});
    ++copy_count;
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    if (stream != expected_stream)
        return cudaErrorInvalidValue;
    if (sync_errors != 0) {
        --sync_errors;
        return cudaErrorLaunchFailure;
    }
    for (const auto& transfer : queued) {
        if (transfer.stream != stream)
            return cudaErrorInvalidValue;
        std::memcpy(transfer.destination, transfer.source, transfer.count);
    }
    queued.clear();
    return cudaSuccess;
}

int main() {
    try {
        using Reader = trtmc::TrtPlanStreamReader;
        bool source_alive = true;
        {
            auto source = std::make_unique<MemoryReader>(4);
            source->alive = &source_alive;
            Reader reader(std::move(source));
            check(source == nullptr && source_alive, "Callback did not retain transferred source");
            char result[4];
            check(reader.read(result, 4, expected_stream) == 4,
                  "Transferred source was unavailable after its caller released ownership");
        }
        check(!source_alive, "Callback leaked its owned source");
        for (const bool bad_attributes : {false, true}) {
            attribute_error = bad_attributes;
            auto source_owner = std::make_unique<MemoryReader>(4096);
            auto& source = *source_owner;
            Reader reader(std::move(source_owner));
            std::vector<char> result(source.bytes.size(), '!');
            if (bad_attributes) {
                check(reader.read(result.data(), result.size(), expected_stream) == -1 &&
                          source.cursor == 0 &&
                          std::all_of(result.begin(), result.end(),
                                      [](char value) { return value == '!'; }),
                      "Pointer-query failure was treated as writable host memory");
                expect_failure(reader);
                continue;
            }
            check(reader.read(result.data(), result.size(), expected_stream) == 4096 &&
                      result == source.bytes,
                  "Host destination bytes changed");
            check(reader.seek(-2, nvinfer1::SeekPosition::kEND),
                  "End-relative callback seek failed");
            check(reader.read(result.data(), 8, expected_stream) == 2 &&
                      result[0] == source.bytes[4094] && result[1] == source.bytes[4095],
                  "Host partial read crossed EOF");
            check(reader.read(result.data(), 1, expected_stream) == 0, "EOF callback changed");
            check(!reader.seek(1, nvinfer1::SeekPosition::kCUR), "Invalid callback seek accepted");
            reader.rethrow_if_failed();
            check(pinned.empty() && queued.empty(), "Host read allocated a device staging buffer");
        }
        attribute_error = false;

        for (const auto type : {cudaMemoryTypeDevice, cudaMemoryTypeManaged}) {
            destination_type = type;
            auto source_owner = std::make_unique<MemoryReader>(3 * Reader::transfer_bytes + 19);
            auto& source = *source_owner;
            std::vector<char> result(source.bytes.size(), '!');
            device = &result;
            const auto before = copy_count;
            {
                Reader reader(std::move(source_owner));
                check(reader.read(result.data(), result.size(), expected_stream) ==
                          static_cast<std::int64_t>(result.size()),
                      "Device callback incomplete");
                reader.rethrow_if_failed();
                check(result == source.bytes && queued.empty(),
                      "Asynchronous bytes were overwritten");
                check(pinned.size() == 1 && largest_pinned == Reader::transfer_bytes &&
                          source.maximum_read <= Reader::transfer_bytes && copy_count - before == 4,
                      "Device reads were not bounded to one reusable staging buffer");
            }
            check(pinned.empty() && !unsafe_free, "Staging was leaked or freed while pending");
            device = nullptr;
        }

        for (int failure = 0; failure != 4; ++failure) {
            auto source_owner = std::make_unique<MemoryReader>(2 * Reader::transfer_bytes + 19);
            auto& source = *source_owner;
            std::vector<char> result(source.bytes.size(), '!');
            device = &result;
            if (failure == 0)
                source.throw_at = Reader::transfer_bytes;
            copy_error = failure == 1;
            sync_errors = failure >= 2 ? failure - 1 : 0;
            {
                Reader reader(std::move(source_owner));
                check(reader.read(result.data(), result.size(), expected_stream) == -1,
                      "Source or CUDA failure was accepted");
                check(!reader.seek(0, nvinfer1::SeekPosition::kSET) &&
                          reader.read(result.data(), 1, expected_stream) == -1,
                      "Failed reader silently resumed");
                expect_failure(reader);
            }
            check(!unsafe_free, "Error cleanup released a pending transfer buffer");
            if (failure == 3) {
                check(pinned.size() == 1 && queued.size() == 1,
                      "Persistent CUDA error did not preserve pending source storage");
                check(cudaStreamSynchronize(expected_stream) == cudaSuccess,
                      "Cannot complete the simulated pending copy");
                check(cudaFreeHost(pinned.begin()->first) == cudaSuccess,
                      "Cannot release the test-owned retained buffer after completion");
            }
            check(pinned.empty() && queued.empty(), "Completed error path leaked staging");
            copy_error = false;
            device = nullptr;
        }
        {
            auto source_owner = std::make_unique<MemoryReader>(4);
            Reader reader(std::move(source_owner));
            check(reader.read(nullptr, 0, expected_stream) == 0, "Zero-byte callback failed");
            check(reader.read(nullptr, -1, expected_stream) == -1, "Negative byte count accepted");
            expect_failure(reader);
        }
        std::cout << "CPU plan stream host/device bytes, stream order, bounds and failure "
                     "lifetimes passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#else

#include "runtime/primitives/trt_common.h"
#include "runtime/tensorrt/trt_logger.h"
#include "runtime/tensorrt/trt_module_impl.h"

#include <NvInfer.h>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace trtmc {

class TrtModuleImplTestPeer {
  public:
    static DType from_trt_dtype(nvinfer1::DataType dtype) {
        return TrtModuleImpl::from_trt_dtype(dtype);
    }
    static void validate_io_tensor_format(nvinfer1::DataType dtype, nvinfer1::TensorFormat format) {
        TrtModuleImpl::validate_io_tensor_format(dtype, format);
    }
    static bool has_no_bindings_or_buffers(const TrtModuleImpl& module) {
        return module.initial_external_bindings_.empty() && module.buffers_.empty();
    }
    static std::size_t allocation_bytes(const nvinfer1::Dims& dims, DType dtype) {
        std::vector<int64_t> shape;
        return TrtModuleImpl::compute_alloc_bytes(dims, dtype, shape);
    }
};

} // namespace trtmc

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_fp8_tensor_storage_and_mapping() {
    const auto dtype = trtmc::TrtModuleImplTestPeer::from_trt_dtype(nvinfer1::DataType::kFP8);
    check(dtype == trtmc::DType::kFloat8E4M3, "TensorRT FP8 maps to E4M3, not float32");
    check(trtmc::dtype_size(dtype) == 1, "E4M3 elements occupy one byte");
    uint8_t values[] = {0x00, 0x38, 0xb8, 0x7e, 0xfe, 0x01};
    const trtmc::Tensor tensor{values, {2, 3}, dtype};
    check(tensor.numel() == 6 && tensor.nbytes() == sizeof(values),
          "FP8 host tensor sizes use byte storage");
    check(trtmc::TrtModuleImplTestPeer::allocation_bytes(nvinfer1::Dims{2, {2, 3}}, dtype) ==
              sizeof(values),
          "FP8 engine buffer allocation uses byte storage");
    check(trtmc::TrtModuleImplTestPeer::from_trt_dtype(nvinfer1::DataType::kINT8) ==
              trtmc::DType::kInt8,
          "signed integer storage remains distinct from E4M3");
    trtmc::TrtModuleImplTestPeer::validate_io_tensor_format(nvinfer1::DataType::kFP8,
                                                            nvinfer1::TensorFormat::kLINEAR);
    for (const auto format : {nvinfer1::TensorFormat::kHWC16, nvinfer1::TensorFormat::kCHW32}) {
        bool rejected = false;
        try {
            trtmc::TrtModuleImplTestPeer::validate_io_tensor_format(nvinfer1::DataType::kFP8,
                                                                    format);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "FP8 vectorized I/O is rejected before byte-buffer allocation");
    }
}

trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_identity(bool dynamic,
                                                          bool vectorized_fp8 = false) {
    static trtmc::TrtLogger logger;
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
        return nullptr;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    const auto dtype = vectorized_fp8 ? nvinfer1::DataType::kFP8 : nvinfer1::DataType::kFLOAT;
    const auto dims =
        vectorized_fp8 ? nvinfer1::Dims{4, {1, 4, 1, 1}} : nvinfer1::Dims{1, {dynamic ? -1 : 4}};
    auto* input = network->addInput("input", dtype, dims);
    auto* identity = network->addIdentity(*input);
    identity->getOutput(0)->setName("output");
    network->markOutput(*identity->getOutput(0));
    if (vectorized_fp8) {
        const auto formats = 1U << static_cast<unsigned>(nvinfer1::TensorFormat::kHWC16);
        input->setAllowedFormats(formats);
        identity->getOutput(0)->setAllowedFormats(formats);
    }

    if (dynamic) {
        auto* profile = builder->createOptimizationProfile();
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}});
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {2}});
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {4}});
        config->addOptimizationProfile(profile);
    }

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

void check_vectorized_fp8_rejected(nvinfer1::ICudaEngine& engine, cudaStream_t stream,
                                   void* external, bool prebind) {
    std::vector<trtmc::ModuleExternalBinding> bindings;
    if (prebind)
        bindings.push_back({"input", external, 16});
    auto* context = engine.createExecutionContext();
    check(context != nullptr, "FP8 fixture creates a valid context before rejection");
    if (!context)
        return;
    trtmc::TrtModuleImpl module(&engine, context, stream, 0, nullptr, bindings);
    check(!module.ok(), "constructor rejects non-linear FP8 I/O");
    check(trtmc::TrtModuleImplTestPeer::has_no_bindings_or_buffers(module),
          "constructor rejects FP8 format before prebinding or buffer allocation");
    check(module.device_ptr("input") == nullptr && module.device_ptr("output") == nullptr,
          "rejected FP8 module exposes no device buffers");
}

void test_constructor_rejects_vectorized_fp8(cudaStream_t stream) {
    // The storage checks above run everywhere. This real-engine fixture needs
    // FP8 tactics (compute capability 8.9 or newer), not merely byte storage.
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        check(false, "FP8 fixture can query the current CUDA device");
        return;
    }
    if (properties.major * 10 + properties.minor < 89) {
        std::cout << "SKIP: FP8 engine fixture requires FP8-capable TensorRT tactics\n";
        return;
    }
    auto engine = build_identity(false, true);
    check(engine != nullptr, "vectorized FP8 engine fixture builds");
    if (!engine)
        return;
    for (const auto* name : {"input", "output"}) {
        check(engine->getTensorDataType(name) == nvinfer1::DataType::kFP8 &&
                  engine->getTensorFormat(name) == nvinfer1::TensorFormat::kHWC16,
              "fixture really exposes vectorized FP8 I/O");
    }
    void* external = nullptr;
    if (cudaMalloc(&external, 16) != cudaSuccess) {
        check(false, "FP8 fixture can allocate a correctly padded external buffer");
        return;
    }
    for (bool prebind : {false, true})
        check_vectorized_fp8_rejected(*engine, stream, external, prebind);
    cudaFree(external);
}

void test_first_forward_allocates_unbound_dynamic_input(nvinfer1::ICudaEngine& engine,
                                                        cudaStream_t stream) {
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
    check(module.device_ptr("input") == nullptr,
          "dynamic input is not allocated during module construction");
    float values[] = {1.0F, 2.0F};
    const auto outputs =
        module.forward({{"input", trtmc::Tensor{values, {2}, trtmc::DType::kFloat32}}});
    check(module.device_ptr("input") != nullptr,
          "first forward allocates and binds an unbound dynamic input");
    const auto found = outputs.find("output");
    check(found != outputs.end() && found->second.numel() == 2,
          "dynamic identity returns its runtime shape");
    if (found != outputs.end() && found->second.numel() == 2) {
        const auto* output = static_cast<const float*>(found->second.data);
        check(std::fabs(output[0] - 1.0F) < 1.0e-6F && std::fabs(output[1] - 2.0F) < 1.0e-6F,
              "dynamic identity returns its input values");
    }
}

void test_external_dynamic_input_stays_external(nvinfer1::ICudaEngine& engine,
                                                cudaStream_t stream) {
    void* external = nullptr;
    if (cudaMalloc(&external, 4 * sizeof(float)) != cudaSuccess) {
        ++failures;
        return;
    }
    {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        check(module.device_ptr("input") == nullptr,
              "second dynamic input also starts without an allocation");
        module.bind_external("input", external, {2});
        float values[] = {3.0F, 4.0F};
        (void)module.forward({{"input", trtmc::Tensor{values, {2}, trtmc::DType::kFloat32}}});
        check(module.device_ptr("input") == external,
              "forward preserves the pre-bound dynamic input buffer");
    }
    cudaFree(external);
}

void test_backend_managed_graph_does_not_enable_nested_capture(nvinfer1::ICudaEngine& engine,
                                                               cudaStream_t stream) {
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream, 0, nullptr, {},
                                true);
    module.enable_cuda_graph();
    check(!module.cuda_graph_active(), "backend-managed graph leaves stream capture disabled");
}

void test_rejected_shape_fails_without_poisoning_the_next_request(nvinfer1::ICudaEngine& engine,
                                                                  cudaStream_t stream) {
    for (bool graph : {false, true}) {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        if (graph)
            module.enable_cuda_graph();
        float initial[] = {1.0F, 2.0F};
        (void)module.forward({{"input", trtmc::Tensor{initial, {2}, trtmc::DType::kFloat32}}});
        float oversized[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        bool rejected = false;
        try {
            (void)module.forward(
                {{"input", trtmc::Tensor{oversized, {5}, trtmc::DType::kFloat32}}});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "out-of-profile shapes must fail instead of returning stale-shape outputs");
        float next[] = {7.0F, 8.0F};
        const auto output =
            module.forward({{"input", trtmc::Tensor{next, {2}, trtmc::DType::kFloat32}}});
        const auto& tensor = output.at("output");
        const auto* values = static_cast<const float*>(tensor.data);
        check(tensor.numel() == 2 && values[0] == 7.0F && values[1] == 8.0F,
              "a rejected shape leaves the next valid request usable");
    }
}

void test_short_static_upload_does_not_keep_the_previous_input(nvinfer1::ICudaEngine& engine,
                                                               cudaStream_t stream) {
    for (bool graph : {false, true}) {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        if (graph)
            module.enable_cuda_graph();
        float full[] = {1.0F, 2.0F, 3.0F, 4.0F};
        (void)module.forward({{"input", trtmc::Tensor{full, {4}, trtmc::DType::kFloat32}}});
        float short_input[] = {7.0F, 8.0F};
        const auto outputs =
            module.forward({{"input", trtmc::Tensor{short_input, {2}, trtmc::DType::kFloat32}}});
        const auto& tensor = outputs.at("output");
        const auto* values = static_cast<const float*>(tensor.data);
        check(tensor.numel() == 4 && values[0] == 7.0F && values[1] == 8.0F,
              "static input keeps the current uploaded prefix");
        check(values[2] == 0.0F && values[3] == 0.0F,
              "short static upload clears bytes from the previous longer request");
    }
}

void test_short_external_static_upload_preserves_its_tail(nvinfer1::ICudaEngine& engine,
                                                          cudaStream_t stream) {
    void* external = nullptr;
    if (cudaMalloc(&external, 4 * sizeof(float)) != cudaSuccess) {
        ++failures;
        return;
    }
    {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        module.bind_external("input", external, {4});
        float full[] = {1.0F, 2.0F, 3.0F, 4.0F};
        (void)module.forward({{"input", trtmc::Tensor{full, {4}, trtmc::DType::kFloat32}}});
        float prefix[] = {7.0F, 8.0F};
        const auto outputs =
            module.forward({{"input", trtmc::Tensor{prefix, {2}, trtmc::DType::kFloat32}}});
        const auto* values = static_cast<const float*>(outputs.at("output").data);
        check(values[0] == 7.0F && values[1] == 8.0F && values[2] == 3.0F && values[3] == 4.0F,
              "short upload preserves externally owned static state beyond its prefix");
        check(module.device_ptr("input") == external, "static input keeps external ownership");
    }
    cudaFree(external);
}

} // namespace

int main() {
    test_fp8_tensor_storage_and_mapping();
    if (failures)
        return failures;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;
    auto engine = build_identity(true);
    auto static_engine = build_identity(false);
    if (!engine || !static_engine)
        return 1;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;
    test_constructor_rejects_vectorized_fp8(stream);
    test_first_forward_allocates_unbound_dynamic_input(*engine, stream);
    test_external_dynamic_input_stays_external(*engine, stream);
    test_backend_managed_graph_does_not_enable_nested_capture(*engine, stream);
    test_rejected_shape_fails_without_poisoning_the_next_request(*engine, stream);
    test_short_static_upload_does_not_keep_the_previous_input(*static_engine, stream);
    test_short_external_static_upload_preserves_its_tail(*static_engine, stream);
    cudaStreamDestroy(stream);
    return failures;
}

#endif
