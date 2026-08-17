/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_bbox_postprocess.h"
#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_preprocess.h"
#include "tools/sam2_native_builder/sam2_image_network.h"

#include <NvInfer.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// These tolerances are part of the diagnostic contract, not values selected
// after observing a run. They allow at most one 1024-space pixel of BF16
// kernel variation and the corresponding maximum original-space scaling.
inline constexpr float kModelCoordinateAbsoluteTolerance = 1.0F;
inline constexpr float kOriginalCoordinateAbsoluteTolerance = 1.25F;
inline constexpr float kScoreAbsoluteTolerance = 1.0F / 512.0F;

inline constexpr std::array<float, 4> kExpectedModelBox{572.0F, 640.0F, 652.0F, 708.0F};
inline constexpr std::array<float, 4> kExpectedOriginalBox{607.75F, 800.0F, 692.75F, 885.0F};
inline constexpr float kExpectedScore = 0.4140625F;
inline constexpr int32_t kExpectedLabel = 1;

class Logger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "TensorRT: " << (message == nullptr ? "" : message) << '\n';
    }
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

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void checkCuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

class CudaBuffer final {
  public:
    explicit CudaBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes_ == 0)
            fail("refusing to allocate an empty CUDA tensor buffer");
        checkCuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
    }

    ~CudaBuffer() {
        if (data_ != nullptr)
            (void)cudaFree(data_);
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr)
                (void)cudaFree(data_);
            data_ = std::exchange(other.data_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    void* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }

  private:
    void* data_{nullptr};
    std::size_t bytes_{0};
};

class CudaStream final {
  public:
    CudaStream() {
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags");
    }

    ~CudaStream() {
        if (stream_ != nullptr)
            (void)cudaStreamDestroy(stream_);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_{nullptr};
};

struct ExpectedTensor {
    std::string name;
    nvinfer1::DataType type;
    std::array<int32_t, 4> dimensions;
    nvinfer1::TensorIOMode mode;
};

struct DeviceTensor {
    ExpectedTensor contract;
    std::size_t element_count;
    CudaBuffer storage;

    DeviceTensor(ExpectedTensor expected, std::size_t count, std::size_t bytes)
        : contract(std::move(expected)), element_count(count), storage(bytes) {}
};

std::vector<ExpectedTensor> expectedTensors() {
    using namespace trtmc::sam2::native;
    std::vector<ExpectedTensor> result;
    result.reserve(1U + kTrackerFpnContracts.size() + kBboxMapContracts.size());
    result.push_back({std::string(kImageInputContract.name), kImageInputContract.type,
                      kImageInputContract.dimensions, nvinfer1::TensorIOMode::kINPUT});
    for (const StaticTensorContract& contract : kTrackerFpnContracts) {
        result.push_back({std::string(contract.name), contract.type, contract.dimensions,
                          nvinfer1::TensorIOMode::kOUTPUT});
    }
    for (const StaticTensorContract& contract : kBboxMapContracts) {
        result.push_back({std::string(contract.name), contract.type, contract.dimensions,
                          nvinfer1::TensorIOMode::kOUTPUT});
    }
    return result;
}

bool sameShape(const nvinfer1::Dims& actual, const std::array<int32_t, 4>& expected) {
    if (actual.nbDims != static_cast<int32_t>(expected.size()))
        return false;
    for (int32_t index = 0; index < actual.nbDims; ++index) {
        if (actual.d[index] != expected[static_cast<std::size_t>(index)])
            return false;
    }
    return true;
}

std::size_t checkedElementCount(const std::array<int32_t, 4>& dimensions) {
    std::size_t count = 1;
    for (const int32_t dimension : dimensions) {
        if (dimension <= 0)
            fail("engine tensor has a non-positive static dimension");
        const std::size_t extent = static_cast<std::size_t>(dimension);
        if (extent > std::numeric_limits<std::size_t>::max() / count)
            fail("engine tensor element count overflows size_t");
        count *= extent;
    }
    return count;
}

std::size_t elementSize(nvinfer1::DataType type) {
    if (type == nvinfer1::DataType::kFLOAT)
        return sizeof(float);
    if (type == nvinfer1::DataType::kBF16)
        return sizeof(uint16_t);
    fail("engine tensor uses an unexpected data type");
}

void validateExactEngineAbi(const nvinfer1::ICudaEngine& engine,
                            const std::vector<ExpectedTensor>& expected) {
    if (engine.getNbIOTensors() != static_cast<int32_t>(expected.size()))
        fail("image engine I/O tensor count drifted from the exact ABI");

    std::map<std::string, const ExpectedTensor*> expected_by_name;
    for (const ExpectedTensor& contract : expected)
        expected_by_name.emplace(contract.name, &contract);

    for (int32_t index = 0; index < engine.getNbIOTensors(); ++index) {
        const char* name = engine.getIOTensorName(index);
        if (name == nullptr)
            fail("image engine exposes a null I/O tensor name");
        const auto found = expected_by_name.find(name);
        if (found == expected_by_name.end())
            fail(std::string("image engine exposes an unexpected I/O tensor: ") + name);
        const ExpectedTensor& contract = *found->second;
        if (engine.getTensorIOMode(name) != contract.mode)
            fail(contract.name + " I/O mode drifted");
        if (engine.getTensorDataType(name) != contract.type)
            fail(contract.name + " data type drifted");
        if (!sameShape(engine.getTensorShape(name), contract.dimensions))
            fail(contract.name + " dimensions drifted");
        if (engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE)
            fail(contract.name + " is not device-resident");
        expected_by_name.erase(found);
    }
    if (!expected_by_name.empty())
        fail("image engine is missing an expected I/O tensor");
}

std::vector<uint8_t> readExactRgb(const std::filesystem::path& path) {
    constexpr std::size_t kPixelChannels = 3;
    constexpr std::size_t kExpectedBytes =
        static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
        static_cast<std::size_t>(trtmc::sam2::kOriginalImageWidth) * kPixelChannels;

    std::error_code error;
    const std::uintmax_t actual_bytes = std::filesystem::file_size(path, error);
    if (error)
        fail("cannot stat decoded RGB frame " + path.string() + ": " + error.message());
    if (actual_bytes != kExpectedBytes) {
        fail("decoded RGB frame must contain exactly " + std::to_string(kExpectedBytes) +
             " bytes, received " + std::to_string(actual_bytes));
    }

    std::ifstream input(path, std::ios::binary);
    if (!input)
        fail("cannot open decoded RGB frame: " + path.string());
    std::vector<uint8_t> rgb(kExpectedBytes);
    input.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    if (!input || input.gcount() != static_cast<std::streamsize>(rgb.size()))
        fail("decoded RGB frame read was truncated");
    return rgb;
}

std::vector<float> rgbToUnitFloat(const std::vector<uint8_t>& rgb) {
    std::vector<float> result(rgb.size());
    constexpr float kScale = 1.0F / 255.0F;
    for (std::size_t index = 0; index < rgb.size(); ++index)
        result[index] = static_cast<float>(rgb[index]) * kScale;
    return result;
}

DeviceTensor& findTensor(std::vector<DeviceTensor>& tensors, std::string_view name) {
    for (DeviceTensor& tensor : tensors) {
        if (tensor.contract.name == name)
            return tensor;
    }
    fail("internal diagnostic tensor lookup failed for " + std::string(name));
}

std::array<int64_t, 4> bboxShape(const DeviceTensor& tensor) {
    std::array<int64_t, 4> result{};
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = tensor.contract.dimensions[index];
    return result;
}

trtmc::Sam2BBoxTensorView bboxView(const DeviceTensor& tensor, const std::vector<uint16_t>& host) {
    return {host.data(), trtmc::Sam2BBoxDataType::kBFloat16, bboxShape(tensor), host.size()};
}

float bfloat16ToFloat(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void printWinningAnchorRaw(
    const trtmc::Sam2BBoxDetection& detection,
    const std::array<std::vector<uint16_t>, trtmc::sam2::native::kBboxMapContracts.size()>& host) {
    constexpr std::array<std::size_t, 3> kAnchorCounts{128U * 128U, 64U * 64U, 32U * 32U};
    constexpr std::array<int32_t, 3> kStrides{8, 16, 32};
    std::size_t level = 0;
    std::size_t local = detection.flattened_anchor_index;
    while (level + 1U < kAnchorCounts.size() && local >= kAnchorCounts[level]) {
        local -= kAnchorCounts[level];
        ++level;
    }
    const std::size_t width = 128U >> level;
    const std::size_t area = kAnchorCounts[level];
    std::cout << "RAW: flat_anchor=" << detection.flattened_anchor_index
              << " stride=" << kStrides[level] << " x=" << local % width << " y=" << local / width
              << " cls_bf16=[";
    for (std::size_t channel = 0; channel < 2U; ++channel) {
        if (channel != 0)
            std::cout << ", ";
        const uint16_t bits = host[level][channel * area + local];
        std::cout << bfloat16ToFloat(bits) << " (0x" << std::hex << bits << std::dec << ')';
    }
    std::cout << "] reg_bf16=[";
    for (std::size_t channel = 0; channel < 4U; ++channel) {
        if (channel != 0)
            std::cout << ", ";
        const uint16_t bits = host[level + 3U][channel * area + local];
        std::cout << bfloat16ToFloat(bits) << " (0x" << std::hex << bits << std::dec << ')';
    }
    std::cout << "]\n";
}

void requireNear(float actual, float expected, float tolerance, std::string_view field) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        fail(std::string(field) + " mismatch: expected " + std::to_string(expected) + " +/- " +
             std::to_string(tolerance) + ", received " + std::to_string(actual));
    }
}

void requireCapturedFrameZero(const trtmc::Sam2BBoxDetection& detection) {
    for (std::size_t coordinate = 0; coordinate < kExpectedModelBox.size(); ++coordinate) {
        requireNear(detection.model_xyxy_1024[coordinate], kExpectedModelBox[coordinate],
                    kModelCoordinateAbsoluteTolerance,
                    "frame0 model-space bbox coordinate " + std::to_string(coordinate));
        requireNear(detection.original_xyxy[coordinate], kExpectedOriginalBox[coordinate],
                    kOriginalCoordinateAbsoluteTolerance,
                    "frame0 original-space bbox coordinate " + std::to_string(coordinate));
    }
    requireNear(detection.score, kExpectedScore, kScoreAbsoluteTolerance, "frame0 score");
    if (detection.label != kExpectedLabel) {
        fail("frame0 label mismatch: expected " + std::to_string(kExpectedLabel) + ", received " +
             std::to_string(detection.label));
    }
}

void printDetection(std::string_view prefix, const trtmc::Sam2BBoxDetection& detection) {
    std::cout << std::fixed << std::setprecision(7) << prefix << " model_xyxy=["
              << detection.model_xyxy_1024[0] << ", " << detection.model_xyxy_1024[1] << ", "
              << detection.model_xyxy_1024[2] << ", " << detection.model_xyxy_1024[3]
              << "] original_xyxy=[" << detection.original_xyxy[0] << ", "
              << detection.original_xyxy[1] << ", " << detection.original_xyxy[2] << ", "
              << detection.original_xyxy[3] << "] score=" << detection.score
              << " label=" << detection.label << '\n';
}

void runDiagnostic(const std::filesystem::path& checkpoint_path,
                   const std::filesystem::path& rgb_path) {
    using namespace trtmc::sam2::native;

    const std::vector<uint8_t> rgb = readExactRgb(rgb_path);
    const std::vector<float> rgb_float = rgbToUnitFloat(rgb);
    const trtmc::sam2::PreprocessedFrame preprocessed = trtmc::sam2::preprocessFrame(
        rgb_float.data(), trtmc::sam2::kOriginalImageHeight, trtmc::sam2::kOriginalImageWidth);
    if (preprocessed.pixel_values.size() != checkedElementCount(kImageInputContract.dimensions))
        fail("native preprocessing returned an unexpected pixel_values element count");

    Logger logger;
    CheckpointReader checkpoint = CheckpointReader::open(checkpoint_path);
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder)
        fail("TensorRT builder creation failed");
    TrtPtr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(sam2NetworkCreationFlags()));
    if (!network)
        fail("TensorRT network creation failed");
    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config)
        fail("TensorRT builder configuration creation failed");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, std::size_t{8} << 30U);

    Sam2ImageNetworkBuilder image_builder(*network, checkpoint);
    const Sam2ImageNetworkOutputs build_report = image_builder.build();
    if (build_report.referenced_tensor_count != kImageNetworkReferencedTensorCount ||
        build_report.checkpoint_tensor_count != kDeliveredCheckpointTensorCount) {
        fail("image network checkpoint-reference inventory drifted");
    }
    std::cout << "GRAPH: layers=" << build_report.added_layer_count
              << " referenced_tensors=" << build_report.referenced_tensor_count << '\n';
    TrtPtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized || serialized->data() == nullptr || serialized->size() == 0)
        fail("TensorRT image plan serialization failed");

    TrtPtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime)
        fail("TensorRT runtime creation failed");
    TrtPtr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    if (!engine)
        fail("TensorRT image plan deserialization failed");

    const std::vector<ExpectedTensor> expected = expectedTensors();
    validateExactEngineAbi(*engine, expected);
    TrtPtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context)
        fail("TensorRT image execution-context creation failed");

    std::vector<DeviceTensor> device_tensors;
    device_tensors.reserve(expected.size());
    for (const ExpectedTensor& contract : expected) {
        const std::size_t count = checkedElementCount(contract.dimensions);
        const std::size_t bytes = count * elementSize(contract.type);
        device_tensors.emplace_back(contract, count, bytes);
    }

    for (DeviceTensor& tensor : device_tensors) {
        if (!context->setTensorAddress(tensor.contract.name.c_str(), tensor.storage.data()))
            fail("setTensorAddress failed for " + tensor.contract.name);
    }

    DeviceTensor& input = findTensor(device_tensors, kImageInputContract.name);
    if (input.storage.bytes() != preprocessed.pixel_values.size() * sizeof(float))
        fail("pixel_values device allocation does not match preprocessing output");

    CudaStream stream;
    checkCuda(cudaMemcpyAsync(input.storage.data(), preprocessed.pixel_values.data(),
                              input.storage.bytes(), cudaMemcpyHostToDevice, stream.get()),
              "pixel_values cudaMemcpyAsync");
    if (!context->enqueueV3(stream.get()))
        fail("TensorRT image enqueueV3 failed");

    std::array<std::vector<uint16_t>, kBboxMapContracts.size()> bbox_host;
    std::array<DeviceTensor*, kBboxMapContracts.size()> bbox_device{};
    for (std::size_t index = 0; index < kBboxMapContracts.size(); ++index) {
        DeviceTensor& tensor = findTensor(device_tensors, kBboxMapContracts[index].name);
        bbox_device[index] = &tensor;
        bbox_host[index].resize(tensor.element_count);
        checkCuda(cudaMemcpyAsync(bbox_host[index].data(), tensor.storage.data(),
                                  tensor.storage.bytes(), cudaMemcpyDeviceToHost, stream.get()),
                  "bbox cudaMemcpyAsync");
    }
    checkCuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize");

    const trtmc::Sam2BBoxRawOutputs raw_outputs{
        bboxView(*bbox_device[0], bbox_host[0]), bboxView(*bbox_device[1], bbox_host[1]),
        bboxView(*bbox_device[2], bbox_host[2]), bboxView(*bbox_device[3], bbox_host[3]),
        bboxView(*bbox_device[4], bbox_host[4]), bboxView(*bbox_device[5], bbox_host[5])};
    const trtmc::Sam2BBoxDetections detections = trtmc::decode_sam2_bbox_outputs(
        raw_outputs, trtmc::sam2::kOriginalImageHeight, trtmc::sam2::kOriginalImageWidth);
    const trtmc::Sam2BBoxDetection& detection =
        trtmc::require_exactly_one_sam2_bbox_detection(detections);
    printWinningAnchorRaw(detection, bbox_host);
    printDetection("OBSERVED: frame0", detection);
    requireCapturedFrameZero(detection);
    printDetection("PASS: frame0", detection);
}

const char* pathArgument(int argc, char** argv, int index, const char* environment_name) {
    if (argc > index)
        return argv[index];
    return std::getenv(environment_name);
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 3) {
        std::cerr << "usage: " << argv[0] << " [checkpoint.pt [frame0.rgb]]\n"
                  << "or set TRTMC_SAM2_CHECKPOINT and TRTMC_SAM2_FRAME0_RGB\n";
        return 2;
    }
    const char* checkpoint_path = pathArgument(argc, argv, 1, "TRTMC_SAM2_CHECKPOINT");
    const char* rgb_path = pathArgument(argc, argv, 2, "TRTMC_SAM2_FRAME0_RGB");
    if (checkpoint_path == nullptr || checkpoint_path[0] == '\0' || rgb_path == nullptr ||
        rgb_path[0] == '\0') {
        std::cerr << "usage: " << argv[0] << " [checkpoint.pt [frame0.rgb]]\n"
                  << "or set TRTMC_SAM2_CHECKPOINT and TRTMC_SAM2_FRAME0_RGB\n";
        return 2;
    }

    try {
        runDiagnostic(checkpoint_path, rgb_path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: native SAM2 image diagnostic: " << error.what() << '\n';
        return 1;
    }
}
