/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_native_video_processor.h"
#include "sam2_golden_fixture.h"
#include "tools/sam2_native_builder/checkpoint_reader.h"
#include "tools/sam2_native_builder/sam2_image_network.h"
#include "tools/sam2_native_builder/sam2_tracker_network.h"
#include "tools/sam2_native_builder/sam2_trt_layers.h"

#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using trtmc::sam2::native::CheckpointReader;

constexpr std::uint64_t kWorkspaceBytes = std::uint64_t{8} << 30U;
constexpr std::size_t kFrameChannels = 3U;
constexpr std::size_t kFramePixels = static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
                                     static_cast<std::size_t>(trtmc::sam2::kOriginalImageWidth);
constexpr std::size_t kFrameBytes = kFramePixels * kFrameChannels;
constexpr std::size_t kAllMaskBytes =
    static_cast<std::size_t>(trtmc::sam2::kFrameCount) * kFramePixels;
constexpr std::int32_t kImageLayerCount = 1139;
constexpr std::array<std::int32_t, 5> kTrackerLayerCounts = {882, 1630, 1652, 1674, 1696};
constexpr std::array<std::string_view, 5> kFrameSha256 = {
    "0bcadde0e5a6f8ba04f79c44f064c5b00d3cd1b250e2f2f3bbf10ef0630a9ce9",
    "0abfd57f9e3886a8c3068bf6bcc353b26d1e3a8a43819a80dfeb00f309b24ec3",
    "9166cc263c3edb262065fa3b98ee062cbf6d781dd656bae13def7f4141b7d025",
    "77525faadfc8a607e4e1556135887caaddd0b64d7cd677fcf47c38ecf9e25a4f",
    "cb0801b490ba13dfb6d36aeef06b049ff67ff11864ef62ccd858a0096d97c6af",
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void requireCuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess)
        fail(std::string(operation) + " failed: " + cudaGetErrorString(status));
}

class Logger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "TensorRT SAM2 diagnostic: " << (message == nullptr ? "" : message)
                      << '\n';
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

class CudaStream final {
  public:
    CudaStream() {
        requireCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
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

class CudaBuffer final {
  public:
    explicit CudaBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes_ == 0U)
            fail("refusing to allocate an empty engine tensor");
        requireCuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
    }

    ~CudaBuffer() {
        if (data_ != nullptr)
            (void)cudaFree(data_);
    }

    CudaBuffer(CudaBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0U)) {}

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr)
                (void)cudaFree(data_);
            data_ = std::exchange(other.data_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0U);
        }
        return *this;
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    void* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }

  private:
    void* data_{nullptr};
    std::size_t bytes_{0U};
};

trtmc::DType runtimeDtype(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT:
        return trtmc::DType::kFloat32;
    case nvinfer1::DataType::kBF16:
        return trtmc::DType::kBFloat16;
    default:
        fail("native diagnostic engine contains an unsupported tensor data type");
    }
}

std::vector<std::int64_t> staticShape(const nvinfer1::Dims& dimensions) {
    if (dimensions.nbDims <= 0)
        fail("native diagnostic engine tensor has an invalid rank");
    std::vector<std::int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dimensions.nbDims));
    for (std::int32_t index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] <= 0)
            fail("native diagnostic requires static positive tensor dimensions");
        shape.push_back(dimensions.d[index]);
    }
    return shape;
}

std::size_t checkedElements(const std::vector<std::int64_t>& shape) {
    std::size_t result = 1U;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0)
            fail("native diagnostic tensor shape contains a non-positive dimension");
        const auto extent = static_cast<std::size_t>(dimension);
        if (extent > std::numeric_limits<std::size_t>::max() / result)
            fail("native diagnostic tensor element count overflowed");
        result *= extent;
    }
    return result;
}

struct Binding {
    Binding(trtmc::TensorInfo metadata, std::size_t bytes)
        : info(std::move(metadata)), device(bytes), host(info.is_input ? 0U : bytes) {}

    trtmc::TensorInfo info;
    CudaBuffer device;
    std::vector<std::uint8_t> host;
};

// This adapter is deliberately test-local. It accepts only fixed device tensors
// of the two data types in the SAM2 contract and exposes no external bindings or
// device-forward shortcuts.
class TestPlanModule final : public trtmc::ITrtModule {
  public:
    TestPlanModule(const std::vector<std::uint8_t>& plan, Logger& logger) : stream_() {
        if (plan.empty())
            fail("cannot deserialize an empty native diagnostic plan");
        runtime_.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime_)
            fail("TensorRT runtime creation failed");
        engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine_)
            fail("TensorRT native diagnostic plan deserialization failed");
        context_.reset(engine_->createExecutionContext());
        if (!context_)
            fail("TensorRT native diagnostic context creation failed");
        if (engine_->getNbOptimizationProfiles() != 1)
            fail("native diagnostic engine must contain exactly one optimization profile");

        for (std::int32_t index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* raw_name = engine_->getIOTensorName(index);
            if (raw_name == nullptr || raw_name[0] == '\0')
                fail("native diagnostic engine contains an unnamed I/O tensor");
            const std::string name(raw_name);
            const auto mode = engine_->getTensorIOMode(raw_name);
            if (mode != nvinfer1::TensorIOMode::kINPUT && mode != nvinfer1::TensorIOMode::kOUTPUT) {
                fail("native diagnostic engine contains an invalid I/O mode");
            }
            if (engine_->getTensorLocation(raw_name) != nvinfer1::TensorLocation::kDEVICE)
                fail("native diagnostic engine contains a non-device I/O tensor");
            trtmc::TensorInfo info{name, staticShape(engine_->getTensorShape(raw_name)),
                                   runtimeDtype(engine_->getTensorDataType(raw_name)),
                                   mode == nvinfer1::TensorIOMode::kINPUT};
            const std::size_t elements = checkedElements(info.shape);
            if (trtmc::dtype_size(info.dtype) >
                std::numeric_limits<std::size_t>::max() / elements) {
                fail("native diagnostic tensor byte count overflowed");
            }
            const std::size_t bytes = elements * trtmc::dtype_size(info.dtype);
            auto inserted = bindings_.try_emplace(name, std::move(info), bytes);
            if (!inserted.second)
                fail("native diagnostic engine contains duplicate I/O tensor names");
            if (!context_->setTensorAddress(raw_name, inserted.first->second.device.data()))
                fail("TensorRT setTensorAddress failed for " + name);
        }
        if (bindings_.empty())
            fail("native diagnostic engine exposes no I/O tensors");
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto expected_inputs = input_info();
        if (inputs.size() != expected_inputs.size())
            fail("native diagnostic module received the wrong input count");
        for (const auto& info : expected_inputs) {
            const auto found = inputs.find(info.name);
            if (found == inputs.end())
                fail("native diagnostic module is missing input " + info.name);
            const auto& tensor = found->second;
            Binding& binding = requiredBinding(info.name);
            if (tensor.data == nullptr || tensor.shape != info.shape ||
                tensor.dtype != info.dtype || tensor.nbytes() != binding.device.bytes()) {
                fail("native diagnostic module input contract drifted for " + info.name);
            }
            requireCuda(cudaMemcpyAsync(binding.device.data(), tensor.data, binding.device.bytes(),
                                        cudaMemcpyHostToDevice, stream_.get()),
                        "input cudaMemcpyAsync");
        }
        if (!context_->enqueueV3(stream_.get()))
            fail("TensorRT native diagnostic enqueueV3 failed");
        for (const auto& info : output_info()) {
            Binding& binding = requiredBinding(info.name);
            requireCuda(cudaMemcpyAsync(binding.host.data(), binding.device.data(),
                                        binding.device.bytes(), cudaMemcpyDeviceToHost,
                                        stream_.get()),
                        "output cudaMemcpyAsync");
        }
        sync();

        trtmc::TensorMap result;
        for (const auto& info : output_info()) {
            Binding& binding = requiredBinding(info.name);
            result.emplace(info.name, trtmc::Tensor{binding.host.data(), info.shape, info.dtype});
        }
        return result;
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override {
        fail("native diagnostic test module does not expose device forward");
    }

    void forward_device_async(const trtmc::DeviceTensorMap&) override {
        fail("native diagnostic test module does not expose asynchronous device forward");
    }

    void forward_async(const trtmc::TensorMap&) override {
        fail("native diagnostic test module does not expose asynchronous host forward");
    }

    void sync() override {
        requireCuda(cudaStreamSynchronize(stream_.get()), "cudaStreamSynchronize");
    }

    cudaStream_t stream() const override { return stream_.get(); }

    void enable_cuda_graph() override {
        fail("native diagnostic test module does not expose CUDA graph capture");
    }

    bool cuda_graph_active() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }

    std::vector<trtmc::TensorInfo> input_info() const override { return metadata(true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return metadata(false); }

    bool has_input(const std::string& name) const override {
        const auto found = bindings_.find(name);
        return found != bindings_.end() && found->second.info.is_input;
    }

    bool has_output(const std::string& name) const override {
        const auto found = bindings_.find(name);
        return found != bindings_.end() && !found->second.info.is_input;
    }

    trtmc::DType tensor_dtype(const std::string& name) const override {
        return requiredBinding(name).info.dtype;
    }

    std::vector<std::int64_t> tensor_shape(const std::string& name) const override {
        return requiredBinding(name).info.shape;
    }

    std::vector<std::int64_t> input_profile_shape(const std::string& name,
                                                  std::int32_t profile_index,
                                                  trtmc::ProfileShapeSelector) const override {
        if (profile_index != 0 || !has_input(name))
            fail("native diagnostic module received an invalid profile-shape query");
        return requiredBinding(name).info.shape;
    }

    std::int32_t optimization_profile_count() const override { return 1; }

    void* device_ptr(const std::string& name) const override {
        return requiredBinding(name).device.data();
    }

    void bind_external(const std::string&, void*) override {
        fail("native diagnostic test module rejects external tensor bindings");
    }

    void bind_external(const std::string&, void*, const std::vector<std::int64_t>&) override {
        fail("native diagnostic test module rejects shaped external tensor bindings");
    }

    std::int32_t input_rank(const std::string& name) const override {
        if (!has_input(name))
            return 0;
        return static_cast<std::int32_t>(requiredBinding(name).info.shape.size());
    }

    bool input_is_dynamic(const std::string& name) const override {
        if (!has_input(name))
            fail("native diagnostic module received an unknown dynamic-shape query");
        return false;
    }

    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return context_ != nullptr; }

    void keep_alive(std::shared_ptr<void> resource) override {
        if (resource != nullptr)
            keep_alive_.push_back(std::move(resource));
    }

  private:
    Binding& requiredBinding(const std::string& name) {
        const auto found = bindings_.find(name);
        if (found == bindings_.end())
            fail("native diagnostic module does not contain tensor " + name);
        return found->second;
    }

    const Binding& requiredBinding(const std::string& name) const {
        const auto found = bindings_.find(name);
        if (found == bindings_.end())
            fail("native diagnostic module does not contain tensor " + name);
        return found->second;
    }

    std::vector<trtmc::TensorInfo> metadata(bool inputs) const {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& entry : bindings_) {
            if (entry.second.info.is_input == inputs)
                result.push_back(entry.second.info);
        }
        return result;
    }

    // Declaration order keeps the CUDA stream alive until every engine,
    // context, allocation, and retained resource has been released.
    CudaStream stream_;
    std::vector<std::shared_ptr<void>> keep_alive_;
    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    std::map<std::string, Binding> bindings_;
};

struct SerializedPlan {
    std::string_view section;
    std::vector<std::uint8_t> bytes;
};

TrtPtr<nvinfer1::INetworkDefinition> createNetwork(nvinfer1::IBuilder& builder) {
    TrtPtr<nvinfer1::INetworkDefinition> network(
        builder.createNetworkV2(trtmc::sam2::native::sam2NetworkCreationFlags()));
    if (!network)
        fail("TensorRT native diagnostic network creation failed");
    return network;
}

std::vector<std::uint8_t> serializeNetwork(nvinfer1::IBuilder& builder,
                                           nvinfer1::INetworkDefinition& network,
                                           std::string_view section) {
    TrtPtr<nvinfer1::IBuilderConfig> config(builder.createBuilderConfig());
    if (!config)
        fail("TensorRT native diagnostic builder configuration failed");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               static_cast<std::size_t>(kWorkspaceBytes));
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    TrtPtr<nvinfer1::IHostMemory> memory(builder.buildSerializedNetwork(network, *config));
    if (!memory || memory->data() == nullptr || memory->size() == 0U)
        fail("TensorRT failed to serialize " + std::string(section));
    std::vector<std::uint8_t> result(memory->size());
    std::memcpy(result.data(), memory->data(), memory->size());
    return result;
}

SerializedPlan buildImagePlan(nvinfer1::IBuilder& builder, const CheckpointReader& checkpoint) {
    using namespace trtmc::sam2::native;
    auto network = createNetwork(builder);
    Sam2ImageNetworkBuilder graph(*network, checkpoint);
    const Sam2ImageNetworkOutputs outputs = graph.build();
    if (network->getNbInputs() != 1 || network->getNbOutputs() != 9 ||
        network->getNbLayers() != kImageLayerCount ||
        outputs.added_layer_count != kImageLayerCount ||
        outputs.cast_layer_count != kImageNetworkCastLayerCount ||
        outputs.checkpoint_tensor_count != kDeliveredCheckpointTensorCount ||
        outputs.referenced_tensor_count != kImageNetworkReferencedTensorCount) {
        fail("native diagnostic image graph inventory drifted");
    }
    SerializedPlan result{trtmc::sam2::kImagePlanSection,
                          serializeNetwork(builder, *network, trtmc::sam2::kImagePlanSection)};
    std::cout << "BUILD: " << result.section << " layers=" << network->getNbLayers()
              << " weights=" << outputs.referenced_tensor_count << " bytes=" << result.bytes.size()
              << '\n';
    return result;
}

SerializedPlan buildTrackerPlan(nvinfer1::IBuilder& builder, const CheckpointReader& checkpoint,
                                std::int32_t history_frames) {
    using namespace trtmc::sam2::native;
    if (history_frames < 0 || history_frames > 4)
        fail("native diagnostic tracker history is outside [0,4]");
    auto network = createNetwork(builder);
    Sam2TrackerNetworkBuilder graph(*network, checkpoint);
    const Sam2TrackerNetworkOutputs outputs =
        history_frames == 0 ? graph.buildPrompt() : graph.buildRecurrent(history_frames);
    const TrackerPlanSpec spec =
        history_frames == 0 ? promptTrackerPlanSpec() : recurrentTrackerPlanSpec(history_frames);
    const std::size_t expected_weights = history_frames == 0 ? 185U : 291U;
    const auto expected_layers = kTrackerLayerCounts[static_cast<std::size_t>(history_frames)];
    if (network->getNbInputs() != static_cast<std::int32_t>(spec.inputs.size()) ||
        network->getNbOutputs() != static_cast<std::int32_t>(spec.outputs.size()) ||
        network->getNbLayers() != expected_layers || outputs.added_layer_count != expected_layers ||
        outputs.referenced_tensor_count != expected_weights || outputs.mask_logits_256 == nullptr ||
        outputs.object_pointer == nullptr || outputs.memory_features == nullptr) {
        fail("native diagnostic tracker graph inventory drifted for history " +
             std::to_string(history_frames));
    }
    SerializedPlan result{spec.plan_section,
                          serializeNetwork(builder, *network, spec.plan_section)};
    std::cout << "BUILD: " << result.section << " layers=" << network->getNbLayers()
              << " weights=" << outputs.referenced_tensor_count << " bytes=" << result.bytes.size()
              << '\n';
    return result;
}

std::vector<std::uint8_t> readExactFrame(const std::filesystem::path& path,
                                         std::string_view expected_sha256) {
    if (std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path))
        fail("decoded RGB frame must be a regular file: " + path.string());
    const std::string actual_sha256 = CheckpointReader::checkpointSha256(path, kFrameBytes);
    if (actual_sha256 != expected_sha256)
        fail("decoded RGB frame SHA-256 mismatch: " + path.string());
    if (std::filesystem::file_size(path) != kFrameBytes)
        fail("decoded RGB frame byte count drifted: " + path.string());
    std::ifstream input(path, std::ios::binary);
    if (!input)
        fail("unable to open decoded RGB frame: " + path.string());
    std::vector<std::uint8_t> bytes(kFrameBytes);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input || input.gcount() != static_cast<std::streamsize>(bytes.size()))
        fail("decoded RGB frame read was truncated: " + path.string());
    return bytes;
}

std::vector<float> toUnitRgb(const std::vector<std::uint8_t>& bytes) {
    std::vector<float> result(bytes.size());
    constexpr float kScale = 1.0F / 255.0F;
    for (std::size_t index = 0; index < bytes.size(); ++index)
        result[index] = static_cast<float>(bytes[index]) * kScale;
    return result;
}

void verifySourceConfig(const std::filesystem::path& path) {
    constexpr std::uint64_t kMaximumConfigBytes = std::uint64_t{1} << 20U;
    const std::string actual = CheckpointReader::checkpointSha256(path, kMaximumConfigBytes);
    if (actual != trtmc::sam2::kConfigSha256)
        fail("SAM2 source config SHA-256 mismatch");
}

void printAccuracy(const trtmc::sam2::test::BboxAccuracy& bbox,
                   const trtmc::sam2::test::MaskAccuracy& masks,
                   const std::array<std::uint64_t, 5>& foreground) {
    std::cout << std::fixed << std::setprecision(7) << "BBOX: iou=" << bbox.iou
              << " max_coordinate_error=" << bbox.max_coordinate_error
              << " score_error=" << bbox.score_error
              << " label_exact=" << (bbox.label_exact ? "true" : "false") << '\n';
    for (std::size_t frame = 0; frame < masks.frame_iou.size(); ++frame) {
        std::cout << "MASK: frame=" << frame << " iou=" << masks.frame_iou[frame]
                  << " foreground=" << foreground[frame] << '\n';
    }
    std::cout << "MASK: macro_iou=" << masks.macro_iou << " global_iou=" << masks.global_iou
              << '\n';
}

void runDiagnostic(const std::filesystem::path& checkpoint_path,
                   const std::filesystem::path& config_path,
                   const std::filesystem::path& golden_path,
                   const std::array<std::filesystem::path, 5>& frame_paths) {
    verifySourceConfig(config_path);
    const auto golden = trtmc::sam2::test::loadGoldenFixture(golden_path);

    std::array<std::vector<float>, 5> frame_storage;
    trtmc::Sam2VideoFrames frames{};
    for (std::size_t frame = 0; frame < frame_storage.size(); ++frame) {
        frame_storage[frame] = toUnitRgb(readExactFrame(frame_paths[frame], kFrameSha256[frame]));
        frames[frame] = {static_cast<std::int32_t>(frame), trtmc::sam2::kOriginalImageHeight,
                         trtmc::sam2::kOriginalImageWidth, frame_storage[frame].data(),
                         frame_storage[frame].size()};
    }

    CheckpointReader checkpoint = CheckpointReader::open(checkpoint_path);
    if (!trtmc::sam2::native::trackerGraphEmissionComplete())
        fail("native tracker graph inventory is incomplete");
    Logger logger;
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder)
        fail("TensorRT native diagnostic builder creation failed");

    std::array<SerializedPlan, 6> plans;
    plans[0] = buildImagePlan(*builder, checkpoint);
    for (std::int32_t history = 0; history <= 4; ++history)
        plans[static_cast<std::size_t>(history + 1)] =
            buildTrackerPlan(*builder, checkpoint, history);
    builder.reset();

    trtmc::sam2::NativeVideoEngineSet engines;
    engines.image = std::make_unique<TestPlanModule>(plans[0].bytes, logger);
    engines.prompt = std::make_unique<TestPlanModule>(plans[1].bytes, logger);
    for (std::size_t history = 0; history < engines.recurrent.size(); ++history) {
        engines.recurrent[history] =
            std::make_unique<TestPlanModule>(plans[history + 2U].bytes, logger);
    }
    plans = {};

    auto processor = trtmc::sam2::makeNativeVideoProcessor(std::move(engines));
    auto prompt = processor.run_bbox_prompt(frames);
    auto results = processor.propagate(prompt, frames);

    trtmc::sam2::test::GoldenBbox candidate_bbox;
    candidate_bbox.original_xyxy = prompt.track.prompt_box_xyxy;
    candidate_bbox.score = prompt.track.detector_score;
    candidate_bbox.label = prompt.track.label;
    constexpr float kModelXScale = static_cast<float>(trtmc::sam2::kModelImageSize) /
                                   static_cast<float>(trtmc::sam2::kOriginalImageWidth);
    constexpr float kModelYScale = static_cast<float>(trtmc::sam2::kModelImageSize) /
                                   static_cast<float>(trtmc::sam2::kOriginalImageHeight);
    candidate_bbox.model_xyxy = {
        candidate_bbox.original_xyxy[0] * kModelXScale,
        candidate_bbox.original_xyxy[1] * kModelYScale,
        candidate_bbox.original_xyxy[2] * kModelXScale,
        candidate_bbox.original_xyxy[3] * kModelYScale,
    };

    std::vector<std::uint8_t> candidate_masks;
    candidate_masks.reserve(kAllMaskBytes);
    std::array<std::uint64_t, 5> foreground{};
    for (std::size_t frame = 0; frame < results.size(); ++frame) {
        if (results[frame].frame_index != static_cast<std::int32_t>(frame) ||
            results[frame].height != trtmc::sam2::kOriginalImageHeight ||
            results[frame].width != trtmc::sam2::kOriginalImageWidth) {
            fail("native video processor returned invalid frame metadata");
        }
        const auto& mask = results[frame].mask.materialize_host(kFramePixels);
        foreground[frame] = static_cast<std::uint64_t>(
            std::count(mask.begin(), mask.end(), static_cast<std::uint8_t>(1U)));
        candidate_masks.insert(candidate_masks.end(), mask.begin(), mask.end());
    }

    const auto bbox_accuracy = trtmc::sam2::test::compareBbox(candidate_bbox, golden);
    const auto mask_accuracy = trtmc::sam2::test::compareMasks(candidate_masks, golden);
    printAccuracy(bbox_accuracy, mask_accuracy, foreground);
    if (!bbox_accuracy.passes())
        fail("native five-frame diagnostic failed the checked-in bbox gate");
    if (!mask_accuracy.passes())
        fail("native five-frame diagnostic failed the checked-in mask gates");
    std::cout << "PASS: native five-frame SAM2 diagnostic\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: " << argv[0]
                  << " checkpoint.pt config.yaml golden-directory frame0.rgb frame1.rgb"
                     " frame2.rgb frame3.rgb frame4.rgb\n";
        return 2;
    }
    try {
        std::array<std::filesystem::path, 5> frames = {argv[4], argv[5], argv[6], argv[7], argv[8]};
        runDiagnostic(argv[1], argv[2], argv[3], frames);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: native SAM2 five-frame diagnostic: " << error.what() << '\n';
        return 1;
    }
}
