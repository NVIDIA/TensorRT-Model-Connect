/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/bundle.h"
#include "trtmc/features.hpp"

#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
class Logger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << message << '\n';
    }
};
struct CudaResources {
    CudaResources() {
        buffers.reserve(3);
        events.reserve(200);
    }
    CudaResources(const CudaResources&) = delete;
    CudaResources& operator=(const CudaResources&) = delete;
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    std::vector<void*> buffers;
    std::vector<cudaEvent_t> events;
    ~CudaResources() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
        for (auto event : events)
            cudaEventDestroy(event);
        for (void* buffer : buffers)
            cudaFree(buffer);
        if (stream)
            cudaStreamDestroy(stream);
    }
    void* allocate(size_t bytes) {
        void* pointer{};
        checked(cudaMalloc(&pointer, bytes));
        buffers.push_back(pointer);
        return pointer;
    }
    cudaEvent_t event() {
        cudaEvent_t value{};
        checked(cudaEventCreate(&value));
        events.push_back(value);
        return value;
    }
};

std::vector<float> read(const char* path, size_t expected) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    if (static_cast<size_t>(file.tellg()) != expected * sizeof(float))
        throw std::invalid_argument("BEV input file has the wrong length");
    std::vector<float> values(expected);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), expected * sizeof(float));
    return values;
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return (static_cast<double>(values[(values.size() - 1) / 2]) + values[values.size() / 2]) / 2;
}

void check_tensor(const nvinfer1::ICudaEngine& engine, const char* name,
                  const std::array<int64_t, 4>& shape) {
    const auto dimensions = engine.getTensorShape(name);
    require(dimensions.nbDims == 4 && std::equal(shape.begin(), shape.end(), dimensions.d) &&
                engine.getTensorDataType(name) == nvinfer1::DataType::kFLOAT,
            "engine-only measurement requires the qualified FP32 BEV I/O contract");
}

void engine_only(const char* bundle, const char* camera_path, const char* lidar_path,
                 const char* output) {
    const auto camera = read(camera_path, 80 * 128 * 128),
               lidar = read(lidar_path, 256 * 128 * 128);
    trtmc::BundleReader reader(bundle);
    require(reader.info().family == "bevfusion" &&
                reader.info().task == TRTMC_TASK_SPATIAL_FEATURES_TO_CLASS_SCORES,
            "engine-only measurement requires a BEVFusion feature-to-map bundle");
    const auto plan = reader.read_section("engine.plan");
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    require(runtime != nullptr, "failed to create TensorRT runtime");
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(plan.data(), plan.size()));
    require(engine != nullptr && engine->getNbIOTensors() == 3, "invalid BEVFusion engine");
    check_tensor(*engine, "camera_bev", {1, 80, 128, 128});
    check_tensor(*engine, "lidar_bev", {1, 256, 128, 128});
    check_tensor(*engine, "map_probabilities", {1, 6, 200, 200});
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    require(context != nullptr, "failed to create TensorRT execution context");
    CudaResources cuda;
    checked(cudaStreamCreateWithFlags(&cuda.stream, cudaStreamNonBlocking));
    void* camera_gpu = cuda.allocate(camera.size() * sizeof(float));
    void* lidar_gpu = cuda.allocate(lidar.size() * sizeof(float));
    std::vector<float> probabilities(6 * 200 * 200);
    void* output_gpu = cuda.allocate(probabilities.size() * sizeof(float));
    checked(cudaMemcpyAsync(camera_gpu, camera.data(), camera.size() * sizeof(float),
                            cudaMemcpyHostToDevice, cuda.stream));
    checked(cudaMemcpyAsync(lidar_gpu, lidar.data(), lidar.size() * sizeof(float),
                            cudaMemcpyHostToDevice, cuda.stream));
    require(context->setTensorAddress("camera_bev", camera_gpu) &&
                context->setTensorAddress("lidar_bev", lidar_gpu) &&
                context->setTensorAddress("map_probabilities", output_gpu),
            "failed to bind resident engine I/O");
    require(context->enqueueV3(cuda.stream), "initial TensorRT enqueue failed");
    checked(cudaStreamSynchronize(cuda.stream));
    // The captured region contains exactly one TensorRT enqueue, with no copies or Task work.
    checked(cudaStreamBeginCapture(cuda.stream, cudaStreamCaptureModeGlobal));
    require(context->enqueueV3(cuda.stream), "captured TensorRT enqueue failed");
    checked(cudaStreamEndCapture(cuda.stream, &cuda.graph));
    checked(cudaGraphInstantiateWithFlags(&cuda.executable, cuda.graph, 0));
    for (int index = 0; index < 20; ++index)
        checked(cudaGraphLaunch(cuda.executable, cuda.stream));
    checked(cudaStreamSynchronize(cuda.stream));
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> events;
    for (int index = 0; index < 100; ++index)
        events.emplace_back(cuda.event(), cuda.event());
    std::vector<float> samples;
    std::vector<double> round_medians;
    for (int round = 0; round < 3; ++round) {
        for (auto [start, end] : events) {
            checked(cudaEventRecord(start, cuda.stream));
            checked(cudaGraphLaunch(cuda.executable, cuda.stream));
            checked(cudaEventRecord(end, cuda.stream));
        }
        checked(cudaStreamSynchronize(cuda.stream));
        std::vector<float> current;
        for (auto [start, end] : events) {
            float milliseconds{};
            checked(cudaEventElapsedTime(&milliseconds, start, end));
            current.push_back(milliseconds);
        }
        round_medians.push_back(median(current));
        samples.insert(samples.end(), current.begin(), current.end());
    }
    std::sort(samples.begin(), samples.end());
    // Output readback and serialization happen only after every measured iteration.
    checked(cudaMemcpyAsync(probabilities.data(), output_gpu, probabilities.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, cuda.stream));
    checked(cudaStreamSynchronize(cuda.stream));
    const std::string values_path = std::string(output) + ".probabilities.f32";
    std::ofstream values(values_path, std::ios::binary);
    values.exceptions(std::ios::badbit | std::ios::failbit);
    values.write(reinterpret_cast<const char*>(probabilities.data()),
                 probabilities.size() * sizeof(float));
    cudaDeviceProp device{};
    int ordinal{};
    checked(cudaGetDevice(&ordinal));
    checked(cudaGetDeviceProperties(&device, ordinal));
    int cuda_version{};
    checked(cudaRuntimeGetVersion(&cuda_version));
    const auto metadata = reader.read_section("runtime.json");
    const nlohmann::json report{
        {"timing_boundary", "trt_engine_cuda_graph_gpu"},
        {"median_ms", median(samples)},
        {"p95_ms", samples[284]},
        {"round_medians_ms", round_medians},
        {"warmup", 20},
        {"iterations", 100},
        {"rounds", 3},
        {"engine_forward_calls_per_iteration", 1},
        {"transfers_included", false},
        {"cuda_runtime_version", cuda_version},
        {"tensorrt_runtime_version", getInferLibVersion()},
        {"runtime_metadata", nlohmann::json::parse(metadata.begin(), metadata.end())},
        {"gpu", device.name},
        {"output_values_file", values_path},
        {"synchronization",
         "CUDA events on the engine stream; stream synchronization after each round"}};
    std::ofstream result(output);
    result.exceptions(std::ios::badbit | std::ios::failbit);
    result << report.dump(2) << '\n';
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 6 && argc != 7)
        return 2;
    try {
        if (argc == 7) {
            require(std::string(argv[6]) == "--engine-only",
                    "unknown BEVFusion qualification option");
            engine_only(argv[1], argv[3], argv[4], argv[5]);
            return 0;
        }
        auto camera = read(argv[3], 80 * 128 * 128), lidar = read(argv[4], 256 * 128 * 128);
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        auto task = model.task<trtmc::SpatialFeaturesToClassScores>();
        trtmc::SpatialFeaturesToClassScoresRequest request{
            {{"camera_bev", {camera.data(), camera.size()}, 80, 128, 128},
             {"lidar_bev", {lidar.data(), lidar.size()}, 256, 128, 128}}};
        auto result = task.run(request);
        const auto original = result.view();
        const std::vector<float> saved(original.values.data,
                                       original.values.data + original.values.size);
        std::reverse(request.inputs.begin(), request.inputs.end());
        auto repeated = task.run(request);
        const auto view = result.view(), next = repeated.view();
        const std::vector<std::string> names{"drivable_area", "ped_crossing", "walkway",
                                             "stop_line",     "carpark_area", "divider"};
        if (view.classes != 6 || view.height != 200 || view.width != 200 ||
            view.values.size != 240000 || view.score_kind != TRTMC_SCORE_PROBABILITY ||
            view.class_names.size != 6 || next.values.size != view.values.size ||
            !std::equal(view.values.data, view.values.data + view.values.size, next.values.data))
            throw std::runtime_error(
                "BEVFusion public Task result or independent-call contract failed");
        for (auto& value : camera)
            value = -value;
        for (auto& value : lidar)
            value = -value;
        auto changed = task.run(request);
        const auto different = changed.view();
        require(different.values.size == saved.size() &&
                    !std::equal(saved.begin(), saved.end(), different.values.data) &&
                    std::equal(saved.begin(), saved.end(), result.view().values.data) &&
                    std::equal(saved.begin(), saved.end(), repeated.view().values.data),
                "BEVFusion later calls must preserve independently owned results");
        std::ofstream output(argv[5], std::ios::binary);
        for (size_t index = 0; index < names.size(); ++index)
            if (std::string_view(view.class_names.data[index].data,
                                 view.class_names.data[index].size) != names[index])
                throw std::runtime_error(
                    "BEVFusion class order differs from the source configuration");
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output.write(reinterpret_cast<const char*>(view.values.data),
                     view.values.size * sizeof(float));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
