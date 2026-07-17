/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kINFO)
            std::cerr << "[TensorRT] " << message << '\n';
    }
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::runtime_error("could not open " + path);
    const auto size = stream.tellg();
    if (size < 0)
        throw std::runtime_error("could not determine size of " + path);
    std::vector<char> result(static_cast<size_t>(size));
    stream.seekg(0);
    if (!stream.read(result.data(), size))
        throw std::runtime_error("could not read " + path);
    return result;
}

void write_file(const std::string& path, const std::vector<char>& data) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(data.data(), static_cast<std::streamsize>(data.size())))
        throw std::runtime_error("could not write " + path);
}

int64_t volume(const nvinfer1::Dims& dimensions) {
    int64_t result = 1;
    for (int32_t index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] <= 0)
            throw std::runtime_error("probe plan has a non-static tensor shape");
        result *= dimensions.d[index];
    }
    return result;
}

struct DeviceBuffer {
    void* pointer{nullptr};
    ~DeviceBuffer() {
        if (pointer != nullptr)
            cudaFree(pointer);
    }
    void allocate(size_t bytes) { check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc"); }
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: " << argv[0]
                  << " PLUGIN PLAN Q_BF16 K_BF16 V_BF16 OUTPUT_BF16 WARMUP ITERATIONS\n";
        return 2;
    }
    try {
        const std::string plugin_path = argv[1];
        const std::string plan_path = argv[2];
        const int warmup = std::stoi(argv[7]);
        const int iterations = std::stoi(argv[8]);
        if (warmup < 0 || iterations <= 0)
            throw std::runtime_error("warmup must be nonnegative and iterations must be positive");

        std::unique_ptr<void, int (*)(void*)> plugin_handle(
            dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL), dlclose);
        if (!plugin_handle)
            throw std::runtime_error(std::string("dlopen failed: ") + dlerror());

        auto plan = read_file(plan_path);
        auto q_host = read_file(argv[3]);
        auto k_host = read_file(argv[4]);
        auto v_host = read_file(argv[5]);
        if (k_host.size() != v_host.size())
            throw std::runtime_error("K/V raw buffer sizes differ");

        Logger logger;
        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        if (!runtime)
            throw std::runtime_error("createInferRuntime failed");
        std::unique_ptr<nvinfer1::ICudaEngine> engine(
            runtime->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine)
            throw std::runtime_error("deserializeCudaEngine failed");
        std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
        if (!context)
            throw std::runtime_error("createExecutionContext failed");

        const auto q_bytes = static_cast<size_t>(volume(engine->getTensorShape("q"))) * 2U;
        const auto k_bytes = static_cast<size_t>(volume(engine->getTensorShape("k"))) * 2U;
        const auto v_bytes = static_cast<size_t>(volume(engine->getTensorShape("v"))) * 2U;
        const auto output_bytes = static_cast<size_t>(volume(engine->getTensorShape("o"))) * 2U;
        if (q_host.size() != q_bytes || k_host.size() != k_bytes || v_host.size() != v_bytes ||
            output_bytes != q_bytes)
            throw std::runtime_error("raw buffers do not match the TensorRT plan shape");

        DeviceBuffer q_device;
        DeviceBuffer k_device;
        DeviceBuffer v_device;
        DeviceBuffer output_device;
        q_device.allocate(q_bytes);
        k_device.allocate(k_bytes);
        v_device.allocate(v_bytes);
        output_device.allocate(output_bytes);
        cudaStream_t stream = nullptr;
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        check_cuda(cudaMemcpyAsync(q_device.pointer, q_host.data(), q_bytes, cudaMemcpyHostToDevice,
                                   stream),
                   "copy Q");
        check_cuda(cudaMemcpyAsync(k_device.pointer, k_host.data(), k_bytes, cudaMemcpyHostToDevice,
                                   stream),
                   "copy K");
        check_cuda(cudaMemcpyAsync(v_device.pointer, v_host.data(), v_bytes, cudaMemcpyHostToDevice,
                                   stream),
                   "copy V");
        if (!context->setTensorAddress("q", q_device.pointer) ||
            !context->setTensorAddress("k", k_device.pointer) ||
            !context->setTensorAddress("v", v_device.pointer) ||
            !context->setTensorAddress("o", output_device.pointer))
            throw std::runtime_error("setTensorAddress failed");
        check_cuda(cudaStreamSynchronize(stream), "input copy synchronize");

        for (int index = 0; index < warmup; ++index) {
            if (!context->enqueueV3(stream))
                throw std::runtime_error("TensorRT warmup enqueue failed");
        }
        check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start, stream), "cudaEventRecord start");
        for (int index = 0; index < iterations; ++index) {
            if (!context->enqueueV3(stream))
                throw std::runtime_error("TensorRT timed enqueue failed");
        }
        check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
        float total_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");

        std::vector<char> output_host(output_bytes);
        check_cuda(cudaMemcpyAsync(output_host.data(), output_device.pointer, output_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "copy output");
        check_cuda(cudaStreamSynchronize(stream), "output copy synchronize");
        write_file(argv[6], output_host);
        std::cout << "{\n"
                  << "  \"kind\": \"wan2_2_ti2v_cudnn_sdpa_trt_cpp_run\",\n"
                  << "  \"warmup\": " << warmup << ",\n"
                  << "  \"iterations\": " << iterations << ",\n"
                  << "  \"total_ms\": " << total_ms << ",\n"
                  << "  \"mean_ms\": " << total_ms / static_cast<float>(iterations) << ",\n"
                  << "  \"output_bytes\": " << output_bytes << "\n"
                  << "}\n";

        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaStreamDestroy(stream);
        context.reset();
        engine.reset();
        runtime.reset();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "wan22_cudnn_sdpa_trt_runner: " << error.what() << '\n';
        return 1;
    }
}
