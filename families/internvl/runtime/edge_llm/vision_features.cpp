/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <edgellm/cpp/common/logger.h>
#include <edgellm/cpp/multimodal/common/multimodalRunner.h>
#include <edgellm/cpp/runtime/streaming.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Stream {
    cudaStream_t value{nullptr};
    Stream() {
        const auto status = cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
    }
    ~Stream() { cudaStreamDestroy(value); }
};
struct CloseLibrary {
    void operator()(void* handle) const noexcept {
        if (handle)
            dlclose(handle);
    }
};
int64_t positive_integer(const char* text) {
    std::size_t end = 0;
    const std::string input(text);
    const auto value = std::stoll(input, &end);
    if (end != input.size() || value <= 0 || value > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("Expected a positive int32 dimension");
    return value;
}
} // namespace

// Test-only: execute Edge's actual visual engine, without an LLM or Model Connect adapter.
int main(int argc, char** argv) {
    try {
        if (argc != 9)
            throw std::invalid_argument("Usage: internvl_edge_vision_features ENGINE CHECKPOINT "
                                        "PLUGIN RGB H W CAPACITY OUTPUT");
        const auto height = positive_integer(argv[5]);
        const auto width = positive_integer(argv[6]);
        const auto capacity = positive_integer(argv[7]);
        const auto image_bytes = static_cast<uint64_t>(height) * width * 3;
        if (std::filesystem::file_size(argv[4]) != image_bytes)
            throw std::invalid_argument("RGB fixture size does not match H W");
        std::unique_ptr<void, CloseLibrary> plugin(
            dlopen(argv[3], RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE));
        if (!plugin)
            throw std::runtime_error(std::string("Cannot load Edge plugin: ") + dlerror());
        using InitializeFn = bool (*)(void*, const char*);
        auto initialize = reinterpret_cast<InitializeFn>(dlsym(plugin.get(), "initEdgellmPlugins"));
        if (!initialize || !initialize(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), ""))
            throw std::runtime_error("Cannot initialize Edge plugin");
        Stream stream;
        trt_edgellm::rt::LLMGenerationRequest request{};
        request.requests.resize(1);
        trt_edgellm::rt::Tensor image({1, height, width, 3}, trt_edgellm::rt::DeviceType::kCPU,
                                      nvinfer1::DataType::kUINT8);
        std::ifstream input(argv[4], std::ios::binary);
        if (!input.read(static_cast<char*>(image.rawPointer()), image_bytes))
            throw std::runtime_error("Cannot read RGB fixture");
        request.requests.front().imageBuffers.emplace_back(std::move(image));
        // Context memory must outlive the runner and all enqueued work.
        trt_edgellm::rt::Tensor context_memory;
        auto runner = trt_edgellm::rt::MultimodalRunner::create(
            (std::filesystem::path(argv[1]) / "visual").string(), 1, capacity, stream.value,
            argv[2]);
        if (!runner)
            throw std::runtime_error("Cannot create Edge visual runner");
        context_memory =
            trt_edgellm::rt::Tensor({runner->getRequiredContextMemorySize()},
                                    trt_edgellm::rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        try {
            if (!runner->setContextMemory(context_memory))
                throw std::runtime_error("Cannot set visual context memory");
            std::vector<std::vector<int32_t>> unused_ids;
            if (!runner->preprocess(request, unused_ids, nullptr, std::nullopt, stream.value,
                                    true) ||
                !runner->infer(stream.value))
                throw std::runtime_error("Edge visual inference failed");
            const auto& output = runner->getOutputEmbedding();
            if (output.getDataType() != nvinfer1::DataType::kHALF ||
                output.getShape().volume() <= 0)
                throw std::runtime_error("Invalid Edge feature shape or dtype");
            std::vector<uint16_t> features(output.getShape().volume());
            const auto bytes = features.size() * sizeof(uint16_t);
            const auto copied = cudaMemcpyAsync(features.data(), output.rawPointer(), bytes,
                                                cudaMemcpyDeviceToHost, stream.value);
            const auto synced = cudaStreamSynchronize(stream.value);
            if (copied != cudaSuccess || synced != cudaSuccess)
                throw std::runtime_error("Cannot read Edge vision features");
            std::ofstream file(argv[8], std::ios::binary);
            file.write(reinterpret_cast<const char*>(features.data()), bytes);
            file.close();
            if (!file)
                throw std::runtime_error("Cannot write Edge vision features");
        } catch (...) {
            cudaStreamSynchronize(stream.value);
            throw;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
