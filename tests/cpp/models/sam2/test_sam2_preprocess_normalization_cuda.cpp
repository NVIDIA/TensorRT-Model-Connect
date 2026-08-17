/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_preprocess.h"
#include "runtime/models/sam2/sam2_preprocess_cuda.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void requireCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0U;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

} // namespace

int main() {
    try {
        int device_count = 0;
        const auto probe = cudaGetDeviceCount(&device_count);
        if (probe == cudaErrorNoDevice || probe == cudaErrorInsufficientDriver ||
            (probe == cudaSuccess && device_count <= 0)) {
            std::cout << "SKIP: CUDA device unavailable for SAM2 RGB8 normalization parity\n";
            return 0;
        }
        requireCuda(probe, "device count query");

        constexpr auto kElements = trtmc::sam2::kSam2Rgb8NormalizationTableElements;
        std::array<std::uint8_t, kElements> values{};
        for (std::size_t channel = 0; channel < trtmc::sam2::kSam2RgbChannels; ++channel) {
            for (std::size_t value = 0; value < trtmc::sam2::kSam2Rgb8ValueCount; ++value)
                values[channel * trtmc::sam2::kSam2Rgb8ValueCount + value] =
                    static_cast<std::uint8_t>(value);
        }
        std::array<float, kElements> observed{};
        const auto& expected = trtmc::sam2::sam2Rgb8NormalizationTable();

        cudaStream_t stream = nullptr;
        std::uint8_t* device_values = nullptr;
        float* device_observed = nullptr;
        float* device_table = nullptr;
        try {
            requireCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
            requireCuda(cudaMalloc(reinterpret_cast<void**>(&device_values), sizeof(values)),
                        "value allocation");
            requireCuda(cudaMalloc(reinterpret_cast<void**>(&device_observed), sizeof(observed)),
                        "output allocation");
            requireCuda(cudaMalloc(reinterpret_cast<void**>(&device_table), sizeof(expected)),
                        "table allocation");
            requireCuda(cudaMemcpyAsync(device_values, values.data(), sizeof(values),
                                        cudaMemcpyHostToDevice, stream),
                        "value upload");
            requireCuda(cudaMemcpyAsync(device_table, expected.data(), sizeof(expected),
                                        cudaMemcpyHostToDevice, stream),
                        "table upload");
            requireCuda(trtmc::sam2::enqueueSam2Rgb8NormalizationQualification(
                            device_values, device_observed, device_table, stream),
                        "qualification launch");
            requireCuda(cudaMemcpyAsync(observed.data(), device_observed, sizeof(observed),
                                        cudaMemcpyDeviceToHost, stream),
                        "output download");
            requireCuda(cudaStreamSynchronize(stream), "qualification synchronize");
            for (std::size_t index = 0; index < kElements; ++index) {
                if (bits(observed[index]) != bits(expected[index]))
                    throw std::runtime_error("RGB8 GPU normalization bit pattern drifted");
            }
        } catch (...) {
            if (device_table != nullptr)
                (void)cudaFree(device_table);
            if (device_observed != nullptr)
                (void)cudaFree(device_observed);
            if (device_values != nullptr)
                (void)cudaFree(device_values);
            if (stream != nullptr)
                (void)cudaStreamDestroy(stream);
            throw;
        }

        requireCuda(cudaFree(device_table), "table free");
        requireCuda(cudaFree(device_observed), "output free");
        requireCuda(cudaFree(device_values), "value free");
        requireCuda(cudaStreamDestroy(stream), "stream destroy");
        std::cout << "SAM2 RGB8 CUDA normalization matched all 768 CPU bit patterns\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
