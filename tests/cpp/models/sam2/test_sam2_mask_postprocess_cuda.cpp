/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_mask_postprocess.h"
#include "runtime/models/sam2/sam2_mask_postprocess_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMaskElements = 256U * 256U;
constexpr std::size_t kPointerElements = 256U;
constexpr std::size_t kMemoryElements = 64U * 64U * 64U;
constexpr std::size_t kOutputElements =
    static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
    static_cast<std::size_t>(trtmc::sam2::kOriginalImageWidth);

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        fail(std::string(operation) + " failed: " + cudaGetErrorString(status));
}

class Stream final {
  public:
    Stream() {
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream create");
    }
    ~Stream() {
        if (stream_ != nullptr)
            (void)cudaStreamDestroy(stream_);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_{nullptr};
};

class DeviceBytes final {
  public:
    explicit DeviceBytes(std::size_t bytes) : bytes_(bytes) {
        checkCuda(cudaMalloc(&data_, bytes_), "device allocation");
    }
    ~DeviceBytes() {
        if (data_ != nullptr)
            (void)cudaFree(data_);
    }
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
    void* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }

  private:
    void* data_{nullptr};
    std::size_t bytes_{0U};
};

struct DeviceFixture {
    Stream stream;
    DeviceBytes mask{kMaskElements * sizeof(float)};
    DeviceBytes pointer{kPointerElements * sizeof(float)};
    DeviceBytes memory{kMemoryElements * sizeof(std::uint16_t)};
    DeviceBytes output{kOutputElements};
    DeviceBytes status{sizeof(std::uint32_t)};
};

std::uint16_t bfloat16(float value) {
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7FFF) + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

std::vector<std::uint8_t> runDevice(DeviceFixture& fixture, const std::vector<float>& logits,
                                    const std::vector<float>& pointers,
                                    const std::vector<std::uint16_t>& memory,
                                    std::uint32_t& status) {
    checkCuda(cudaMemcpyAsync(fixture.mask.data(), logits.data(), fixture.mask.bytes(),
                              cudaMemcpyHostToDevice, fixture.stream.get()),
              "mask upload");
    checkCuda(cudaMemcpyAsync(fixture.pointer.data(), pointers.data(), fixture.pointer.bytes(),
                              cudaMemcpyHostToDevice, fixture.stream.get()),
              "pointer upload");
    checkCuda(cudaMemcpyAsync(fixture.memory.data(), memory.data(), fixture.memory.bytes(),
                              cudaMemcpyHostToDevice, fixture.stream.get()),
              "memory upload");
    checkCuda(
        cudaMemsetAsync(fixture.status.data(), 0, fixture.status.bytes(), fixture.stream.get()),
        "status reset");
    checkCuda(trtmc::sam2::enqueueSam2ValidateAndResizeMask(
                  static_cast<const float*>(fixture.mask.data()),
                  static_cast<const float*>(fixture.pointer.data()),
                  static_cast<const std::uint16_t*>(fixture.memory.data()),
                  static_cast<std::uint8_t*>(fixture.output.data()),
                  static_cast<std::uint32_t*>(fixture.status.data()), fixture.stream.get()),
              "mask postprocess enqueue");

    std::vector<std::uint8_t> output(kOutputElements);
    checkCuda(cudaMemcpyAsync(output.data(), fixture.output.data(), output.size(),
                              cudaMemcpyDeviceToHost, fixture.stream.get()),
              "mask download");
    checkCuda(cudaMemcpyAsync(&status, fixture.status.data(), sizeof(status),
                              cudaMemcpyDeviceToHost, fixture.stream.get()),
              "status download");
    checkCuda(cudaStreamSynchronize(fixture.stream.get()), "stream synchronize");
    return output;
}

void checkParity(DeviceFixture& fixture, std::vector<float> logits, const char* label) {
    std::vector<float> pointers(kPointerElements, 1.0F);
    std::vector<std::uint16_t> memory(kMemoryElements, bfloat16(1.0F));
    const auto expected = trtmc::sam2::resizeAndThresholdMask(logits.data(), 256, 256,
                                                              trtmc::sam2::kOriginalImageHeight,
                                                              trtmc::sam2::kOriginalImageWidth);
    std::uint32_t status = UINT32_MAX;
    const auto actual = runDevice(fixture, logits, pointers, memory, status);
    if (status != 0U || actual != expected)
        fail(std::string("CUDA mask parity drifted for ") + label);
}

void testParity(DeviceFixture& fixture) {
    checkParity(fixture, std::vector<float>(kMaskElements, 1.0F), "positive constant");
    checkParity(fixture, std::vector<float>(kMaskElements, -1.0F), "negative constant");

    std::vector<float> structured(kMaskElements);
    for (std::size_t index = 0; index < structured.size(); ++index) {
        const auto signed_value = static_cast<std::int32_t>((index * 37U) % 1001U) - 500;
        structured[index] = static_cast<float>(signed_value) / 127.0F;
    }
    for (std::size_t index = 0; index < structured.size(); index += 257U)
        structured[index] = 0.0F;
    checkParity(fixture, std::move(structured), "structured logits");

    std::vector<float> threshold_edge(kMaskElements, 0x1.5be326p-6F);
    for (std::size_t row = 0; row < 256U; ++row)
        threshold_edge[row * 256U + 4U] = -0x1.aeb7b6p-7F;
    const auto edge_expected = trtmc::sam2::resizeAndThresholdMask(
        threshold_edge.data(), 256, 256, trtmc::sam2::kOriginalImageHeight,
        trtmc::sam2::kOriginalImageWidth);
    if (edge_expected[17U] != 0U)
        fail("threshold-edge host oracle unexpectedly contracted interpolation arithmetic");
    checkParity(fixture, std::move(threshold_edge), "separately-rounded threshold edge");

    std::vector<float> contraction_witness(kMaskElements, 0x1.2b6cfep+0F);
    for (std::size_t row = 0; row < 256U; ++row)
        contraction_witness[row * 256U + 1U] = -0x1.82c21cp+3F;
    const auto contraction_expected = trtmc::sam2::resizeAndThresholdMask(
        contraction_witness.data(), 256, 256, trtmc::sam2::kOriginalImageHeight,
        trtmc::sam2::kOriginalImageWidth);
    if (contraction_expected[2U] != 0U)
        fail("contraction witness did not retain separate rounding");
    checkParity(fixture, std::move(contraction_witness), "release-build contraction witness");
}

void testFiniteStatus(DeviceFixture& fixture) {
    std::vector<float> logits(kMaskElements, 1.0F);
    std::vector<float> pointers(kPointerElements, 1.0F);
    std::vector<std::uint16_t> memory(kMemoryElements, bfloat16(1.0F));
    logits[7] = std::numeric_limits<float>::quiet_NaN();
    pointers[11] = std::numeric_limits<float>::infinity();
    memory[13] = UINT16_C(0x7F80);
    std::uint32_t status = 0U;
    (void)runDevice(fixture, logits, pointers, memory, status);
    const auto required = trtmc::sam2::kSam2DeviceStatusMaskLogitsNonFinite |
                          trtmc::sam2::kSam2DeviceStatusObjectPointerNonFinite |
                          trtmc::sam2::kSam2DeviceStatusMemoryFeaturesNonFinite |
                          trtmc::sam2::kSam2DeviceStatusMaskResizeNonFinite;
    if ((status & required) != required)
        fail("CUDA tracker finite-status bits were incomplete");
}

} // namespace

int main() {
    try {
        std::int32_t device_count = 0;
        const auto probe = cudaGetDeviceCount(&device_count);
        if (probe != cudaSuccess || device_count <= 0) {
            std::cout << "SKIP: CUDA device unavailable for SAM2 mask parity\n";
            return 0;
        }
        DeviceFixture fixture;
        testParity(fixture);
        testFiniteStatus(fixture);
        std::cout << "PASS: exact CUDA SAM2 mask postprocess parity and finite status\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
