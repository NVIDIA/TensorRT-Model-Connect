/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/tensorrt/plan_stream_reader.h"

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
