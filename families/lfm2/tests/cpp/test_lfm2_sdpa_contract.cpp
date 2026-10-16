/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exercise the actual family metadata validator without initializing CUDA or
// constructing a cuDNN graph. Extent payload addresses are deliberately unreadable.
#include "families/lfm2/runtime/sdpa_byok.cpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

struct View {
    std::vector<int64_t> shape;
    DLTensor tensor;

    View(uintptr_t address, std::vector<int64_t> dimensions, DLDataType dtype)
        : shape(std::move(dimensions)), tensor{reinterpret_cast<void*>(address),
                                               DLDevice{kDLCUDA, 0},
                                               static_cast<int32_t>(shape.size()),
                                               dtype,
                                               shape.data(),
                                               nullptr,
                                               0} {}

    tvm::ffi::TensorView get() { return tvm::ffi::TensorView(&tensor); }
};

struct Inputs {
    View q{0x100000000ULL, {1, 32, 1, 64}, {kDLBfloat, 16, 1}};
    View k{0x200000000ULL, {1, 8, 32, 64}, {kDLBfloat, 16, 1}};
    View v{0x300000000ULL, {1, 8, 32, 64}, {kDLBfloat, 16, 1}};
    View extent{0, {21}, {kDLInt, 32, 1}};
    View output{0x400000000ULL, {1, 32, 1, 64}, {kDLBfloat, 16, 1}};

    Geometry validate() {
        return validate_contract(q.get(), k.get(), v.get(), extent.get(), output.get(), 0);
    }
};

int failures = 0;
int checks = 0;
void expect(bool condition, const char* name) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << name << '\n';
    }
}

template <typename Mutation>
void rejected(const char* name, Mutation mutation) {
    Inputs inputs;
    mutation(inputs);
    bool caught = false;
    try {
        (void)inputs.validate();
    } catch (const std::exception&) {
        caught = true;
    }
    expect(caught, name);
}

} // namespace

int main() {
    Inputs inputs;
    const auto initial = inputs.validate();
    expect(initial.length == 21 && initial.capacity == 32, "actual extent metadata is used");
    inputs.extent.tensor.data = reinterpret_cast<void*>(uintptr_t{1});
    expect(inputs.validate().length == 21, "extent payload is never read");
    inputs.extent.shape[0] = 1;
    expect(inputs.validate().length == 1, "minimum active length");
    inputs.extent.shape[0] = kMaximumCapacity;
    inputs.k.shape[2] = inputs.v.shape[2] = kMaximumCapacity;
    expect(inputs.validate().length == kMaximumCapacity, "full declared capacity");

    rejected("zero active length", [](Inputs& x) { x.extent.shape[0] = 0; });
    rejected("negative active length", [](Inputs& x) { x.extent.shape[0] = -1; });
    rejected("active length exceeds cache", [](Inputs& x) { x.extent.shape[0] = 33; });
    rejected("zero capacity", [](Inputs& x) { x.k.shape[2] = x.v.shape[2] = 0; });
    rejected("capacity above model limit",
             [](Inputs& x) { x.k.shape[2] = x.v.shape[2] = kMaximumCapacity + 1; });
    rejected("extent must be rank one", [](Inputs& x) { x.extent.tensor.ndim = 0; });
    rejected("extent must be INT32", [](Inputs& x) { x.extent.tensor.dtype = {kDLFloat, 32, 1}; });
    rejected("extent must use same CUDA device",
             [](Inputs& x) { x.extent.tensor.device.device_id = 1; });
    rejected("CPU extent is not the bridge contract",
             [](Inputs& x) { x.extent.tensor.device.device_type = kDLCPU; });
    rejected("query requires BF16", [](Inputs& x) { x.q.tensor.dtype = {kDLFloat, 16, 1}; });
    rejected("query heads are fixed", [](Inputs& x) { x.q.shape[1] = 8; });
    rejected("single query required", [](Inputs& x) { x.q.shape[2] = 2; });
    rejected("key rank required", [](Inputs& x) { x.k.tensor.ndim = 3; });
    rejected("key head geometry", [](Inputs& x) { x.k.shape[1] = 32; });
    rejected("key value capacity must agree", [](Inputs& x) { x.v.shape[2] = 31; });
    rejected("output shape must match query", [](Inputs& x) { x.output.shape[3] = 32; });
    rejected("null data tensor", [](Inputs& x) { x.q.tensor.data = nullptr; });
    rejected("data device mismatch", [](Inputs& x) { x.v.tensor.device.device_id = 1; });
    rejected("query output alias", [](Inputs& x) { x.output.tensor.data = x.q.tensor.data; });
    rejected("key value alias", [](Inputs& x) { x.v.tensor.data = x.k.tensor.data; });
    rejected("overlapping cache range", [](Inputs& x) {
        x.v.tensor.data = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(x.k.tensor.data) + 2);
    });
    rejected("byte offset overflow",
             [](Inputs& x) { x.q.tensor.byte_offset = std::numeric_limits<uint64_t>::max(); });
    int64_t strided[4] = {2048, 128, 64, 1};
    rejected("noncontiguous query", [&](Inputs& x) { x.q.tensor.strides = strided; });
    int64_t extent_stride[1] = {2};
    rejected("noncontiguous extent descriptor",
             [&](Inputs& x) { x.extent.tensor.strides = extent_stride; });
    std::cout << checks << " CPU metadata checks; " << failures << " failures; no CUDA calls\n";
    return failures == 0 ? 0 : 1;
}
