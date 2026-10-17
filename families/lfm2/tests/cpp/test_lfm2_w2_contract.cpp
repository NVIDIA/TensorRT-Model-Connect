/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Validate the actual descriptor contract without creating a CUDA context.
#include "families/lfm2/runtime/w2_byok.cpp"

#include <iostream>
#include <vector>

namespace {

struct View {
    std::vector<int64_t> shape;
    DLTensor tensor;

    View(uintptr_t address, std::vector<int64_t> dimensions)
        : shape(std::move(dimensions)), tensor{reinterpret_cast<void*>(address),
                                               DLDevice{kDLCUDA, 0},
                                               static_cast<int32_t>(shape.size()),
                                               DLDataType{kDLBfloat, 16, 1},
                                               shape.data(),
                                               nullptr,
                                               0} {}

    tvm::ffi::TensorView get() { return tvm::ffi::TensorView(&tensor); }
};

struct Inputs {
    View x{0x100000000ULL, {17, 10752}};
    View weight{0x200000000ULL, {2048, 10752}};
    View shape_like{0, {17, 2048}};
    View output{0x300000000ULL, {17, 2048}};

    int64_t validate() {
        return validate_rows_contract(x.get(), weight.get(), shape_like.get(), output.get(), 0);
    }
};

int checks = 0;
int failures = 0;
void expect(bool result, const char* name) {
    ++checks;
    if (!result) {
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
    expect(inputs.validate() == 17, "batched prefill rows");
    inputs.shape_like.tensor.data = reinterpret_cast<void*>(uintptr_t{1});
    expect(inputs.validate() == 17, "shape carrier payload is never read");
    inputs.x.shape[0] = inputs.shape_like.shape[0] = inputs.output.shape[0] = 1;
    expect(inputs.validate() == 1, "single decode row");
    inputs.x.shape[0] = inputs.shape_like.shape[0] = inputs.output.shape[0] = 131072;
    expect(inputs.validate() == 131072, "full declared sequence capacity");
    rejected("zero rows", [](Inputs& x) { x.x.shape[0] = 0; });
    rejected("negative rows", [](Inputs& x) { x.x.shape[0] = -1; });
    rejected("rows exceed GEMM int domain",
             [](Inputs& x) { x.x.shape[0] = int64_t{std::numeric_limits<int>::max()} + 1; });
    rejected("wrong input width", [](Inputs& x) { x.x.shape[1] = 2048; });
    rejected("wrong weight orientation", [](Inputs& x) {
        x.weight.shape[0] = 10752;
        x.weight.shape[1] = 2048;
    });
    rejected("carrier row mismatch", [](Inputs& x) { x.shape_like.shape[0] = 1; });
    rejected("carrier width mismatch", [](Inputs& x) { x.shape_like.shape[1] = 10752; });
    rejected("output row mismatch", [](Inputs& x) { x.output.shape[0] = 1; });
    rejected("output width mismatch", [](Inputs& x) { x.output.shape[1] = 10752; });
    rejected("carrier rank mismatch", [](Inputs& x) { x.shape_like.tensor.ndim = 1; });
    rejected("carrier dtype mismatch",
             [](Inputs& x) { x.shape_like.tensor.dtype = {kDLFloat, 32, 1}; });
    rejected("carrier device mismatch",
             [](Inputs& x) { x.shape_like.tensor.device.device_id = 1; });
    rejected("null matrix input", [](Inputs& x) { x.x.tensor.data = nullptr; });
    rejected("null weight", [](Inputs& x) { x.weight.tensor.data = nullptr; });
    rejected("null output", [](Inputs& x) { x.output.tensor.data = nullptr; });
    int64_t strides[2] = {4096, 1};
    rejected("noncontiguous shape carrier",
             [&](Inputs& x) { x.shape_like.tensor.strides = strides; });
    std::cout << checks << " CPU metadata checks; " << failures << " failures; no CUDA calls\n";
    return failures == 0 ? 0 : 1;
}
