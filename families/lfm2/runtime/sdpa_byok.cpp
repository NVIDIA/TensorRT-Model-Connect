/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 *
 * The dense SDPA graph follows PyTorch commit
 * 7661cd9c6b841b62b7f411aa52ec51f05457263b,
 * aten/src/ATen/native/cudnn/MHA.cpp.
 *
 * From PyTorch:
 *
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin,
 * Jason Weston) Copyright (c) 2006      Idiap Research Institute (Samy Bengio) Copyright (c)
 * 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 *
 * From Caffe2:
 *
 * Copyright (c) 2016-present, Facebook Inc. All rights reserved.
 *
 * All contributions by Facebook:
 * Copyright (c) 2016 Facebook Inc.
 *
 * All contributions by Google:
 * Copyright (c) 2015 Google Inc.
 * All rights reserved.
 *
 * All contributions by Yangqing Jia:
 * Copyright (c) 2015 Yangqing Jia
 * All rights reserved.
 *
 * All contributions by Kakao Brain:
 * Copyright 2019-2020 Kakao Brain
 *
 * All contributions by Cruise LLC:
 * Copyright (c) 2022 Cruise LLC.
 * All rights reserved.
 *
 * All contributions by Tri Dao:
 * Copyright (c) 2024 Tri Dao.
 * All rights reserved.
 *
 * All contributions by Arm:
 * Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
 *
 * All contributions from Caffe:
 * Copyright(c) 2013, 2014, 2015, the respective contributors
 * All rights reserved.
 *
 * All other contributions:
 * Copyright(c) 2015, 2016 the respective contributors
 * All rights reserved.
 *
 * Caffe2 uses a copyright model similar to Caffe: each contributor holds
 * copyright over their contributions to Caffe2. The project versioning records
 * all such contribution and copyright details. If a contributor wants to further
 * mark their specific copyright on a particular contribution, they should
 * indicate their copyright solely in the commit message of the change when it is
 * committed.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
 *    and IDIAP Research Institute nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cudnn_frontend.h>
#include <exception>
#include <limits>
#include <memory>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <unordered_map>

namespace {

namespace fe = cudnn_frontend;
constexpr std::size_t kWorkspaceLimit = std::size_t{1} << 30;
constexpr int64_t kMaximumCapacity = 131072;

void require(bool value, const char* message) {
    if (!value)
        TVM_FFI_THROW(ValueError) << message;
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << operation << ": " << cudaGetErrorString(status);
}

void check(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS)
        TVM_FFI_THROW(RuntimeError) << operation << ": " << cudnnGetErrorString(status);
}

void check(fe::error_t status, const char* operation) {
    if (!status.is_good())
        TVM_FFI_THROW(RuntimeError) << operation << ": " << status.get_message();
}

void* pointer(tvm::ffi::TensorView value) {
    const auto address = reinterpret_cast<uintptr_t>(value.data_ptr());
    require(value.byte_offset() <= std::numeric_limits<uintptr_t>::max() - address,
            "LFM2 SDPA tensor byte offset overflows");
    return reinterpret_cast<void*>(address + value.byte_offset());
}

struct Geometry {
    int64_t length;
    int64_t capacity;
};

void require_device(tvm::ffi::TensorView value, int device) {
    require(value.device().device_type == kDLCUDA && value.device().device_id == device,
            "LFM2 SDPA tensors must use the current CUDA device");
}

void require_bf16(tvm::ffi::TensorView value, const std::array<int64_t, 4>& shape, int device) {
    require_device(value, device);
    const auto dtype = value.dtype();
    require(value.ndim() == 4 && value.IsContiguous() && dtype.code == kDLBfloat &&
                dtype.bits == 16 && dtype.lanes == 1,
            "LFM2 SDPA requires contiguous rank-four BF16 tensors");
    for (int index = 0; index < 4; ++index)
        require(value.size(index) == shape[index], "LFM2 SDPA tensor geometry mismatch");
    require(value.data_ptr() != nullptr, "LFM2 SDPA data tensor has a null pointer");
}

Geometry validate_contract(tvm::ffi::TensorView query, tvm::ffi::TensorView key,
                           tvm::ffi::TensorView value, tvm::ffi::TensorView extent,
                           tvm::ffi::TensorView output, int device) {
    require_device(extent, device);
    const auto dtype = extent.dtype();
    require(extent.ndim() == 1 && extent.IsContiguous() && dtype.code == kDLInt &&
                dtype.bits == 32 && dtype.lanes == 1,
            "LFM2 active_length_extent must be rank-one INT32 metadata");
    // Only descriptor metadata is read. The extent's pointer and values have
    // no mathematical meaning and may not contain initialized token data.
    const int64_t length = extent.size(0);
    require(key.ndim() == 4, "LFM2 SDPA key must have rank four");
    const int64_t capacity = key.size(2);
    require(capacity > 0 && capacity <= kMaximumCapacity && length > 0 && length <= capacity,
            "LFM2 SDPA requires 1 <= active length <= capacity <= 131072");
    require_bf16(query, {1, 32, 1, 64}, device);
    require_bf16(key, {1, 8, capacity, 64}, device);
    require_bf16(value, {1, 8, capacity, 64}, device);
    require_bf16(output, {1, 32, 1, 64}, device);
    const std::size_t cache_bytes = static_cast<std::size_t>(8 * capacity * 64 * 2);
    const std::array<std::pair<uintptr_t, std::size_t>, 4> ranges{{
        {reinterpret_cast<uintptr_t>(pointer(query)), 4096},
        {reinterpret_cast<uintptr_t>(pointer(key)), cache_bytes},
        {reinterpret_cast<uintptr_t>(pointer(value)), cache_bytes},
        {reinterpret_cast<uintptr_t>(pointer(output)), 4096},
    }};
    for (const auto& range : ranges)
        require(range.first <= std::numeric_limits<uintptr_t>::max() - range.second,
                "LFM2 SDPA pointer range overflows");
    for (std::size_t a = 0; a < ranges.size(); ++a)
        for (std::size_t b = a + 1; b < ranges.size(); ++b)
            require(ranges[a].first + ranges[a].second <= ranges[b].first ||
                        ranges[b].first + ranges[b].second <= ranges[a].first,
                    "LFM2 SDPA data tensors must not overlap");
    return {length, capacity};
}

std::unique_ptr<fe::graph::Graph> build_graph(cudnnHandle_t handle, int64_t length) {
    auto graph = std::make_unique<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto scale = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_uid(5)
                                   .set_name("Attn_scale")
                                   .set_dim({1, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_is_pass_by_value(true)
                                   .set_data_type(fe::DataType_t::FLOAT));
    auto options = fe::graph::SDPA_attributes()
                       .set_name("CUDNN_SDPA")
                       .set_generate_stats(false)
                       .set_causal_mask(false)
                       .set_attn_scale(scale);
    auto query = graph->tensor(fe::graph::Tensor_attributes().set_uid(0).set_name("Q"));
    auto key = graph->tensor(fe::graph::Tensor_attributes().set_uid(1).set_name("K"));
    auto value = graph->tensor(fe::graph::Tensor_attributes().set_uid(2).set_name("V"));
    auto [output, stats] = graph->sdpa(query, key, value, options);
    require(!stats, "LFM2 SDPA inference unexpectedly created a statistics tensor");
    output->set_uid(3).set_output(true);
    query->set_dim({1, 32, 1, 64}).set_stride({2048, 64, 64, 1});
    key->set_dim({1, 8, length, 64}).set_stride({8 * length * 64, length * 64, 64, 1});
    value->set_dim({1, 8, length, 64}).set_stride({8 * length * 64, length * 64, 64, 1});
    output->set_dim({1, 32, 1, 64}).set_stride({2048, 64, 64, 1});
    check(graph->validate(), "LFM2 SDPA validate");
    check(graph->build_operation_graph(handle), "LFM2 SDPA build operation graph");
    check(graph->create_execution_plans({fe::HeurMode_t::A}), "LFM2 SDPA create plans");
    check(graph->check_support(handle), "LFM2 SDPA check support");
    check(graph->build_plans(handle), "LFM2 SDPA build default plan");
    return graph;
}

struct Buffer {
    void* data = nullptr;
    std::size_t size = 0;

    // The caller waits for the completion event before changing allocations.
    void grow(std::size_t required) {
        if (required <= size)
            return;
        if (data) {
            check(cudaFree(data), "LFM2 SDPA release scratch");
            data = nullptr;
            size = 0;
        }
        check(cudaMalloc(&data, required), "LFM2 SDPA allocate scratch");
        size = required;
    }
};

struct State {
    int device;
    cudnnHandle_t handle = nullptr;
    cudaEvent_t completion = nullptr;
    cudaStream_t stream = nullptr;
    bool pending = false;
    int64_t current_length = 0;
    std::unique_ptr<fe::graph::Graph> graph;
    Buffer packed;
    Buffer workspace;

    explicit State(int device_id) : device(device_id) {
        check(cudnnCreate(&handle), "LFM2 SDPA create handle");
        const auto status = cudaEventCreateWithFlags(&completion, cudaEventDisableTiming);
        if (status != cudaSuccess) {
            cudnnDestroy(handle);
            handle = nullptr;
            check(status, "LFM2 SDPA create completion event");
        }
    }

    ~State() {
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess ||
            (previous != device && cudaSetDevice(device) != cudaSuccess)) {
            // During CUDA teardown the owning context may already be gone.
            // Do not destroy descriptors or allocations on another device.
            graph.release();
            return;
        }
        if (pending)
            cudaEventSynchronize(completion);
        graph.reset();
        if (packed.data)
            cudaFree(packed.data);
        if (workspace.data)
            cudaFree(workspace.data);
        cudnnDestroy(handle);
        cudaEventDestroy(completion);
        if (previous != device)
            cudaSetDevice(previous);
    }

    void wait() {
        if (pending) {
            check(cudaEventSynchronize(completion), "LFM2 SDPA wait for prior use");
            pending = false;
        }
    }

    void prepare(Geometry geometry, cudaStream_t next_stream) {
        const auto packing_bytes = static_cast<std::size_t>(2 * 8 * geometry.capacity * 64 * 2);
        if (stream != next_stream || current_length != geometry.length ||
            packed.size < packing_bytes) {
            // The owned event remains valid even if the previous borrowed
            // stream has since been destroyed by its model owner.
            wait();
            if (stream != next_stream) {
                check(cudnnSetStream(handle, next_stream), "LFM2 SDPA set stream");
                stream = next_stream;
            }
        }
        if (current_length == geometry.length) {
            packed.grow(packing_bytes);
            return;
        }
        graph.reset();
        current_length = 0;
        auto next = build_graph(handle, geometry.length);
        int64_t bytes = 0;
        check(next->get_workspace_size(bytes), "LFM2 SDPA workspace size");
        require(bytes >= 0 && static_cast<uint64_t>(bytes) <= kWorkspaceLimit,
                "LFM2 cuDNN SDPA workspace exceeds the 1 GiB resource limit; no alternate plan");
        // Reserve the fixed bundle capacity once, rather than reallocating
        // for every successive token length. The graph still attends exact T.
        packed.grow(packing_bytes);
        workspace.grow(static_cast<std::size_t>(bytes));
        graph = std::move(next);
        current_length = geometry.length;
    }

    void execute(tvm::ffi::TensorView query, tvm::ffi::TensorView key, tvm::ffi::TensorView value,
                 tvm::ffi::TensorView output, Geometry geometry) {
        std::exception_ptr failure;
        try {
            const std::size_t width = static_cast<std::size_t>(geometry.length * 64 * 2);
            const std::size_t pitch = static_cast<std::size_t>(geometry.capacity * 64 * 2);
            auto* packed_value = static_cast<unsigned char*>(packed.data) + 8 * width;
            check(cudaMemcpy2DAsync(packed.data, width, pointer(key), pitch, width, 8,
                                    cudaMemcpyDeviceToDevice, stream),
                  "LFM2 SDPA pack active keys");
            check(cudaMemcpy2DAsync(packed_value, width, pointer(value), pitch, width, 8,
                                    cudaMemcpyDeviceToDevice, stream),
                  "LFM2 SDPA pack active values");
            float scale = 0.125F;
            std::unordered_map<int64_t, void*> pointers{{0, pointer(query)},
                                                        {1, packed.data},
                                                        {2, packed_value},
                                                        {3, pointer(output)},
                                                        {5, &scale}};
            check(graph->execute(handle, pointers, workspace.data), "LFM2 SDPA execute");
        } catch (...) {
            failure = std::current_exception();
        }
        // Cover queued copies even when cuDNN reports an execution error.
        const auto status = cudaEventRecord(completion, stream);
        if (status == cudaSuccess) {
            pending = true;
        } else {
            // This stream is still borrowed by the active call, unlike a
            // future handoff or TLS teardown. Drain before propagating error.
            cudaStreamSynchronize(stream);
            pending = false;
            check(status, "LFM2 SDPA record completion");
        }
        if (failure)
            std::rethrow_exception(failure);
    }
};

State& state(int device) {
    // Device IDs are runtime-owned and finite. No length or stream is a key.
    thread_local std::unordered_map<int, std::unique_ptr<State>> states;
    auto& value = states[device];
    if (!value)
        value = std::make_unique<State>(device);
    return *value;
}

void run(tvm::ffi::TensorView query, tvm::ffi::TensorView key, tvm::ffi::TensorView value,
         tvm::ffi::TensorView active_length_extent, tvm::ffi::TensorView output) {
    int device = -1;
    check(cudaGetDevice(&device), "LFM2 SDPA current device");
    const auto geometry =
        validate_contract(query, key, value, active_length_extent, output, device);
    auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    require(stream != nullptr, "LFM2 SDPA requires the inherited decoder stream");
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream, &capture), "LFM2 SDPA capture status");
    require(capture == cudaStreamCaptureStatusNone,
            "LFM2 cuDNN SDPA does not support CUDA stream capture");
    auto& owner = state(device);
    owner.prepare(geometry, stream);
    owner.execute(query, key, value, output, geometry);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
