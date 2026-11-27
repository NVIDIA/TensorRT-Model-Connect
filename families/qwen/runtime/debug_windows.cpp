/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/debug_windows.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

// Layer index of a debug_hidden_<i> output, or -1 for any other tensor name.
int32_t debug_hidden_layer(const std::string& name) {
    static const std::string prefix = "debug_hidden_";
    if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0)
        return -1;
    const std::string digits = name.substr(prefix.size());
    if (digits.size() > 9 || !std::all_of(digits.begin(), digits.end(),
                                          [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
        return -1;
    }
    return std::stoi(digits);
}

} // namespace

void QwenDebugWindows::bind(ITrtModule& module, int64_t min_rows) {
    windows_.clear();
    stream_ = module.stream();
    for (const auto& info : module.output_info()) {
        const int32_t layer = debug_hidden_layer(info.name);
        if (layer < 0)
            continue;
        const auto shape = module.tensor_shape(info.name);
        if (module.tensor_dtype(info.name) != DType::kFloat32 || shape.size() != 2 ||
            shape[0] <= 0 || shape[0] < min_rows || shape[1] <= 0) {
            throw std::runtime_error("Qwen debug output '" + info.name +
                                     "' must be float32 [rows, hidden] with at least " +
                                     std::to_string(min_rows) + " rows");
        }
        windows_.push_back(Window{info.name, layer, shape[1],
                                  DeviceTensor::zeros(shape, DType::kFloat32, stream_)});
        void* const buffer = windows_.back().buffer.data();
        if (buffer == nullptr)
            throw std::runtime_error("Qwen debug output allocation failed for '" + info.name + "'");
        module.bind_external(info.name, buffer);
        if (module.device_ptr(info.name) != buffer)
            throw std::runtime_error("Qwen engine did not bind debug output '" + info.name + "'");
    }
    std::sort(windows_.begin(), windows_.end(),
              [](const Window& a, const Window& b) { return a.layer < b.layer; });
}

void QwenDebugWindows::append_rows(std::size_t index, int64_t rows, std::vector<float>& out) const {
    const auto& window = windows_.at(index);
    if (rows < 0 || rows > window.buffer.shape().front())
        throw std::out_of_range("Qwen debug output '" + window.name + "' row count out of range");
    const auto count = static_cast<std::size_t>(rows * window.hidden);
    const auto offset = out.size();
    out.resize(offset + count);
    if (count == 0)
        return;
    if (cudaMemcpyAsync(out.data() + offset, window.buffer.data(), count * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
        cudaStreamSynchronize(stream_) != cudaSuccess) {
        throw std::runtime_error("Qwen debug output read failed for '" + window.name + "'");
    }
}

} // namespace trtmc
