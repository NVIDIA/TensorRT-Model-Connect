/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "decoder_layer.h"

#include <cstring>
#include <stdexcept>

namespace trtmc::qwen3_8_flash_next {
namespace {
SavedTensor save(const Tensor& tensor, const Tensor& expected) {
    if (!tensor.data || tensor.shape != expected.shape || tensor.dtype != expected.dtype)
        throw std::runtime_error("Unexpected Flash-Next decoder output shape or dtype");
    SavedTensor out{std::vector<char>(tensor.nbytes()), tensor.shape, tensor.dtype};
    std::memcpy(out.bytes.data(), tensor.data, out.bytes.size());
    return out;
}
} // namespace

LayerOutput run_decoder_layer(ITrtModule& layer, ITrtModule& expert, ITrtModule& combine,
                              const TensorMap& inputs,
                              const std::function<std::vector<char>(std::int32_t)>& load_expert) {
    const auto& hyper = inputs.at("hyper");
    const auto routed = layer.forward(inputs);
    auto answers = run_experts(expert, routed.at("tokens"), routed.at("selected"), load_expert);
    const auto rows = routed.at("tokens").shape.at(0);
    const TensorMap parts{{"skip", routed.at("skip")},
                          {"shared", routed.at("shared")},
                          {"inject", routed.at("inject")},
                          {"weights", routed.at("weights")},
                          {"experts", {answers.data(), {rows, 10, 2560}, DType::kFloat16}}};
    const auto mixed = combine.forward(parts);
    LayerOutput next{save(mixed.at("hyper_out"), hyper), {}};
    for (const std::string name :
         {"history", "state", "key", "value", "index_key", "lexical_history"}) {
        const auto in = inputs.find(name);
        if (in != inputs.end())
            next.state.emplace(name, save(routed.at("new_" + name), in->second));
    }
    return next;
}
} // namespace trtmc::qwen3_8_flash_next
