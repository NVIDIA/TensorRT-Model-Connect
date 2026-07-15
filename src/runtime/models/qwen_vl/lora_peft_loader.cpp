/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen_vl/lora_peft_loader.h"

#include "runtime/models/qwen_vl/lora_peft_artifact.h"

#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace trtmc {
namespace {

using qwen_vl::PeftLoraConfig;
using qwen_vl::PeftTensorInfo;

struct PeftPair {
    const PeftTensorInfo* a{nullptr};
    const PeftTensorInfo* b{nullptr};
};

constexpr const char* kSupportedModules[] = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
};

const std::unordered_map<std::string, std::string> kRuntimeWeightNames = {
    {"q_proj", "w_q"},       {"k_proj", "w_k"},   {"v_proj", "w_v"},       {"o_proj", "w_o"},
    {"gate_proj", "w_gate"}, {"up_proj", "w_up"}, {"down_proj", "w_down"},
};

std::size_t checked_numel(const std::vector<int64_t>& shape, const std::string& label) {
    if (shape.empty())
        throw std::runtime_error("Qwen-VL LoRA: " + label + " has an empty shape");
    std::size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0)
            throw std::runtime_error("Qwen-VL LoRA: " + label + " has a non-positive shape");
        const auto unsigned_dim = static_cast<std::size_t>(dim);
        if (count > std::numeric_limits<std::size_t>::max() / unsigned_dim)
            throw std::runtime_error("Qwen-VL LoRA: " + label + " shape overflows size_t");
        count *= unsigned_dim;
    }
    return count;
}

uint32_t round_shift_right(uint32_t value, int shift) {
    if (shift <= 0)
        return value;
    const uint32_t base = value >> shift;
    const uint32_t mask = (uint32_t{1} << shift) - 1U;
    const uint32_t remainder = value & mask;
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    return base +
           static_cast<uint32_t>(remainder > halfway || (remainder == halfway && (base & 1U) != 0));
}

uint16_t fp32_to_fp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
    const uint32_t absolute = bits & 0x7FFFFFFFU;
    const uint32_t source_exp = (absolute >> 23U) & 0xFFU;
    const uint32_t mantissa = absolute & 0x7FFFFFU;
    if (source_exp == 0xFFU)
        return static_cast<uint16_t>(sign | (mantissa == 0 ? 0x7C00U : 0x7E00U));
    if (source_exp == 0)
        return sign;

    const int exponent = static_cast<int>(source_exp) - 127;
    if (exponent > 15)
        return static_cast<uint16_t>(sign | 0x7C00U);
    if (exponent < -24)
        return sign;
    if (exponent < -14) {
        const int shift = 13 + (-14 - exponent);
        const uint32_t rounded = round_shift_right(mantissa | 0x800000U, shift);
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t half_exp = static_cast<uint32_t>(exponent + 15);
    uint32_t half_mantissa = round_shift_right(mantissa, 13);
    if (half_mantissa == 0x400U) {
        half_mantissa = 0;
        ++half_exp;
        if (half_exp >= 31U)
            return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (half_exp << 10U) | half_mantissa);
}

uint16_t fp32_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

void write_scalar(std::vector<uint8_t>& bytes, DType dtype, std::size_t index, float value) {
    if (dtype == DType::kFloat32) {
        std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(value));
        return;
    }
    uint16_t converted = 0;
    if (dtype == DType::kFloat16)
        converted = fp32_to_fp16(value);
    else if (dtype == DType::kBFloat16)
        converted = fp32_to_bf16(value);
    else
        throw std::runtime_error("Qwen-VL LoRA: engine LoRA inputs must be floating point");
    std::memcpy(bytes.data() + index * sizeof(uint16_t), &converted, sizeof(converted));
}

std::set<std::string> parse_target_modules(const PeftLoraConfig& config) {
    const std::set<std::string> supported(std::begin(kSupportedModules),
                                          std::end(kSupportedModules));
    std::set<std::string> targets;
    if (config.target_modules.empty()) {
        targets = supported;
    } else {
        targets.insert(config.target_modules.begin(), config.target_modules.end());
    }
    for (const auto& target : targets) {
        if (supported.find(target) == supported.end())
            throw std::runtime_error("Qwen-VL LoRA: unsupported target module " + target);
    }
    return targets;
}

std::string runtime_input_name(char side, int layer, const std::string& module) {
    return std::string("lora_") + static_cast<char>(side == 'A' ? 'a' : 'b') + "_layer_" +
           std::to_string(layer) + "_" + kRuntimeWeightNames.at(module);
}

QwenVlHostLoraAdapter::Buffer make_buffer(const TensorInfo& info) {
    QwenVlHostLoraAdapter::Buffer output;
    output.shape = info.shape;
    output.dtype = info.dtype;
    output.bytes.resize(checked_numel(info.shape, info.name) * dtype_size(info.dtype), uint8_t{0});
    return output;
}

} // namespace

TensorMap QwenVlHostLoraAdapter::tensor_views() {
    TensorMap views;
    views.reserve(buffers.size());
    for (auto& [name, buffer] : buffers)
        views.emplace(name, Tensor{buffer.bytes.data(), buffer.shape, buffer.dtype});
    return views;
}

QwenVlHostLoraAdapter qwen_vl_load_peft_lora_adapter(const std::string& adapter_dir,
                                                     const std::vector<TensorInfo>& engine_inputs) {
    const auto artifact = qwen_vl::PeftLoraArtifact::load(adapter_dir);
    const int rank = artifact.config().rank;
    const float scale = static_cast<float>(artifact.config().scale());
    const auto targets = parse_target_modules(artifact.config());

    std::unordered_map<std::string, TensorInfo> expected;
    expected.reserve(engine_inputs.size());
    for (const auto& info : engine_inputs)
        expected.emplace(info.name, info);
    if (expected.empty())
        throw std::runtime_error("Qwen-VL LoRA: engine has no dynamic LoRA inputs");

    const std::regex weight_pattern(
        R"(.*layers\.([0-9]+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_([AB])(?:\.[^.]+)?\.weight$)");
    std::map<std::pair<int, std::string>, PeftPair> pairs;
    for (const auto& tensor : artifact.tensors()) {
        std::smatch match;
        if (!std::regex_match(tensor.name, match, weight_pattern))
            continue;
        const std::string module = match[3].str();
        if (targets.find(module) == targets.end())
            continue;
        auto& pair = pairs[{std::stoi(match[1].str()), module}];
        if (match[4].str() == "A")
            pair.a = &tensor;
        else
            pair.b = &tensor;
    }
    if (pairs.empty())
        throw std::runtime_error("Qwen-VL LoRA: no supported decoder LoRA tensors were found");

    QwenVlHostLoraAdapter adapter;
    for (const auto& [key, pair] : pairs) {
        const int layer = key.first;
        const std::string& module = key.second;
        if (pair.a == nullptr || pair.b == nullptr)
            throw std::runtime_error("Qwen-VL LoRA: incomplete A/B pair for layer " +
                                     std::to_string(layer) + " " + module);
        if (pair.a->shape.size() != 2 || pair.b->shape.size() != 2 || pair.a->shape[0] != rank ||
            pair.b->shape[1] != rank)
            throw std::runtime_error("Qwen-VL LoRA: rank or shape mismatch for layer " +
                                     std::to_string(layer) + " " + module);

        const std::string a_name = runtime_input_name('A', layer, module);
        const std::string b_name = runtime_input_name('B', layer, module);
        const auto a_it = expected.find(a_name);
        const auto b_it = expected.find(b_name);
        if (a_it == expected.end() || b_it == expected.end())
            throw std::runtime_error("Qwen-VL LoRA: adapter targets " + module + " at layer " +
                                     std::to_string(layer) +
                                     " but the engine was not built with matching inputs");
        const TensorInfo& a_info = a_it->second;
        const TensorInfo& b_info = b_it->second;
        if (a_info.shape.size() != 2 || b_info.shape.size() != 2 ||
            a_info.shape[0] != pair.a->shape[1] || b_info.shape[1] != pair.b->shape[0] ||
            a_info.shape[1] != b_info.shape[0] || rank > a_info.shape[1])
            throw std::runtime_error(
                "Qwen-VL LoRA: adapter dimensions do not match engine inputs for " + module +
                " at layer " + std::to_string(layer));
        if (a_info.dtype != b_info.dtype)
            throw std::runtime_error("Qwen-VL LoRA: engine A/B input dtypes differ for " + module);

        auto a_output = make_buffer(a_info);
        auto b_output = make_buffer(b_info);
        const std::size_t input_size = static_cast<std::size_t>(pair.a->shape[1]);
        const std::size_t output_size = static_cast<std::size_t>(pair.b->shape[0]);
        const std::size_t max_rank = static_cast<std::size_t>(a_info.shape[1]);
        for (std::size_t input = 0; input < input_size; ++input) {
            for (int r = 0; r < rank; ++r) {
                const float value = artifact.read_float(
                    pair.a->name, static_cast<std::size_t>(r) * input_size + input);
                write_scalar(a_output.bytes, a_output.dtype, input * max_rank + r, value);
            }
        }
        for (int r = 0; r < rank; ++r) {
            for (std::size_t output = 0; output < output_size; ++output) {
                const float value =
                    artifact.read_float(pair.b->name, output * static_cast<std::size_t>(rank) +
                                                          static_cast<std::size_t>(r));
                write_scalar(b_output.bytes, b_output.dtype,
                             static_cast<std::size_t>(r) * output_size + output, value * scale);
            }
        }
        adapter.buffers.emplace(a_name, std::move(a_output));
        adapter.buffers.emplace(b_name, std::move(b_output));
    }
    return adapter;
}

} // namespace trtmc
