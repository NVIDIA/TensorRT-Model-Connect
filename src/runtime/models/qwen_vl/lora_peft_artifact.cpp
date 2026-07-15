/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen_vl/lora_peft_artifact.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace trtmc {
namespace qwen_vl {

namespace {

using json = nlohmann::json;

std::vector<uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("PEFT LoRA: cannot open " + path.string());
    const auto end = input.tellg();
    if (end <= 0)
        throw std::runtime_error("PEFT LoRA: empty file " + path.string());
    std::vector<uint8_t> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input)
        throw std::runtime_error("PEFT LoRA: failed to read " + path.string());
    return bytes;
}

json read_json_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("PEFT LoRA: cannot open " + path.string());
    try {
        return json::parse(input);
    } catch (const json::exception& exc) {
        throw std::runtime_error("PEFT LoRA: invalid JSON in " + path.string() + ": " + exc.what());
    }
}

std::size_t checked_numel(const std::vector<int64_t>& shape, const std::string& label) {
    if (shape.empty())
        throw std::runtime_error("PEFT LoRA: " + label + " has an empty shape");
    std::size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0)
            throw std::runtime_error("PEFT LoRA: " + label + " has a non-positive shape");
        const auto unsigned_dim = static_cast<std::size_t>(dim);
        if (count > std::numeric_limits<std::size_t>::max() / unsigned_dim)
            throw std::runtime_error("PEFT LoRA: " + label + " shape overflows size_t");
        count *= unsigned_dim;
    }
    return count;
}

std::size_t dtype_size(PeftTensorDType dtype) {
    return dtype == PeftTensorDType::kFloat32 ? sizeof(float) : sizeof(uint16_t);
}

PeftTensorDType parse_dtype(const std::string& dtype) {
    if (dtype == "F16")
        return PeftTensorDType::kFloat16;
    if (dtype == "BF16")
        return PeftTensorDType::kBFloat16;
    if (dtype == "F32")
        return PeftTensorDType::kFloat32;
    throw std::runtime_error("PEFT LoRA: unsupported safetensors dtype " + dtype);
}

uint64_t read_u64_le(const uint8_t* data) {
    uint64_t value = 0;
    for (int index = 7; index >= 0; --index)
        value = (value << 8U) | data[index];
    return value;
}

uint16_t load_u16(const uint8_t* data) {
    uint16_t value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

float fp16_to_fp32(uint16_t bits) {
    const bool negative = (bits & 0x8000U) != 0;
    const uint32_t exponent = (bits >> 10U) & 0x1FU;
    const uint32_t mantissa = bits & 0x3FFU;
    float value = 0.0F;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31U) {
        value = mantissa == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F,
                           static_cast<int>(exponent) - 15);
    }
    return negative ? -value : value;
}

float bf16_to_fp32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float output;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

bool config_flag(const json& config, const char* key) {
    return config.contains(key) && !config.at(key).is_null() && config.at(key).get<bool>();
}

bool config_nonempty(const json& config, const char* key) {
    return config.contains(key) && !config.at(key).is_null() && !config.at(key).empty();
}

PeftLoraConfig parse_config(const json& config) {
    if (!config.is_object())
        throw std::runtime_error("PEFT LoRA: adapter_config.json must contain an object");
    try {
        if (config.value("peft_type", std::string("LORA")) != "LORA")
            throw std::runtime_error("PEFT LoRA: adapter peft_type must be LORA");
        if (config_flag(config, "use_dora") || config_flag(config, "use_rslora") ||
            config_flag(config, "use_qalora"))
            throw std::runtime_error("PEFT LoRA: DoRA, rsLoRA, and QALoRA are not supported");
        if (config.value("fan_in_fan_out", false))
            throw std::runtime_error("PEFT LoRA: fan_in_fan_out is not supported");
        if (config.value("bias", std::string("none")) != "none" || config.value("lora_bias", false))
            throw std::runtime_error("PEFT LoRA: adapted bias tensors are not supported");
        if (config_nonempty(config, "modules_to_save") || config_nonempty(config, "rank_pattern") ||
            config_nonempty(config, "alpha_pattern"))
            throw std::runtime_error(
                "PEFT LoRA: modules_to_save and per-module rank/alpha patterns are not supported");

        PeftLoraConfig parsed;
        parsed.rank = config.value("r", 0);
        if (parsed.rank <= 0)
            throw std::runtime_error("PEFT LoRA: adapter rank must be positive");
        parsed.alpha = config.value("lora_alpha", static_cast<double>(parsed.rank));

        if (!config.contains("target_modules") || config.at("target_modules").is_null()) {
            return parsed;
        }
        const auto& targets = config.at("target_modules");
        if (targets.is_string()) {
            parsed.target_modules.push_back(targets.get<std::string>());
        } else if (targets.is_array()) {
            for (const auto& target : targets)
                parsed.target_modules.push_back(target.get<std::string>());
        } else {
            throw std::runtime_error("PEFT LoRA: target_modules must be a string or array");
        }
        std::unordered_set<std::string> unique;
        std::vector<std::string> deduplicated;
        deduplicated.reserve(parsed.target_modules.size());
        for (auto& target : parsed.target_modules) {
            if (target.empty())
                throw std::runtime_error("PEFT LoRA: target_modules contains an empty name");
            if (unique.insert(target).second)
                deduplicated.push_back(std::move(target));
        }
        parsed.target_modules = std::move(deduplicated);
        return parsed;
    } catch (const json::exception& exc) {
        throw std::runtime_error(std::string("PEFT LoRA: invalid adapter config: ") + exc.what());
    }
}

} // namespace

struct PeftLoraArtifact::Impl {
    struct Entry {
        std::size_t info_index{0};
        std::size_t begin{0};
        std::size_t end{0};
    };

    PeftLoraConfig config;
    std::vector<uint8_t> bytes;
    std::size_t data_begin{0};
    std::vector<PeftTensorInfo> tensor_infos;
    std::unordered_map<std::string, Entry> entries;
};

PeftLoraArtifact::PeftLoraArtifact(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

PeftLoraArtifact::~PeftLoraArtifact() = default;
PeftLoraArtifact::PeftLoraArtifact(PeftLoraArtifact&&) noexcept = default;
PeftLoraArtifact& PeftLoraArtifact::operator=(PeftLoraArtifact&&) noexcept = default;

PeftLoraArtifact PeftLoraArtifact::load(const std::string& adapter_dir) {
    namespace fs = std::filesystem;
    const fs::path root(adapter_dir);
    if (!fs::is_directory(root))
        throw std::runtime_error("PEFT LoRA: adapter path is not a directory: " + adapter_dir);

    auto impl = std::make_unique<Impl>();
    impl->config = parse_config(read_json_file(root / "adapter_config.json"));
    impl->bytes = read_binary_file(root / "adapter_model.safetensors");
    if (impl->bytes.size() < sizeof(uint64_t))
        throw std::runtime_error("PEFT LoRA: safetensors file is too small");

    const uint64_t header_size_u64 = read_u64_le(impl->bytes.data());
    if (header_size_u64 > std::numeric_limits<std::size_t>::max() - sizeof(uint64_t))
        throw std::runtime_error("PEFT LoRA: safetensors header size overflows");
    const auto header_size = static_cast<std::size_t>(header_size_u64);
    impl->data_begin = sizeof(uint64_t) + header_size;
    if (impl->data_begin > impl->bytes.size())
        throw std::runtime_error("PEFT LoRA: truncated safetensors header");

    json header;
    try {
        const auto* begin = reinterpret_cast<const char*>(impl->bytes.data() + sizeof(uint64_t));
        header = json::parse(begin, begin + header_size);
    } catch (const json::exception& exc) {
        throw std::runtime_error(std::string("PEFT LoRA: invalid safetensors header: ") +
                                 exc.what());
    }
    if (!header.is_object())
        throw std::runtime_error("PEFT LoRA: safetensors header must be an object");

    impl->tensor_infos.reserve(header.size());
    try {
        for (auto item = header.begin(); item != header.end(); ++item) {
            if (item.key() == "__metadata__")
                continue;
            const auto& value = item.value();
            if (!value.is_object() || !value.contains("dtype") || !value.contains("shape") ||
                !value.contains("data_offsets"))
                throw std::runtime_error("PEFT LoRA: malformed tensor header for " + item.key());

            PeftTensorInfo info;
            info.name = item.key();
            info.dtype = parse_dtype(value.at("dtype").get<std::string>());
            info.shape = value.at("shape").get<std::vector<int64_t>>();
            info.element_count = checked_numel(info.shape, info.name);
            const auto offsets = value.at("data_offsets").get<std::vector<std::size_t>>();
            if (offsets.size() != 2 || offsets[0] > offsets[1] ||
                offsets[1] > impl->bytes.size() - impl->data_begin)
                throw std::runtime_error("PEFT LoRA: invalid data offsets for " + info.name);
            if (info.element_count >
                std::numeric_limits<std::size_t>::max() / dtype_size(info.dtype))
                throw std::runtime_error("PEFT LoRA: byte size overflows for " + info.name);
            const std::size_t expected_bytes = info.element_count * dtype_size(info.dtype);
            if (offsets[1] - offsets[0] != expected_bytes)
                throw std::runtime_error("PEFT LoRA: byte size mismatch for " + info.name);

            const std::size_t info_index = impl->tensor_infos.size();
            impl->tensor_infos.push_back(std::move(info));
            impl->entries.emplace(item.key(), Impl::Entry{info_index, offsets[0], offsets[1]});
        }
    } catch (const json::exception& exc) {
        throw std::runtime_error(std::string("PEFT LoRA: malformed safetensors header: ") +
                                 exc.what());
    }
    return PeftLoraArtifact(std::move(impl));
}

const PeftLoraConfig& PeftLoraArtifact::config() const {
    return impl_->config;
}

const std::vector<PeftTensorInfo>& PeftLoraArtifact::tensors() const {
    return impl_->tensor_infos;
}

float PeftLoraArtifact::read_float(const std::string& tensor_name, std::size_t index) const {
    const auto found = impl_->entries.find(tensor_name);
    if (found == impl_->entries.end())
        throw std::invalid_argument("PEFT LoRA: unknown tensor " + tensor_name);
    const auto& info = impl_->tensor_infos[found->second.info_index];
    if (index >= info.element_count)
        throw std::out_of_range("PEFT LoRA: tensor index is out of range for " + tensor_name);
    const uint8_t* data = impl_->bytes.data() + impl_->data_begin + found->second.begin;
    if (info.dtype == PeftTensorDType::kFloat32) {
        float value;
        std::memcpy(&value, data + index * sizeof(float), sizeof(value));
        return value;
    }
    const uint16_t value = load_u16(data + index * sizeof(uint16_t));
    return info.dtype == PeftTensorDType::kFloat16 ? fp16_to_fp32(value) : bf16_to_fp32(value);
}

} // namespace qwen_vl
} // namespace trtmc
