/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "refit_weights.h"

#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace trtmc::qwen {
namespace {
using json = nlohmann::json;

void require(bool valid, const char* message) {
    if (!valid)
        throw std::runtime_error(std::string("qwen refit: ") + message);
}

std::uint64_t integer(const json& value) {
    require(value.is_number_integer() &&
                (value.is_number_unsigned() || value.get<std::int64_t>() >= 0),
            "expected a nonnegative integer");
    return value.get<std::uint64_t>();
}

DType dtype(const std::string& value) {
    if (value == "BF16")
        return DType::kBFloat16;
    if (value == "F16")
        return DType::kFloat16;
    if (value == "F32")
        return DType::kFloat32;
    throw std::runtime_error("qwen refit: unsupported storage dtype");
}
} // namespace

QwenRefitSource::QwenRefitSource(const BundleReader& reader) {
    const auto* metadata = reader.find_section("refit.json");
    const auto* payload = reader.find_section("refit_weights");
    if (!metadata && !payload)
        return;
    require(metadata && payload, "both metadata and payload are required");
    require(metadata->length <= (16U << 20), "metadata is too large");
    const auto bytes = reader.read_section("refit.json");
    const auto manifest = json::parse(bytes.begin(), bytes.end());
    require(manifest.is_object() && manifest.size() == 3, "invalid manifest fields");
    require(integer(manifest.at("version")) == 1, "unsupported schema version");
    const auto& weights = manifest.at("weights");
    const auto& engines = manifest.at("engines");
    require(weights.is_object() && !weights.empty(), "empty or invalid weights");
    require(engines.is_object() && engines.size() == 2 && engines.contains("engine.plan") &&
                engines.contains("prefill.plan"),
            "expected both split engine bindings");
    storage_ = reader.map_section("refit_weights");
    RefitWeightMap views;
    for (const auto& [name, value] : weights.items()) {
        require(!name.empty() && name.size() < 4096 && name.find('\0') == std::string::npos,
                "invalid weight name");
        require(value.is_object() && value.size() == 4, "invalid tensor fields");
        const auto type = dtype(value.at("dtype").get<std::string>());
        const auto offset = integer(value.at("offset"));
        const auto nbytes = integer(value.at("nbytes"));
        const auto& shape = value.at("shape");
        require(shape.is_array() && !shape.empty(), "invalid tensor shape");
        std::uint64_t count = 1;
        const auto max_count =
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / dtype_size(type);
        for (const auto& dimension : shape) {
            const auto size = integer(dimension);
            require(size > 0 && size <= max_count / count, "tensor shape overflows");
            count *= size;
        }
        require(nbytes == count * dtype_size(type), "shape and byte length disagree");
        require(offset <= storage_.size && nbytes <= storage_.size - offset,
                "tensor exceeds payload bounds");
        views.emplace(name, RefitWeightView{type, storage_.data.get() + offset,
                                            static_cast<std::int64_t>(count)});
    }
    std::set<std::string> used;
    for (const auto& [engine, names] : engines.items()) {
        require(names.is_array() && !names.empty(), "empty or invalid engine bindings");
        auto& bindings = engines_[engine];
        for (const auto& item : names) {
            const auto name = item.get<std::string>();
            const auto value = views.find(name);
            require(value != views.end(), "binding names an absent weight");
            require(bindings.emplace(name, value->second).second, "duplicate engine binding");
            used.insert(name);
        }
    }
    require(used.size() == views.size(), "unreferenced payload weights");
}

const RefitWeightMap* QwenRefitSource::weights_for(const std::string& engine) const {
    if (engines_.empty())
        return nullptr;
    return &engines_.at(engine);
}

} // namespace trtmc::qwen
