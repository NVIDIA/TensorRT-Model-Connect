/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/history_snapshot.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace trtmc::hstu {
namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}
void little_endian_host() {
    const std::uint16_t value = 1;
    require(*reinterpret_cast<const std::uint8_t*>(&value) == 1,
            "snapshot requires little-endian host");
}
} // namespace
void HistoryHash::update(const void* pointer, std::size_t size) {
    const auto* data = static_cast<const std::uint8_t*>(pointer);
    for (std::size_t i = 0; i < size; ++i) {
        first_ = (first_ ^ data[i]) * 1099511628211ull;
        second_ = (second_ ^ data[i]) * 14029467366897019727ull;
    }
}
std::string HistoryHash::finish() const {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << first_ << std::setw(16) << second_;
    return out.str();
}
std::string history_data_key(const void* data, std::size_t size) {
    HistoryHash hash;
    hash.update(data, size);
    return hash.finish();
}
std::vector<std::uint8_t> encode_history_snapshot(HistorySnapshot snapshot) {
    little_endian_host();
    auto& meta = snapshot.metadata;
    meta["format"] = "hstu_history_v2";
    meta["cache_schema"] = 1;
    meta["tensors"] = nlohmann::json::array();
    std::size_t offset = 0;
    for (const auto& tensor : snapshot.tensors) {
        meta["tensors"].push_back(
            {{"name", tensor.name},
             {"dtype", dtype_name(tensor.dtype)},
             {"shape", tensor.shape},
             {"offset", offset},
             {"bytes", tensor.bytes.size()},
             {"checksum", history_data_key(tensor.bytes.data(), tensor.bytes.size())}});
        offset += tensor.bytes.size();
    }
    const auto header = meta.dump();
    require(header.size() <= 65536 && offset <= kHistorySnapshotLimit - header.size() - 16,
            "snapshot exceeds size limit");
    std::vector<std::uint8_t> result{'H', 'S', 'T', 'U', 'K', 'V', '2', 0};
    const auto size = std::uint64_t(header.size());
    for (unsigned i = 0; i < 8; ++i)
        result.push_back(std::uint8_t(size >> (i * 8)));
    result.insert(result.end(), header.begin(), header.end());
    for (const auto& tensor : snapshot.tensors)
        result.insert(result.end(), tensor.bytes.begin(), tensor.bytes.end());
    return result;
}
HistorySnapshot decode_history_snapshot(const std::vector<std::uint8_t>& data,
                                        const std::string& identity, const nlohmann::json& specs,
                                        std::int64_t max_batch) {
    little_endian_host();
    require(data.size() >= 16 && data.size() <= kHistorySnapshotLimit, "invalid snapshot size");
    require(std::memcmp(data.data(), "HSTUKV2\0", 8) == 0, "invalid snapshot magic");
    std::uint64_t size = 0;
    for (unsigned i = 0; i < 8; ++i)
        size |= std::uint64_t(data[8 + i]) << (8 * i);
    require(size > 0 && size <= 65536 && size <= data.size() - 16,
            "invalid snapshot manifest length");
    HistorySnapshot result;
    auto& meta = result.metadata;
    meta = nlohmann::json::parse(data.begin() + 16, data.begin() + 16 + size);
    require(meta.at("format") == "hstu_history_v2" && meta.at("cache_schema") == 1,
            "unsupported history snapshot schema");
    require(meta.at("bundle_id") == identity, "incompatible history snapshot bundle");
    const auto batch = meta.at("batch").get<std::int64_t>();
    require(batch > 0 && batch <= max_batch, "invalid history snapshot batch");
    for (const auto* key : {"revisions", "history_key"}) {
        require(meta.at(key).is_array() && meta.at(key).size() == std::size_t(batch),
                "invalid history revisions");
        for (const auto& value : meta.at(key)) {
            require(value.is_string(), "history revision must be a string");
            const auto text = value.get<std::string>();
            require(!text.empty() && text.size() <= 256, "invalid history revision length");
            if (std::string(key) == "history_key")
                require(text.size() == 32 &&
                            text.find_first_not_of("0123456789abcdef") == std::string::npos,
                        "invalid history feature digest");
        }
    }
    require(meta.at("tensors").is_array() && meta.at("tensors").size() == specs.size(),
            "wrong snapshot tensor count");
    const auto start = std::size_t(16 + size);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const auto& spec = specs.at(i);
        const auto& entry = meta.at("tensors").at(i);
        auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
        shape.at(0) = batch;
        const auto dtype = parse_dtype(spec.at("dtype"));
        const auto bytes = checked_bytes(shape, dtype);
        require(entry.at("name") == spec.at("name") && entry.at("dtype") == spec.at("dtype") &&
                    entry.at("shape") == shape,
                "invalid history tensor schema");
        require(entry.at("offset") == offset && entry.at("bytes") == bytes &&
                    offset <= data.size() - start && bytes <= data.size() - start - offset,
                "invalid history payload range");
        const auto* pointer = data.data() + start + offset;
        require(entry.at("checksum") == history_data_key(pointer, bytes),
                "history snapshot checksum mismatch");
        for (std::size_t j = 0; j < bytes; j += 4) {
            if (dtype == DType::kFloat32) {
                float value;
                std::memcpy(&value, pointer + j, 4);
                require(std::isfinite(value), "nonfinite history cache value");
            } else if (dtype == DType::kInt32) {
                std::int32_t value;
                std::memcpy(&value, pointer + j, 4);
                require(value == 0 || value == 1, "invalid history mask");
            }
        }
        if (dtype == DType::kInt64) {
            for (std::size_t j = 0; j < bytes; j += sizeof(std::int64_t)) {
                std::int64_t value;
                std::memcpy(&value, pointer + j, sizeof(value));
                require(value >= 0 && value <= (std::int64_t{1} << 40),
                        "invalid history query timestamp");
            }
        }
        FeatureTensor tensor{spec.at("name"), dtype, shape, {}};
        tensor.bytes.assign(pointer, pointer + bytes);
        result.tensors.push_back(std::move(tensor));
        offset += bytes;
    }
    require(offset == data.size() - start, "unexpected trailing history payload");
    return result;
}
} // namespace trtmc::hstu
