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
constexpr std::array<std::uint32_t, 64> constants{
    {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
     0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
     0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
     0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
     0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
     0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
     0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
     0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
     0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
     0xc67178f2}};
std::uint32_t rotr(std::uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}
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
void HistoryHash::block(const std::uint8_t* data) {
    std::array<std::uint32_t, 64> words{};
    for (unsigned i = 0; i < 16; ++i)
        words[i] = (std::uint32_t(data[i * 4]) << 24) | (std::uint32_t(data[i * 4 + 1]) << 16) |
                   (std::uint32_t(data[i * 4 + 2]) << 8) | data[i * 4 + 3];
    for (unsigned i = 16; i < 64; ++i) {
        const auto a = words[i - 15], b = words[i - 2];
        words[i] = words[i - 16] + (rotr(a, 7) ^ rotr(a, 18) ^ (a >> 3)) + words[i - 7] +
                   (rotr(b, 17) ^ rotr(b, 19) ^ (b >> 10));
    }
    auto a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    auto e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (unsigned i = 0; i < 64; ++i) {
        const auto t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ ((~e) & g)) +
                        constants[i] + words[i];
        const auto t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}
void HistoryHash::update(const void* pointer, std::size_t size) {
    const auto* data = static_cast<const std::uint8_t*>(pointer);
    bytes_ += size;
    while (size) {
        const auto n = std::min(size, buffer_.size() - count_);
        std::memcpy(buffer_.data() + count_, data, n);
        count_ += n;
        data += n;
        size -= n;
        if (count_ == buffer_.size()) {
            block(buffer_.data());
            count_ = 0;
        }
    }
}
std::string HistoryHash::finish() {
    const auto bits = bytes_ * 8;
    const std::uint8_t one = 0x80, zero = 0;
    update(&one, 1);
    while (count_ != 56)
        update(&zero, 1);
    std::array<std::uint8_t, 8> length{};
    for (unsigned i = 0; i < 8; ++i)
        length[7 - i] = std::uint8_t(bits >> (i * 8));
    update(length.data(), length.size());
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (auto word : state_)
        result << std::setw(8) << word;
    return result.str();
}
std::string history_sha256(const void* data, std::size_t size) {
    HistoryHash hash;
    hash.update(data, size);
    return hash.finish();
}
std::string history_file_sha256(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open history identity source");
    HistoryHash hash;
    std::array<char, 65536> buffer{};
    while (file) {
        file.read(buffer.data(), buffer.size());
        hash.update(buffer.data(), std::size_t(file.gcount()));
    }
    if (!file.eof())
        throw std::runtime_error("cannot read history identity source");
    return hash.finish();
}
std::vector<std::uint8_t> encode_history_snapshot(HistorySnapshot snapshot) {
    little_endian_host();
    auto& meta = snapshot.metadata;
    meta["format"] = "hstu_history_v1";
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
             {"sha256", history_sha256(tensor.bytes.data(), tensor.bytes.size())}});
        offset += tensor.bytes.size();
    }
    const auto header = meta.dump();
    require(header.size() <= 65536 && offset <= kHistorySnapshotLimit - header.size() - 16,
            "snapshot exceeds size limit");
    std::vector<std::uint8_t> result{'H', 'S', 'T', 'U', 'K', 'V', '1', 0};
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
    require(std::memcmp(data.data(), "HSTUKV1\0", 8) == 0, "invalid snapshot magic");
    std::uint64_t size = 0;
    for (unsigned i = 0; i < 8; ++i)
        size |= std::uint64_t(data[8 + i]) << (8 * i);
    require(size > 0 && size <= 65536 && size <= data.size() - 16,
            "invalid snapshot manifest length");
    HistorySnapshot result;
    auto& meta = result.metadata;
    meta = nlohmann::json::parse(data.begin() + 16, data.begin() + 16 + size);
    require(meta.at("format") == "hstu_history_v1" && meta.at("cache_schema") == 1,
            "unsupported history snapshot schema");
    require(meta.at("bundle_sha256") == identity, "incompatible history snapshot bundle");
    const auto batch = meta.at("batch").get<std::int64_t>();
    require(batch > 0 && batch <= max_batch, "invalid history snapshot batch");
    for (const auto* key : {"revisions", "history_sha256"}) {
        require(meta.at(key).is_array() && meta.at(key).size() == std::size_t(batch),
                "invalid history revisions");
        for (const auto& value : meta.at(key)) {
            require(value.is_string(), "history revision must be a string");
            const auto text = value.get<std::string>();
            require(!text.empty() && text.size() <= 256, "invalid history revision length");
            if (std::string(key) == "history_sha256")
                require(text.size() == 64 &&
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
        require(entry.at("sha256") == history_sha256(pointer, bytes),
                "history snapshot checksum mismatch");
        for (std::size_t j = 0; j < bytes; j += 4) {
            if (dtype == DType::kFloat32) {
                float value;
                std::memcpy(&value, pointer + j, 4);
                require(std::isfinite(value), "nonfinite history cache value");
            } else {
                std::int32_t value;
                std::memcpy(&value, pointer + j, 4);
                require(value == 0 || value == 1, "invalid history mask");
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
