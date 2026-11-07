/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "refit_weights.h"

#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace trtmc {

namespace {

using json = nlohmann::json;

std::uint64_t read_u64_le(const char* bytes) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    return value;
}

DType dtype_from_safetensors(const std::string& name) {
    if (name == "F16")
        return DType::kFloat16;
    if (name == "F32")
        return DType::kFloat32;
    if (name == "BF16")
        return DType::kBFloat16;
    throw std::runtime_error("[trtmc] refit_weights: unsupported dtype " + name);
}

} // namespace

QwenRefitSource parse_qwen_refit_weights(const std::vector<char>& bytes) {
    QwenRefitSource source;
    RefitWeightMap& result = source.weights_;
    if (bytes.empty())
        return source; // baked bundle -- nothing to refit
    if (bytes.size() < sizeof(std::uint64_t))
        throw std::runtime_error("[trtmc] refit_weights: section is too small");

    const std::uint64_t header_size = read_u64_le(bytes.data());
    if (header_size > bytes.size() - sizeof(std::uint64_t))
        throw std::runtime_error("[trtmc] refit_weights: truncated safetensors header");
    const std::size_t data_begin = sizeof(std::uint64_t) + static_cast<std::size_t>(header_size);

    json header;
    try {
        const char* begin = bytes.data() + sizeof(std::uint64_t);
        header = json::parse(begin, begin + header_size);
    } catch (const json::exception& exc) {
        throw std::runtime_error(std::string("[trtmc] refit_weights: invalid header: ") +
                                 exc.what());
    }
    if (!header.is_object())
        throw std::runtime_error("[trtmc] refit_weights: header must be an object");

    const std::size_t payload = bytes.size() - data_begin;
    for (auto item = header.begin(); item != header.end(); ++item) {
        if (item.key() == "__metadata__")
            continue;
        const json& entry = item.value();
        if (!entry.is_object() || !entry.contains("dtype") || !entry.contains("shape") ||
            !entry.contains("data_offsets")) {
            throw std::runtime_error("[trtmc] refit_weights: malformed entry for " + item.key());
        }

        const DType dtype = dtype_from_safetensors(entry["dtype"].get<std::string>());
        const auto& offsets = entry["data_offsets"];
        if (!offsets.is_array() || offsets.size() != 2)
            throw std::runtime_error("[trtmc] refit_weights: bad data_offsets for " + item.key());
        const auto begin_off = offsets[0].get<std::uint64_t>();
        const auto end_off = offsets[1].get<std::uint64_t>();
        if (end_off < begin_off || end_off > payload)
            throw std::runtime_error("[trtmc] refit_weights: entry " + item.key() +
                                     " runs past the end of the section");

        std::int64_t count = 1;
        for (const auto& dim : entry["shape"])
            count *= dim.get<std::int64_t>();

        const std::size_t span = static_cast<std::size_t>(end_off - begin_off);
        const std::size_t element = dtype_size(dtype);
        if (element == 0 || span != static_cast<std::size_t>(count) * element) {
            throw std::runtime_error("[trtmc] refit_weights: entry " + item.key() +
                                     " byte span does not match shape x dtype");
        }

        RefitWeightView view;
        view.dtype = dtype;
        view.data = bytes.data() + data_begin + begin_off;
        view.count = count;
        result.emplace(item.key(), view);
    }
    return source;
}

QwenRefitSource::~QwenRefitSource() {
    for (const auto& m : mappings_)
        if (m.base != nullptr && m.base != MAP_FAILED)
            ::munmap(m.base, m.size);
}

QwenRefitSource::QwenRefitSource(QwenRefitSource&& other) noexcept
    : mappings_(std::move(other.mappings_)), weights_(std::move(other.weights_)) {
    other.mappings_.clear();
    other.weights_.clear();
}

QwenRefitSource& QwenRefitSource::operator=(QwenRefitSource&& other) noexcept {
    if (this != &other) {
        this->~QwenRefitSource();
        mappings_ = std::move(other.mappings_);
        weights_ = std::move(other.weights_);
        other.mappings_.clear();
        other.weights_.clear();
    }
    return *this;
}

std::size_t QwenRefitSource::mapped_bytes() const {
    std::size_t total = 0;
    for (const auto& m : mappings_)
        total += m.size;
    return total;
}

QwenRefitSource load_qwen_refit_from_manifest(const std::string& manifest_json) {
    namespace fs = std::filesystem;
    QwenRefitSource source;

    json manifest;
    try {
        manifest = json::parse(manifest_json);
    } catch (const json::exception& exc) {
        throw std::runtime_error(std::string("[trtmc] refit_manifest: invalid JSON: ") +
                                 exc.what());
    }
    if (!manifest.is_object() || !manifest.contains("weights") || !manifest.contains("files"))
        throw std::runtime_error("[trtmc] refit_manifest: missing 'weights' or 'files'");

    // The recorded directory is where the checkpoint lived at build time; an
    // env override lets a relocated bundle name its own copy.
    std::string dir = manifest.value("checkpoint_dir", std::string{});
    if (const char* override_dir = std::getenv(kQwenRefitCheckpointEnv);
        override_dir != nullptr && *override_dir != '\0') {
        dir = override_dir;
    }
    if (dir.empty() || !fs::is_directory(dir)) {
        throw std::runtime_error(
            "[trtmc] refit_manifest: checkpoint directory '" + dir +
            "' not found; set " + kQwenRefitCheckpointEnv + " to its location");
    }

    // Map each shard once, failing closed on any size mismatch: a truncated or
    // substituted checkpoint must not be refit as if it were correct.
    std::map<std::string, std::pair<const char*, std::size_t>> mapped;
    for (auto item = manifest["files"].begin(); item != manifest["files"].end(); ++item) {
        const fs::path path = fs::path(dir) / item.key();
        const auto expected = item.value().value("nbytes", std::uint64_t{0});

        struct stat st {};
        if (::stat(path.c_str(), &st) != 0)
            throw std::runtime_error("[trtmc] refit_manifest: cannot stat " + path.string());
        if (expected != 0 && static_cast<std::uint64_t>(st.st_size) != expected) {
            throw std::runtime_error(
                "[trtmc] refit_manifest: " + path.string() + " is " +
                std::to_string(st.st_size) + " bytes, manifest expects " +
                std::to_string(expected) + " -- checkpoint does not match the bundle");
        }

        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            throw std::runtime_error("[trtmc] refit_manifest: cannot open " + path.string());
        void* base = ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ,
                            MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED)
            throw std::runtime_error("[trtmc] refit_manifest: cannot map " + path.string());
        source.mappings_.push_back({base, static_cast<std::size_t>(st.st_size)});
        mapped.emplace(item.key(),
                       std::make_pair(static_cast<const char*>(base),
                                      static_cast<std::size_t>(st.st_size)));
    }

    for (auto item = manifest["weights"].begin(); item != manifest["weights"].end(); ++item) {
        const json& entry = item.value();
        const auto file = entry.value("file", std::string{});
        const auto offset = entry.value("offset", std::uint64_t{0});
        const auto nbytes = entry.value("nbytes", std::uint64_t{0});
        const auto it = mapped.find(file);
        if (it == mapped.end())
            throw std::runtime_error("[trtmc] refit_manifest: entry " + item.key() +
                                     " names unmapped file " + file);
        if (offset > it->second.second || nbytes > it->second.second - offset)
            throw std::runtime_error("[trtmc] refit_manifest: entry " + item.key() +
                                     " runs past the end of " + file);

        const DType dtype = dtype_from_safetensors(entry.value("dtype", std::string{}));
        const std::size_t element = dtype_size(dtype);
        if (element == 0 || nbytes % element != 0)
            throw std::runtime_error("[trtmc] refit_manifest: entry " + item.key() +
                                     " byte span is not a whole number of elements");

        RefitWeightView view;
        view.dtype = dtype;
        view.data = it->second.first + offset;
        view.count = static_cast<std::int64_t>(nbytes / element);
        source.weights_.emplace(item.key(), view);
    }
    return source;
}

} // namespace trtmc
