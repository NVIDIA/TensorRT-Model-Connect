/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/bundle/bundle_format.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace trtmc {

namespace {

using BundleSectionLocation = std::pair<std::uint64_t, std::uint64_t>;
using BundleSectionEntry = std::pair<std::string, BundleSectionLocation>;
using BundleSectionTable = std::vector<BundleSectionEntry>;

uint64_t read_u64_le(std::ifstream& in) {
    unsigned char bytes[8];
    in.read(reinterpret_cast<char*>(bytes), 8);
    if (!in) {
        throw std::runtime_error("Failed to read uint64 from bundle file");
    }
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | bytes[i];
    }
    return value;
}

void require_exact_keys(const nlohmann::json& object,
                        std::initializer_list<std::string_view> expected,
                        const std::string& context) {
    if (!object.is_object())
        throw std::runtime_error(context + " must be a JSON object");

    for (const auto& [key, value] : object.items()) {
        (void)value;
        const bool known = std::find(expected.begin(), expected.end(), key) != expected.end();
        if (!known)
            throw std::runtime_error(context + " contains unsupported field '" + key + "'");
    }
    for (const auto key : expected) {
        if (!object.contains(std::string(key)))
            throw std::runtime_error(context + " missing required field '" + std::string(key) +
                                     "'");
    }
}

std::string require_nonempty_string(const nlohmann::json& object, const char* field,
                                    const std::string& context) {
    const auto& value = object.at(field);
    if (!value.is_string())
        throw std::runtime_error(context + " field '" + field + "' must be a string");
    std::string result = value.get<std::string>();
    if (result.empty())
        throw std::runtime_error(context + " field '" + field + "' must not be empty");
    return result;
}

std::uint64_t require_uint64(const nlohmann::json& object, const char* field,
                             const std::string& context) {
    const auto& value = object.at(field);
    if (value.is_number_unsigned())
        return value.get<std::uint64_t>();
    if (value.is_number_integer()) {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value >= 0)
            return static_cast<std::uint64_t>(signed_value);
    }
    throw std::runtime_error(context + " field '" + field + "' must be a non-negative integer");
}

BundleInfo BundleInfoFromJson(const std::string& json, BundleSectionTable& sections_out) {
    nlohmann::json header;
    try {
        header = nlohmann::json::parse(json);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("Invalid bundle header JSON: " + std::string(error.what()));
    }

    require_exact_keys(header, {"format", "family", "task", "backend", "sections"},
                       "Bundle header");
    const std::uint64_t format = require_uint64(header, "format", "Bundle header");
    if (format != 1)
        throw std::runtime_error("Unsupported bundle format: " + std::to_string(format));

    BundleInfo info;
    info.format = 1;
    info.family = require_nonempty_string(header, "family", "Bundle header");
    info.task = require_nonempty_string(header, "task", "Bundle header");
    info.backend = require_nonempty_string(header, "backend", "Bundle header");

    const auto& sections = header.at("sections");
    if (!sections.is_object())
        throw std::runtime_error("Bundle header field 'sections' must be a JSON object");

    sections_out.clear();
    info.sections.clear();
    sections_out.reserve(sections.size());
    info.sections.reserve(sections.size());
    for (const auto& [name, descriptor] : sections.items()) {
        if (name.empty())
            throw std::runtime_error("Bundle section name must not be empty");
        const std::string context = "Bundle section '" + name + "'";
        require_exact_keys(descriptor, {"offset", "length"}, context);
        const std::uint64_t offset = require_uint64(descriptor, "offset", context);
        const std::uint64_t length = require_uint64(descriptor, "length", context);
        sections_out.push_back({name, {offset, length}});
        info.sections.emplace_back(name, offset, length);
    }

    return info;
}

std::uint64_t checked_section_file_offset(const BundleSectionInfo& section,
                                          std::uint64_t data_start, std::uint64_t file_size,
                                          const std::string& path) {
    if (data_start > file_size || section.offset > file_size - data_start ||
        section.length > file_size - data_start - section.offset) {
        throw std::runtime_error("Bundle section '" + section.name +
                                 "' extends outside file: " + path);
    }
    if (section.length > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        section.length > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Bundle section '" + section.name +
                                 "' is too large to read: " + path);
    }
    return data_start + section.offset;
}

} // namespace

BundleReader::BundleReader(std::string bundle_path) {
    std::error_code error;
    path_ = std::filesystem::absolute(std::filesystem::path(bundle_path), error)
                .lexically_normal()
                .string();
    if (error)
        throw std::runtime_error("Failed to resolve bundle path: " + error.message());
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open bundle file: " + path_);
    }

    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in || std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) != 0) {
        throw std::runtime_error("Invalid bundle magic in: " + path_);
    }

    const uint64_t header_length = read_u64_le(in);
    if (header_length > 100 * 1024 * 1024) {
        throw std::runtime_error("Bundle header too large: " + path_);
    }

    std::string header_json(static_cast<std::size_t>(header_length), '\0');
    in.read(header_json.data(), static_cast<std::streamsize>(header_length));
    if (!in) {
        throw std::runtime_error("Failed to read bundle header: " + path_);
    }

    BundleSectionTable section_table;
    info_ = BundleInfoFromJson(header_json, section_table);

    in.seekg(0, std::ios::end);
    const auto file_end = in.tellg();
    if (file_end < 0)
        throw std::runtime_error("Failed to determine bundle size: " + path_);
    file_size_ = static_cast<std::uint64_t>(file_end);
    data_offset_ = kBundleHeaderOffset + header_length;
    for (const auto& section : info_.sections)
        (void)checked_section_file_offset(section, data_offset_, file_size_, path_);
}

const BundleSectionInfo* BundleReader::find_section(std::string_view name) const noexcept {
    for (const auto& section : info_.sections) {
        if (section.name == name)
            return &section;
    }
    return nullptr;
}

std::vector<char> BundleReader::read_section(std::string_view name) const {
    const auto* section = find_section(name);
    if (section == nullptr)
        throw std::runtime_error("Bundle section not found: " + std::string(name));
    const std::uint64_t file_offset =
        checked_section_file_offset(*section, data_offset_, file_size_, path_);
    if (file_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
        throw std::runtime_error("Bundle section '" + section->name +
                                 "' has an unsupported file offset: " + path_);

    std::vector<char> data(static_cast<std::size_t>(section->length));
    if (data.empty())
        return data;
    std::ifstream in(path_, std::ios::binary);
    if (!in)
        throw std::runtime_error("Failed to open bundle file: " + path_);
    in.seekg(static_cast<std::streamoff>(file_offset));
    in.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!in) {
        throw std::runtime_error("Failed to read bundle section '" + section->name +
                                 "' from: " + path_);
    }
    return data;
}

BundleInfo InspectBundle(const std::string& bundle_path) {
    return BundleReader(bundle_path).info();
}

} // namespace trtmc
