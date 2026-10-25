/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/bundle/bundle_format.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
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

class SectionPlanReader final : public IPlanReader {
    struct CloseFile {
        void operator()(std::FILE* file) const noexcept { (void)std::fclose(file); }
    };

  public:
    SectionPlanReader(const std::string& path, std::uint64_t file_size, std::uint64_t offset,
                      std::uint64_t length)
        : file_(std::fopen(path.c_str(), "rb")), offset_(offset), length_(length) {
        if (!file_)
            throw std::runtime_error("Cannot open bundle section stream: " + path);
        const int fd = ::fileno(file_.get());
        if (::fcntl(fd, F_SETFD, FD_CLOEXEC) != 0 || ::fstat(fd, &identity_) != 0 ||
            !S_ISREG(identity_.st_mode) || identity_.st_size < 0 ||
            static_cast<std::uint64_t>(identity_.st_size) != file_size || offset > file_size ||
            length > file_size - offset ||
            file_size > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
            throw std::runtime_error("Bundle changed or cannot be streamed: " + path);
    }

    std::uint64_t size() const noexcept override { return length_; }

    std::size_t read(void* destination, std::size_t count) override {
        verify();
        if (count == 0 || cursor_ == length_)
            return 0;
        if (destination == nullptr)
            throw std::invalid_argument("Bundle section read requires a destination");
        const auto requested =
            static_cast<std::size_t>(std::min<std::uint64_t>(count, length_ - cursor_));
        std::size_t done = 0;
        while (done < requested) {
            const auto chunk = std::min<std::size_t>(
                requested - done, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
            const auto received =
                ::pread(::fileno(file_.get()), static_cast<char*>(destination) + done, chunk,
                        static_cast<off_t>(offset_ + cursor_ + done));
            if (received < 0 && errno == EINTR)
                continue;
            if (received <= 0)
                throw std::runtime_error("Failed reading complete bundle section bytes");
            done += static_cast<std::size_t>(received);
        }
        verify();
        cursor_ += done;
        return done;
    }

    bool seek(std::int64_t offset, PlanSeekOrigin origin) override {
        verify();
        std::uint64_t base;
        switch (origin) {
        case PlanSeekOrigin::begin:
            base = 0;
            break;
        case PlanSeekOrigin::current:
            base = cursor_;
            break;
        case PlanSeekOrigin::end:
            base = length_;
            break;
        default:
            return false;
        }
        // The bundle validator already limits section lengths to streamsize.
        const auto signed_base = static_cast<std::int64_t>(base);
        if (offset < -signed_base || offset > static_cast<std::int64_t>(length_ - base))
            return false;
        cursor_ = static_cast<std::uint64_t>(signed_base + offset);
        return true;
    }

  private:
    void verify() const {
        struct stat now{};
        if (::fstat(::fileno(file_.get()), &now) != 0 || now.st_dev != identity_.st_dev ||
            now.st_ino != identity_.st_ino || now.st_size != identity_.st_size ||
            now.st_mtim.tv_sec != identity_.st_mtim.tv_sec ||
            now.st_mtim.tv_nsec != identity_.st_mtim.tv_nsec ||
            now.st_ctim.tv_sec != identity_.st_ctim.tv_sec ||
            now.st_ctim.tv_nsec != identity_.st_ctim.tv_nsec)
            throw std::runtime_error("Bundle file changed during streamed section loading");
    }

    std::unique_ptr<std::FILE, CloseFile> file_;
    struct stat identity_{};
    std::uint64_t offset_;
    std::uint64_t length_;
    std::uint64_t cursor_{0};
};

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

std::unique_ptr<IPlanReader> BundleReader::open_section(std::string_view name) const {
    const auto* section = find_section(name);
    if (section == nullptr)
        throw std::runtime_error("Bundle section not found: " + std::string(name));
    const auto offset = checked_section_file_offset(*section, data_offset_, file_size_, path_);
    return std::make_unique<SectionPlanReader>(path_, file_size_, offset, section->length);
}

void BundleReader::copy_section(std::string_view name, std::ostream& output) const {
    const auto* section = find_section(name);
    if (section == nullptr)
        throw std::runtime_error("Bundle section not found: " + std::string(name));
    const auto offset = checked_section_file_offset(*section, data_offset_, file_size_, path_);
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
        throw std::runtime_error("Bundle section has an unsupported file offset: " + path_);
    std::ifstream input(path_, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(offset));
    if (!input || !output)
        throw std::runtime_error("Cannot copy bundle section: " + std::string(name));
    std::array<char, 64 * 1024> buffer;
    auto remaining = section->length;
    while (remaining != 0) {
        const auto count =
            static_cast<std::streamsize>(std::min<std::uint64_t>(remaining, buffer.size()));
        input.read(buffer.data(), count);
        if (!input)
            throw std::runtime_error("Failed reading bundle section: " + std::string(name));
        output.write(buffer.data(), count);
        if (!output)
            throw std::runtime_error("Failed writing bundle section: " + std::string(name));
        remaining -= static_cast<std::uint64_t>(count);
    }
}

BundleInfo InspectBundle(const std::string& bundle_path) {
    return BundleReader(bundle_path).info();
}

} // namespace trtmc
