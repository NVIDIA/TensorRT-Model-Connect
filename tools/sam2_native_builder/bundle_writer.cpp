/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle_writer.h"

#include "durable_file_writer.h"
#include "runtime/models/sam2/sam2_engine_contract.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace trtmc::sam2::native {

namespace {

constexpr std::array<unsigned char, 8> kBundleMagic = {'B', 'U', 'N', 'D', 'L', 'E', '\x01', 0};

std::string escapeJson(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    constexpr char kHex[] = "0123456789abcdef";
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (byte < 0x20U) {
                result += "\\u00";
                result.push_back(kHex[byte >> 4U]);
                result.push_back(kHex[byte & 0x0fU]);
            } else {
                result.push_back(static_cast<char>(byte));
            }
        }
    }
    return result;
}

void requireMetadata(const BundleMetadata& metadata) {
    if (metadata.model_id.empty() || metadata.trt_version.empty() || metadata.trt_abi.empty() ||
        metadata.gpu_name.empty() || metadata.created_at.empty()) {
        throw BundleWriteError("SAM2 native bundle metadata fields must not be empty");
    }
}

std::vector<std::string_view> requiredSectionNames() {
    std::vector<std::string_view> names;
    names.reserve(kRequiredPlanSections.size() + 2U);
    names.insert(names.end(), kRequiredPlanSections.begin(), kRequiredPlanSections.end());
    names.push_back(kConfigSection);
    names.push_back(kBuildReceiptSection);
    return names;
}

std::vector<const BundleSectionView*>
validateAndOrderSections(const std::vector<BundleSectionView>& sections) {
    const auto required = requiredSectionNames();
    if (sections.size() != required.size()) {
        throw BundleWriteError(
            "SAM2 native bundle requires exactly six plans, config, and receipt");
    }

    std::unordered_map<std::string_view, const BundleSectionView*> indexed;
    indexed.reserve(sections.size());
    for (const auto& section : sections) {
        if (section.name.empty())
            throw BundleWriteError("SAM2 native bundle section name must not be empty");
        if (section.data == nullptr || section.size == 0)
            throw BundleWriteError("SAM2 native bundle sections must not be empty");
        if (!indexed.emplace(section.name, &section).second)
            throw BundleWriteError("SAM2 native bundle contains a duplicate section");
    }

    std::vector<const BundleSectionView*> ordered;
    ordered.reserve(required.size());
    for (const auto name : required) {
        const auto found = indexed.find(name);
        if (found == indexed.end())
            throw BundleWriteError("SAM2 native bundle is missing required section: " +
                                   std::string(name));
        ordered.push_back(found->second);
        indexed.erase(found);
    }
    if (!indexed.empty())
        throw BundleWriteError("SAM2 native bundle contains an unsupported section");
    return ordered;
}

std::string makeHeader(const BundleMetadata& metadata,
                       const std::vector<const BundleSectionView*>& ordered) {
    std::ostringstream header;
    header << "{\"model_id\":\"" << escapeJson(metadata.model_id)
           << "\",\"model_type\":\"sam2_video_tracking\",\"family\":\"sam2\""
              ",\"precision\":\"mixed_bf16_fp32\",\"trt_version\":\""
           << escapeJson(metadata.trt_version) << "\",\"trt_abi\":\""
           << escapeJson(metadata.trt_abi) << "\",\"gpu_name\":\"" << escapeJson(metadata.gpu_name)
           << "\",\"created_at\":\"" << escapeJson(metadata.created_at)
           << "\",\"runtime_strategy\":\"" << kStrategyName << "\",\"sections\":{";

    std::uint64_t offset = 0;
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        const auto& section = *ordered[index];
        const auto size = static_cast<std::uint64_t>(section.size);
        if (size > std::numeric_limits<std::uint64_t>::max() - offset)
            throw BundleWriteError("SAM2 native bundle payload size overflows uint64");
        if (index != 0)
            header << ',';
        internal::Sha256 hash;
        hash.update(section.data, section.size);
        header << '\"' << escapeJson(section.name) << "\":{\"offset\":" << offset
               << ",\"size\":" << size << ",\"sha256\":\"" << hash.hex_digest() << "\"}";
        offset += size;
    }
    header << "}}";
    const auto result = header.str();
    if (result.size() > 100U * 1024U * 1024U)
        throw BundleWriteError("SAM2 native bundle header exceeds the format limit");
    return result;
}

std::array<std::uint8_t, 8> u64LittleEndian(std::uint64_t value) {
    std::array<std::uint8_t, 8> result{};
    for (unsigned int shift = 0; shift < 64U; shift += 8U)
        result[shift / 8U] = static_cast<std::uint8_t>(value >> shift);
    return result;
}

struct CanonicalBundleFacts {
    std::string sha256;
    std::uint64_t size_bytes{0U};
};

CanonicalBundleFacts canonicalBundleFacts(const std::string& header,
                                          const std::vector<const BundleSectionView*>& ordered) {
    internal::Sha256 hash;
    hash.update(kBundleMagic.data(), kBundleMagic.size());
    const auto header_size = u64LittleEndian(static_cast<std::uint64_t>(header.size()));
    hash.update(header_size.data(), header_size.size());
    hash.update(header.data(), header.size());

    std::uint64_t size = kBundleMagic.size() + header_size.size();
    if (header.size() > std::numeric_limits<std::uint64_t>::max() - size)
        throw BundleWriteError("SAM2 native bundle size overflows uint64");
    size += static_cast<std::uint64_t>(header.size());
    for (const auto* section : ordered) {
        hash.update(section->data, section->size);
        if (section->size > std::numeric_limits<std::uint64_t>::max() - size)
            throw BundleWriteError("SAM2 native bundle size overflows uint64");
        size += static_cast<std::uint64_t>(section->size);
    }
    return {hash.hex_digest(), size};
}

#if defined(__linux__)

std::string systemError(std::string_view action, int error) {
    return std::string(action) + ": " + std::strerror(error);
}

void writeBytes(int descriptor, const void* data, std::size_t size, std::string_view what) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t written = 0;
    constexpr std::size_t kChunk = 64U * 1024U * 1024U;
    while (written != size) {
        const auto count = std::min(kChunk, size - written);
        const ssize_t result = ::write(descriptor, bytes + written, count);
        if (result < 0) {
            if (errno == EINTR)
                continue;
            throw BundleWriteError(
                systemError("Failed to write SAM2 native bundle " + std::string(what), errno));
        }
        if (result == 0)
            throw BundleWriteError("Failed to make progress writing SAM2 native bundle " +
                                   std::string(what));
        written += static_cast<std::size_t>(result);
    }
}

bool sameCompletedFileState(const struct stat& left, const struct stat& right) noexcept {
    return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
           left.st_mode == right.st_mode && left.st_size == right.st_size &&
           left.st_mtim.tv_sec == right.st_mtim.tv_sec &&
           left.st_mtim.tv_nsec == right.st_mtim.tv_nsec &&
           left.st_ctim.tv_sec == right.st_ctim.tv_sec &&
           left.st_ctim.tv_nsec == right.st_ctim.tv_nsec;
}

BundlePublicationFacts authenticateCompletedDescriptor(int descriptor,
                                                       const CanonicalBundleFacts& expected) {
    struct stat before{};
    if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) || before.st_size < 0 ||
        static_cast<std::uint64_t>(before.st_size) != expected.size_bytes) {
        throw BundleWriteError("Completed SAM2 native bundle descriptor size or type drifted");
    }

    internal::Sha256 hash;
    std::array<std::uint8_t, 1024U * 1024U> buffer{};
    std::uint64_t offset = 0U;
    while (offset != expected.size_bytes) {
        const auto count = static_cast<std::size_t>(
            std::min<std::uint64_t>(buffer.size(), expected.size_bytes - offset));
        ssize_t read_count = -1;
        do {
            read_count = ::pread(descriptor, buffer.data(), count, static_cast<off_t>(offset));
        } while (read_count < 0 && errno == EINTR);
        if (read_count <= 0)
            throw BundleWriteError("Failed to hash the completed SAM2 native bundle descriptor");
        hash.update(buffer.data(), static_cast<std::size_t>(read_count));
        offset += static_cast<std::uint64_t>(read_count);
    }

    struct stat after{};
    if (::fstat(descriptor, &after) != 0 || !sameCompletedFileState(before, after)) {
        throw BundleWriteError("SAM2 native bundle descriptor changed while it was authenticated");
    }
    const std::string actual_sha256 = hash.hex_digest();
    if (actual_sha256 != expected.sha256)
        throw BundleWriteError("Completed SAM2 native bundle SHA-256 drifted from written bytes");
    return {actual_sha256, expected.size_bytes, static_cast<std::uint64_t>(after.st_dev),
            static_cast<std::uint64_t>(after.st_ino), true};
}

#else

void writeU64LittleEndian(std::ostream& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64U; shift += 8U)
        output.put(static_cast<char>((value >> shift) & 0xffU));
}

void writeBytes(std::ostream& output, const void* data, std::size_t size, std::string_view what) {
    const auto* bytes = static_cast<const char*>(data);
    std::size_t written = 0;
    constexpr std::size_t kChunk = 64U * 1024U * 1024U;
    while (written != size) {
        const auto count = std::min(kChunk, size - written);
        output.write(bytes + written, static_cast<std::streamsize>(count));
        if (!output)
            throw BundleWriteError("Failed to write SAM2 native bundle " + std::string(what));
        written += count;
    }
}

class TemporaryFile final {
  public:
    explicit TemporaryFile(std::filesystem::path path) : path_(std::move(path)) {}
    ~TemporaryFile() {
        if (!committed_) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }
    const std::filesystem::path& path() const noexcept { return path_; }
    void commit() noexcept { committed_ = true; }

  private:
    std::filesystem::path path_;
    bool committed_{false};
};

#endif

} // namespace

BundlePublicationFacts writeSam2NativeBundle(const std::filesystem::path& destination,
                                             const BundleMetadata& metadata,
                                             const std::vector<BundleSectionView>& sections) {
    requireMetadata(metadata);
    if (destination.empty() || !destination.has_filename())
        throw BundleWriteError("SAM2 native bundle destination must name a file");

    const auto ordered = validateAndOrderSections(sections);
    const auto header = makeHeader(metadata, ordered);
    const auto expected = canonicalBundleFacts(header, ordered);

#if defined(__linux__)
    try {
        BundlePublicationFacts publication;
        (void)durable_file::writeExclusiveDurably(
            destination, "SAM2 native bundle",
            [&](int descriptor) {
                writeBytes(descriptor, kBundleMagic.data(), kBundleMagic.size(), "magic");
                const auto header_size = u64LittleEndian(static_cast<std::uint64_t>(header.size()));
                writeBytes(descriptor, header_size.data(), header_size.size(), "header length");
                writeBytes(descriptor, header.data(), header.size(), "header");
                for (const auto* section : ordered)
                    writeBytes(descriptor, section->data, section->size, section->name);
            },
            [&](int descriptor) {
                publication = authenticateCompletedDescriptor(descriptor, expected);
            });
        return publication;
    } catch (const durable_file::WriteError& error) {
        throw BundleWriteError(error.what());
    }
#else
    if (std::filesystem::exists(destination))
        throw BundleWriteError("SAM2 native bundle destination already exists");
    const auto parent = destination.parent_path();
    if (!parent.empty() && !std::filesystem::is_directory(parent))
        throw BundleWriteError("SAM2 native bundle destination directory does not exist");
    auto temporary_path = destination;
    temporary_path += ".tmp";
    if (std::filesystem::exists(temporary_path))
        throw BundleWriteError("SAM2 native bundle temporary path already exists");
    TemporaryFile temporary(temporary_path);
    {
        std::ofstream output(temporary.path(), std::ios::binary | std::ios::out | std::ios::trunc);
        if (!output)
            throw BundleWriteError("Failed to create SAM2 native bundle temporary file");
        writeBytes(output, kBundleMagic.data(), kBundleMagic.size(), "magic");
        writeU64LittleEndian(output, static_cast<std::uint64_t>(header.size()));
        writeBytes(output, header.data(), header.size(), "header");
        for (const auto* section : ordered)
            writeBytes(output, section->data, section->size, section->name);
        output.flush();
        if (!output)
            throw BundleWriteError("Failed to flush SAM2 native bundle");
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary.path(), destination, rename_error);
    if (rename_error)
        throw BundleWriteError("Failed to publish SAM2 native bundle: " + rename_error.message());
    temporary.commit();
    return {expected.sha256, expected.size_bytes, 0U, 0U, false};
#endif
}

} // namespace trtmc::sam2::native
