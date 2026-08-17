/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "checkpoint_reader.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace trtmc::sam2::native {
namespace {

constexpr std::uint32_t kLocalHeaderSignature = 0x04034b50U;
constexpr std::uint32_t kCentralHeaderSignature = 0x02014b50U;
constexpr std::uint32_t kDataDescriptorSignature = 0x08074b50U;
constexpr std::uint32_t kEndOfCentralDirectorySignature = 0x06054b50U;
constexpr std::uint32_t kZip64EndSignature = 0x06064b50U;
constexpr std::uint32_t kZip64LocatorSignature = 0x07064b50U;
constexpr std::uint16_t kZip64ExtraId = 0x0001U;
constexpr std::uint16_t kPyTorchAlignmentExtraId = 0x4246U;
constexpr std::uint16_t kExpectedFlags = 0x0808U; // UTF-8 names and data descriptors.

[[noreturn]] void fail(const std::string& message) {
    throw CheckpointError(message);
}

std::string systemError(const std::string& action, const std::filesystem::path& path) {
    return action + " '" + path.string() + "': " + std::strerror(errno);
}

std::uint64_t checkedAdd(std::uint64_t left, std::uint64_t right, const char* context) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        fail(std::string("integer overflow while ") + context);
    return left + right;
}

std::uint64_t checkedMultiply(std::uint64_t left, std::uint64_t right, const char* context) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
        fail(std::string("integer overflow while ") + context);
    return left * right;
}

std::size_t checkedSize(std::uint64_t value, const char* context) {
    if (value > std::numeric_limits<std::size_t>::max())
        fail(std::string("size_t overflow while ") + context);
    return static_cast<std::size_t>(value);
}

void requireRange(std::size_t offset, std::size_t length, std::size_t size, const char* context) {
    if (offset > size || length > size - offset)
        fail(std::string("truncated checkpoint while reading ") + context);
}

std::uint16_t readU16(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint16_t>(bytes[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t readU32(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint64_t readU64(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint64_t>(readU32(bytes)) |
           (static_cast<std::uint64_t>(readU32(bytes + 4U)) << 32U);
}

std::uint32_t rotateRight(std::uint32_t value, unsigned int amount) noexcept {
    return (value >> amount) | (value << (32U - amount));
}

void sha256Transform(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) noexcept {
    static constexpr std::array<std::uint32_t, 64> constants = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
        0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
        0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
        0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
        0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
        0xc67178f2U};
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16U; ++index) {
        const std::uint8_t* word = block + index * 4U;
        schedule[index] = (static_cast<std::uint32_t>(word[0]) << 24U) |
                          (static_cast<std::uint32_t>(word[1]) << 16U) |
                          (static_cast<std::uint32_t>(word[2]) << 8U) |
                          static_cast<std::uint32_t>(word[3]);
    }
    for (std::size_t index = 16U; index < schedule.size(); ++index) {
        const std::uint32_t s0 = rotateRight(schedule[index - 15U], 7U) ^
                                 rotateRight(schedule[index - 15U], 18U) ^
                                 (schedule[index - 15U] >> 3U);
        const std::uint32_t s1 = rotateRight(schedule[index - 2U], 17U) ^
                                 rotateRight(schedule[index - 2U], 19U) ^
                                 (schedule[index - 2U] >> 10U);
        schedule[index] = schedule[index - 16U] + s0 + schedule[index - 7U] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t index = 0; index < schedule.size(); ++index) {
        const std::uint32_t sum1 = rotateRight(e, 6U) ^ rotateRight(e, 11U) ^ rotateRight(e, 25U);
        const std::uint32_t choice = (e & f) ^ ((~e) & g);
        const std::uint32_t temporary1 = h + sum1 + choice + constants[index] + schedule[index];
        const std::uint32_t sum0 = rotateRight(a, 2U) ^ rotateRight(a, 13U) ^ rotateRight(a, 22U);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

std::string sha256Hex(const std::uint8_t* data, std::size_t size) {
    std::array<std::uint32_t, 8> state = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                          0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    const std::size_t full_bytes = size - (size % 64U);
    for (std::size_t offset = 0; offset < full_bytes; offset += 64U)
        sha256Transform(state, data + offset);

    std::array<std::uint8_t, 128> tail{};
    const std::size_t remainder = size - full_bytes;
    if (remainder != 0U)
        std::memcpy(tail.data(), data + full_bytes, remainder);
    tail[remainder] = 0x80U;
    const std::size_t padded_bytes = remainder < 56U ? 64U : 128U;
    const std::uint64_t bit_length = checkedMultiply(size, 8U, "hashing checkpoint");
    for (std::size_t index = 0; index < 8U; ++index)
        tail[padded_bytes - 1U - index] = static_cast<std::uint8_t>(bit_length >> (index * 8U));
    for (std::size_t offset = 0; offset < padded_bytes; offset += 64U)
        sha256Transform(state, tail.data() + offset);

    static constexpr char digits[] = "0123456789abcdef";
    std::string result(64U, '0');
    std::size_t output = 0;
    for (const std::uint32_t word : state) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            result[output++] = digits[(word >> static_cast<unsigned int>(shift)) & 0x0fU];
        }
    }
    return result;
}

bool isCanonicalSha256(std::string_view digest) noexcept {
    if (digest.size() != 64U)
        return false;
    return std::all_of(digest.begin(), digest.end(), [](char character) {
        return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
    });
}

class FileDescriptor final {
  public:
    explicit FileDescriptor(int value) noexcept : value_(value) {}
    ~FileDescriptor() {
        if (value_ >= 0)
            ::close(value_);
    }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    int get() const noexcept { return value_; }

  private:
    int value_{-1};
};

// The checkpoint is authenticated and parsed from an owned byte snapshot. In
// particular, tensor views must not retain a file-backed mapping: another
// writer can modify or truncate such a mapping after authentication, changing
// TensorRT weights or delivering SIGBUS while the plans are serialized.
class OwnedFileSnapshot final {
  public:
    OwnedFileSnapshot() = default;
    OwnedFileSnapshot(OwnedFileSnapshot&&) noexcept = default;
    OwnedFileSnapshot& operator=(OwnedFileSnapshot&&) noexcept = default;
    OwnedFileSnapshot(const OwnedFileSnapshot&) = delete;
    OwnedFileSnapshot& operator=(const OwnedFileSnapshot&) = delete;

    static OwnedFileSnapshot openReadOnly(const std::filesystem::path& path,
                                          std::uint64_t max_bytes) {
        int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#endif
        const int raw_fd = ::open(path.c_str(), flags);
        if (raw_fd < 0)
            fail(systemError("unable to open checkpoint", path));
        const FileDescriptor fd(raw_fd);

        struct stat status{};
        if (::fstat(fd.get(), &status) != 0)
            fail(systemError("unable to stat checkpoint", path));
        if (!S_ISREG(status.st_mode))
            fail("checkpoint must be a regular file: '" + path.string() + "'");
        if (status.st_size <= 0)
            fail("checkpoint is empty: '" + path.string() + "'");
        const auto file_size = static_cast<std::uint64_t>(status.st_size);
        if (file_size > max_bytes || file_size > std::numeric_limits<std::size_t>::max())
            fail("checkpoint exceeds the configured archive size limit");

        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
        std::size_t copied = 0;
        constexpr std::size_t kReadChunkBytes = 64U * 1024U * 1024U;
        while (copied != bytes.size()) {
            const std::size_t request = std::min(kReadChunkBytes, bytes.size() - copied);
            const ssize_t count =
                ::pread(fd.get(), bytes.data() + copied, request, static_cast<off_t>(copied));
            if (count < 0) {
                if (errno == EINTR)
                    continue;
                fail(systemError("unable to snapshot checkpoint", path));
            }
            if (count == 0)
                fail("checkpoint changed while its immutable snapshot was being created");
            copied += static_cast<std::size_t>(count);
        }

        struct stat current{};
        if (::fstat(fd.get(), &current) != 0)
            fail(systemError("unable to re-stat checkpoint after snapshot", path));
        if (current.st_dev != status.st_dev || current.st_ino != status.st_ino ||
            current.st_size != status.st_size) {
            fail("checkpoint changed while its immutable snapshot was being created");
        }
        return OwnedFileSnapshot(std::move(bytes));
    }

    const std::uint8_t* data() const noexcept { return bytes_.data(); }

    std::size_t size() const noexcept { return bytes_.size(); }

  private:
    explicit OwnedFileSnapshot(std::vector<std::uint8_t> bytes) : bytes_(std::move(bytes)) {}

    std::vector<std::uint8_t> bytes_;
};

const std::array<std::uint32_t, 256>& crcTable() {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> result{};
        for (std::uint32_t index = 0; index < result.size(); ++index) {
            std::uint32_t value = index;
            for (int bit = 0; bit < 8; ++bit)
                value = (value & 1U) != 0U ? (value >> 1U) ^ 0xedb88320U : value >> 1U;
            result[index] = value;
        }
        return result;
    }();
    return table;
}

std::uint32_t crc32(const std::uint8_t* data, std::size_t size) noexcept {
    std::uint32_t crc = 0xffffffffU;
    const auto& table = crcTable();
    for (std::size_t index = 0; index < size; ++index)
        crc = table[(crc ^ data[index]) & 0xffU] ^ (crc >> 8U);
    return crc ^ 0xffffffffU;
}

bool isSafeArchiveName(const std::string& name) {
    if (name.empty() || name.front() == '/' || name.back() == '/' ||
        name.find('\\') != std::string::npos || name.find(':') != std::string::npos)
        return false;
    std::size_t segment_begin = 0;
    for (std::size_t index = 0; index <= name.size(); ++index) {
        if (index != name.size() && name[index] != '/') {
            const unsigned char byte = static_cast<unsigned char>(name[index]);
            if (byte < 0x21U || byte > 0x7eU)
                return false;
            continue;
        }
        const std::string_view segment(name.data() + segment_begin, index - segment_begin);
        if (segment.empty() || segment == "." || segment == "..")
            return false;
        segment_begin = index + 1U;
    }
    return true;
}

struct ZipEntry {
    std::string name;
    std::uint16_t version_needed{0};
    std::uint16_t flags{0};
    std::uint16_t method{0};
    std::uint16_t modified_time{0};
    std::uint16_t modified_date{0};
    std::uint32_t crc{0};
    std::uint32_t size{0};
    std::uint32_t local_offset{0};
    std::size_t data_offset{0};
    std::size_t record_end{0};
};

class ZipArchive final {
  public:
    ZipArchive(const OwnedFileSnapshot& file, const ReaderLimits& limits) : file_(file) {
        parse(limits);
    }

    const std::vector<ZipEntry>& entries() const noexcept { return entries_; }

    const ZipEntry& requireEntry(const std::string& name) const {
        const auto found = by_name_.find(name);
        if (found == by_name_.end())
            fail("checkpoint ZIP is missing required member '" + name + "'");
        return entries_[found->second];
    }

    const std::uint8_t* data(const ZipEntry& entry) const noexcept {
        return file_.data() + entry.data_offset;
    }

  private:
    void parse(const ReaderLimits& limits) {
        const std::size_t size = file_.size();
        if (size < 22U)
            fail("checkpoint is too small to contain a ZIP end record");
        const std::size_t eocd_offset = size - 22U;
        const std::uint8_t* eocd = file_.data() + eocd_offset;
        if (readU32(eocd) != kEndOfCentralDirectorySignature)
            fail("checkpoint ZIP must have one comment-free end record at end of file");
        if (readU16(eocd + 20U) != 0U)
            fail("checkpoint ZIP comments are not supported");
        const std::uint16_t disk = readU16(eocd + 4U);
        const std::uint16_t central_disk = readU16(eocd + 6U);
        const std::uint16_t disk_entries = readU16(eocd + 8U);
        const std::uint16_t total_entries = readU16(eocd + 10U);
        const std::uint32_t central_size_u32 = readU32(eocd + 12U);
        const std::uint32_t central_offset_u32 = readU32(eocd + 16U);
        if (disk != 0U || central_disk != 0U || disk_entries != total_entries)
            fail("multi-disk ZIP checkpoints are not supported");
        if (total_entries == UINT16_MAX || central_size_u32 == UINT32_MAX ||
            central_offset_u32 == UINT32_MAX)
            fail("ZIP64 checkpoints are not supported");
        if (total_entries == 0U || total_entries > limits.max_archive_members)
            fail("checkpoint ZIP member count is outside the configured limit");

        const std::size_t central_offset = static_cast<std::size_t>(central_offset_u32);
        const std::size_t central_size = static_cast<std::size_t>(central_size_u32);
        requireRange(central_offset, central_size, size, "ZIP central directory");
        std::size_t central_record_end = eocd_offset;
        if (eocd_offset >= 20U &&
            readU32(file_.data() + eocd_offset - 20U) == kZip64LocatorSignature) {
            const std::size_t locator_offset = eocd_offset - 20U;
            const std::uint8_t* locator = file_.data() + locator_offset;
            const std::uint64_t zip64_offset_u64 = readU64(locator + 8U);
            if (readU32(locator + 4U) != 0U || readU32(locator + 16U) != 1U ||
                zip64_offset_u64 > std::numeric_limits<std::size_t>::max())
                fail("invalid ZIP64 end-record locator");
            const std::size_t zip64_offset = static_cast<std::size_t>(zip64_offset_u64);
            requireRange(zip64_offset, 56U, locator_offset, "ZIP64 end record");
            const std::uint8_t* zip64 = file_.data() + zip64_offset;
            if (readU32(zip64) != kZip64EndSignature || readU64(zip64 + 4U) != 44U ||
                zip64_offset + 56U != locator_offset)
                fail("unsupported extensible ZIP64 end record");
            if (readU16(zip64 + 14U) > 45U || readU32(zip64 + 16U) != 0U ||
                readU32(zip64 + 20U) != 0U)
                fail("invalid multi-disk ZIP64 end record");
            // PyTorch emits this redundant fixed-size ZIP64 envelope even for
            // small archives. Accept it only when every 64-bit field exactly
            // repeats the classic EOCD; substantive ZIP64 remains unsupported.
            if (readU64(zip64 + 24U) != disk_entries || readU64(zip64 + 32U) != total_entries ||
                readU64(zip64 + 40U) != central_size_u32 ||
                readU64(zip64 + 48U) != central_offset_u32)
                fail("ZIP64 and classic end records disagree");
            central_record_end = zip64_offset;
        }
        if (central_offset > central_record_end ||
            central_size != central_record_end - central_offset)
            fail("checkpoint ZIP central-directory offset or size drifted");

        entries_.reserve(total_entries);
        std::unordered_set<std::string> names;
        std::size_t cursor = central_offset;
        std::uint32_t previous_local_offset = 0;
        for (std::uint16_t index = 0; index < total_entries; ++index) {
            requireRange(cursor, 46U, eocd_offset, "ZIP central member header");
            const std::uint8_t* header = file_.data() + cursor;
            if (readU32(header) != kCentralHeaderSignature)
                fail("invalid ZIP central member signature");
            const std::uint16_t version_made = readU16(header + 4U);
            ZipEntry entry;
            entry.version_needed = readU16(header + 6U);
            entry.flags = readU16(header + 8U);
            entry.method = readU16(header + 10U);
            entry.modified_time = readU16(header + 12U);
            entry.modified_date = readU16(header + 14U);
            entry.crc = readU32(header + 16U);
            const std::uint32_t compressed_size = readU32(header + 20U);
            entry.size = readU32(header + 24U);
            const std::uint16_t name_size = readU16(header + 28U);
            const std::uint16_t extra_size = readU16(header + 30U);
            const std::uint16_t comment_size = readU16(header + 32U);
            const std::uint16_t starting_disk = readU16(header + 34U);
            const std::uint32_t external_attributes = readU32(header + 38U);
            entry.local_offset = readU32(header + 42U);

            if (entry.version_needed >= 45U || compressed_size == UINT32_MAX ||
                entry.size == UINT32_MAX || entry.local_offset == UINT32_MAX ||
                starting_disk == UINT16_MAX)
                fail("ZIP64 checkpoints are not supported");
            if (entry.flags != kExpectedFlags) {
                if ((entry.flags & 0x0001U) != 0U || (entry.flags & 0x0040U) != 0U)
                    fail("encrypted ZIP members are not supported");
                fail("checkpoint ZIP member flags do not match the PyTorch stored format");
            }
            if (entry.method != 0U)
                fail("compressed ZIP members are not supported");
            if (compressed_size != entry.size)
                fail("stored ZIP member compressed and uncompressed sizes differ");
            if (starting_disk != 0U)
                fail("multi-disk ZIP members are not supported");
            if (extra_size != 0U || comment_size != 0U)
                fail("checkpoint ZIP central members contain unsupported metadata");

            const std::size_t variable_size =
                static_cast<std::size_t>(name_size) + extra_size + comment_size;
            requireRange(cursor + 46U, variable_size, eocd_offset, "ZIP central member name");
            entry.name.assign(reinterpret_cast<const char*>(header + 46U), name_size);
            if (!isSafeArchiveName(entry.name))
                fail("checkpoint ZIP contains an unsafe member name");
            const std::uint8_t creator_system = static_cast<std::uint8_t>(version_made >> 8U);
            const std::uint32_t unix_type = (external_attributes >> 16U) & 0170000U;
            if (creator_system == 3U && unix_type == 0120000U)
                fail("checkpoint ZIP symlink members are not supported");
            if (!names.insert(entry.name).second)
                fail("checkpoint ZIP contains duplicate member '" + entry.name + "'");
            if (index == 0U) {
                if (entry.local_offset != 0U)
                    fail("checkpoint ZIP contains an unsupported leading payload");
            } else if (entry.local_offset <= previous_local_offset) {
                fail("checkpoint ZIP local records are reordered or overlapping");
            }
            previous_local_offset = entry.local_offset;
            entries_.push_back(std::move(entry));
            cursor += 46U + variable_size;
        }
        if (cursor != central_record_end)
            fail("checkpoint ZIP central-directory entry count drifted");

        for (std::size_t index = 0; index < entries_.size(); ++index) {
            parseLocalRecord(entries_[index], index + 1U < entries_.size()
                                                  ? entries_[index + 1U].local_offset
                                                  : central_offset);
            by_name_.emplace(entries_[index].name, index);
        }
    }

    void parseLocalRecord(ZipEntry& entry, std::size_t expected_end) {
        const std::size_t local_offset = static_cast<std::size_t>(entry.local_offset);
        requireRange(local_offset, 30U, file_.size(), "ZIP local member header");
        const std::uint8_t* header = file_.data() + local_offset;
        if (readU32(header) != kLocalHeaderSignature)
            fail("invalid ZIP local member signature for '" + entry.name + "'");
        const std::uint16_t version_needed = readU16(header + 4U);
        const std::uint16_t flags = readU16(header + 6U);
        const std::uint16_t method = readU16(header + 8U);
        const std::uint16_t modified_time = readU16(header + 10U);
        const std::uint16_t modified_date = readU16(header + 12U);
        const std::uint32_t local_crc = readU32(header + 14U);
        const std::uint32_t local_compressed_size = readU32(header + 18U);
        const std::uint32_t local_size = readU32(header + 22U);
        const std::uint16_t name_size = readU16(header + 26U);
        const std::uint16_t extra_size = readU16(header + 28U);

        if (version_needed != entry.version_needed || flags != entry.flags ||
            method != entry.method || modified_time != entry.modified_time ||
            modified_date != entry.modified_date)
            fail("ZIP central/local metadata drift for '" + entry.name + "'");
        if (local_crc != 0U || local_compressed_size != 0U || local_size != 0U)
            fail("ZIP data-descriptor member has unexpected local sizes for '" + entry.name + "'");
        const std::size_t variable_size = static_cast<std::size_t>(name_size) + extra_size;
        requireRange(local_offset + 30U, variable_size, file_.size(), "ZIP local member name");
        const std::string local_name(reinterpret_cast<const char*>(header + 30U), name_size);
        if (local_name != entry.name)
            fail("ZIP central/local name drift for '" + entry.name + "'");

        const std::uint8_t* extra = header + 30U + name_size;
        if (extra_size < 4U || readU16(extra) != kPyTorchAlignmentExtraId ||
            readU16(extra + 2U) != static_cast<std::uint16_t>(extra_size - 4U))
            fail("ZIP local member lacks the expected PyTorch alignment field");
        for (std::size_t index = 4U; index < extra_size; ++index) {
            if (extra[index] != static_cast<std::uint8_t>('Z'))
                fail("ZIP local PyTorch alignment padding drifted");
        }
        entry.data_offset = local_offset + 30U + variable_size;
        if ((entry.data_offset & 63U) != 0U)
            fail("ZIP local PyTorch payload is not 64-byte aligned");
        requireRange(entry.data_offset, entry.size, file_.size(), "ZIP stored member data");
        const std::size_t descriptor_offset = entry.data_offset + entry.size;
        requireRange(descriptor_offset, 16U, file_.size(), "ZIP data descriptor");
        const std::uint8_t* descriptor = file_.data() + descriptor_offset;
        if (readU32(descriptor) != kDataDescriptorSignature ||
            readU32(descriptor + 4U) != entry.crc || readU32(descriptor + 8U) != entry.size ||
            readU32(descriptor + 12U) != entry.size)
            fail("ZIP data descriptor drift for '" + entry.name + "'");
        entry.record_end = descriptor_offset + 16U;
        if (entry.record_end != expected_end)
            fail("ZIP local record boundary drift for '" + entry.name + "'");
        if (crc32(file_.data() + entry.data_offset, entry.size) != entry.crc)
            fail("ZIP CRC mismatch for '" + entry.name + "'");
    }

    const OwnedFileSnapshot& file_;
    std::vector<ZipEntry> entries_;
    std::unordered_map<std::string, std::size_t> by_name_;
};

std::uint64_t parseCanonicalDecimal(std::string_view text, const char* context) {
    if (text.empty() || (text.size() > 1U && text.front() == '0'))
        fail(std::string("non-canonical decimal ") + context);
    std::uint64_t value = 0;
    for (const char character : text) {
        if (character < '0' || character > '9')
            fail(std::string("invalid decimal ") + context);
        value = checkedMultiply(value, 10U, context);
        value = checkedAdd(value, static_cast<std::uint64_t>(character - '0'), context);
    }
    return value;
}

struct ArchiveLayout {
    std::string root;
    const ZipEntry* pickle{nullptr};
    const ZipEntry* version{nullptr};
    std::unordered_map<std::string, const ZipEntry*> storages;
};

ArchiveLayout inspectArchiveLayout(const ZipArchive& archive, const ReaderLimits& limits) {
    ArchiveLayout layout;
    for (const ZipEntry& entry : archive.entries()) {
        constexpr std::string_view suffix = "/data.pkl";
        if (entry.name.size() > suffix.size() &&
            entry.name.compare(entry.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            if (layout.pickle != nullptr)
                fail("checkpoint ZIP must contain exactly one data.pkl member");
            layout.root = entry.name.substr(0, entry.name.size() - suffix.size());
            if (layout.root.empty() || layout.root.find('/') != std::string::npos)
                fail("checkpoint ZIP must have exactly one archive root");
            layout.pickle = &entry;
        }
    }
    if (layout.pickle == nullptr)
        fail("checkpoint ZIP must contain exactly one data.pkl member");
    if (layout.pickle->size > limits.max_pickle_bytes)
        fail("checkpoint pickle exceeds the configured size limit");

    const std::string version_name = layout.root + "/version";
    const std::string storage_prefix = layout.root + "/data/";
    std::map<std::uint64_t, const ZipEntry*> ordered_storages;
    for (const ZipEntry& entry : archive.entries()) {
        if (entry.name == layout.pickle->name)
            continue;
        if (entry.name == version_name) {
            if (layout.version != nullptr)
                fail("checkpoint ZIP contains duplicate version members");
            layout.version = &entry;
            continue;
        }
        if (entry.name.compare(0, storage_prefix.size(), storage_prefix) == 0) {
            const std::string key = entry.name.substr(storage_prefix.size());
            const std::uint64_t numeric_key = parseCanonicalDecimal(key, "storage key");
            if (!ordered_storages.emplace(numeric_key, &entry).second)
                fail("checkpoint ZIP contains duplicate numeric storage keys");
            layout.storages.emplace(key, &entry);
            continue;
        }
        fail("checkpoint ZIP contains unsupported member '" + entry.name + "'");
    }
    if (layout.version == nullptr)
        fail("checkpoint ZIP is missing its serialization version");
    if (layout.version->size != 2U || archive.data(*layout.version)[0] != '3' ||
        archive.data(*layout.version)[1] != '\n')
        fail("checkpoint uses an unsupported PyTorch serialization version");
    if (ordered_storages.empty())
        fail("checkpoint ZIP contains no tensor storages");
    std::uint64_t expected_key = 0;
    for (const auto& item : ordered_storages) {
        if (item.first != expected_key)
            fail("checkpoint ZIP storage keys are not consecutive from zero");
        ++expected_key;
    }
    return layout;
}

enum class ValueKind {
    kInteger,
    kBoolean,
    kString,
    kTuple,
    kDictionary,
    kGlobal,
    kStorage,
    kOrderedDictionary,
    kTensor,
};

enum class GlobalKind {
    kRebuildTensorV2,
    kFloatStorage,
    kLongStorage,
    kOrderedDictionary,
};

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct DictionaryValue {
    std::vector<std::pair<std::string, ValuePtr>> items;
    std::unordered_set<std::string> keys;
};

struct StorageValue {
    DType dtype{DType::kFloat32};
    std::string key;
    std::uint64_t elements{0};
};

struct TensorValue {
    StorageValue storage;
    std::uint64_t storage_offset{0};
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
};

struct Value {
    ValueKind kind{ValueKind::kInteger};
    std::int64_t integer{0};
    bool boolean{false};
    std::string string;
    std::vector<ValuePtr> tuple;
    std::shared_ptr<DictionaryValue> dictionary;
    GlobalKind global{GlobalKind::kRebuildTensorV2};
    StorageValue storage;
    TensorValue tensor;
};

ValuePtr makeValue(ValueKind kind) {
    auto value = std::make_shared<Value>();
    value->kind = kind;
    return value;
}

ValuePtr makeInteger(std::int64_t integer) {
    auto value = makeValue(ValueKind::kInteger);
    value->integer = integer;
    return value;
}

ValuePtr makeBoolean(bool boolean) {
    auto value = makeValue(ValueKind::kBoolean);
    value->boolean = boolean;
    return value;
}

ValuePtr makeString(std::string string) {
    auto value = makeValue(ValueKind::kString);
    value->string = std::move(string);
    return value;
}

ValuePtr makeTuple(std::vector<ValuePtr> tuple) {
    auto value = makeValue(ValueKind::kTuple);
    value->tuple = std::move(tuple);
    return value;
}

ValuePtr makeDictionary() {
    auto value = makeValue(ValueKind::kDictionary);
    value->dictionary = std::make_shared<DictionaryValue>();
    return value;
}

ValuePtr makeGlobal(GlobalKind global) {
    auto value = makeValue(ValueKind::kGlobal);
    value->global = global;
    return value;
}

const std::string& requireString(const ValuePtr& value, const char* context) {
    if (value == nullptr || value->kind != ValueKind::kString)
        fail(std::string(context) + " must be a string");
    return value->string;
}

std::int64_t requireInteger(const ValuePtr& value, const char* context) {
    if (value == nullptr || value->kind != ValueKind::kInteger)
        fail(std::string(context) + " must be an integer");
    return value->integer;
}

const std::vector<ValuePtr>& requireTuple(const ValuePtr& value, const char* context) {
    if (value == nullptr || value->kind != ValueKind::kTuple)
        fail(std::string(context) + " must be a tuple");
    return value->tuple;
}

class PickleParser final {
  public:
    PickleParser(const std::uint8_t* data, std::size_t size, const ReaderLimits& limits)
        : data_(data), size_(size), limits_(limits) {}

    ValuePtr parse() {
        while (position_ < size_) {
            const std::size_t opcode_offset = position_;
            const std::uint8_t opcode = getByte("pickle opcode");
            switch (opcode) {
            case 0x80U: // PROTO
                parseProtocol(opcode_offset);
                break;
            case '}': // EMPTY_DICT
                push(makeDictionary());
                break;
            case 'q': // BINPUT
                memoize(getByte("BINPUT index"), false);
                break;
            case 'r': // LONG_BINPUT
                memoize(getU32("LONG_BINPUT index"), true);
                break;
            case 'h': // BINGET
                recall(getByte("BINGET index"), false);
                break;
            case 'j': // LONG_BINGET
                recall(getU32("LONG_BINGET index"), true);
                break;
            case 'X': // BINUNICODE
                push(makeString(getString()));
                break;
            case 'c': // GLOBAL
                parseGlobal();
                break;
            case '(': // MARK
                if (marks_.size() >= limits_.max_pickle_stack)
                    fail("pickle MARK stack exceeds the configured limit");
                marks_.push_back(stack_.size());
                break;
            case 'J': { // BININT
                const std::uint32_t raw = getU32("BININT");
                const std::int64_t integer = raw <= static_cast<std::uint32_t>(INT32_MAX)
                                                 ? static_cast<std::int64_t>(raw)
                                                 : static_cast<std::int64_t>(raw) - 4294967296LL;
                push(makeInteger(integer));
                break;
            }
            case 'K': // BININT1
                push(makeInteger(getByte("BININT1")));
                break;
            case 'M': // BININT2
                push(makeInteger(getU16("BININT2")));
                break;
            case 'Q': // BINPERSID
                parsePersistentId();
                break;
            case 0x89U: // NEWFALSE
                push(makeBoolean(false));
                break;
            case ')': // EMPTY_TUPLE
                push(makeTuple({}));
                break;
            case 't': // TUPLE
                buildMarkedTuple();
                break;
            case 0x85U: // TUPLE1
                buildFixedTuple(1U);
                break;
            case 0x86U: // TUPLE2
                buildFixedTuple(2U);
                break;
            case 0x87U: // TUPLE3
                buildFixedTuple(3U);
                break;
            case 'R': // REDUCE
                reduce();
                break;
            case 's': // SETITEM
                setItem();
                break;
            case 'u': // SETITEMS
                setItems();
                break;
            case '.': // STOP
                return stop();
            default: {
                std::ostringstream message;
                message << "unsupported pickle opcode 0x" << std::hex
                        << static_cast<unsigned int>(opcode) << " at byte " << std::dec
                        << opcode_offset;
                fail(message.str());
            }
            }
        }
        fail("checkpoint pickle ended without STOP");
    }

  private:
    std::uint8_t getByte(const char* context) {
        requireRange(position_, 1U, size_, context);
        return data_[position_++];
    }

    std::uint16_t getU16(const char* context) {
        requireRange(position_, 2U, size_, context);
        const std::uint16_t value = readU16(data_ + position_);
        position_ += 2U;
        return value;
    }

    std::uint32_t getU32(const char* context) {
        requireRange(position_, 4U, size_, context);
        const std::uint32_t value = readU32(data_ + position_);
        position_ += 4U;
        return value;
    }

    std::string getLine(const char* context) {
        const std::size_t begin = position_;
        while (position_ < size_ && data_[position_] != '\n') {
            const std::uint8_t byte = data_[position_];
            if (byte < 0x21U || byte > 0x7eU)
                fail(std::string("non-ASCII text in pickle ") + context);
            ++position_;
            if (position_ - begin > 128U)
                fail(std::string("oversized pickle ") + context);
        }
        if (position_ == size_)
            fail(std::string("unterminated pickle ") + context);
        std::string line(reinterpret_cast<const char*>(data_ + begin), position_ - begin);
        ++position_;
        return line;
    }

    std::string getString() {
        const std::uint32_t length = getU32("BINUNICODE length");
        if (length > limits_.max_string_bytes)
            fail("pickle string exceeds the configured size limit");
        requireRange(position_, length, size_, "BINUNICODE payload");
        for (std::uint32_t index = 0; index < length; ++index) {
            const std::uint8_t byte = data_[position_ + index];
            if (byte < 0x20U || byte > 0x7eU)
                fail("checkpoint pickle strings must be printable ASCII");
        }
        std::string result(reinterpret_cast<const char*>(data_ + position_), length);
        position_ += length;
        return result;
    }

    void push(ValuePtr value) {
        if (stack_.size() >= limits_.max_pickle_stack)
            fail("pickle stack exceeds the configured limit");
        stack_.push_back(std::move(value));
    }

    ValuePtr pop(const char* context) {
        if (stack_.empty())
            fail(std::string("pickle stack underflow while ") + context);
        ValuePtr result = std::move(stack_.back());
        stack_.pop_back();
        return result;
    }

    void parseProtocol(std::size_t opcode_offset) {
        if (protocol_seen_ || opcode_offset != 0U || !stack_.empty() || !memo_.empty() ||
            !marks_.empty())
            fail("pickle PROTO must be the first opcode");
        if (getByte("PROTO version") != 2U)
            fail("checkpoint pickle must use protocol 2");
        protocol_seen_ = true;
    }

    void memoize(std::uint32_t index, bool long_form) {
        if (stack_.empty())
            fail("pickle memo write requires a stack value");
        if (memo_.size() >= limits_.max_pickle_memo)
            fail("pickle memo exceeds the configured limit");
        if (index != memo_.size())
            fail("pickle memo indices must be unique and consecutive");
        if ((long_form && index < 256U) || (!long_form && index >= 256U))
            fail("pickle memo index uses a non-canonical opcode");
        memo_.push_back(stack_.back());
    }

    void recall(std::uint32_t index, bool long_form) {
        if (index >= memo_.size())
            fail("pickle memo read references an undefined index");
        if ((long_form && index < 256U) || (!long_form && index >= 256U))
            fail("pickle memo read uses a non-canonical opcode");
        push(memo_[index]);
    }

    void parseGlobal() {
        const std::string module = getLine("GLOBAL module");
        const std::string name = getLine("GLOBAL name");
        if (module == "torch._utils" && name == "_rebuild_tensor_v2")
            push(makeGlobal(GlobalKind::kRebuildTensorV2));
        else if (module == "torch" && name == "FloatStorage")
            push(makeGlobal(GlobalKind::kFloatStorage));
        else if (module == "torch" && name == "LongStorage")
            push(makeGlobal(GlobalKind::kLongStorage));
        else if (module == "collections" && name == "OrderedDict")
            push(makeGlobal(GlobalKind::kOrderedDictionary));
        else
            fail("unsupported pickle GLOBAL '" + module + " " + name + "'");
    }

    void parsePersistentId() {
        const ValuePtr persistent_id = pop("processing BINPERSID");
        const auto& fields = requireTuple(persistent_id, "persistent storage id");
        if (fields.size() != 5U || requireString(fields[0], "persistent id tag") != "storage")
            fail("persistent id is not the expected five-field storage tuple");
        if (fields[1]->kind != ValueKind::kGlobal ||
            (fields[1]->global != GlobalKind::kFloatStorage &&
             fields[1]->global != GlobalKind::kLongStorage))
            fail("persistent storage type must be FloatStorage or LongStorage");
        const std::string& key = requireString(fields[2], "persistent storage key");
        parseCanonicalDecimal(key, "persistent storage key");
        if (requireString(fields[3], "persistent storage location") != "cpu")
            fail("persistent storage location must be cpu");
        const std::int64_t elements = requireInteger(fields[4], "persistent storage size");
        if (elements <= 0)
            fail("persistent storage size must be positive");

        auto value = makeValue(ValueKind::kStorage);
        value->storage.dtype =
            fields[1]->global == GlobalKind::kFloatStorage ? DType::kFloat32 : DType::kInt64;
        value->storage.key = key;
        value->storage.elements = static_cast<std::uint64_t>(elements);
        push(std::move(value));
    }

    void buildMarkedTuple() {
        if (marks_.empty())
            fail("pickle TUPLE has no MARK");
        const std::size_t mark = marks_.back();
        marks_.pop_back();
        if (mark > stack_.size())
            fail("pickle MARK is outside the stack");
        std::vector<ValuePtr> values;
        values.reserve(stack_.size() - mark);
        for (std::size_t index = mark; index < stack_.size(); ++index)
            values.push_back(std::move(stack_[index]));
        stack_.resize(mark);
        push(makeTuple(std::move(values)));
    }

    void buildFixedTuple(std::size_t count) {
        if (stack_.size() < count)
            fail("pickle fixed tuple underflows the stack");
        std::vector<ValuePtr> values(count);
        for (std::size_t index = count; index > 0U; --index)
            values[index - 1U] = pop("building a fixed tuple");
        push(makeTuple(std::move(values)));
    }

    std::vector<std::int64_t> integerTuple(const ValuePtr& value, const char* context) {
        const auto& values = requireTuple(value, context);
        if (values.size() > limits_.max_dimensions)
            fail(std::string(context) + " exceeds the configured dimension limit");
        std::vector<std::int64_t> result;
        result.reserve(values.size());
        for (const ValuePtr& item : values)
            result.push_back(requireInteger(item, context));
        return result;
    }

    void reduce() {
        const ValuePtr arguments = pop("reading REDUCE arguments");
        const ValuePtr callable = pop("reading REDUCE callable");
        if (callable->kind != ValueKind::kGlobal)
            fail("pickle REDUCE callable is not an allowed GLOBAL");
        const auto& fields = requireTuple(arguments, "REDUCE arguments");
        if (callable->global == GlobalKind::kOrderedDictionary) {
            if (!fields.empty())
                fail("OrderedDict REDUCE arguments must be empty");
            push(makeValue(ValueKind::kOrderedDictionary));
            return;
        }
        if (callable->global != GlobalKind::kRebuildTensorV2)
            fail("storage GLOBAL cannot be invoked with REDUCE");
        if (fields.size() != 6U || fields[0]->kind != ValueKind::kStorage)
            fail("_rebuild_tensor_v2 must receive the expected six arguments");
        const std::int64_t storage_offset = requireInteger(fields[1], "tensor storage offset");
        if (storage_offset < 0)
            fail("tensor storage offset must be non-negative");
        std::vector<std::int64_t> shape = integerTuple(fields[2], "tensor shape");
        std::vector<std::int64_t> strides = integerTuple(fields[3], "tensor strides");
        if (shape.size() != strides.size())
            fail("tensor shape and stride ranks differ");
        for (const std::int64_t dimension : shape) {
            if (dimension < 0)
                fail("tensor dimensions must be non-negative");
        }
        for (const std::int64_t stride : strides) {
            if (stride < 0)
                fail("negative tensor strides are not supported");
        }
        if (fields[4]->kind != ValueKind::kBoolean || fields[4]->boolean)
            fail("checkpoint tensors must have requires_grad=false");
        if (fields[5]->kind != ValueKind::kOrderedDictionary)
            fail("checkpoint tensor hooks must be an empty OrderedDict");

        auto value = makeValue(ValueKind::kTensor);
        value->tensor.storage = fields[0]->storage;
        value->tensor.storage_offset = static_cast<std::uint64_t>(storage_offset);
        value->tensor.shape = std::move(shape);
        value->tensor.strides = std::move(strides);
        push(std::move(value));
    }

    static void insertDictionaryItem(const ValuePtr& dictionary, const ValuePtr& key,
                                     ValuePtr value) {
        if (dictionary->kind != ValueKind::kDictionary || dictionary->dictionary == nullptr)
            fail("pickle SETITEM target is not a dictionary");
        const std::string& string_key = requireString(key, "dictionary key");
        if (!dictionary->dictionary->keys.insert(string_key).second)
            fail("pickle dictionary contains duplicate key '" + string_key + "'");
        dictionary->dictionary->items.emplace_back(string_key, std::move(value));
    }

    void setItem() {
        const ValuePtr value = pop("reading SETITEM value");
        const ValuePtr key = pop("reading SETITEM key");
        if (stack_.empty())
            fail("pickle SETITEM has no dictionary");
        insertDictionaryItem(stack_.back(), key, value);
    }

    void setItems() {
        if (marks_.empty())
            fail("pickle SETITEMS has no MARK");
        const std::size_t mark = marks_.back();
        marks_.pop_back();
        if (mark == 0U || mark > stack_.size() || ((stack_.size() - mark) & 1U) != 0U)
            fail("pickle SETITEMS has malformed key/value pairs");
        const ValuePtr dictionary = stack_[mark - 1U];
        for (std::size_t index = mark; index < stack_.size(); index += 2U)
            insertDictionaryItem(dictionary, stack_[index], stack_[index + 1U]);
        stack_.resize(mark);
    }

    ValuePtr stop() {
        if (!protocol_seen_)
            fail("checkpoint pickle has no PROTO opcode");
        if (position_ != size_)
            fail("checkpoint pickle has trailing data after STOP");
        if (!marks_.empty() || stack_.size() != 1U)
            fail("checkpoint pickle STOP has an invalid stack state");
        return stack_.back();
    }

    const std::uint8_t* data_;
    std::size_t size_;
    const ReaderLimits& limits_;
    std::size_t position_{0};
    bool protocol_seen_{false};
    std::vector<ValuePtr> stack_;
    std::vector<std::size_t> marks_;
    std::vector<ValuePtr> memo_;
};

bool isContiguous(const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& strides,
                  std::uint64_t logical_elements) {
    if (logical_elements == 0U)
        return true;
    std::uint64_t expected_stride = 1U;
    for (std::size_t index = shape.size(); index > 0U; --index) {
        const std::uint64_t dimension = static_cast<std::uint64_t>(shape[index - 1U]);
        if (dimension > 1U && static_cast<std::uint64_t>(strides[index - 1U]) != expected_stride)
            return false;
        expected_stride = checkedMultiply(expected_stride, dimension, "checking contiguity");
    }
    return true;
}

struct StorageRecord {
    DType dtype{DType::kFloat32};
    std::uint64_t elements{0};
    std::size_t data_offset{0};
    std::size_t bytes{0};
};

const ValuePtr& requireModelDictionary(const ValuePtr& root) {
    if (root == nullptr || root->kind != ValueKind::kDictionary || root->dictionary == nullptr ||
        root->dictionary->items.size() != 1U || root->dictionary->items[0].first != "model")
        fail("checkpoint pickle root must be exactly {'model': state_dict}");
    const ValuePtr& model = root->dictionary->items[0].second;
    if (model == nullptr || model->kind != ValueKind::kDictionary || model->dictionary == nullptr)
        fail("checkpoint pickle model value must be a dictionary");
    return model;
}

} // namespace

struct CheckpointReader::Impl {
    OwnedFileSnapshot file;
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, std::size_t> tensor_indices;
    std::unordered_map<std::string, StorageRecord> storages;
};

const char* dtypeName(DType dtype) noexcept {
    switch (dtype) {
    case DType::kFloat32:
        return "float32";
    case DType::kInt64:
        return "int64";
    }
    return "unknown";
}

std::size_t elementSize(DType dtype) noexcept {
    switch (dtype) {
    case DType::kFloat32:
        return 4U;
    case DType::kInt64:
        return 8U;
    }
    return 0U;
}

std::string CheckpointReader::checkpointSha256(const std::filesystem::path& path,
                                               std::uint64_t max_archive_bytes) {
    OwnedFileSnapshot file = OwnedFileSnapshot::openReadOnly(path, max_archive_bytes);
    return sha256Hex(file.data(), file.size());
}

CheckpointReader CheckpointReader::open(const std::filesystem::path& path, ReaderLimits limits) {
    return open(path, kSupportedCheckpointSha256, limits);
}

CheckpointReader CheckpointReader::open(const std::filesystem::path& path,
                                        std::string_view expected_sha256, ReaderLimits limits) {
    if (limits.max_archive_bytes == 0U || limits.max_archive_members == 0U ||
        limits.max_pickle_bytes == 0U || limits.max_pickle_stack == 0U ||
        limits.max_pickle_memo == 0U || limits.max_string_bytes == 0U || limits.max_tensors == 0U ||
        limits.max_dimensions == 0U || limits.max_tensor_logical_bytes == 0U)
        fail("checkpoint reader limits must all be positive");
    if (!isCanonicalSha256(expected_sha256))
        fail("expected checkpoint SHA-256 must be 64 lowercase hexadecimal characters");

    OwnedFileSnapshot file = OwnedFileSnapshot::openReadOnly(path, limits.max_archive_bytes);
    const std::string actual_sha256 = sha256Hex(file.data(), file.size());
    if (actual_sha256 != expected_sha256)
        fail("checkpoint SHA-256 mismatch: expected " + std::string(expected_sha256) + ", got " +
             actual_sha256);
    const ZipArchive archive(file, limits);
    const ArchiveLayout layout = inspectArchiveLayout(archive, limits);
    PickleParser parser(archive.data(*layout.pickle), layout.pickle->size, limits);
    const ValuePtr root = parser.parse();
    const ValuePtr& model = requireModelDictionary(root);
    const auto& model_items = model->dictionary->items;
    if (model_items.empty() || model_items.size() > limits.max_tensors)
        fail("checkpoint tensor count is outside the configured limit");

    auto impl = std::make_unique<Impl>();
    impl->tensors.reserve(model_items.size());
    impl->tensor_indices.reserve(model_items.size());
    impl->storages.reserve(layout.storages.size());

    for (const auto& item : model_items) {
        const std::string& name = item.first;
        const ValuePtr& value = item.second;
        if (name.empty() || name.size() > limits.max_string_bytes)
            fail("checkpoint contains an invalid tensor name");
        if (value == nullptr || value->kind != ValueKind::kTensor)
            fail("state_dict value '" + name + "' is not a tensor");
        const TensorValue& tensor = value->tensor;
        const auto archive_storage = layout.storages.find(tensor.storage.key);
        if (archive_storage == layout.storages.end())
            fail("tensor '" + name + "' references missing storage '" + tensor.storage.key + "'");
        const ZipEntry& storage_entry = *archive_storage->second;
        const std::size_t item_size = elementSize(tensor.storage.dtype);
        const std::uint64_t expected_storage_bytes =
            checkedMultiply(tensor.storage.elements, item_size, "validating storage byte size");
        if (expected_storage_bytes != storage_entry.size)
            fail("storage '" + tensor.storage.key + "' byte size disagrees with its pickle");

        const auto existing_storage = impl->storages.find(tensor.storage.key);
        if (existing_storage == impl->storages.end()) {
            impl->storages.emplace(tensor.storage.key,
                                   StorageRecord{tensor.storage.dtype, tensor.storage.elements,
                                                 storage_entry.data_offset, storage_entry.size});
        } else if (existing_storage->second.dtype != tensor.storage.dtype ||
                   existing_storage->second.elements != tensor.storage.elements ||
                   existing_storage->second.data_offset != storage_entry.data_offset) {
            fail("shared storage '" + tensor.storage.key + "' has inconsistent pickle metadata");
        }

        std::uint64_t logical_elements = 1U;
        for (const std::int64_t dimension : tensor.shape)
            logical_elements =
                checkedMultiply(logical_elements, static_cast<std::uint64_t>(dimension),
                                "computing tensor element count");

        std::uint64_t maximum_delta = 0U;
        if (logical_elements != 0U) {
            for (std::size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
                const std::uint64_t extent = static_cast<std::uint64_t>(tensor.shape[dimension]);
                const std::uint64_t stride = static_cast<std::uint64_t>(tensor.strides[dimension]);
                maximum_delta = checkedAdd(
                    maximum_delta,
                    checkedMultiply(extent - 1U, stride, "computing tensor storage span"),
                    "computing tensor storage span");
            }
        }
        if (logical_elements == 0U) {
            if (tensor.storage_offset > tensor.storage.elements)
                fail("empty tensor '" + name + "' starts beyond its storage");
        } else {
            const std::uint64_t maximum_index = checkedAdd(tensor.storage_offset, maximum_delta,
                                                           "validating tensor storage bounds");
            if (maximum_index >= tensor.storage.elements)
                fail("tensor '" + name + "' addresses beyond its storage");
        }
        const std::uint64_t logical_bytes_u64 =
            checkedMultiply(logical_elements, item_size, "computing tensor logical bytes");
        if (logical_bytes_u64 > limits.max_tensor_logical_bytes)
            fail("tensor '" + name + "' exceeds the configured logical byte limit");
        const std::uint64_t span_elements =
            logical_elements == 0U ? 0U
                                   : checkedAdd(maximum_delta, 1U, "computing tensor backing span");
        const std::uint64_t span_bytes_u64 =
            checkedMultiply(span_elements, item_size, "computing tensor backing span");
        const std::uint64_t byte_offset = checkedMultiply(tensor.storage_offset, item_size,
                                                          "computing tensor storage byte offset");
        if (checkedAdd(byte_offset, span_bytes_u64, "validating tensor byte span") >
            storage_entry.size)
            fail("tensor '" + name + "' byte span exceeds its ZIP storage member");

        TensorInfo info;
        info.name = name;
        info.dtype = tensor.storage.dtype;
        info.storage_key = tensor.storage.key;
        info.storage_offset = tensor.storage_offset;
        info.storage_elements = tensor.storage.elements;
        info.shape = tensor.shape;
        info.strides = tensor.strides;
        info.logical_elements = logical_elements;
        info.logical_bytes = checkedSize(logical_bytes_u64, "storing tensor logical bytes");
        info.storage_span_bytes = checkedSize(span_bytes_u64, "storing tensor backing span");
        info.contiguous = isContiguous(info.shape, info.strides, logical_elements);
        const std::size_t tensor_index = impl->tensors.size();
        if (!impl->tensor_indices.emplace(info.name, tensor_index).second)
            fail("checkpoint contains duplicate tensor name '" + info.name + "'");
        impl->tensors.push_back(std::move(info));
    }

    if (impl->storages.size() != layout.storages.size())
        fail("checkpoint ZIP contains storages not referenced by the state_dict");
    impl->file = std::move(file);
    return CheckpointReader(std::move(impl));
}

CheckpointReader::CheckpointReader(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

CheckpointReader::~CheckpointReader() = default;
CheckpointReader::CheckpointReader(CheckpointReader&&) noexcept = default;
CheckpointReader& CheckpointReader::operator=(CheckpointReader&&) noexcept = default;

std::size_t CheckpointReader::tensorCount() const noexcept {
    return impl_ == nullptr ? 0U : impl_->tensors.size();
}

std::size_t CheckpointReader::storageCount() const noexcept {
    return impl_ == nullptr ? 0U : impl_->storages.size();
}

std::vector<std::string> CheckpointReader::tensorNames() const {
    if (impl_ == nullptr)
        fail("checkpoint reader has been moved from");
    std::vector<std::string> names;
    names.reserve(impl_->tensors.size());
    for (const TensorInfo& tensor : impl_->tensors)
        names.push_back(tensor.name);
    return names;
}

const TensorInfo& CheckpointReader::tensorInfo(std::string_view name) const {
    if (impl_ == nullptr)
        fail("checkpoint reader has been moved from");
    const auto found = impl_->tensor_indices.find(std::string(name));
    if (found == impl_->tensor_indices.end())
        fail("checkpoint tensor not found: '" + std::string(name) + "'");
    return impl_->tensors[found->second];
}

WeightView CheckpointReader::tensor(std::string_view name) const {
    const TensorInfo& info = tensorInfo(name);
    const StorageRecord& storage = impl_->storages.at(info.storage_key);
    const std::size_t byte_offset =
        checkedSize(checkedMultiply(info.storage_offset, elementSize(info.dtype),
                                    "computing tensor view byte offset"),
                    "computing tensor view byte offset");
    WeightView view;
    view.data = impl_->file.data() + storage.data_offset + byte_offset;
    view.bytes = info.logical_bytes;
    view.dtype = info.dtype;
    view.shape = info.shape;
    view.strides = info.strides;
    view.contiguous = info.contiguous;
    view.storage_span_bytes = info.storage_span_bytes;
    return view;
}

WeightView CheckpointReader::requireTensor(std::string_view name, DType dtype,
                                           const std::vector<std::int64_t>& shape) const {
    const TensorInfo& info = tensorInfo(name);
    if (info.dtype != dtype)
        fail("checkpoint tensor '" + std::string(name) + "' has dtype " + dtypeName(info.dtype) +
             ", expected " + dtypeName(dtype));
    if (info.shape != shape)
        fail("checkpoint tensor '" + std::string(name) + "' has an unexpected shape");
    return tensor(name);
}

WeightView CheckpointReader::requireTensor(std::string_view name, DType dtype,
                                           std::initializer_list<std::int64_t> shape) const {
    return requireTensor(name, dtype, std::vector<std::int64_t>(shape));
}

std::vector<std::uint8_t> CheckpointReader::copyTensor(std::string_view name) const {
    const TensorInfo& info = tensorInfo(name);
    const WeightView view = tensor(name);
    std::vector<std::uint8_t> result(info.logical_bytes);
    if (result.empty())
        return result;
    if (info.contiguous) {
        std::memcpy(result.data(), view.data, result.size());
        return result;
    }

    const std::size_t item_size = elementSize(info.dtype);
    const auto* source = static_cast<const std::uint8_t*>(view.data);
    for (std::uint64_t linear = 0; linear < info.logical_elements; ++linear) {
        std::uint64_t coordinate_source = linear;
        std::uint64_t storage_delta = 0U;
        for (std::size_t dimension = info.shape.size(); dimension > 0U; --dimension) {
            const std::uint64_t extent = static_cast<std::uint64_t>(info.shape[dimension - 1U]);
            const std::uint64_t coordinate = coordinate_source % extent;
            coordinate_source /= extent;
            storage_delta =
                checkedAdd(storage_delta,
                           checkedMultiply(coordinate,
                                           static_cast<std::uint64_t>(info.strides[dimension - 1U]),
                                           "gathering non-contiguous tensor"),
                           "gathering non-contiguous tensor");
        }
        const std::size_t source_offset = checkedSize(
            checkedMultiply(storage_delta, item_size, "gathering non-contiguous tensor"),
            "gathering non-contiguous tensor");
        const std::size_t destination_offset =
            checkedSize(checkedMultiply(linear, item_size, "gathering non-contiguous tensor"),
                        "gathering non-contiguous tensor");
        std::memcpy(result.data() + destination_offset, source + source_offset, item_size);
    }
    return result;
}

} // namespace trtmc::sam2::native
