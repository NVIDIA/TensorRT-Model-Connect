/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tools/sam2_native_builder/checkpoint_reader.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using trtmc::sam2::native::CheckpointError;
using trtmc::sam2::native::CheckpointReader;
using trtmc::sam2::native::DType;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

template <typename Function>
void expectError(Function&& function, std::string_view expected, const std::string& message) {
    try {
        function();
    } catch (const CheckpointError& error) {
        check(std::string_view(error.what()).find(expected) != std::string_view::npos,
              message + " (unexpected error: " + error.what() + ")");
        return;
    }
    check(false, message + " (no error)");
}

void appendU16(std::vector<std::uint8_t>& output, std::uint16_t value) {
    output.push_back(static_cast<std::uint8_t>(value));
    output.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void appendU32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32U; shift += 8U)
        output.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64U; shift += 8U)
        output.push_back(static_cast<std::uint8_t>(value >> shift));
}

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

std::uint32_t crc32(const std::vector<std::uint8_t>& bytes) {
    std::uint32_t crc = 0xffffffffU;
    for (const std::uint8_t byte : bytes)
        crc = crcTable()[(crc ^ byte) & 0xffU] ^ (crc >> 8U);
    return crc ^ 0xffffffffU;
}

void appendText(std::vector<std::uint8_t>& output, std::string_view text) {
    output.insert(output.end(), text.begin(), text.end());
}

void appendBinUnicode(std::vector<std::uint8_t>& output, std::string_view text) {
    output.push_back('X');
    appendU32(output, static_cast<std::uint32_t>(text.size()));
    appendText(output, text);
}

void appendGlobal(std::vector<std::uint8_t>& output, std::string_view module,
                  std::string_view name) {
    output.push_back('c');
    appendText(output, module);
    output.push_back('\n');
    appendText(output, name);
    output.push_back('\n');
}

void appendInteger(std::vector<std::uint8_t>& output, std::uint64_t value) {
    check(value <= UINT32_MAX, "synthetic pickle integer fits BININT");
    if (value <= UINT8_MAX) {
        output.push_back('K');
        output.push_back(static_cast<std::uint8_t>(value));
    } else if (value <= UINT16_MAX) {
        output.push_back('M');
        appendU16(output, static_cast<std::uint16_t>(value));
    } else {
        output.push_back('J');
        appendU32(output, static_cast<std::uint32_t>(value));
    }
}

struct TensorSpec {
    std::string name;
    std::string storage_key;
    std::string storage_global{"FloatStorage"};
    std::uint64_t storage_elements{0};
    std::uint64_t storage_offset{0};
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> strides;
};

void appendIntegerTuple(std::vector<std::uint8_t>& output,
                        const std::vector<std::uint64_t>& values) {
    output.push_back('(');
    for (const std::uint64_t value : values)
        appendInteger(output, value);
    output.push_back('t');
}

void appendTensor(std::vector<std::uint8_t>& output, const TensorSpec& tensor) {
    appendGlobal(output, "torch._utils", "_rebuild_tensor_v2");
    output.push_back('(');
    output.push_back('(');
    appendBinUnicode(output, "storage");
    appendGlobal(output, "torch", tensor.storage_global);
    appendBinUnicode(output, tensor.storage_key);
    appendBinUnicode(output, "cpu");
    appendInteger(output, tensor.storage_elements);
    output.push_back('t');
    output.push_back('Q');
    appendInteger(output, tensor.storage_offset);
    appendIntegerTuple(output, tensor.shape);
    appendIntegerTuple(output, tensor.strides);
    output.push_back(0x89U); // NEWFALSE
    appendGlobal(output, "collections", "OrderedDict");
    output.push_back(')');
    output.push_back('R');
    output.push_back('t');
    output.push_back('R');
}

std::vector<std::uint8_t> makePickle(const std::vector<TensorSpec>& tensors) {
    std::vector<std::uint8_t> output = {0x80U, 0x02U, static_cast<std::uint8_t>('}')};
    appendBinUnicode(output, "model");
    output.push_back('}');
    output.push_back('(');
    for (const TensorSpec& tensor : tensors) {
        appendBinUnicode(output, tensor.name);
        appendTensor(output, tensor);
    }
    output.push_back('u');
    output.push_back('s');
    output.push_back('.');
    return output;
}

template <typename T>
std::vector<std::uint8_t> rawBytes(const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::vector<std::uint8_t> output(values.size() * sizeof(T));
    if (!output.empty())
        std::memcpy(output.data(), values.data(), output.size());
    return output;
}

struct SourceEntry {
    std::string name;
    std::vector<std::uint8_t> data;
};

struct ZipOptions {
    std::uint16_t flags{0x0808U};
    std::uint16_t method{0U};
    bool local_method_drift{false};
    bool duplicate_storage{false};
    bool traversal_member{false};
    bool redundant_zip64{true};
    bool zip64_mismatch{false};
    bool zip64_entry_sentinel{false};
};

struct CentralEntry {
    SourceEntry source;
    std::uint32_t crc{0};
    std::uint32_t local_offset{0};
};

std::vector<std::uint8_t> makeArchive(const std::vector<std::uint8_t>& pickle,
                                      const std::vector<std::vector<std::uint8_t>>& storages,
                                      ZipOptions options = {}) {
    const std::string root = "synthetic_sam2";
    std::vector<SourceEntry> sources;
    sources.push_back({root + "/data.pkl", pickle});
    for (std::size_t index = 0; index < storages.size(); ++index)
        sources.push_back({root + "/data/" + std::to_string(index), storages[index]});
    sources.push_back({root + "/version", {'3', '\n'}});
    if (options.duplicate_storage)
        sources.push_back({root + "/data/0", storages.front()});
    if (options.traversal_member)
        sources.push_back({root + "/../escape", {0U}});

    std::vector<std::uint8_t> output;
    std::vector<CentralEntry> central;
    for (SourceEntry& source : sources) {
        check(source.data.size() <= UINT32_MAX && output.size() <= UINT32_MAX,
              "synthetic ZIP uses classic member sizes");
        CentralEntry record;
        record.source = std::move(source);
        record.crc = crc32(record.source.data);
        record.local_offset = static_cast<std::uint32_t>(output.size());
        appendU32(output, 0x04034b50U);
        appendU16(output, 0U);
        appendU16(output, options.flags);
        appendU16(output, options.local_method_drift ? 1U : options.method);
        appendU16(output, 0U);
        appendU16(output, 0U);
        appendU32(output, 0U);
        appendU32(output, 0U);
        appendU32(output, 0U);
        appendU16(output, static_cast<std::uint16_t>(record.source.name.size()));

        const std::size_t before_extra = output.size() + 2U + record.source.name.size();
        const std::size_t extra_body = (64U - ((before_extra + 4U) & 63U)) & 63U;
        const std::uint16_t extra_size = static_cast<std::uint16_t>(4U + extra_body);
        appendU16(output, extra_size);
        appendText(output, record.source.name);
        appendU16(output, 0x4246U);
        appendU16(output, static_cast<std::uint16_t>(extra_body));
        output.insert(output.end(), extra_body, static_cast<std::uint8_t>('Z'));
        check((output.size() & 63U) == 0U, "synthetic PyTorch member is aligned");
        output.insert(output.end(), record.source.data.begin(), record.source.data.end());
        appendU32(output, 0x08074b50U);
        appendU32(output, record.crc);
        appendU32(output, static_cast<std::uint32_t>(record.source.data.size()));
        appendU32(output, static_cast<std::uint32_t>(record.source.data.size()));
        central.push_back(std::move(record));
    }

    const std::uint32_t central_offset = static_cast<std::uint32_t>(output.size());
    for (std::size_t index = 0; index < central.size(); ++index) {
        const CentralEntry& record = central[index];
        appendU32(output, 0x02014b50U);
        appendU16(output, 0U);
        appendU16(output, 0U);
        appendU16(output, options.flags);
        appendU16(output, options.method);
        appendU16(output, 0U);
        appendU16(output, 0U);
        appendU32(output, record.crc);
        const std::uint32_t member_size = static_cast<std::uint32_t>(record.source.data.size());
        appendU32(output, options.zip64_entry_sentinel && index == 0U ? UINT32_MAX : member_size);
        appendU32(output, options.zip64_entry_sentinel && index == 0U ? UINT32_MAX : member_size);
        appendU16(output, static_cast<std::uint16_t>(record.source.name.size()));
        appendU16(output, 0U);
        appendU16(output, 0U);
        appendU16(output, 0U);
        appendU16(output, 0U);
        appendU32(output, 0U);
        appendU32(output, record.local_offset);
        appendText(output, record.source.name);
    }
    const std::uint32_t central_size = static_cast<std::uint32_t>(output.size()) - central_offset;
    check(central.size() <= UINT16_MAX, "synthetic ZIP member count fits classic EOCD");
    if (options.redundant_zip64) {
        const std::uint64_t zip64_offset = output.size();
        appendU32(output, 0x06064b50U);
        appendU64(output, 44U);
        appendU16(output, 45U);
        appendU16(output, 45U);
        appendU32(output, 0U);
        appendU32(output, 0U);
        appendU64(output, central.size());
        appendU64(output, central.size() + (options.zip64_mismatch ? 1U : 0U));
        appendU64(output, central_size);
        appendU64(output, central_offset);
        appendU32(output, 0x07064b50U);
        appendU32(output, 0U);
        appendU64(output, zip64_offset);
        appendU32(output, 1U);
    }
    appendU32(output, 0x06054b50U);
    appendU16(output, 0U);
    appendU16(output, 0U);
    appendU16(output, static_cast<std::uint16_t>(central.size()));
    appendU16(output, static_cast<std::uint16_t>(central.size()));
    appendU32(output, central_size);
    appendU32(output, central_offset);
    appendU16(output, 0U);
    return output;
}

class TempArchive final {
  public:
    explicit TempArchive(const std::vector<std::uint8_t>& bytes) {
        static unsigned int sequence = 0;
        path_ = std::filesystem::temp_directory_path() /
                ("sam2_checkpoint_reader_" + std::to_string(::getpid()) + "_" +
                 std::to_string(sequence++) + ".pt");
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        check(static_cast<bool>(output), "open synthetic checkpoint for writing");
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        output.close();
        check(static_cast<bool>(output), "write synthetic checkpoint");
    }

    ~TempArchive() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept { return path_; }

  private:
    std::filesystem::path path_;
};

CheckpointReader openSynthetic(const TempArchive& archive) {
    return CheckpointReader::open(archive.path(),
                                  CheckpointReader::checkpointSha256(archive.path()));
}

std::vector<TensorSpec> validTensorSpecs() {
    return {{"weight", "0", "FloatStorage", 6U, 0U, {2U, 2U}, {3U, 1U}},
            {"index", "1", "LongStorage", 2U, 0U, {2U}, {1U}}};
}

std::vector<std::vector<std::uint8_t>> validStorages() {
    return {rawBytes<float>({0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F}),
            rawBytes<std::int64_t>({11, 22})};
}

void testRestrictedReaderAndStridedCopy() {
    const TempArchive archive(makeArchive(makePickle(validTensorSpecs()), validStorages()));
    CheckpointReader reader = openSynthetic(archive);
    check(reader.tensorCount() == 2U && reader.storageCount() == 2U,
          "reader reports synthetic tensor and storage counts");
    check(reader.tensorNames() == std::vector<std::string>({"weight", "index"}),
          "reader preserves exact state_dict names and order");

    const auto& info = reader.tensorInfo("weight");
    check(info.dtype == DType::kFloat32 && info.storage_key == "0" && info.storage_offset == 0U &&
              info.storage_elements == 6U && info.shape == std::vector<std::int64_t>({2, 2}) &&
              info.strides == std::vector<std::int64_t>({3, 1}) && info.logical_bytes == 16U &&
              info.storage_span_bytes == 20U && !info.contiguous,
          "reader exposes complete bounded non-contiguous tensor metadata");
    const auto view = reader.requireTensor("weight", DType::kFloat32, {2, 2});
    check(view.data != nullptr && view.bytes == 16U && view.storage_span_bytes == 20U &&
              !view.contiguous,
          "reader returns a zero-copy bounded strided view");
    const auto copied = reader.copyTensor("weight");
    check(copied.size() == 16U, "strided copy has tightly packed logical size");
    std::array<float, 4> gathered{};
    std::memcpy(gathered.data(), copied.data(), copied.size());
    check(gathered == std::array<float, 4>({0.0F, 1.0F, 3.0F, 4.0F}),
          "strided copy gathers logical elements without copying storage holes");

    const auto integers = reader.requireTensor("index", DType::kInt64, {2});
    check(integers.contiguous && integers.bytes == 16U && integers.storage_span_bytes == 16U,
          "reader exposes an int64 contiguous zero-copy view");
    expectError([&] { reader.requireTensor("weight", DType::kInt64, {2, 2}); }, "dtype",
                "requireTensor rejects dtype drift");
    expectError([&] { reader.requireTensor("weight", DType::kFloat32, {4}); }, "shape",
                "requireTensor rejects shape drift");
    expectError([&] { reader.tensor("missing"); }, "not found",
                "named lookup rejects absent tensors");
}

void testTensorViewsOwnAuthenticatedSnapshot() {
    const TempArchive archive(makeArchive(makePickle(validTensorSpecs()), validStorages()));
    CheckpointReader reader = openSynthetic(archive);
    const auto expected_weight = reader.copyTensor("weight");
    const auto expected_index = reader.copyTensor("index");

    check(::truncate(archive.path().c_str(), 0) == 0,
          "truncate synthetic checkpoint after authenticated open");

    check(reader.copyTensor("weight") == expected_weight &&
              reader.copyTensor("index") == expected_index,
          "authenticated tensor views remain backed by an immutable owned snapshot");
    const auto weight = reader.requireTensor("weight", DType::kFloat32, {2, 2});
    std::array<float, 4> gathered{};
    const auto copied = reader.copyTensor("weight");
    std::memcpy(gathered.data(), copied.data(), copied.size());
    check(weight.data != nullptr && gathered == std::array<float, 4>({0.0F, 1.0F, 3.0F, 4.0F}),
          "borrowed weight pointers cannot observe same-inode truncation after authentication");
}

void testShaAuthentication() {
    const TempArchive archive(makeArchive(makePickle(validTensorSpecs()), validStorages()));
    const std::string digest = CheckpointReader::checkpointSha256(archive.path());
    check(digest.size() == 64U, "SHA helper returns a canonical digest");
    std::string wrong_digest(64U, '0');
    if (wrong_digest == digest)
        wrong_digest[0] = '1';
    expectError([&] { CheckpointReader::open(archive.path(), wrong_digest); }, "SHA-256 mismatch",
                "reader authenticates bytes before ZIP parsing");
    expectError([&] { CheckpointReader::open(archive.path(), "not-a-sha"); },
                "64 lowercase hexadecimal", "reader rejects ambiguous expected digests");
}

void testZipEnvelopeRejections() {
    const auto pickle = makePickle(validTensorSpecs());
    const auto storages = validStorages();
    auto reject = [&](ZipOptions options, std::string_view expected, const std::string& message) {
        const TempArchive archive(makeArchive(pickle, storages, options));
        expectError([&] { openSynthetic(archive); }, expected, message);
    };

    ZipOptions compressed;
    compressed.method = 8U;
    reject(compressed, "compressed", "reader rejects compressed members");

    ZipOptions encrypted;
    encrypted.flags = 0x0809U;
    reject(encrypted, "encrypted", "reader rejects encrypted members");

    ZipOptions duplicate;
    duplicate.duplicate_storage = true;
    reject(duplicate, "duplicate", "reader rejects duplicate archive names");

    ZipOptions traversal;
    traversal.traversal_member = true;
    reject(traversal, "unsafe", "reader rejects archive path traversal");

    ZipOptions local_drift;
    local_drift.local_method_drift = true;
    reject(local_drift, "central/local metadata drift",
           "reader rejects local and central record drift");

    ZipOptions zip64_mismatch;
    zip64_mismatch.zip64_mismatch = true;
    reject(zip64_mismatch, "ZIP64 and classic", "reader rejects inconsistent ZIP64 metadata");

    ZipOptions substantive_zip64;
    substantive_zip64.zip64_entry_sentinel = true;
    reject(substantive_zip64, "ZIP64", "reader rejects substantive ZIP64 member sizes");

    auto corrupt_payload = makeArchive(pickle, storages);
    check(corrupt_payload.size() > 64U, "synthetic ZIP has a first aligned payload");
    corrupt_payload[64] ^= 0x01U;
    const TempArchive corrupt_archive(corrupt_payload);
    expectError([&] { openSynthetic(corrupt_archive); }, "CRC mismatch",
                "reader rejects member payload drift after SHA authentication");
}

void testPickleAndStorageRejections() {
    const auto storages = validStorages();

    auto wrong_protocol = makePickle(validTensorSpecs());
    wrong_protocol[1] = 1U;
    {
        const TempArchive archive(makeArchive(wrong_protocol, storages));
        expectError([&] { openSynthetic(archive); }, "protocol 2",
                    "reader rejects pickle protocol drift");
    }

    auto unsupported_opcode = makePickle(validTensorSpecs());
    unsupported_opcode[2] = 'N';
    {
        const TempArchive archive(makeArchive(unsupported_opcode, storages));
        expectError([&] { openSynthetic(archive); }, "unsupported pickle opcode",
                    "reader rejects pickle opcodes outside the observed grammar");
    }

    auto unsupported_global = validTensorSpecs();
    unsupported_global[0].storage_global = "DoubleStorage";
    {
        const TempArchive archive(makeArchive(makePickle(unsupported_global), storages));
        expectError([&] { openSynthetic(archive); }, "unsupported pickle GLOBAL",
                    "reader rejects pickle globals outside the observed allowlist");
    }

    auto out_of_bounds = validTensorSpecs();
    out_of_bounds[0].strides = {6U, 1U};
    {
        const TempArchive archive(makeArchive(makePickle(out_of_bounds), storages));
        expectError([&] { openSynthetic(archive); }, "beyond its storage",
                    "reader checks full non-contiguous tensor bounds");
    }

    auto short_storages = storages;
    short_storages[0].resize(short_storages[0].size() - sizeof(float));
    {
        const TempArchive archive(makeArchive(makePickle(validTensorSpecs()), short_storages));
        expectError([&] { openSynthetic(archive); }, "byte size disagrees",
                    "reader rejects pickle/storage byte-size drift");
    }

    auto duplicate_names = validTensorSpecs();
    duplicate_names[1].name = duplicate_names[0].name;
    {
        const TempArchive archive(makeArchive(makePickle(duplicate_names), storages));
        expectError([&] { openSynthetic(archive); }, "duplicate key",
                    "reader rejects duplicate state_dict tensor names");
    }
}

void testOptionalDeliveredCheckpoint() {
    const char* path = std::getenv("TRTMC_SAM2_CHECKPOINT");
    if (path == nullptr || *path == '\0')
        path = std::getenv("SAM2_CHECKPOINT");
    if (path == nullptr || *path == '\0') {
        std::cout << "SKIP: set TRTMC_SAM2_CHECKPOINT for delivered-checkpoint validation\n";
        return;
    }

    CheckpointReader reader = CheckpointReader::open(path);
    check(reader.tensorCount() == 603U, "delivered checkpoint contains exactly 603 tensors");
    check(reader.storageCount() == 595U, "delivered checkpoint contains exactly 595 storages");
    std::size_t float_tensors = 0;
    std::size_t long_tensors = 0;
    for (const std::string& name : reader.tensorNames()) {
        const auto& tensor = reader.tensorInfo(name);
        if (tensor.dtype == DType::kFloat32)
            ++float_tensors;
        else if (tensor.dtype == DType::kInt64)
            ++long_tensors;
    }
    check(float_tensors == 591U && long_tensors == 12U,
          "delivered checkpoint contains exactly 591 FP32 and 12 int64 tensors");
    const auto positional =
        reader.requireTensor("maskmem_tpos_enc", DType::kFloat32, {7, 1, 1, 64});
    check(positional.contiguous && positional.bytes == 1792U,
          "delivered checkpoint exposes the expected leading SAM2 tensor");
}

} // namespace

int main() {
    testRestrictedReaderAndStridedCopy();
    testTensorViewsOwnAuthenticatedSnapshot();
    testShaAuthentication();
    testZipEnvelopeRejections();
    testPickleAndStorageRejections();
    testOptionalDeliveredCheckpoint();
    std::cout << "PASS: strict native SAM2 checkpoint reader\n";
    return 0;
}
