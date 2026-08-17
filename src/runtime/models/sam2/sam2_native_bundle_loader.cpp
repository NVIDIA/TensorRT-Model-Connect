/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_native_bundle_loader.h"

#include "bundle/bundle_format.h"
#include "runtime/models/sam2/sam2_engine_contract.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace trtmc::sam2 {

namespace {

using Json = nlohmann::json;
using ContractList = std::vector<TensorContract>;

constexpr std::string_view kModelId = "sam2.1-hiera-small-bbox";
constexpr std::string_view kModelType = "sam2_video_tracking";
constexpr std::string_view kFamily = "sam2";
constexpr std::string_view kPrecision = "mixed_bf16_fp32";
constexpr std::size_t kMaximumHeaderBytes = 100U * 1024U * 1024U;
constexpr std::size_t kMaximumConfigBytes = 1U * 1024U * 1024U;
constexpr std::size_t kMaximumReceiptBytes = 4U * 1024U * 1024U;
constexpr std::uint64_t kMaximumBundleBytes = UINT64_C(2) * 1024U * 1024U * 1024U;

struct RawHeader {
    std::string text;
    std::uint64_t file_size{0};
};

[[noreturn]] void fail(const std::string& message) {
    throw NativeBundleLoadError(message);
}

#if defined(__linux__)

class ScopedDescriptor final {
  public:
    explicit ScopedDescriptor(int value = -1) noexcept : value_(value) {}
    ~ScopedDescriptor() {
        if (value_ >= 0)
            ::close(value_);
    }

    ScopedDescriptor(const ScopedDescriptor&) = delete;
    ScopedDescriptor& operator=(const ScopedDescriptor&) = delete;

    int get() const noexcept { return value_; }
    void reset(int value) noexcept {
        if (value_ >= 0)
            ::close(value_);
        value_ = value;
    }

  private:
    int value_{-1};
};

bool sameSnapshotSource(const struct stat& before, const struct stat& after) {
    return before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
           before.st_mode == after.st_mode && before.st_size == after.st_size &&
           before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
           before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
           before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
           before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
}

std::string copySnapshotBytes(int source, int destination, std::uint64_t size) {
    trtmc::internal::Sha256 hash;
    std::array<char, 1024U * 1024U> buffer{};
    std::uint64_t offset = 0;
    while (offset != size) {
        const auto count =
            static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), size - offset));
        ssize_t read_count = -1;
        do {
            read_count = ::pread(source, buffer.data(), count, static_cast<off_t>(offset));
        } while (read_count < 0 && errno == EINTR);
        if (read_count <= 0 || static_cast<std::size_t>(read_count) != count)
            fail("failed to read a stable SAM2 native bundle snapshot");
        hash.update(buffer.data(), static_cast<std::size_t>(read_count));

        std::size_t written = 0;
        while (written != count) {
            ssize_t write_count = -1;
            do {
                write_count = ::pwrite(destination, buffer.data() + written, count - written,
                                       static_cast<off_t>(offset + written));
            } while (write_count < 0 && errno == EINTR);
            if (write_count <= 0)
                fail("failed to write the SAM2 native bundle snapshot");
            written += static_cast<std::size_t>(write_count);
        }
        offset += count;
    }
    return hash.hex_digest();
}

class SealedBundleSnapshot final {
  public:
    SealedBundleSnapshot(const std::string& path, std::string_view expected_sha256) {
        int source_flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
        source_flags |= O_NOFOLLOW;
#endif
        const ScopedDescriptor source(::open(path.c_str(), source_flags));
        if (source.get() < 0)
            fail("failed to open the SAM2 native bundle snapshot source");

        struct stat before{};
        if (::fstat(source.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
            before.st_size < static_cast<off_t>(kBundleHeaderOffset) ||
            static_cast<std::uint64_t>(before.st_size) > kMaximumBundleBytes) {
            fail("SAM2 native bundle snapshot source is not a supported regular file");
        }
        const auto size = static_cast<std::uint64_t>(before.st_size);
        size_ = size;

        const int descriptor = static_cast<int>(
            ::syscall(SYS_memfd_create, "trtmc-sam2-bundle", MFD_CLOEXEC | MFD_ALLOW_SEALING));
        if (descriptor < 0)
            fail("failed to create a sealable SAM2 native bundle snapshot");
        descriptor_.reset(descriptor);
        if (::ftruncate(descriptor_.get(), static_cast<off_t>(size)) != 0)
            fail("failed to size the SAM2 native bundle snapshot");
        sha256_ = copySnapshotBytes(source.get(), descriptor_.get(), size);

        struct stat after{};
        if (::fstat(source.get(), &after) != 0 || !sameSnapshotSource(before, after))
            fail("SAM2 native bundle changed while its snapshot was captured");

        constexpr int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
        if (::fcntl(descriptor_.get(), F_ADD_SEALS, seals) != 0)
            fail("failed to seal the SAM2 native bundle snapshot");
        const int applied_seals = ::fcntl(descriptor_.get(), F_GET_SEALS);
        if (applied_seals < 0 || (applied_seals & seals) != seals) {
            fail("failed to seal the SAM2 native bundle snapshot");
        }
        if (!expected_sha256.empty() && sha256_ != expected_sha256)
            fail("SAM2 native bundle sealed snapshot full SHA-256 mismatch");
        path_ = "/proc/self/fd/" + std::to_string(descriptor_.get());
    }

    const std::string& path() const noexcept { return path_; }
    const std::string& sha256() const noexcept { return sha256_; }
    std::uint64_t size() const noexcept { return size_; }

  private:
    ScopedDescriptor descriptor_;
    std::string path_;
    std::string sha256_;
    std::uint64_t size_{0U};
};

#else

class SealedBundleSnapshot final {
  public:
    SealedBundleSnapshot(const std::string&, std::string_view) {
        fail("SAM2 native bundle immutable snapshots require Linux");
    }
    const std::string& path() const noexcept { return path_; }
    const std::string& sha256() const noexcept { return sha256_; }
    std::uint64_t size() const noexcept { return size_; }

  private:
    std::string path_;
    std::string sha256_;
    std::uint64_t size_{0U};
};

#endif

std::vector<std::string_view> requiredSectionNames() {
    std::vector<std::string_view> result;
    result.reserve(kRequiredPlanSections.size() + 2U);
    result.insert(result.end(), kRequiredPlanSections.begin(), kRequiredPlanSections.end());
    result.push_back(kConfigSection);
    result.push_back(kBuildReceiptSection);
    return result;
}

std::uint64_t readU64LittleEndian(std::istream& input) {
    std::array<unsigned char, 8> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input)
        fail("SAM2 native bundle header length is truncated");
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index)
        result |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
    return result;
}

RawHeader readRawHeader(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        fail("failed to open SAM2 native bundle header");
    const auto end = input.tellg();
    if (end < 0)
        fail("failed to determine SAM2 native bundle size");
    const auto file_size = static_cast<std::uint64_t>(end);
    input.seekg(0);

    std::array<unsigned char, 8> magic{};
    input.read(reinterpret_cast<char*>(magic.data()), static_cast<std::streamsize>(magic.size()));
    if (!input || std::memcmp(magic.data(), kBundleMagic, magic.size()) != 0)
        fail("SAM2 native bundle magic is invalid");
    const auto header_size = readU64LittleEndian(input);
    if (header_size > kMaximumHeaderBytes ||
        header_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        header_size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
        header_size > file_size - std::min<std::uint64_t>(file_size, kBundleHeaderOffset)) {
        fail("SAM2 native bundle header size is invalid");
    }
    std::string text(static_cast<std::size_t>(header_size), '\0');
    input.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!input)
        fail("SAM2 native bundle header is truncated");
    return {std::move(text), file_size};
}

Json parseStrictJson(const std::string& text, std::size_t maximum_size, std::string_view context) {
    if (text.empty() || text.size() > maximum_size)
        fail("SAM2 " + std::string(context) + " size is invalid");
    std::vector<std::unordered_set<std::string>> object_keys;
    Json::parser_callback_t callback = [&](int, Json::parse_event_t event, Json& value) {
        if (event == Json::parse_event_t::object_start)
            object_keys.emplace_back();
        if (event == Json::parse_event_t::key) {
            const auto key = value.get<std::string>();
            if (object_keys.empty() || !object_keys.back().insert(key).second)
                fail("SAM2 " + std::string(context) + " contains duplicate key: " + key);
        }
        if (event == Json::parse_event_t::object_end) {
            if (object_keys.empty())
                fail("SAM2 " + std::string(context) + " object nesting is invalid");
            object_keys.pop_back();
        }
        return true;
    };
    try {
        return Json::parse(text, callback);
    } catch (const NativeBundleLoadError&) {
        throw;
    } catch (const Json::exception& error) {
        fail("SAM2 " + std::string(context) + " is invalid JSON: " + error.what());
    }
}

void requireExactKeys(const Json& object, std::initializer_list<std::string_view> expected,
                      std::string_view context) {
    if (!object.is_object())
        fail("SAM2 " + std::string(context) + " must be an object");
    std::set<std::string> expected_keys;
    for (const auto key : expected)
        expected_keys.emplace(key);
    std::set<std::string> actual_keys;
    for (auto item = object.begin(); item != object.end(); ++item)
        actual_keys.insert(item.key());
    if (actual_keys != expected_keys)
        fail("SAM2 " + std::string(context) + " field set drifted");
}

const Json& requireObject(const Json& object, std::string_view field, std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || !found->is_object())
        fail("SAM2 " + std::string(context) + " requires object field " + std::string(field));
    return *found;
}

const Json& requireArray(const Json& object, std::string_view field, std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || !found->is_array())
        fail("SAM2 " + std::string(context) + " requires array field " + std::string(field));
    return *found;
}

std::string requireString(const Json& object, std::string_view field, std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || !found->is_string())
        fail("SAM2 " + std::string(context) + " requires string field " + std::string(field));
    const auto result = found->get<std::string>();
    if (result.empty())
        fail("SAM2 " + std::string(context) + " contains empty field " + std::string(field));
    return result;
}

bool requireBool(const Json& object, std::string_view field, std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || !found->is_boolean())
        fail("SAM2 " + std::string(context) + " requires boolean field " + std::string(field));
    return found->get<bool>();
}

std::uint64_t requireUint64(const Json& object, std::string_view field, std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || (!found->is_number_unsigned() && !found->is_number_integer())) {
        fail("SAM2 " + std::string(context) + " requires integer field " + std::string(field));
    }
    if (found->is_number_integer() && found->get<std::int64_t>() < 0)
        fail("SAM2 " + std::string(context) + " contains negative field " + std::string(field));
    try {
        return found->get<std::uint64_t>();
    } catch (const Json::exception&) {
        fail("SAM2 " + std::string(context) +
             " integer field is out of range: " + std::string(field));
    }
}

std::int32_t requireInt32(const Json& object, std::string_view field, std::string_view context) {
    const auto value = requireUint64(object, field, context);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
        fail("SAM2 " + std::string(context) +
             " integer field is out of range: " + std::string(field));
    return static_cast<std::int32_t>(value);
}

void requireStringValue(const Json& object, std::string_view field, std::string_view expected,
                        std::string_view context) {
    if (requireString(object, field, context) != expected)
        fail("SAM2 " + std::string(context) + " field mismatch: " + std::string(field));
}

void requireIntValue(const Json& object, std::string_view field, std::int32_t expected,
                     std::string_view context) {
    if (requireInt32(object, field, context) != expected)
        fail("SAM2 " + std::string(context) + " field mismatch: " + std::string(field));
}

bool isLowercaseSha256(std::string_view value) {
    return value.size() == 64U && std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

std::string requireSha256(const Json& object, std::string_view field, std::string_view context) {
    const auto value = requireString(object, field, context);
    if (!isLowercaseSha256(value))
        fail("SAM2 " + std::string(context) + " SHA-256 field is invalid: " + std::string(field));
    return value;
}

std::string sha256(const void* data, std::size_t size) {
    trtmc::internal::Sha256 hash;
    hash.update(data, size);
    return hash.hex_digest();
}

bool isCanonicalUtcTimestamp(std::string_view value) {
    if (value.size() != 20U || value[4] != '-' || value[7] != '-' || value[10] != 'T' ||
        value[13] != ':' || value[16] != ':' || value[19] != 'Z') {
        return false;
    }
    constexpr std::array<std::size_t, 14> digits = {0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18};
    return std::all_of(digits.begin(), digits.end(), [value](std::size_t index) {
        return std::isdigit(static_cast<unsigned char>(value[index])) != 0;
    });
}

std::string sectionText(const BundleSection& section, std::size_t limit, std::string_view context) {
    if (section.data.empty() || section.data.size() > limit)
        fail("SAM2 " + std::string(context) + " section size is invalid");
    return std::string(section.data.begin(), section.data.end());
}

void validateRuntimeTarget(const NativeBundleRuntimeTarget& target) {
    if (target.tensorrt_version != kTargetTensorRtVersion ||
        target.tensorrt_abi != kTargetTensorRtAbi || target.gpu_name != kTargetGpuName ||
        target.compute_capability != kTargetComputeCapability) {
        fail("SAM2 native bundle runtime target does not match the pinned TRT 11.1 L4 target");
    }
}

void validateHeaderIdentity(const Json& header, const BundleFile& bundle) {
    requireExactKeys(header,
                     {"model_id", "model_type", "family", "precision", "trt_version", "trt_abi",
                      "gpu_name", "created_at", "runtime_strategy", "sections"},
                     "bundle header");
    requireStringValue(header, "model_id", kModelId, "bundle header");
    requireStringValue(header, "model_type", kModelType, "bundle header");
    requireStringValue(header, "family", kFamily, "bundle header");
    requireStringValue(header, "precision", kPrecision, "bundle header");
    requireStringValue(header, "runtime_strategy", kStrategyName, "bundle header");
    requireStringValue(header, "trt_version", kTargetTensorRtVersion, "bundle header");
    requireStringValue(header, "trt_abi", kTargetTensorRtAbi, "bundle header");
    requireStringValue(header, "gpu_name", kTargetGpuName, "bundle header");
    const auto created_at = requireString(header, "created_at", "bundle header");
    if (!isCanonicalUtcTimestamp(created_at))
        fail("SAM2 bundle header created_at is not canonical UTC");

    const auto& info = bundle.info;
    if (info.model_id != kModelId || info.model_type != kModelType || info.family != kFamily ||
        info.precision != kPrecision || info.runtime_strategy != kStrategyName ||
        info.trt_version != kTargetTensorRtVersion || info.trt_abi != kTargetTensorRtAbi ||
        info.gpu_name != kTargetGpuName || info.created_at != created_at) {
        fail("SAM2 generic bundle metadata does not match the authenticated header");
    }
}

struct AuthenticatedSections {
    std::array<const BundleSection*, 6> plans{};
    const BundleSection* config{nullptr};
    const BundleSection* receipt{nullptr};
    std::array<std::string, 8> digests{};
};

AuthenticatedSections authenticateSections(const Json& header, const RawHeader& raw,
                                           const BundleFile& bundle) {
    const auto names = requiredSectionNames();
    if (bundle.sections.size() != names.size() || bundle.info.sections.size() != names.size())
        fail("SAM2 native bundle requires exactly six plans, config, and receipt");
    const auto& header_sections = requireObject(header, "sections", "bundle header");
    if (header_sections.size() != names.size())
        fail("SAM2 native bundle header section count drifted");

    std::set<std::string> expected_names;
    for (const auto name : names)
        expected_names.emplace(name);
    std::set<std::string> actual_names;
    for (auto item = header_sections.begin(); item != header_sections.end(); ++item)
        actual_names.insert(item.key());
    if (actual_names != expected_names)
        fail("SAM2 native bundle header section names drifted");

    AuthenticatedSections result;
    std::uint64_t canonical_offset = 0;
    for (std::size_t index = 0; index < names.size(); ++index) {
        const std::string expected_name(names[index]);
        const auto& generic_info = bundle.info.sections[index];
        const auto& loaded = bundle.sections[index];
        if (generic_info.name != expected_name || loaded.name != expected_name)
            fail("SAM2 native bundle section order drifted");
        const auto found = header_sections.find(expected_name);
        if (found == header_sections.end())
            fail("SAM2 native bundle header is missing a required section");
        requireExactKeys(*found, {"offset", "size", "sha256"}, "bundle section metadata");
        const auto offset = requireUint64(*found, "offset", "bundle section metadata");
        const auto size = requireUint64(*found, "size", "bundle section metadata");
        const auto expected_digest = requireSha256(*found, "sha256", "bundle section metadata");
        if (offset != canonical_offset || generic_info.offset != offset ||
            generic_info.size != size || size == 0U || size != loaded.data.size()) {
            fail("SAM2 native bundle section offset or size drifted: " + expected_name);
        }
        const auto actual_digest = sha256(loaded.data.data(), loaded.data.size());
        if (actual_digest != expected_digest)
            fail("SAM2 native bundle section SHA-256 mismatch: " + expected_name);
        if (size > std::numeric_limits<std::uint64_t>::max() - canonical_offset)
            fail("SAM2 native bundle payload size overflowed");
        canonical_offset += size;
        result.digests[index] = actual_digest;
        if (index < result.plans.size())
            result.plans[index] = &loaded;
        else if (index == result.plans.size())
            result.config = &loaded;
        else
            result.receipt = &loaded;
    }
    if (canonical_offset > std::numeric_limits<std::uint64_t>::max() - kBundleHeaderOffset ||
        raw.text.size() >
            std::numeric_limits<std::uint64_t>::max() - kBundleHeaderOffset - canonical_offset ||
        kBundleHeaderOffset + static_cast<std::uint64_t>(raw.text.size()) + canonical_offset !=
            raw.file_size) {
        fail("SAM2 native bundle has trailing, missing, or unbound payload bytes");
    }
    return result;
}

struct Qualification {
    std::string state;
    bool runtime_eligible{false};
    bool golden_parity_verified{false};
};

Qualification validateConfig(const Json& config) {
    requireExactKeys(config,
                     {"schema_version", "family", "model_id", "engine_contract_version",
                      "runtime_strategy", "precision", "checkpoint_sha256", "source_config_sha256",
                      "golden_manifest_sha256", "frame_count", "selected_object_count",
                      "model_image_size", "original_image_height", "original_image_width",
                      "plan_sections", "qualification", "runtime_eligible"},
                     "embedded config");
    requireIntValue(config, "schema_version", 1, "embedded config");
    requireStringValue(config, "family", kFamily, "embedded config");
    requireStringValue(config, "model_id", kModelId, "embedded config");
    requireIntValue(config, "engine_contract_version",
                    static_cast<std::int32_t>(kEngineContractVersion), "embedded config");
    requireStringValue(config, "runtime_strategy", kStrategyName, "embedded config");
    requireStringValue(config, "precision", kPrecision, "embedded config");
    requireStringValue(config, "checkpoint_sha256", kCheckpointSha256, "embedded config");
    requireStringValue(config, "source_config_sha256", kConfigSha256, "embedded config");
    requireStringValue(config, "golden_manifest_sha256", kGoldenManifestSha256, "embedded config");
    requireIntValue(config, "frame_count", kFrameCount, "embedded config");
    requireIntValue(config, "selected_object_count", kSelectedObjectCount, "embedded config");
    requireIntValue(config, "model_image_size", kModelImageSize, "embedded config");
    requireIntValue(config, "original_image_height", kOriginalImageHeight, "embedded config");
    requireIntValue(config, "original_image_width", kOriginalImageWidth, "embedded config");

    const auto& plan_sections = requireArray(config, "plan_sections", "embedded config");
    if (plan_sections.size() != kRequiredPlanSections.size())
        fail("SAM2 embedded config plan section count drifted");
    for (std::size_t index = 0; index < plan_sections.size(); ++index) {
        if (!plan_sections[index].is_string() ||
            plan_sections[index].get<std::string>() != kRequiredPlanSections[index]) {
            fail("SAM2 embedded config plan section order drifted");
        }
    }

    Qualification result;
    result.state = requireString(config, "qualification", "embedded config");
    result.runtime_eligible = requireBool(config, "runtime_eligible", "embedded config");
    if (result.state != "unqualified" || result.runtime_eligible)
        fail("SAM2 native bundle requires exact unqualified golden config facts");
    return result;
}

void validateReceiptQualification(const Json& receipt, const Qualification& config_qualification) {
    const auto& qualification = requireObject(receipt, "qualification", "build receipt");
    requireExactKeys(qualification, {"state", "runtime_eligible", "golden_parity_verified"},
                     "build receipt qualification");
    const auto state = requireString(qualification, "state", "build receipt qualification");
    const bool runtime_eligible =
        requireBool(qualification, "runtime_eligible", "build receipt qualification");
    const bool parity =
        requireBool(qualification, "golden_parity_verified", "build receipt qualification");
    if (state != config_qualification.state ||
        runtime_eligible != config_qualification.runtime_eligible ||
        parity != config_qualification.golden_parity_verified) {
        fail("SAM2 config and receipt qualification facts disagree");
    }
    if (state != "unqualified" || runtime_eligible || parity)
        fail("SAM2 native bundle requires exact unqualified golden receipt facts");
}

void validateReceiptAssets(const Json& receipt, const std::string& config_digest) {
    const auto& assets = requireObject(receipt, "assets", "build receipt");
    requireExactKeys(assets,
                     {"checkpoint_sha256", "source_config_sha256", "golden_manifest_sha256",
                      "embedded_config_sha256"},
                     "build receipt assets");
    requireStringValue(assets, "checkpoint_sha256", kCheckpointSha256, "build receipt assets");
    requireStringValue(assets, "source_config_sha256", kConfigSha256, "build receipt assets");
    requireStringValue(assets, "golden_manifest_sha256", kGoldenManifestSha256,
                       "build receipt assets");
    requireStringValue(assets, "embedded_config_sha256", config_digest, "build receipt assets");
}

void validateReceiptTarget(const Json& receipt, const Json& header) {
    const auto& build = requireObject(receipt, "build", "build receipt");
    requireExactKeys(build,
                     {"created_at_utc", "workspace_bytes", "network_mode", "tf32_enabled",
                      "plan_profiling_verbosity", "tensorrt_version", "tensorrt_abi",
                      "cuda_runtime_version", "cuda_driver_version", "gpu"},
                     "build receipt build facts");
    const auto created_at = requireString(build, "created_at_utc", "build receipt build facts");
    if (!isCanonicalUtcTimestamp(created_at) ||
        created_at != requireString(header, "created_at", "bundle header")) {
        fail("SAM2 build receipt timestamp does not match the bundle header");
    }
    if (requireUint64(build, "workspace_bytes", "build receipt build facts") == 0U)
        fail("SAM2 build receipt workspace size is invalid");
    requireStringValue(build, "network_mode", "strongly_typed", "build receipt build facts");
    if (requireBool(build, "tf32_enabled", "build receipt build facts"))
        fail("SAM2 build receipt enabled TF32");
    requireStringValue(build, "plan_profiling_verbosity", kPlanProfilingVerbosity,
                       "build receipt build facts");
    requireStringValue(build, "tensorrt_version", kTargetTensorRtVersion,
                       "build receipt build facts");
    requireStringValue(build, "tensorrt_abi", kTargetTensorRtAbi, "build receipt build facts");
    (void)requireString(build, "cuda_runtime_version", "build receipt build facts");
    (void)requireString(build, "cuda_driver_version", "build receipt build facts");

    const auto& gpu = requireObject(build, "gpu", "build receipt build facts");
    requireExactKeys(gpu, {"device", "name", "compute_capability", "global_memory_bytes"},
                     "build receipt GPU facts");
    (void)requireInt32(gpu, "device", "build receipt GPU facts");
    requireStringValue(gpu, "name", kTargetGpuName, "build receipt GPU facts");
    requireStringValue(gpu, "compute_capability", kTargetComputeCapability,
                       "build receipt GPU facts");
    if (requireUint64(gpu, "global_memory_bytes", "build receipt GPU facts") == 0U)
        fail("SAM2 build receipt GPU memory fact is invalid");
}

void validateReceiptGraphs(const Json& receipt, const AuthenticatedSections& sections) {
    const auto& graphs = requireArray(receipt, "graphs", "build receipt");
    if (graphs.size() != kRequiredPlanSections.size())
        fail("SAM2 build receipt graph count drifted");
    constexpr std::array<std::string_view, 6> kinds = {"image",     "prompt",    "recurrent",
                                                       "recurrent", "recurrent", "recurrent"};
    constexpr std::array<std::int32_t, 6> histories = {0, 0, 1, 2, 3, 4};
    constexpr std::array<std::int32_t, 6> inputs = {1, 4, 5, 5, 5, 5};
    constexpr std::array<std::int32_t, 6> outputs = {9, 3, 3, 3, 3, 3};
    constexpr std::array<std::uint64_t, 6> layers = {1139U, 882U, 1630U, 1652U, 1674U, 1696U};
    constexpr std::array<std::uint64_t, 6> referenced_tensors = {282U, 185U, 291U,
                                                                 291U, 291U, 291U};
    for (std::size_t index = 0; index < graphs.size(); ++index) {
        const auto& graph = graphs[index];
        if (index == 0U) {
            requireExactKeys(graph,
                             {"section",
                              "kind",
                              "history_frames",
                              "inputs",
                              "outputs",
                              "layers",
                              "convolution_layers",
                              "activation_layers",
                              "pooling_layers",
                              "element_wise_layers",
                              "shuffle_layers",
                              "constant_layers",
                              "slice_layers",
                              "resize_layers",
                              "normalization_layers",
                              "cast_layers",
                              "matrix_multiply_layers",
                              "softmax_layers",
                              "plugin_v3_layers",
                              "attention_input_layers",
                              "attention_output_layers",
                              "referenced_checkpoint_tensors",
                              "serialized_bytes",
                              "serialized_sha256",
                              "graph_complete"},
                             "build receipt image graph");
        } else {
            requireExactKeys(graph,
                             {"section", "kind", "history_frames", "inputs", "outputs", "layers",
                              "referenced_checkpoint_tensors", "serialized_bytes",
                              "serialized_sha256", "graph_complete"},
                             "build receipt tracker graph");
        }
        requireStringValue(graph, "section", kRequiredPlanSections[index], "build receipt graph");
        requireStringValue(graph, "kind", kinds[index], "build receipt graph");
        requireIntValue(graph, "history_frames", histories[index], "build receipt graph");
        requireIntValue(graph, "inputs", inputs[index], "build receipt graph");
        requireIntValue(graph, "outputs", outputs[index], "build receipt graph");
        if (requireUint64(graph, "layers", "build receipt graph") != layers[index] ||
            requireUint64(graph, "referenced_checkpoint_tensors", "build receipt graph") !=
                referenced_tensors[index]) {
            fail("SAM2 build receipt exact graph construction facts drifted");
        }
        if (index == 0U &&
            (requireUint64(graph, "convolution_layers", "build receipt image graph") != 23U ||
             requireUint64(graph, "activation_layers", "build receipt image graph") != 28U ||
             requireUint64(graph, "pooling_layers", "build receipt image graph") != 6U ||
             requireUint64(graph, "element_wise_layers", "build receipt image graph") != 130U ||
             requireUint64(graph, "shuffle_layers", "build receipt image graph") != 313U ||
             requireUint64(graph, "constant_layers", "build receipt image graph") != 216U ||
             requireUint64(graph, "slice_layers", "build receipt image graph") != 67U ||
             requireUint64(graph, "resize_layers", "build receipt image graph") != 2U ||
             requireUint64(graph, "normalization_layers", "build receipt image graph") != 32U ||
             requireUint64(graph, "cast_layers", "build receipt image graph") != 223U ||
             requireUint64(graph, "matrix_multiply_layers", "build receipt image graph") != 67U ||
             requireUint64(graph, "softmax_layers", "build receipt image graph") != 0U ||
             requireUint64(graph, "plugin_v3_layers", "build receipt image graph") != 0U ||
             requireUint64(graph, "attention_input_layers", "build receipt image graph") != 16U ||
             requireUint64(graph, "attention_output_layers", "build receipt image graph") != 16U)) {
            fail("SAM2 build receipt exact image layer-type facts drifted");
        }
        if (requireUint64(graph, "serialized_bytes", "build receipt graph") !=
            sections.plans[index]->data.size()) {
            fail("SAM2 build receipt serialized size mismatch");
        }
        requireStringValue(graph, "serialized_sha256", sections.digests[index],
                           "build receipt graph");
        if (!requireBool(graph, "graph_complete", "build receipt graph"))
            fail("SAM2 build receipt contains an incomplete graph");
    }
}

void validateReceiptImageAttention(const Json& receipt) {
    const auto& attention = requireObject(receipt, "image_attention", "build receipt");
    requireExactKeys(attention,
                     {"implementation", "operator", "api", "block_count", "head_dimension",
                      "query_form", "key_value_form", "output_form", "normalization", "causal_mask",
                      "decomposable", "fused_kernel_intent", "metadata_prefix",
                      "metadata_index_width", "q_scale_formula", "k_scale_formula",
                      "effective_score_scale", "scale_dtype"},
                     "build receipt image attention");
    requireStringValue(attention, "implementation", "tensorrt_iattention_v2",
                       "build receipt image attention");
    requireStringValue(attention, "operator", "IAttention", "build receipt image attention");
    requireStringValue(attention, "api", "addAttentionV2", "build receipt image attention");
    requireIntValue(attention, "block_count", 16, "build receipt image attention");
    requireIntValue(attention, "head_dimension", 96, "build receipt image attention");
    requireStringValue(attention, "query_form", "padded_bhnd", "build receipt image attention");
    requireStringValue(attention, "key_value_form", "padded_bhnd", "build receipt image attention");
    requireStringValue(attention, "output_form", "padded_bhnd", "build receipt image attention");
    requireStringValue(attention, "normalization", "softmax", "build receipt image attention");
    requireStringValue(attention, "causal_mask", "none", "build receipt image attention");
    if (requireBool(attention, "decomposable", "build receipt image attention"))
        fail("SAM2 build receipt image attention is decomposable");
    if (!requireBool(attention, "fused_kernel_intent", "build receipt image attention"))
        fail("SAM2 build receipt image attention lacks fused-kernel intent");
    requireStringValue(attention, "metadata_prefix", kImageAttentionMetadataPrefix,
                       "build receipt image attention");
    requireIntValue(attention, "metadata_index_width", kImageAttentionMetadataIndexWidth,
                    "build receipt image attention");
    requireStringValue(attention, "q_scale_formula", "1/sqrt(head_dimension)",
                       "build receipt image attention");
    requireStringValue(attention, "k_scale_formula", "none", "build receipt image attention");
    requireStringValue(attention, "effective_score_scale", "1/sqrt(head_dimension)",
                       "build receipt image attention");
    requireStringValue(attention, "scale_dtype", "bf16", "build receipt image attention");
}

void validateReceipt(const Json& receipt, const Json& header, const AuthenticatedSections& sections,
                     const Qualification& config_qualification) {
    requireExactKeys(receipt,
                     {"schema_version", "family", "model_id", "qualification", "assets", "build",
                      "image_attention", "graphs"},
                     "build receipt");
    requireIntValue(receipt, "schema_version", 1, "build receipt");
    requireStringValue(receipt, "family", kFamily, "build receipt");
    requireStringValue(receipt, "model_id", kModelId, "build receipt");
    validateReceiptQualification(receipt, config_qualification);
    validateReceiptAssets(receipt, sections.digests[6]);
    validateReceiptTarget(receipt, header);
    validateReceiptImageAttention(receipt);
    validateReceiptGraphs(receipt, sections);
}

void validateQualificationRecordBinding(const NativeQualificationRecord& record,
                                        const SealedBundleSnapshot& snapshot, const Json& header,
                                        const Json& config, const Json& receipt,
                                        const AuthenticatedSections& sections,
                                        const NativeBundleRuntimeTarget& runtime_target) {
    if (record.bundle.sha256 != snapshot.sha256())
        fail("SAM2 qualification record bundle SHA-256 binding mismatch");
    if (record.bundle.size_bytes != snapshot.size())
        fail("SAM2 qualification record bundle size binding mismatch");
    if (record.bundle.embedded_config_sha256 != sections.digests[6])
        fail("SAM2 qualification record embedded config SHA-256 binding mismatch");
    if (record.bundle.build_receipt_sha256 != sections.digests[7])
        fail("SAM2 qualification record build receipt SHA-256 binding mismatch");
    for (std::size_t index = 0; index < record.bundle.plans.size(); ++index) {
        if (record.bundle.plans[index].section != kRequiredPlanSections[index] ||
            record.bundle.plans[index].sha256 != sections.digests[index]) {
            fail("SAM2 qualification record six-plan binding mismatch");
        }
    }

    const auto& scope = record.scope;
    if (scope.family != requireString(header, "family", "bundle header") ||
        scope.model_id != requireString(header, "model_id", "bundle header") ||
        scope.runtime_strategy != requireString(header, "runtime_strategy", "bundle header") ||
        scope.precision != requireString(header, "precision", "bundle header") ||
        scope.gpu_name != requireString(header, "gpu_name", "bundle header") ||
        scope.tensorrt_version != requireString(header, "trt_version", "bundle header") ||
        scope.tensorrt_abi != requireString(header, "trt_abi", "bundle header") ||
        scope.engine_contract_version !=
            requireUint64(config, "engine_contract_version", "embedded config") ||
        scope.compute_capability != runtime_target.compute_capability ||
        scope.gpu_name != runtime_target.gpu_name ||
        scope.tensorrt_version != runtime_target.tensorrt_version ||
        scope.tensorrt_abi != runtime_target.tensorrt_abi) {
        fail("SAM2 qualification record scope binding mismatch");
    }

    const auto& receipt_assets = requireObject(receipt, "assets", "build receipt");
    if (record.accuracy_evidence.golden_manifest_sha256 !=
            requireString(config, "golden_manifest_sha256", "embedded config") ||
        record.accuracy_evidence.golden_manifest_sha256 !=
            requireString(receipt_assets, "golden_manifest_sha256", "build receipt assets")) {
        fail("SAM2 qualification record semantic policy evidence binding mismatch");
    }
}

DType runtimeDtype(TensorDataType type) {
    switch (type) {
    case TensorDataType::kFloat32:
        return DType::kFloat32;
    case TensorDataType::kBFloat16:
        return DType::kBFloat16;
    }
    fail("SAM2 native engine contract contains an unsupported data type");
}

std::vector<int64_t> runtimeShape(const TensorContract& contract) {
    std::vector<int64_t> result;
    result.reserve(contract.rank);
    for (std::uint8_t index = 0; index < contract.rank; ++index)
        result.push_back(contract.dimensions[index]);
    return result;
}

const TensorContract* findContract(const ContractList& contracts, const std::string& name) {
    const auto found = std::find_if(contracts.begin(), contracts.end(),
                                    [&](const auto& contract) { return contract.name == name; });
    return found == contracts.end() ? nullptr : &*found;
}

void validateDirection(const ITrtModule& module, const ContractList& contracts, bool input,
                       std::string_view section) {
    const auto info = input ? module.input_info() : module.output_info();
    if (info.size() != contracts.size())
        fail("SAM2 module tensor count drifted: " + std::string(section));
    for (const auto& tensor : info) {
        const auto* contract = findContract(contracts, tensor.name);
        const auto duplicates = std::count_if(info.begin(), info.end(), [&](const auto& candidate) {
            return candidate.name == tensor.name;
        });
        if (contract == nullptr || duplicates != 1 || tensor.is_input != input ||
            tensor.dtype != runtimeDtype(contract->data_type) ||
            tensor.shape != runtimeShape(*contract)) {
            fail("SAM2 module tensor metadata drifted: " + std::string(section));
        }
    }
    for (const auto& contract : contracts) {
        const std::string name(contract.name);
        const bool expected_direction = input ? module.has_input(name) : module.has_output(name);
        const bool other_direction = input ? module.has_output(name) : module.has_input(name);
        if (!expected_direction || other_direction ||
            module.tensor_dtype(name) != runtimeDtype(contract.data_type) ||
            module.tensor_shape(name) != runtimeShape(contract)) {
            fail("SAM2 module binding metadata drifted: " + std::string(section));
        }
    }
}

void validateModule(const ITrtModule* module, const ContractList& inputs,
                    const ContractList& outputs, std::string_view section) {
    if (module == nullptr || !module->ok())
        fail("SAM2 module creation failed: " + std::string(section));
    if (module->optimization_profile_count() != 1 || module->profile_idx() != 0)
        fail("SAM2 module optimization profile contract drifted: " + std::string(section));
    validateDirection(*module, inputs, true, section);
    validateDirection(*module, outputs, false, section);
}

ContractList imageInputs() {
    return {kPixelValues};
}

ContractList imageOutputs() {
    ContractList result(kTrackerFpn.begin(), kTrackerFpn.end());
    result.insert(result.end(), kBboxMaps.begin(), kBboxMaps.end());
    return result;
}

ContractList promptInputs() {
    ContractList result(kTrackerFpn.begin(), kTrackerFpn.end());
    result.push_back(kBoxPrompt);
    return result;
}

ContractList trackerOutputs() {
    return {kMaskLogits256, kObjectPointer, kMemoryFeatures};
}

ContractList recurrentInputs(std::int32_t history) {
    ContractList result(kTrackerFpn.begin(), kTrackerFpn.end());
    result.push_back(historyMemoryFeatures(history));
    result.push_back(historyObjectPointers(history));
    return result;
}

std::unique_ptr<ITrtModule> createModule(const NativePlanModuleFactory& factory,
                                         const BundleSection& plan) {
    try {
        return factory(plan.name, plan.data.data(), plan.data.size());
    } catch (const std::exception& error) {
        fail("SAM2 module factory failed for " + plan.name + ": " + error.what());
    } catch (...) {
        fail("SAM2 module factory failed for " + plan.name);
    }
}

NativeVideoEngineSet createEngineSet(const AuthenticatedSections& sections,
                                     const NativePlanModuleFactory& factory) {
    NativeVideoEngineSet result;
    result.image = createModule(factory, *sections.plans[0]);
    validateModule(result.image.get(), imageInputs(), imageOutputs(), kRequiredPlanSections[0]);
    result.prompt = createModule(factory, *sections.plans[1]);
    validateModule(result.prompt.get(), promptInputs(), trackerOutputs(), kRequiredPlanSections[1]);
    for (std::size_t index = 0; index < result.recurrent.size(); ++index) {
        result.recurrent[index] = createModule(factory, *sections.plans[index + 2U]);
        validateModule(result.recurrent[index].get(),
                       recurrentInputs(static_cast<std::int32_t>(index + 1U)), trackerOutputs(),
                       kRequiredPlanSections[index + 2U]);
    }
    return result;
}

NativeVideoEngineSet loadNativeVideoEngineSetFromBundleImpl(
    const std::string& bundle_path, std::string_view expected_bundle_sha256,
    const NativeBundleRuntimeTarget& runtime_target, const NativePlanModuleFactory& module_factory,
    const NativeQualificationRecord* qualification_record) {
    if (bundle_path.empty())
        fail("SAM2 native bundle path must not be empty");
    if (!expected_bundle_sha256.empty() && !isLowercaseSha256(expected_bundle_sha256))
        fail("SAM2 native bundle expected full SHA-256 is invalid");
    if (!module_factory)
        fail("SAM2 native bundle module factory must not be empty");
    validateRuntimeTarget(runtime_target);

    try {
        const SealedBundleSnapshot snapshot(bundle_path, expected_bundle_sha256);
        if (qualification_record != nullptr &&
            snapshot.size() != qualification_record->bundle.size_bytes) {
            fail("SAM2 qualification record bundle size binding mismatch");
        }
        const BundleFile bundle = ReadBundleFile(snapshot.path());
        const RawHeader raw = readRawHeader(snapshot.path());
        const Json header = parseStrictJson(raw.text, kMaximumHeaderBytes, "bundle header");
        validateHeaderIdentity(header, bundle);
        const AuthenticatedSections sections = authenticateSections(header, raw, bundle);
        const Json config =
            parseStrictJson(sectionText(*sections.config, kMaximumConfigBytes, "embedded config"),
                            kMaximumConfigBytes, "embedded config");
        const Qualification qualification = validateConfig(config);
        const Json receipt =
            parseStrictJson(sectionText(*sections.receipt, kMaximumReceiptBytes, "build receipt"),
                            kMaximumReceiptBytes, "build receipt");
        validateReceipt(receipt, header, sections, qualification);
        if (qualification_record != nullptr) {
            validateQualificationRecordBinding(*qualification_record, snapshot, header, config,
                                               receipt, sections, runtime_target);
        }
        return createEngineSet(sections, module_factory);
    } catch (const NativeBundleLoadError&) {
        throw;
    } catch (const std::exception& error) {
        fail("failed to load SAM2 native bundle: " + std::string(error.what()));
    }
}

} // namespace

NativeVideoEngineSet
loadDiagnosticNativeVideoEngineSetFromBundle(const std::string& bundle_path,
                                             const NativeBundleRuntimeTarget& runtime_target,
                                             const NativePlanModuleFactory& module_factory) {
    return loadNativeVideoEngineSetFromBundleImpl(bundle_path, {}, runtime_target, module_factory,
                                                  nullptr);
}

NativeVideoEngineSet loadDiagnosticNativeVideoEngineSetFromBundleWithExpectedSha256(
    const std::string& bundle_path, std::string_view expected_bundle_sha256,
    const NativeBundleRuntimeTarget& runtime_target,
    const NativePlanModuleFactory& module_factory) {
    if (expected_bundle_sha256.empty())
        fail("SAM2 native bundle expected full SHA-256 must not be empty");
    return loadNativeVideoEngineSetFromBundleImpl(bundle_path, expected_bundle_sha256,
                                                  runtime_target, module_factory, nullptr);
}

NativeVideoEngineSet loadProductionQualifiedNativeVideoEngineSetFromBundle(
    const std::string& bundle_path, const std::string& qualification_record_path,
    const NativeBundleRuntimeTarget& runtime_target,
    const NativePlanModuleFactory& module_factory) {
    try {
        const auto authority = qualification_internal::authorizeProductionNativeQualificationRecord(
            qualification_record_path);
        return loadNativeVideoEngineSetFromBundleImpl(bundle_path, authority.record.bundle.sha256,
                                                      runtime_target, module_factory,
                                                      &authority.record);
    } catch (const NativeBundleLoadError&) {
        throw;
    } catch (const std::exception& error) {
        fail("failed to authorize SAM2 native production bundle: " + std::string(error.what()));
    }
}

#if defined(TRTMC_SAM2_TEST_QUALIFICATION_AUTHORITY)
NativeVideoEngineSet loadProductionQualifiedNativeVideoEngineSetFromBundleForTest(
    const std::string& bundle_path, const std::string& qualification_record_path,
    const NativeBundleRuntimeTarget& runtime_target, const NativePlanModuleFactory& module_factory,
    const qualification_internal::NativeQualificationTestPin& pin) {
    try {
        const auto authority = qualification_internal::authorizeNativeQualificationRecordForTest(
            qualification_record_path, pin);
        return loadNativeVideoEngineSetFromBundleImpl(bundle_path, authority.record.bundle.sha256,
                                                      runtime_target, module_factory,
                                                      &authority.record);
    } catch (const NativeBundleLoadError&) {
        throw;
    } catch (const std::exception& error) {
        fail("failed to authorize SAM2 native production bundle: " + std::string(error.what()));
    }
}
#endif

} // namespace trtmc::sam2
