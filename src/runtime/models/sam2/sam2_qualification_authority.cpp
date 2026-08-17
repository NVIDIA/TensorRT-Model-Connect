/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_qualification_authority.h"

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_qualification_pin_provider.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace trtmc::sam2 {

namespace {

using Json = nlohmann::json;

constexpr std::string_view kModelId = "sam2.1-hiera-small-bbox";
constexpr std::string_view kFamily = "sam2";
constexpr std::string_view kPrecision = "mixed_bf16_fp32";
constexpr std::size_t kMaximumRecordBytes = 1024U * 1024U;
constexpr std::size_t kMaximumAuthorityIdBytes = 128U;

class NativeQualificationError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message) {
    throw NativeQualificationError("SAM2 native qualification record " + message);
}

bool isLowercaseSha256(std::string_view value) {
    return value.size() == 64U && std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

bool isCanonicalAuthorityId(std::string_view value) {
    if (value.empty() || value.size() > kMaximumAuthorityIdBytes || value.front() < 'a' ||
        value.front() > 'z') {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= 'a' && character <= 'z') || (character >= '0' && character <= '9') ||
               character == '.' || character == '_' || character == '-';
    });
}

bool isCanonicalUtcTimestamp(std::string_view value) {
    if (value.size() != 20U || value[4] != '-' || value[7] != '-' || value[10] != 'T' ||
        value[13] != ':' || value[16] != ':' || value[19] != 'Z') {
        return false;
    }
    constexpr std::array<std::size_t, 14> digits = {0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18};
    if (!std::all_of(digits.begin(), digits.end(), [value](std::size_t index) {
            return std::isdigit(static_cast<unsigned char>(value[index])) != 0;
        })) {
        return false;
    }
    const auto number = [value](std::size_t offset) {
        return (value[offset] - '0') * 10 + (value[offset + 1U] - '0');
    };
    const int month = number(5U);
    const int day = number(8U);
    const int hour = number(11U);
    const int minute = number(14U);
    const int second = number(17U);
    return month >= 1 && month <= 12 && day >= 1 && day <= 31 && hour <= 23 && minute <= 59 &&
           second <= 59;
}

Json parseStrictJson(const std::string& text) {
    if (text.empty() || text.size() > kMaximumRecordBytes)
        fail("size is invalid");
    std::vector<std::unordered_set<std::string>> object_keys;
    Json::parser_callback_t callback = [&](int, Json::parse_event_t event, Json& value) {
        if (event == Json::parse_event_t::object_start)
            object_keys.emplace_back();
        if (event == Json::parse_event_t::key) {
            const auto key = value.get<std::string>();
            if (object_keys.empty() || !object_keys.back().insert(key).second)
                fail("contains duplicate key: " + key);
        }
        if (event == Json::parse_event_t::object_end) {
            if (object_keys.empty())
                fail("object nesting is invalid");
            object_keys.pop_back();
        }
        return true;
    };
    try {
        return Json::parse(text, callback);
    } catch (const NativeQualificationError&) {
        throw;
    } catch (const Json::exception& error) {
        fail("is invalid JSON: " + std::string(error.what()));
    }
}

void requireExactKeys(const Json& object, std::initializer_list<std::string_view> expected,
                      std::string_view context) {
    if (!object.is_object())
        fail(std::string(context) + " must be an object");
    std::set<std::string> expected_keys;
    for (const auto key : expected)
        expected_keys.emplace(key);
    std::set<std::string> actual_keys;
    for (auto item = object.begin(); item != object.end(); ++item)
        actual_keys.insert(item.key());
    if (actual_keys != expected_keys)
        fail(std::string(context) + " field set drifted");
}

const Json& requireObject(const Json& object, std::string_view field, std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || !found->is_object())
        fail(std::string(context) + " requires object field " + std::string(field));
    return *found;
}

const Json& requireArray(const Json& object, std::string_view field, std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || !found->is_array())
        fail(std::string(context) + " requires array field " + std::string(field));
    return *found;
}

std::string requireString(const Json& object, std::string_view field, std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || !found->is_string())
        fail(std::string(context) + " requires string field " + std::string(field));
    const auto result = found->get<std::string>();
    if (result.empty())
        fail(std::string(context) + " contains empty field " + std::string(field));
    return result;
}

bool requireBool(const Json& object, std::string_view field, std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || !found->is_boolean())
        fail(std::string(context) + " requires boolean field " + std::string(field));
    return found->get<bool>();
}

std::uint64_t requireUint64(const Json& object, std::string_view field, std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || (!found->is_number_unsigned() && !found->is_number_integer()))
        fail(std::string(context) + " requires integer field " + std::string(field));
    if (found->is_number_integer() && found->get<std::int64_t>() < 0)
        fail(std::string(context) + " contains negative field " + std::string(field));
    try {
        return found->get<std::uint64_t>();
    } catch (const Json::exception&) {
        fail(std::string(context) + " integer field is out of range: " + std::string(field));
    }
}

std::uint32_t requireUint32(const Json& object, std::string_view field, std::string_view context) {
    const auto value = requireUint64(object, field, context);
    if (value > std::numeric_limits<std::uint32_t>::max())
        fail(std::string(context) + " integer field is out of range: " + std::string(field));
    return static_cast<std::uint32_t>(value);
}

std::string requireSha256(const Json& object, std::string_view field, std::string_view context) {
    const auto value = requireString(object, field, context);
    if (!isLowercaseSha256(value))
        fail(std::string(context) + " SHA-256 field is invalid: " + std::string(field));
    return value;
}

NativeQualificationRecord parseRecord(const Json& root) {
    requireExactKeys(root,
                     {"schema_version", "artifact_type", "authority_id", "authority_serial",
                      "self_authorizing", "scope", "bundle", "accuracy_evidence",
                      "generated_at_utc"},
                     "root");

    NativeQualificationRecord result;
    const auto schema = requireUint64(root, "schema_version", "root");
    if (schema != static_cast<std::uint64_t>(kNativeQualificationRecordSchemaVersion))
        fail("schema_version is not supported");
    result.schema_version = static_cast<std::int32_t>(schema);
    result.artifact_type = requireString(root, "artifact_type", "root");
    if (result.artifact_type != kNativeQualificationRecordArtifactType)
        fail("artifact_type is not supported");
    result.authority_id = requireString(root, "authority_id", "root");
    if (!isCanonicalAuthorityId(result.authority_id))
        fail("authority_id is not bounded canonical ASCII");
    result.authority_serial = requireUint64(root, "authority_serial", "root");
    if (result.authority_serial == 0U)
        fail("authority_serial must be positive");
    result.self_authorizing = requireBool(root, "self_authorizing", "root");
    if (result.self_authorizing)
        fail("must not be self-authorizing");
    result.generated_at_utc = requireString(root, "generated_at_utc", "root");
    if (!isCanonicalUtcTimestamp(result.generated_at_utc))
        fail("generated_at_utc is not canonical UTC");

    const auto& scope = requireObject(root, "scope", "root");
    requireExactKeys(scope,
                     {"family", "model_id", "engine_contract_version", "runtime_strategy",
                      "precision", "gpu_name", "compute_capability", "tensorrt_version",
                      "tensorrt_abi"},
                     "scope");
    result.scope.family = requireString(scope, "family", "scope");
    result.scope.model_id = requireString(scope, "model_id", "scope");
    result.scope.engine_contract_version = requireUint32(scope, "engine_contract_version", "scope");
    result.scope.runtime_strategy = requireString(scope, "runtime_strategy", "scope");
    result.scope.precision = requireString(scope, "precision", "scope");
    result.scope.gpu_name = requireString(scope, "gpu_name", "scope");
    result.scope.compute_capability = requireString(scope, "compute_capability", "scope");
    result.scope.tensorrt_version = requireString(scope, "tensorrt_version", "scope");
    result.scope.tensorrt_abi = requireString(scope, "tensorrt_abi", "scope");
    if (result.scope.family != kFamily || result.scope.model_id != kModelId ||
        result.scope.engine_contract_version != kEngineContractVersion ||
        result.scope.runtime_strategy != kStrategyName || result.scope.precision != kPrecision ||
        result.scope.gpu_name != kTargetGpuName ||
        result.scope.compute_capability != kTargetComputeCapability ||
        result.scope.tensorrt_version != kTargetTensorRtVersion ||
        result.scope.tensorrt_abi != kTargetTensorRtAbi) {
        fail("scope does not match the compiled SAM2 TRT 11.1 L4 contract");
    }

    const auto& bundle = requireObject(root, "bundle", "root");
    requireExactKeys(
        bundle, {"sha256", "size_bytes", "embedded_config_sha256", "build_receipt_sha256", "plans"},
        "bundle");
    result.bundle.sha256 = requireSha256(bundle, "sha256", "bundle");
    result.bundle.size_bytes = requireUint64(bundle, "size_bytes", "bundle");
    if (result.bundle.size_bytes == 0U)
        fail("bundle size_bytes must be positive");
    result.bundle.embedded_config_sha256 =
        requireSha256(bundle, "embedded_config_sha256", "bundle");
    result.bundle.build_receipt_sha256 = requireSha256(bundle, "build_receipt_sha256", "bundle");
    const auto& plans = requireArray(bundle, "plans", "bundle");
    if (plans.size() != kRequiredPlanSections.size())
        fail("bundle must bind exactly six plans");
    for (std::size_t index = 0; index < plans.size(); ++index) {
        requireExactKeys(plans[index], {"section", "sha256"}, "bundle plan");
        result.bundle.plans[index].section = requireString(plans[index], "section", "bundle plan");
        result.bundle.plans[index].sha256 = requireSha256(plans[index], "sha256", "bundle plan");
        if (result.bundle.plans[index].section != kRequiredPlanSections[index])
            fail("bundle plan section order drifted");
    }

    const auto& evidence = requireObject(root, "accuracy_evidence", "root");
    requireExactKeys(evidence,
                     {"receipt_sha256", "receipt_size_bytes", "regular_receipt_sha256",
                      "regular_receipt_size_bytes", "mode", "policy_id", "replay_count",
                      "frames_per_replay", "reset_before_each_replay", "all_semantic_gates_passed",
                      "timing_performed", "golden_manifest_sha256", "golden_masks_sha256",
                      "benchmark_executable_sha256", "benchmark_source_manifest_sha256",
                      "benchmark_source_closure_sha256"},
                     "accuracy_evidence");
    auto& output = result.accuracy_evidence;
    output.receipt_sha256 = requireSha256(evidence, "receipt_sha256", "accuracy_evidence");
    output.receipt_size_bytes = requireUint64(evidence, "receipt_size_bytes", "accuracy_evidence");
    if (output.receipt_size_bytes == 0U)
        fail("accuracy_evidence receipt_size_bytes must be positive");
    output.regular_receipt_sha256 =
        requireSha256(evidence, "regular_receipt_sha256", "accuracy_evidence");
    output.regular_receipt_size_bytes =
        requireUint64(evidence, "regular_receipt_size_bytes", "accuracy_evidence");
    if (output.regular_receipt_size_bytes == 0U)
        fail("accuracy_evidence regular_receipt_size_bytes must be positive");
    output.mode = requireString(evidence, "mode", "accuracy_evidence");
    output.policy_id = requireString(evidence, "policy_id", "accuracy_evidence");
    output.replay_count = requireUint32(evidence, "replay_count", "accuracy_evidence");
    output.frames_per_replay = requireUint32(evidence, "frames_per_replay", "accuracy_evidence");
    output.reset_before_each_replay =
        requireBool(evidence, "reset_before_each_replay", "accuracy_evidence");
    output.all_semantic_gates_passed =
        requireBool(evidence, "all_semantic_gates_passed", "accuracy_evidence");
    output.timing_performed = requireBool(evidence, "timing_performed", "accuracy_evidence");
    output.golden_manifest_sha256 =
        requireSha256(evidence, "golden_manifest_sha256", "accuracy_evidence");
    output.golden_masks_sha256 =
        requireSha256(evidence, "golden_masks_sha256", "accuracy_evidence");
    output.benchmark_executable_sha256 =
        requireSha256(evidence, "benchmark_executable_sha256", "accuracy_evidence");
    output.benchmark_source_manifest_sha256 =
        requireSha256(evidence, "benchmark_source_manifest_sha256", "accuracy_evidence");
    output.benchmark_source_closure_sha256 =
        requireSha256(evidence, "benchmark_source_closure_sha256", "accuracy_evidence");
    if (output.mode != "accuracy_only" || output.policy_id != kNativeSemanticAccuracyPolicyId ||
        output.replay_count != 3U || output.frames_per_replay != 5U ||
        !output.reset_before_each_replay || !output.all_semantic_gates_passed ||
        output.timing_performed || output.golden_manifest_sha256 != kGoldenManifestSha256 ||
        output.golden_masks_sha256 != kGoldenMasksSha256) {
        fail("accuracy_evidence does not match semantic policy sam2_semantic_accuracy_v1");
    }

    return result;
}

Json recordJson(const NativeQualificationRecord& record) {
    Json root;
    root["schema_version"] = record.schema_version;
    root["artifact_type"] = record.artifact_type;
    root["authority_id"] = record.authority_id;
    root["authority_serial"] = record.authority_serial;
    root["self_authorizing"] = record.self_authorizing;
    root["scope"] = {
        {"family", record.scope.family},
        {"model_id", record.scope.model_id},
        {"engine_contract_version", record.scope.engine_contract_version},
        {"runtime_strategy", record.scope.runtime_strategy},
        {"precision", record.scope.precision},
        {"gpu_name", record.scope.gpu_name},
        {"compute_capability", record.scope.compute_capability},
        {"tensorrt_version", record.scope.tensorrt_version},
        {"tensorrt_abi", record.scope.tensorrt_abi},
    };
    Json plans = Json::array();
    for (const auto& plan : record.bundle.plans)
        plans.push_back({{"section", plan.section}, {"sha256", plan.sha256}});
    root["bundle"] = {
        {"sha256", record.bundle.sha256},
        {"size_bytes", record.bundle.size_bytes},
        {"embedded_config_sha256", record.bundle.embedded_config_sha256},
        {"build_receipt_sha256", record.bundle.build_receipt_sha256},
        {"plans", std::move(plans)},
    };
    const auto& evidence = record.accuracy_evidence;
    root["accuracy_evidence"] = {
        {"receipt_sha256", evidence.receipt_sha256},
        {"receipt_size_bytes", evidence.receipt_size_bytes},
        {"regular_receipt_sha256", evidence.regular_receipt_sha256},
        {"regular_receipt_size_bytes", evidence.regular_receipt_size_bytes},
        {"mode", evidence.mode},
        {"policy_id", evidence.policy_id},
        {"replay_count", evidence.replay_count},
        {"frames_per_replay", evidence.frames_per_replay},
        {"reset_before_each_replay", evidence.reset_before_each_replay},
        {"all_semantic_gates_passed", evidence.all_semantic_gates_passed},
        {"timing_performed", evidence.timing_performed},
        {"golden_manifest_sha256", evidence.golden_manifest_sha256},
        {"golden_masks_sha256", evidence.golden_masks_sha256},
        {"benchmark_executable_sha256", evidence.benchmark_executable_sha256},
        {"benchmark_source_manifest_sha256", evidence.benchmark_source_manifest_sha256},
        {"benchmark_source_closure_sha256", evidence.benchmark_source_closure_sha256},
    };
    root["generated_at_utc"] = record.generated_at_utc;
    return root;
}

struct RecordSnapshot {
    std::string bytes;
    std::string sha256;
};

#if defined(__linux__)

class ScopedDescriptor final {
  public:
    explicit ScopedDescriptor(int value) noexcept : value_(value) {}
    ~ScopedDescriptor() {
        if (value_ >= 0)
            ::close(value_);
    }
    ScopedDescriptor(const ScopedDescriptor&) = delete;
    ScopedDescriptor& operator=(const ScopedDescriptor&) = delete;
    int get() const noexcept { return value_; }

  private:
    int value_{-1};
};

bool sameSource(const struct stat& before, const struct stat& after) {
    return before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
           before.st_mode == after.st_mode && before.st_size == after.st_size &&
           before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
           before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
           before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
           before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
}

RecordSnapshot snapshotRecord(const std::string& path) {
    if (path.empty())
        fail("path must not be empty");
    int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const ScopedDescriptor descriptor(::open(path.c_str(), flags));
    if (descriptor.get() < 0)
        fail("failed to open a no-follow snapshot source");
    struct stat before{};
    if (::fstat(descriptor.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
        before.st_size <= 0 || static_cast<std::uint64_t>(before.st_size) > kMaximumRecordBytes) {
        fail("snapshot source is not a supported regular file");
    }

    RecordSnapshot result;
    result.bytes.resize(static_cast<std::size_t>(before.st_size));
    trtmc::internal::Sha256 hash;
    std::size_t offset = 0U;
    while (offset != result.bytes.size()) {
        ssize_t count = -1;
        do {
            count = ::pread(descriptor.get(), result.bytes.data() + offset,
                            result.bytes.size() - offset, static_cast<off_t>(offset));
        } while (count < 0 && errno == EINTR);
        if (count <= 0)
            fail("failed to read a stable snapshot");
        hash.update(result.bytes.data() + offset, static_cast<std::size_t>(count));
        offset += static_cast<std::size_t>(count);
    }
    struct stat after{};
    if (::fstat(descriptor.get(), &after) != 0 || !sameSource(before, after))
        fail("changed while its snapshot was captured");
    result.sha256 = hash.hex_digest();
    return result;
}

#else

RecordSnapshot snapshotRecord(const std::string&) {
    fail("immutable snapshots require Linux");
}

#endif

NativeQualificationRecord parseCanonicalRecord(const RecordSnapshot& snapshot) {
    const Json root = parseStrictJson(snapshot.bytes);
    const std::string canonical = root.dump() + '\n';
    if (canonical != snapshot.bytes)
        fail("is not canonical compact key-sorted JSON with one trailing newline");
    return parseRecord(root);
}

template <typename Pin>
qualification_internal::AuthorizedNativeQualificationRecord
authorizeSnapshot(const RecordSnapshot& snapshot, const Pin& pin) {
    if (!isCanonicalAuthorityId(pin.authority_id) || pin.minimum_authority_serial == 0U ||
        !isLowercaseSha256(pin.record_sha256)) {
        fail("compiled authority pin is invalid");
    }
    if (snapshot.sha256 != pin.record_sha256)
        fail("SHA-256 does not match the compiled authority pin");
    NativeQualificationRecord record = parseCanonicalRecord(snapshot);
    if (record.authority_id != pin.authority_id)
        fail("authority_id does not match the compiled authority pin");
    if (record.authority_serial < pin.minimum_authority_serial)
        fail("authority_serial is below the compiled minimum");
    return {std::move(record), snapshot.sha256, static_cast<std::uint64_t>(snapshot.bytes.size())};
}

} // namespace

std::string makeCanonicalNativeQualificationRecord(const NativeQualificationRecord& record) {
    const Json root = recordJson(record);
    (void)parseRecord(root);
    return root.dump() + '\n';
}

namespace qualification_internal {

AuthorizedNativeQualificationRecord
authorizeProductionNativeQualificationRecord(const std::string& record_path) {
    const RecordSnapshot snapshot = snapshotRecord(record_path);
    const auto pins = productionNativeQualificationPins();
    if (pins.size != 0U && pins.data == nullptr)
        fail("compiled production authority pin provider is invalid");
    const NativeQualificationStaticPin* found = nullptr;
    for (std::size_t index = 0U; index < pins.size; ++index) {
        if (pins.data[index].record_sha256 == snapshot.sha256) {
            found = &pins.data[index];
            break;
        }
    }
    if (found == nullptr)
        fail("has no active compiled production authority pin");
    return authorizeSnapshot(snapshot, *found);
}

#if defined(TRTMC_SAM2_TEST_QUALIFICATION_AUTHORITY)
AuthorizedNativeQualificationRecord
authorizeNativeQualificationRecordForTest(const std::string& record_path,
                                          const NativeQualificationTestPin& pin) {
    return authorizeSnapshot(snapshotRecord(record_path), pin);
}
#endif

} // namespace qualification_internal

} // namespace trtmc::sam2
