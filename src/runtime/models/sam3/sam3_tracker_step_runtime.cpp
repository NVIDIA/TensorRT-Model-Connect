/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam3/sam3_tracker_step_runtime.h"

#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

constexpr std::array<uint32_t, 64> kSha256RoundConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

constexpr std::array<uint32_t, 8> kSha256InitialState{
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
};

uint32_t rotate_right(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32U - count));
}

uint32_t read_big_endian_word(const uint8_t* bytes) {
    return (static_cast<uint32_t>(bytes[0]) << 24U) | (static_cast<uint32_t>(bytes[1]) << 16U) |
           (static_cast<uint32_t>(bytes[2]) << 8U) | static_cast<uint32_t>(bytes[3]);
}

void expand_sha256_schedule(std::array<uint32_t, 64>& schedule) {
    for (std::size_t index = 16; index < schedule.size(); ++index) {
        const uint32_t s0 = rotate_right(schedule[index - 15], 7U) ^
                            rotate_right(schedule[index - 15], 18U) ^ (schedule[index - 15] >> 3U);
        const uint32_t s1 = rotate_right(schedule[index - 2], 17U) ^
                            rotate_right(schedule[index - 2], 19U) ^ (schedule[index - 2] >> 10U);
        schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
    }
}

void compress_sha256_block(std::array<uint32_t, 8>& state, const uint8_t* block) {
    std::array<uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16; ++index)
        schedule[index] = read_big_endian_word(block + index * 4U);
    expand_sha256_schedule(schedule);

    auto working = state;
    for (std::size_t index = 0; index < schedule.size(); ++index) {
        const uint32_t s1 = rotate_right(working[4], 6U) ^ rotate_right(working[4], 11U) ^
                            rotate_right(working[4], 25U);
        const uint32_t choice = (working[4] & working[5]) ^ (~working[4] & working[6]);
        const uint32_t temporary1 =
            working[7] + s1 + choice + kSha256RoundConstants[index] + schedule[index];
        const uint32_t s0 = rotate_right(working[0], 2U) ^ rotate_right(working[0], 13U) ^
                            rotate_right(working[0], 22U);
        const uint32_t majority =
            (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);
        const uint32_t temporary2 = s0 + majority;
        for (std::size_t word = working.size() - 1; word > 0; --word)
            working[word] = working[word - 1];
        working[4] += temporary1;
        working[0] = temporary1 + temporary2;
    }
    for (std::size_t index = 0; index < state.size(); ++index)
        state[index] += working[index];
}

std::vector<uint8_t> pad_sha256_message(const std::vector<char>& data) {
    std::vector<uint8_t> padded(data.begin(), data.end());
    padded.push_back(0x80U);
    while ((padded.size() % 64U) != 56U)
        padded.push_back(0U);
    const uint64_t bit_length = static_cast<uint64_t>(data.size()) * 8U;
    for (int32_t shift = 56; shift >= 0; shift -= 8)
        padded.push_back(
            static_cast<uint8_t>((bit_length >> static_cast<uint32_t>(shift)) & 0xffU));
    return padded;
}

std::string sha256_state_hex(const std::array<uint32_t, 8>& state) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const uint32_t word : state)
        stream << std::setw(8) << word;
    return stream.str();
}

const std::vector<char>& require_section(const BundleFile& bundle, std::string_view name) {
    for (const auto& section : bundle.sections) {
        if (section.name == name && !section.data.empty())
            return section.data;
    }
    throw std::runtime_error("SAM3 tracker-step bundle is missing section " + std::string(name));
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

bool is_safe_section_name(std::string_view name) {
    if (name.empty() || name.front() == '.' || name.find("..") != std::string_view::npos)
        return false;
    for (const unsigned char character : name) {
        if (!(std::isalnum(character) || character == '.' || character == '_' || character == '-'))
            return false;
    }
    return true;
}

bool is_safe_global_name(std::string_view name) {
    return is_safe_section_name(name);
}

bool is_sha256(std::string_view value) {
    if (value.size() != 64)
        return false;
    for (const unsigned char character : value) {
        if (!std::isdigit(character) && !(character >= 'a' && character <= 'f'))
            return false;
    }
    return true;
}

void require_sha(const std::vector<char>& data, const std::string& expected,
                 const std::string& section) {
    if (!is_sha256(expected) || sam3_tracker_step_sha256_hex(data) != expected)
        throw std::runtime_error("SAM3 tracker-step SHA-256 mismatch for " + section);
}

void validate_package_storage(const Sam3TrackerStepPackageSpec& package) {
    if ((package.batch_size != 1 && package.batch_size != 2) ||
        (package.stage != "encoder" && package.stage != "decoder") ||
        !is_safe_global_name(package.package_global) || !is_safe_section_name(package.section) ||
        !ends_with(package.section, ".pt2"))
        throw std::runtime_error("SAM3 tracker-step package has an invalid contract");
}

void validate_package_global(const Sam3TrackerStepPackageSpec& package) {
    const std::string batch = std::to_string(package.batch_size);
    const std::string prefix = package.stage == "encoder"
                                   ? "trtmc.sam3.tracker_encoder.b" + batch + ".m1_10.p1_19."
                                   : "trtmc.sam3.tracker_decoder.b" + batch + ".static.";
    constexpr std::size_t kGlobalDigestCharacters = 20;
    if (!is_sha256(package.sha256) || !starts_with(package.package_global, prefix) ||
        package.package_global.size() != prefix.size() + kGlobalDigestCharacters ||
        package.package_global.substr(prefix.size()) !=
            package.sha256.substr(0, kGlobalDigestCharacters))
        throw std::runtime_error("SAM3 tracker-step package global does not match its batch");
}

Sam3TrackerStepPackageSpec parse_package(const nlohmann::json& object) {
    Sam3TrackerStepPackageSpec package;
    package.stage = object.at("stage").get<std::string>();
    package.package_global = object.at("package_global").get<std::string>();
    package.section = object.at("section").get<std::string>();
    package.sha256 = object.at("sha256").get<std::string>();
    package.batch_size = object.at("batch_size").get<int32_t>();
    validate_package_storage(package);
    validate_package_global(package);
    return package;
}

nlohmann::json expected_memory_input_abi() {
    return nlohmann::json::parse(
        R"([{"policy":"soft","tensors":[{"name":"tracker_feature_2","dtype":"float32","shape":[1,256,72,72]},{"name":"final_mask","dtype":"float32","shape":["B",1,288,288]},{"name":"object_score_logits","dtype":"float32","shape":["B",1]},{"name":"suppress_area_shrinkage","dtype":"int32","shape":["B",1]}]},{"policy":"hard","tensors":[{"name":"tracker_feature_2","dtype":"float32","shape":[1,256,72,72]},{"name":"owned_tracker_mask","dtype":"float32","shape":["B",1,1008,1008]},{"name":"object_score_logits","dtype":"float32","shape":["B",1]},{"name":"suppress_area_shrinkage","dtype":"int32","shape":["B",1]}]}])");
}

nlohmann::json expected_memory_mask_policy() {
    return nlohmann::json::parse(
        R"({"soft":"288 bilinear 1152, clamp rejected rows to <=-10, sigmoid, scale 20, bias -10","hard":"globally owned binary FP32 1008, scale 20, bias -10, antialiased bilinear 1152; suppression input ignored","b1_layout":[2,5184,1,64],"b2_layout":[2,2,5184,64],"stored_precision":"bfloat16 rounded then promoted to float32 carrier"})");
}

nlohmann::json expected_memory_implementation() {
    return nlohmann::json::parse(
        R"({"library":"transformers","model_class":"Sam3TrackerVideoModel","module":"Sam3TrackerVideoMemoryEncoder","license":"Apache-2.0","source_import_policy":"transformers-only"})");
}

std::string expected_memory_section(std::string_view policy, int32_t batch_size) {
    return "sam3_tracker_memory_" + std::string(policy) + "_b" + std::to_string(batch_size) +
           ".pt2";
}

bool is_memory_policy(std::string_view policy) {
    return policy == "soft" || policy == "hard";
}

bool is_supported_tracker_batch(int32_t batch_size) {
    return batch_size == 1 || batch_size == 2;
}

void validate_memory_package_contract(const Sam3TrackerMemoryPackageSpec& package) {
    if (!is_memory_policy(package.policy))
        throw std::runtime_error("SAM3 tracker-memory package has an invalid contract");
    if (!is_supported_tracker_batch(package.batch_size))
        throw std::runtime_error("SAM3 tracker-memory package has an invalid contract");
    if (package.hard_mask != (package.policy == "hard"))
        throw std::runtime_error("SAM3 tracker-memory package has an invalid contract");
    if (package.section != expected_memory_section(package.policy, package.batch_size))
        throw std::runtime_error("SAM3 tracker-memory package has an invalid contract");
    if (!is_safe_section_name(package.section))
        throw std::runtime_error("SAM3 tracker-memory package has an invalid contract");
    if (!is_safe_global_name(package.package_global))
        throw std::runtime_error("SAM3 tracker-memory package has an invalid contract");
    if (!is_sha256(package.sha256))
        throw std::runtime_error("SAM3 tracker-memory package has an invalid contract");
}

void validate_memory_package_global(const Sam3TrackerMemoryPackageSpec& package) {
    const std::string prefix = "trtmc.sam3.tracker_memory." + package.policy + ".b" +
                               std::to_string(package.batch_size) + ".fixed.";
    constexpr std::size_t kGlobalDigestCharacters = 20;
    if (!starts_with(package.package_global, prefix))
        throw std::runtime_error("SAM3 tracker-memory package global does not match its content");
    if (package.package_global.size() != prefix.size() + kGlobalDigestCharacters)
        throw std::runtime_error("SAM3 tracker-memory package global does not match its content");
    if (package.package_global.substr(prefix.size()) !=
        package.sha256.substr(0, kGlobalDigestCharacters))
        throw std::runtime_error("SAM3 tracker-memory package global does not match its content");
}

void validate_memory_package_filename(const nlohmann::json& object,
                                      const Sam3TrackerMemoryPackageSpec& package) {
    const std::string expected_filename = "sam3_tracker_memory_" + package.policy + "_b" +
                                          std::to_string(package.batch_size) + "_" +
                                          package.sha256 + ".pt2";
    if (object.at("filename").get<std::string>() != expected_filename)
        throw std::runtime_error("SAM3 tracker-memory package filename is not content-addressed");
}

void validate_memory_tensor_contract(const nlohmann::json& object, std::string_view policy,
                                     int32_t batch_size) {
    if (!object.is_object() || object.size() != 10 || object.at("fixed_shape") != true)
        throw std::runtime_error("SAM3 tracker-memory package contract is incomplete");
    const auto expected_inputs = nlohmann::json::parse(
        R"([{"name":"tracker_feature_2","dtype":"float32","shape":[1,256,72,72]},{"name":"final_mask","dtype":"float32","shape":[1,1,288,288]},{"name":"object_score_logits","dtype":"float32","shape":[1,1]},{"name":"suppress_area_shrinkage","dtype":"int32","shape":[1,1]}])");
    auto inputs = expected_inputs;
    if (policy == "hard") {
        inputs.at(1).at("name") = "owned_tracker_mask";
        inputs.at(1).at("shape").at(2) = 1008;
        inputs.at(1).at("shape").at(3) = 1008;
    }
    inputs.at(1).at("shape").at(0) = batch_size;
    inputs.at(2).at("shape").at(0) = batch_size;
    inputs.at(3).at("shape").at(0) = batch_size;
    nlohmann::json output_shape = batch_size == 1 ? nlohmann::json::array({2, 5184, 1, 64})
                                                  : nlohmann::json::array({2, 2, 5184, 64});
    nlohmann::json outputs = nlohmann::json::array();
    outputs.push_back({{"name", "packed_memory_and_position"},
                       {"dtype", "float32"},
                       {"shape", std::move(output_shape)}});
    if (object.at("inputs") != inputs || object.at("outputs") != outputs ||
        object.at("hard_mask").get<bool>() != (policy == "hard"))
        throw std::runtime_error("SAM3 tracker-memory package tensor ABI mismatch");
}

Sam3TrackerMemoryPackageSpec parse_memory_package(const nlohmann::json& object) {
    Sam3TrackerMemoryPackageSpec package;
    package.policy = object.at("policy").get<std::string>();
    package.package_global = object.at("package_global").get<std::string>();
    package.section = object.at("section").get<std::string>();
    package.sha256 = object.at("sha256").get<std::string>();
    package.batch_size = object.at("batch_size").get<int32_t>();
    package.hard_mask = object.at("hard_mask").get<bool>();
    validate_memory_package_contract(package);
    validate_memory_package_global(package);
    validate_memory_package_filename(object, package);
    validate_memory_tensor_contract(object, package.policy, package.batch_size);
    return package;
}

nlohmann::json expected_resize_implementation() {
    return nlohmann::json::parse(
        R"({"library":"torch","operator":"torch.nn.functional.interpolate","mode":"bilinear","align_corners":false,"source_size":288,"target_size":1008})");
}

nlohmann::json expected_resize_input_abi() {
    return nlohmann::json::parse(
        R"([{"name":"tracker_mask","dtype":"float32","shape":["B",1,288,288]}])");
}

nlohmann::json expected_resize_output_abi() {
    return nlohmann::json::parse(
        R"([{"name":"resized_tracker_mask","dtype":"float32","shape":["B",1,1008,1008]}])");
}

std::string expected_resize_section(int32_t batch_size) {
    return "sam3_hard_mask_resize_b" + std::to_string(batch_size) + ".pt2";
}

void validate_resize_package_contract(const Sam3HardMaskResizePackageSpec& package) {
    if (!is_supported_tracker_batch(package.batch_size) || !is_sha256(package.sha256) ||
        !is_safe_global_name(package.package_global) || !is_safe_section_name(package.section) ||
        package.section != expected_resize_section(package.batch_size)) {
        throw std::runtime_error("SAM3 hard-mask resize package has an invalid contract");
    }
}

void validate_resize_package_global(const Sam3HardMaskResizePackageSpec& package) {
    const std::string prefix =
        "trtmc.sam3.tracker_memory.resize.b" + std::to_string(package.batch_size) + ".fixed.";
    constexpr std::size_t kGlobalDigestCharacters = 20;
    if (!starts_with(package.package_global, prefix) ||
        package.package_global.size() != prefix.size() + kGlobalDigestCharacters ||
        package.package_global.substr(prefix.size()) !=
            package.sha256.substr(0, kGlobalDigestCharacters)) {
        throw std::runtime_error("SAM3 hard-mask resize global does not match its content");
    }
}

void validate_resize_package_filename(const nlohmann::json& object,
                                      const Sam3HardMaskResizePackageSpec& package) {
    const std::string expected_filename = "sam3_hard_mask_resize_b" +
                                          std::to_string(package.batch_size) + "_" +
                                          package.sha256 + ".pt2";
    if (object.at("filename").get<std::string>() != expected_filename)
        throw std::runtime_error("SAM3 hard-mask resize filename is not content-addressed");
}

Sam3HardMaskResizePackageSpec parse_resize_package(const nlohmann::json& object) {
    if (!object.is_object() || object.size() != 5)
        throw std::runtime_error("SAM3 hard-mask resize package contract is incomplete");
    Sam3HardMaskResizePackageSpec package;
    package.package_global = object.at("package_global").get<std::string>();
    package.section = object.at("section").get<std::string>();
    package.sha256 = object.at("sha256").get<std::string>();
    package.batch_size = object.at("batch_size").get<int32_t>();
    validate_resize_package_contract(package);
    validate_resize_package_global(package);
    validate_resize_package_filename(object, package);
    return package;
}

std::vector<char> decode_sha256(std::string_view value) {
    if (!is_sha256(value))
        throw std::runtime_error("SAM3 tracker-step pipeline contains an invalid SHA-256");
    auto nibble = [](char character) -> uint8_t {
        if (character >= '0' && character <= '9')
            return static_cast<uint8_t>(character - '0');
        return static_cast<uint8_t>(character - 'a' + 10);
    };
    std::vector<char> result;
    result.reserve(value.size() / 2);
    for (std::size_t index = 0; index < value.size(); index += 2) {
        const uint8_t byte =
            static_cast<uint8_t>((nibble(value[index]) << 4U) | nibble(value[index + 1]));
        result.push_back(static_cast<char>(byte));
    }
    return result;
}

std::string pipeline_sha256(const Sam3TrackerStepPipelineSpec& pipeline) {
    constexpr std::string_view kDomain = "trtmc.sam3.tracker_step.split_aoti.v1";
    std::vector<char> payload(kDomain.begin(), kDomain.end());
    payload.push_back('\0');
    const auto encoder = decode_sha256(pipeline.encoder_sha256);
    const auto decoder = decode_sha256(pipeline.decoder_sha256);
    payload.insert(payload.end(), encoder.begin(), encoder.end());
    payload.insert(payload.end(), decoder.begin(), decoder.end());
    return sam3_tracker_step_sha256_hex(payload);
}

void validate_pipeline(const Sam3TrackerStepPipelineSpec& pipeline) {
    if ((pipeline.batch_size != 1 && pipeline.batch_size != 2) ||
        !is_safe_global_name(pipeline.global_name) || !is_sha256(pipeline.encoder_sha256) ||
        !is_sha256(pipeline.decoder_sha256))
        throw std::runtime_error("SAM3 tracker-step pipeline has an invalid contract");
    const std::string prefix =
        "trtmc.sam3.tracker_step.b" + std::to_string(pipeline.batch_size) + ".split_aoti.";
    constexpr std::size_t kGlobalDigestCharacters = 20;
    const std::string expected_digest = pipeline_sha256(pipeline);
    if (!starts_with(pipeline.global_name, prefix) ||
        pipeline.global_name.size() != prefix.size() + kGlobalDigestCharacters ||
        pipeline.global_name.substr(prefix.size()) !=
            expected_digest.substr(0, kGlobalDigestCharacters))
        throw std::runtime_error("SAM3 tracker-step pipeline global does not match both packages");
}

Sam3TrackerStepPipelineSpec parse_pipeline(const nlohmann::json& object) {
    Sam3TrackerStepPipelineSpec pipeline;
    pipeline.global_name = object.at("global_name").get<std::string>();
    pipeline.encoder_sha256 = object.at("encoder_sha256").get<std::string>();
    pipeline.decoder_sha256 = object.at("decoder_sha256").get<std::string>();
    pipeline.batch_size = object.at("batch_size").get<int32_t>();
    validate_pipeline(pipeline);
    return pipeline;
}

void validate_plugin_manifest_fields(const Sam3TrackerStepRuntimeManifest& manifest) {
    if (manifest.schema_version != 1 || manifest.step_scope != kSam3TrackerStepScope ||
        manifest.plugin_section != kSam3TrackerStepNativePluginSection ||
        manifest.plugin_type != "Sam3TrackerStepFfi" || manifest.plugin_version != "2" ||
        !is_sha256(manifest.plugin_sha256))
        throw std::runtime_error("SAM3 tracker-step plugin manifest is incompatible");
}

bool is_known_version(std::string_view version) {
    return !version.empty() && version != "unknown";
}

bool valid_memory_metrics(const nlohmann::json& metrics, double minimum_cosine,
                          double maximum_relative_l2) {
    const double cosine = metrics.at("cosine").get<double>();
    const double relative_l2 = metrics.at("relative_l2").get<double>();
    const double maximum_absolute_error = metrics.at("maximum_absolute_error").get<double>();
    return std::isfinite(cosine) && std::isfinite(relative_l2) &&
           std::isfinite(maximum_absolute_error) && cosine >= minimum_cosine &&
           relative_l2 <= maximum_relative_l2;
}

constexpr double kMinimumMemoryValidationCosine = 0.999;
constexpr double kMaximumMemoryValidationRelativeL2 = 0.02;

void validate_memory_validation_contract(const nlohmann::json& validation) {
    if (validation.at("reference").get<std::string>() !=
        "same Transformers module eager execution before cache publication")
        throw std::runtime_error("SAM3 tracker-memory package validation contract mismatch");
    if (validation.at("minimum_cosine").get<double>() != kMinimumMemoryValidationCosine)
        throw std::runtime_error("SAM3 tracker-memory package validation contract mismatch");
    if (validation.at("maximum_relative_l2").get<double>() != kMaximumMemoryValidationRelativeL2)
        throw std::runtime_error("SAM3 tracker-memory package validation contract mismatch");
}

void validate_memory_validation_case_identity(const nlohmann::json& value,
                                              std::unordered_set<std::string>& variants) {
    const std::string policy = value.at("policy").get<std::string>();
    const int32_t batch_size = value.at("batch_size").get<int32_t>();
    if (!is_memory_policy(policy))
        throw std::runtime_error("SAM3 tracker-memory package validation case failed");
    if (!is_supported_tracker_batch(batch_size))
        throw std::runtime_error("SAM3 tracker-memory package validation case failed");
    if (value.at("hard_mask").get<bool>() != (policy == "hard"))
        throw std::runtime_error("SAM3 tracker-memory package validation case failed");
    if (!value.at("passed").get<bool>())
        throw std::runtime_error("SAM3 tracker-memory package validation case failed");
    if (!variants.insert(policy + ":" + std::to_string(batch_size)).second)
        throw std::runtime_error("SAM3 tracker-memory package validation case failed");
    if (!valid_memory_metrics(value, kMinimumMemoryValidationCosine,
                              kMaximumMemoryValidationRelativeL2))
        throw std::runtime_error("SAM3 tracker-memory package validation case failed");
}

void validate_memory_validation_case_planes(const nlohmann::json& planes) {
    if (!planes.is_object())
        throw std::runtime_error("SAM3 tracker-memory plane validation case failed");
    if (planes.size() != 2)
        throw std::runtime_error("SAM3 tracker-memory plane validation case failed");
    if (!planes.contains("memory"))
        throw std::runtime_error("SAM3 tracker-memory plane validation case failed");
    if (!planes.contains("position"))
        throw std::runtime_error("SAM3 tracker-memory plane validation case failed");
    if (!valid_memory_metrics(planes.at("memory"), kMinimumMemoryValidationCosine,
                              kMaximumMemoryValidationRelativeL2))
        throw std::runtime_error("SAM3 tracker-memory plane validation case failed");
    if (!valid_memory_metrics(planes.at("position"), kMinimumMemoryValidationCosine,
                              kMaximumMemoryValidationRelativeL2))
        throw std::runtime_error("SAM3 tracker-memory plane validation case failed");
}

void validate_memory_package_validation(const nlohmann::json& validation) {
    validate_memory_validation_contract(validation);
    const auto& cases = validation.at("cases");
    if (!cases.is_array())
        throw std::runtime_error("SAM3 tracker-memory package validation is incomplete");
    if (cases.size() != 4)
        throw std::runtime_error("SAM3 tracker-memory package validation is incomplete");
    std::unordered_set<std::string> variants;
    for (const auto& value : cases) {
        validate_memory_validation_case_identity(value, variants);
        validate_memory_validation_case_planes(value.at("planes"));
    }
}

void validate_memory_producer_fields(const Sam3TrackerMemoryAotiManifest& memory) {
    if (!is_known_version(memory.torch_version))
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (!is_known_version(memory.transformers_version))
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (!is_known_version(memory.cuda_version))
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (memory.host_architecture.empty())
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (memory.aoti_abi_version == 0)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (memory.compute_capability_major <= 0)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (memory.compute_capability_minor < 0)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
}

void validate_memory_producer_versions(const Sam3TrackerMemoryAotiManifest& memory,
                                       const Sam3TrackerStepRuntimeManifest& step) {
    if (memory.torch_version != step.torch_version)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (memory.transformers_version != step.transformers_version)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (memory.cuda_version != step.cuda_version)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (memory.host_architecture != step.host_architecture)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
}

void validate_memory_producer_abi(const Sam3TrackerMemoryAotiManifest& memory,
                                  const Sam3TrackerStepRuntimeManifest& step) {
    if (memory.torch_cxx11_abi != step.torch_cxx11_abi)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (memory.aoti_abi_version != step.aoti_abi_version)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (memory.compute_capability_major != step.compute_capability_major)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
    if (memory.compute_capability_minor != step.compute_capability_minor)
        throw std::runtime_error("SAM3 tracker-memory/step producer ABI mismatch");
}

void validate_memory_producer_matches_step(const Sam3TrackerMemoryAotiManifest& memory,
                                           const Sam3TrackerStepRuntimeManifest& step) {
    validate_memory_producer_fields(memory);
    validate_memory_producer_versions(memory, step);
    validate_memory_producer_abi(memory, step);
}

void validate_producer_manifest_fields(const Sam3TrackerStepRuntimeManifest& manifest) {
    if (!is_known_version(manifest.torch_version) ||
        !is_known_version(manifest.transformers_version) ||
        !is_known_version(manifest.tvm_ffi_version) ||
        !is_known_version(manifest.tensorrt_version) || !is_known_version(manifest.cuda_version) ||
        manifest.host_architecture.empty() || manifest.aoti_abi_version == 0 ||
        manifest.compute_capability_major <= 0 || manifest.compute_capability_minor < 0)
        throw std::runtime_error("SAM3 tracker-step runtime manifest is incompatible");
}

void validate_manifest_fields(const Sam3TrackerStepRuntimeManifest& manifest) {
    validate_plugin_manifest_fields(manifest);
    validate_producer_manifest_fields(manifest);
}

std::filesystem::path artifact_cache_directory(const std::string& manifest_sha) {
    return std::filesystem::temp_directory_path() / "trtmc-sam3-tracker-step" /
           manifest_sha.substr(0, 20);
}

class ScopedFileDescriptor {
  public:
    explicit ScopedFileDescriptor(int descriptor) : descriptor_(descriptor) {}
    ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
    ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;
    ~ScopedFileDescriptor() {
        if (descriptor_ >= 0)
            (void)::close(descriptor_);
    }
    int get() const noexcept { return descriptor_; }

  private:
    int descriptor_{-1};
};

class ArtifactCacheLock {
  public:
    explicit ArtifactCacheLock(const std::filesystem::path& directory)
        : descriptor_(::open((directory / ".materialize.lock").c_str(),
                             O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR)) {
        if (descriptor_.get() < 0 || ::flock(descriptor_.get(), LOCK_EX) != 0)
            throw std::runtime_error("SAM3 tracker-step could not lock its artifact cache");
    }
    ArtifactCacheLock(const ArtifactCacheLock&) = delete;
    ArtifactCacheLock& operator=(const ArtifactCacheLock&) = delete;
    ~ArtifactCacheLock() {
        if (descriptor_.get() >= 0)
            (void)::flock(descriptor_.get(), LOCK_UN);
    }

  private:
    ScopedFileDescriptor descriptor_;
};

std::vector<char> read_regular_file(const std::filesystem::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0)
        throw std::runtime_error("SAM3 tracker-step could not inspect an artifact cache file");
    ScopedFileDescriptor scoped(descriptor);
    struct stat status{};
    if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0)
        throw std::runtime_error("SAM3 tracker-step cache entry is not a regular file");
    std::vector<char> bytes(static_cast<std::size_t>(status.st_size));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::read(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            throw std::runtime_error("SAM3 tracker-step could not read an artifact cache file");
        offset += static_cast<std::size_t>(count);
    }
    return bytes;
}

std::pair<std::filesystem::path, int>
reserve_temporary_artifact(const std::filesystem::path& destination) {
    std::string pattern = destination.string() + ".tmp.XXXXXX";
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    const int descriptor = ::mkstemp(mutable_pattern.data());
    if (descriptor < 0)
        throw std::runtime_error("SAM3 tracker-step could not reserve an artifact cache file");
    if (::fcntl(descriptor, F_SETFD, FD_CLOEXEC) != 0) {
        (void)::close(descriptor);
        (void)::unlink(mutable_pattern.data());
        throw std::runtime_error("SAM3 tracker-step could not secure an artifact cache file");
    }
    return {std::filesystem::path(mutable_pattern.data()), descriptor};
}

void write_all(int descriptor, const std::vector<char>& data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const ssize_t count = ::write(descriptor, data.data() + offset, data.size() - offset);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            throw std::runtime_error("SAM3 tracker-step could not write an artifact cache file");
        offset += static_cast<std::size_t>(count);
    }
}

bool cached_artifact_matches(const std::filesystem::path& destination,
                             const std::string& expected_sha) {
    std::error_code status_error;
    const auto status = std::filesystem::symlink_status(destination, status_error);
    if (!status_error && status.type() != std::filesystem::file_type::not_found) {
        if (status.type() != std::filesystem::file_type::regular ||
            sam3_tracker_step_sha256_hex(read_regular_file(destination)) != expected_sha)
            throw std::runtime_error("SAM3 tracker-step cache contains conflicting content");
        return true;
    }
    if (status_error && status_error != std::errc::no_such_file_or_directory)
        throw std::runtime_error("SAM3 tracker-step could not inspect its artifact cache");
    return false;
}

void publish_artifact(const std::filesystem::path& temporary, int descriptor,
                      const std::filesystem::path& destination, const std::vector<char>& data) {
    try {
        {
            ScopedFileDescriptor scoped(descriptor);
            write_all(descriptor, data);
            if (::fsync(descriptor) != 0)
                throw std::runtime_error(
                    "SAM3 tracker-step could not flush an artifact cache file");
        }
        if (::rename(temporary.c_str(), destination.c_str()) != 0)
            throw std::runtime_error("SAM3 tracker-step could not publish an artifact cache file");
    } catch (...) {
        (void)::unlink(temporary.c_str());
        throw;
    }
}

void write_artifact(const std::filesystem::path& destination, const std::vector<char>& data,
                    const std::string& expected_sha) {
    if (cached_artifact_matches(destination, expected_sha))
        return;
    auto [temporary, descriptor] = reserve_temporary_artifact(destination);
    publish_artifact(temporary, descriptor, destination, data);
}

using RegisterPipeline = int (*)(const char*, const char*, const char*, const char*, const char*,
                                 int32_t);
using RegisterMemoryPackage = int (*)(const char*, const char*, const char*, const char*, int32_t);
using AotiAbiVersion = uint64_t (*)();
using PluginVersion = const char* (*)();
using DependencyVersion = const char* (*)();
using TorchCxx11Abi = int32_t (*)();

struct NativePluginApi {
    RegisterPipeline register_pipeline{nullptr};
    RegisterMemoryPackage register_memory_package{nullptr};
    AotiAbiVersion aoti_abi{nullptr};
    PluginVersion plugin_version{nullptr};
    DependencyVersion torch_version{nullptr};
    DependencyVersion tvm_ffi_version{nullptr};
    DependencyVersion tensorrt_version{nullptr};
    TorchCxx11Abi torch_cxx11_abi{nullptr};
};

template <typename Function>
Function require_symbol(void* library, const char* name) {
    dlerror();
    auto function = reinterpret_cast<Function>(dlsym(library, name));
    const char* error = dlerror();
    if (function == nullptr || error != nullptr)
        throw std::runtime_error(std::string("SAM3 tracker-step native plugin lacks ") + name);
    return function;
}

int32_t validate_runtime_device(const Sam3TrackerStepRuntimeManifest& manifest) {
    int32_t device_id = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device_id) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device_id) != cudaSuccess)
        throw std::runtime_error("SAM3 tracker-step could not inspect the CUDA device");
    if (properties.major != manifest.compute_capability_major ||
        properties.minor != manifest.compute_capability_minor)
        throw std::runtime_error("SAM3 tracker-step package targets another compute capability");
    return device_id;
}

void materialize_runtime_artifacts(const BundleFile& bundle,
                                   const Sam3TrackerStepRuntimeManifest& manifest,
                                   const Sam3TrackerMemoryAotiManifest& memory_manifest,
                                   const Sam3HardMaskResizeAotiManifest& resize_manifest,
                                   const std::filesystem::path& cache_directory) {
    std::filesystem::create_directories(cache_directory);
    if (std::filesystem::symlink_status(cache_directory).type() !=
        std::filesystem::file_type::directory)
        throw std::runtime_error("SAM3 tracker-step artifact cache is not a directory");
    std::filesystem::permissions(cache_directory, std::filesystem::perms::owner_all);
    ArtifactCacheLock cache_lock(cache_directory);
    const auto plugin_path = cache_directory / "libtrtmc_sam3_tracker_step_native_plugin.so";
    write_artifact(plugin_path, require_section(bundle, manifest.plugin_section),
                   manifest.plugin_sha256);
    for (const auto& package : manifest.packages)
        write_artifact(cache_directory / package.section, require_section(bundle, package.section),
                       package.sha256);
    for (const auto& package : memory_manifest.packages)
        write_artifact(cache_directory / package.section, require_section(bundle, package.section),
                       package.sha256);
    for (const auto& package : resize_manifest.packages)
        write_artifact(cache_directory / package.section, require_section(bundle, package.section),
                       package.sha256);
}

void* open_native_plugin(const std::filesystem::path& plugin_path) {
    dlerror();
    constexpr int load_flags = RTLD_NOW | RTLD_GLOBAL
#ifdef RTLD_NODELETE
                               | RTLD_NODELETE
#endif
        ;
    void* library = dlopen(plugin_path.c_str(), load_flags);
    if (library != nullptr)
        return library;
    const char* error = dlerror();
    throw std::runtime_error(std::string("SAM3 tracker-step native plugin load failed: ") +
                             (error != nullptr ? error : plugin_path.string()));
}

NativePluginApi load_native_plugin_api(void* library) {
    NativePluginApi api;
    api.plugin_version =
        require_symbol<PluginVersion>(library, "trtmc_sam3_tracker_step_plugin_version");
    api.aoti_abi =
        require_symbol<AotiAbiVersion>(library, "trtmc_sam3_tracker_step_aoti_abi_version");
    api.register_pipeline =
        require_symbol<RegisterPipeline>(library, "trtmc_sam3_tracker_step_register_pipeline");
    api.register_memory_package = require_symbol<RegisterMemoryPackage>(
        library, "trtmc_sam3_tracker_memory_register_package");
    api.torch_version =
        require_symbol<DependencyVersion>(library, "trtmc_sam3_tracker_step_torch_version");
    api.tvm_ffi_version =
        require_symbol<DependencyVersion>(library, "trtmc_sam3_tracker_step_tvm_ffi_version");
    api.tensorrt_version =
        require_symbol<DependencyVersion>(library, "trtmc_sam3_tracker_step_tensorrt_version");
    api.torch_cxx11_abi =
        require_symbol<TorchCxx11Abi>(library, "trtmc_sam3_tracker_step_torch_cxx11_abi");
    return api;
}

bool version_matches(const std::string& expected, DependencyVersion actual) {
    return actual() != nullptr && expected == actual();
}

void validate_native_plugin_api(const NativePluginApi& api,
                                const Sam3TrackerStepRuntimeManifest& manifest) {
    const int32_t cxx11_abi = api.torch_cxx11_abi();
    if (!version_matches(manifest.plugin_version, api.plugin_version) ||
        !version_matches(manifest.torch_version, api.torch_version) ||
        !version_matches(manifest.tvm_ffi_version, api.tvm_ffi_version) ||
        !version_matches(manifest.tensorrt_version, api.tensorrt_version) ||
        (cxx11_abi != 0 && cxx11_abi != 1) ||
        manifest.torch_cxx11_abi != static_cast<bool>(cxx11_abi) ||
        manifest.aoti_abi_version != api.aoti_abi())
        throw std::runtime_error("SAM3 tracker-step native plugin ABI mismatch");
}

const Sam3TrackerStepPackageSpec& find_package(const Sam3TrackerStepRuntimeManifest& manifest,
                                               int32_t batch_size, std::string_view stage) {
    const Sam3TrackerStepPackageSpec* match = nullptr;
    for (const auto& package : manifest.packages) {
        if (package.batch_size == batch_size && package.stage == stage) {
            if (match != nullptr)
                throw std::runtime_error("SAM3 tracker-step package pairing is ambiguous");
            match = &package;
        }
    }
    if (match == nullptr)
        throw std::runtime_error("SAM3 tracker-step pipeline is missing a stage package");
    return *match;
}

void assign_compute_capability(const nlohmann::json& producer, std::string_view key, int32_t& major,
                               int32_t& minor, const char* error_message) {
    const auto capability = producer.at(key).get<std::vector<int32_t>>();
    if (capability.size() != 2)
        throw std::runtime_error(error_message);
    major = capability[0];
    minor = capability[1];
}

Sam3TrackerStepRuntimeManifest parse_step_manifest_header(const nlohmann::json& parsed) {
    Sam3TrackerStepRuntimeManifest manifest;
    manifest.schema_version = parsed.at("schema_version").get<int32_t>();
    manifest.step_scope = parsed.at("step_scope").get<std::string>();
    const auto& plugin = parsed.at("plugin");
    manifest.plugin_section = plugin.at("section").get<std::string>();
    manifest.plugin_sha256 = plugin.at("sha256").get<std::string>();
    manifest.plugin_type = plugin.at("type").get<std::string>();
    manifest.plugin_version = plugin.at("version").get<std::string>();
    const auto& producer = parsed.at("producer");
    manifest.torch_version = producer.at("torch_version").get<std::string>();
    manifest.transformers_version = producer.at("transformers_version").get<std::string>();
    manifest.tvm_ffi_version = producer.at("tvm_ffi_version").get<std::string>();
    manifest.tensorrt_version = producer.at("tensorrt_version").get<std::string>();
    manifest.cuda_version = producer.at("cuda_version").get<std::string>();
    manifest.host_architecture = producer.at("host_architecture").get<std::string>();
    manifest.torch_cxx11_abi = producer.at("torch_cxx11_abi").get<bool>();
    manifest.aoti_abi_version = producer.at("aoti_abi_version").get<uint64_t>();
    assign_compute_capability(producer, "compute_capability", manifest.compute_capability_major,
                              manifest.compute_capability_minor,
                              "SAM3 tracker-step compute capability must have two components");
    validate_manifest_fields(manifest);
    return manifest;
}

void validate_step_package_uniqueness(const Sam3TrackerStepPackageSpec& package,
                                      std::unordered_set<std::string>& stage_batches,
                                      std::unordered_set<std::string>& package_globals,
                                      std::unordered_set<std::string>& sections) {
    const std::string stage_batch = package.stage + ":" + std::to_string(package.batch_size);
    if (!stage_batches.insert(stage_batch).second)
        throw std::runtime_error("SAM3 tracker-step package entries must be unique");
    if (!package_globals.insert(package.package_global).second)
        throw std::runtime_error("SAM3 tracker-step package entries must be unique");
    if (!sections.insert(package.section).second)
        throw std::runtime_error("SAM3 tracker-step package entries must be unique");
}

void parse_step_packages(const BundleFile& bundle, const nlohmann::json& parsed,
                         Sam3TrackerStepRuntimeManifest& manifest) {
    const auto& packages = parsed.at("packages");
    if (!packages.is_array())
        throw std::runtime_error(
            "SAM3 tracker-step runtime requires encoder/decoder packages for B1 and B2");
    if (packages.size() != manifest.packages.size())
        throw std::runtime_error(
            "SAM3 tracker-step runtime requires encoder/decoder packages for B1 and B2");
    std::unordered_set<std::string> stage_batches;
    std::unordered_set<std::string> package_globals;
    std::unordered_set<std::string> sections;
    for (std::size_t index = 0; index < manifest.packages.size(); ++index) {
        manifest.packages[index] = parse_package(packages.at(index));
        const auto& package = manifest.packages[index];
        validate_step_package_uniqueness(package, stage_batches, package_globals, sections);
        require_sha(require_section(bundle, package.section), package.sha256, package.section);
    }
}

void validate_step_pipeline_uniqueness(const Sam3TrackerStepPipelineSpec& pipeline,
                                       std::unordered_set<int32_t>& pipeline_batches,
                                       std::unordered_set<std::string>& pipeline_globals) {
    if (!pipeline_batches.insert(pipeline.batch_size).second)
        throw std::runtime_error("SAM3 tracker-step pipeline entries must be unique");
    if (!pipeline_globals.insert(pipeline.global_name).second)
        throw std::runtime_error("SAM3 tracker-step pipeline entries must be unique");
}

void validate_step_pipeline_packages(const Sam3TrackerStepRuntimeManifest& manifest,
                                     const Sam3TrackerStepPipelineSpec& pipeline) {
    const auto& encoder = find_package(manifest, pipeline.batch_size, "encoder");
    const auto& decoder = find_package(manifest, pipeline.batch_size, "decoder");
    if (pipeline.encoder_sha256 != encoder.sha256)
        throw std::runtime_error(
            "SAM3 tracker-step pipeline does not reference its batch packages");
    if (pipeline.decoder_sha256 != decoder.sha256)
        throw std::runtime_error(
            "SAM3 tracker-step pipeline does not reference its batch packages");
}

void parse_step_pipelines(const nlohmann::json& parsed, Sam3TrackerStepRuntimeManifest& manifest) {
    const auto& pipelines = parsed.at("pipelines");
    if (!pipelines.is_array())
        throw std::runtime_error("SAM3 tracker-step runtime requires B1 and B2 pipelines");
    if (pipelines.size() != manifest.pipelines.size())
        throw std::runtime_error("SAM3 tracker-step runtime requires B1 and B2 pipelines");
    std::unordered_set<int32_t> pipeline_batches;
    std::unordered_set<std::string> pipeline_globals;
    for (std::size_t index = 0; index < manifest.pipelines.size(); ++index) {
        manifest.pipelines[index] = parse_pipeline(pipelines.at(index));
        const auto& pipeline = manifest.pipelines[index];
        validate_step_pipeline_uniqueness(pipeline, pipeline_batches, pipeline_globals);
        validate_step_pipeline_packages(manifest, pipeline);
    }
}

void validate_memory_manifest_contract(const nlohmann::json& parsed) {
    if (!parsed.is_object())
        throw std::runtime_error("SAM3 tracker-memory AOTI manifest contract mismatch");
    if (parsed.size() != 11)
        throw std::runtime_error("SAM3 tracker-memory AOTI manifest contract mismatch");
    if (parsed.at("implementation") != expected_memory_implementation())
        throw std::runtime_error("SAM3 tracker-memory AOTI manifest contract mismatch");
    if (parsed.at("input_abi") != expected_memory_input_abi())
        throw std::runtime_error("SAM3 tracker-memory AOTI manifest contract mismatch");
    if (parsed.at("mask_policy") != expected_memory_mask_policy())
        throw std::runtime_error("SAM3 tracker-memory AOTI manifest contract mismatch");
}

void validate_memory_manifest_identity(const Sam3TrackerMemoryAotiManifest& manifest) {
    if (manifest.schema_version != 2)
        throw std::runtime_error("SAM3 tracker-memory AOTI manifest is incompatible");
    if (manifest.scope != kSam3TrackerMemoryScope)
        throw std::runtime_error("SAM3 tracker-memory AOTI manifest is incompatible");
    if (manifest.artifact_format != "torch.aot_inductor.package.pt2")
        throw std::runtime_error("SAM3 tracker-memory AOTI manifest is incompatible");
    if (!is_sha256(manifest.model_sha256))
        throw std::runtime_error("SAM3 tracker-memory AOTI manifest is incompatible");
    if (!is_sha256(manifest.exporter_sha256))
        throw std::runtime_error("SAM3 tracker-memory AOTI manifest is incompatible");
}

void parse_memory_manifest_producer(const nlohmann::json& producer,
                                    Sam3TrackerMemoryAotiManifest& manifest) {
    if (!producer.is_object())
        throw std::runtime_error("SAM3 tracker-memory producer ABI is incomplete");
    if (producer.size() != 7)
        throw std::runtime_error("SAM3 tracker-memory producer ABI is incomplete");
    manifest.torch_version = producer.at("torch_version").get<std::string>();
    manifest.transformers_version = producer.at("transformers_version").get<std::string>();
    manifest.cuda_version = producer.at("cuda_version").get<std::string>();
    manifest.host_architecture = producer.at("host_architecture").get<std::string>();
    manifest.torch_cxx11_abi = producer.at("torch_cxx11_abi").get<bool>();
    manifest.aoti_abi_version = producer.at("torch_aoti_abi_version").get<uint64_t>();
    assign_compute_capability(producer, "compute_capability", manifest.compute_capability_major,
                              manifest.compute_capability_minor,
                              "SAM3 tracker-memory compute capability is incomplete");
}

Sam3TrackerMemoryAotiManifest
parse_memory_manifest_header(const nlohmann::json& parsed,
                             const Sam3TrackerStepRuntimeManifest& step_manifest) {
    Sam3TrackerMemoryAotiManifest manifest;
    manifest.schema_version = parsed.at("schema_version").get<int32_t>();
    manifest.scope = parsed.at("scope").get<std::string>();
    manifest.artifact_format = parsed.at("artifact_format").get<std::string>();
    manifest.model_sha256 = parsed.at("model_sha256").get<std::string>();
    manifest.exporter_sha256 = parsed.at("exporter_sha256").get<std::string>();
    validate_memory_manifest_identity(manifest);
    parse_memory_manifest_producer(parsed.at("producer"), manifest);
    validate_memory_producer_matches_step(manifest, step_manifest);
    return manifest;
}

void validate_memory_package_uniqueness(const Sam3TrackerMemoryPackageSpec& package,
                                        std::unordered_set<std::string>& variants,
                                        std::unordered_set<std::string>& globals,
                                        std::unordered_set<std::string>& sections) {
    const std::string variant = package.policy + ":" + std::to_string(package.batch_size);
    if (!variants.insert(variant).second)
        throw std::runtime_error("SAM3 tracker-memory package entries must be unique");
    if (!globals.insert(package.package_global).second)
        throw std::runtime_error("SAM3 tracker-memory package entries must be unique");
    if (!sections.insert(package.section).second)
        throw std::runtime_error("SAM3 tracker-memory package entries must be unique");
}

void parse_memory_packages(const BundleFile& bundle, const nlohmann::json& parsed,
                           Sam3TrackerMemoryAotiManifest& manifest) {
    const auto& packages = parsed.at("packages");
    if (!packages.is_array())
        throw std::runtime_error("SAM3 tracker-memory runtime requires soft/hard B1/B2 packages");
    if (packages.size() != manifest.packages.size())
        throw std::runtime_error("SAM3 tracker-memory runtime requires soft/hard B1/B2 packages");
    std::unordered_set<std::string> variants;
    std::unordered_set<std::string> globals;
    std::unordered_set<std::string> sections;
    for (std::size_t index = 0; index < manifest.packages.size(); ++index) {
        manifest.packages[index] = parse_memory_package(packages.at(index));
        const auto& package = manifest.packages[index];
        validate_memory_package_uniqueness(package, variants, globals, sections);
        require_sha(require_section(bundle, package.section), package.sha256, package.section);
    }
    const std::unordered_set<std::string> expected_variants{"soft:1", "hard:1", "soft:2", "hard:2"};
    if (variants != expected_variants)
        throw std::runtime_error(
            "SAM3 tracker-memory runtime requires exactly soft/hard B1/B2 packages");
}

bool has_valid_resize_producer(const Sam3HardMaskResizeAotiManifest& resize) {
    return is_known_version(resize.torch_version) &&
           is_known_version(resize.transformers_version) && is_known_version(resize.cuda_version) &&
           !resize.host_architecture.empty() && resize.aoti_abi_version != 0 &&
           resize.compute_capability_major > 0 && resize.compute_capability_minor >= 0;
}

bool resize_producer_matches_step(const Sam3HardMaskResizeAotiManifest& resize,
                                  const Sam3TrackerStepRuntimeManifest& step) {
    return resize.torch_version == step.torch_version &&
           resize.transformers_version == step.transformers_version &&
           resize.cuda_version == step.cuda_version &&
           resize.host_architecture == step.host_architecture &&
           resize.torch_cxx11_abi == step.torch_cxx11_abi &&
           resize.aoti_abi_version == step.aoti_abi_version &&
           resize.compute_capability_major == step.compute_capability_major &&
           resize.compute_capability_minor == step.compute_capability_minor;
}

void validate_resize_producer_matches_step(const Sam3HardMaskResizeAotiManifest& resize,
                                           const Sam3TrackerStepRuntimeManifest& step) {
    if (!has_valid_resize_producer(resize) || !resize_producer_matches_step(resize, step)) {
        throw std::runtime_error("SAM3 hard-mask resize/step producer ABI mismatch");
    }
}

void validate_resize_manifest_contract(const nlohmann::json& parsed) {
    if (!parsed.is_object() || parsed.size() != 11 ||
        parsed.at("implementation") != expected_resize_implementation() ||
        parsed.at("input_abi") != expected_resize_input_abi() ||
        parsed.at("output_abi") != expected_resize_output_abi()) {
        throw std::runtime_error("SAM3 hard-mask resize AOTI manifest contract mismatch");
    }
}

Sam3HardMaskResizeAotiManifest
parse_resize_manifest_header(const nlohmann::json& parsed,
                             const Sam3TrackerStepRuntimeManifest& step_manifest) {
    Sam3HardMaskResizeAotiManifest manifest;
    manifest.schema_version = parsed.at("schema_version").get<int32_t>();
    manifest.scope = parsed.at("scope").get<std::string>();
    manifest.artifact_format = parsed.at("artifact_format").get<std::string>();
    manifest.exporter_sha256 = parsed.at("exporter_sha256").get<std::string>();
    if (manifest.schema_version != 1 || manifest.scope != kSam3HardMaskResizeScope ||
        manifest.artifact_format != "torch.aot_inductor.package.pt2" ||
        !is_sha256(manifest.exporter_sha256)) {
        throw std::runtime_error("SAM3 hard-mask resize AOTI manifest is incompatible");
    }
    const auto& producer = parsed.at("producer");
    if (!producer.is_object() || producer.size() != 7)
        throw std::runtime_error("SAM3 hard-mask resize producer ABI is incomplete");
    manifest.torch_version = producer.at("torch_version").get<std::string>();
    manifest.transformers_version = producer.at("transformers_version").get<std::string>();
    manifest.cuda_version = producer.at("cuda_version").get<std::string>();
    manifest.host_architecture = producer.at("host_architecture").get<std::string>();
    manifest.torch_cxx11_abi = producer.at("torch_cxx11_abi").get<bool>();
    manifest.aoti_abi_version = producer.at("torch_aoti_abi_version").get<uint64_t>();
    assign_compute_capability(producer, "compute_capability", manifest.compute_capability_major,
                              manifest.compute_capability_minor,
                              "SAM3 hard-mask resize compute capability is incomplete");
    if (parsed.at("host_architecture").get<std::string>() != manifest.host_architecture)
        throw std::runtime_error("SAM3 hard-mask resize host architecture is inconsistent");
    validate_resize_producer_matches_step(manifest, step_manifest);
    return manifest;
}

void parse_resize_packages(const BundleFile& bundle, const nlohmann::json& parsed,
                           Sam3HardMaskResizeAotiManifest& manifest) {
    const auto& packages = parsed.at("packages");
    if (!packages.is_array() || packages.size() != manifest.packages.size())
        throw std::runtime_error("SAM3 hard-mask resize runtime requires B1/B2 packages");
    std::unordered_set<int32_t> batches;
    std::unordered_set<std::string> globals;
    std::unordered_set<std::string> sections;
    for (std::size_t index = 0; index < manifest.packages.size(); ++index) {
        manifest.packages[index] = parse_resize_package(packages.at(index));
        const auto& package = manifest.packages[index];
        if (!batches.insert(package.batch_size).second ||
            !globals.insert(package.package_global).second ||
            !sections.insert(package.section).second) {
            throw std::runtime_error("SAM3 hard-mask resize package entries must be unique");
        }
        require_sha(require_section(bundle, package.section), package.sha256, package.section);
    }
    if (batches != std::unordered_set<int32_t>{1, 2})
        throw std::runtime_error("SAM3 hard-mask resize runtime requires exactly B1/B2 packages");
}

void validate_resize_validation_header(const nlohmann::json& validation,
                                       double maximum_absolute_error) {
    if (!validation.is_object() || validation.size() != 3 ||
        validation.at("reference").get<std::string>() != "same torch.interpolate eager execution" ||
        validation.at("maximum_absolute_error").get<double>() != maximum_absolute_error) {
        throw std::runtime_error("SAM3 hard-mask resize validation contract mismatch");
    }
}

void validate_resize_validation_case(const nlohmann::json& value, double maximum_absolute_error,
                                     std::unordered_set<int32_t>& batches) {
    if (!value.is_object() || value.size() != 3 || !value.at("passed").get<bool>())
        throw std::runtime_error("SAM3 hard-mask resize validation case failed");
    const int32_t batch_size = value.at("batch_size").get<int32_t>();
    const double error = value.at("maximum_absolute_error").get<double>();
    if (!is_supported_tracker_batch(batch_size) || !batches.insert(batch_size).second ||
        !std::isfinite(error) || error > maximum_absolute_error) {
        throw std::runtime_error("SAM3 hard-mask resize validation case failed");
    }
}

void validate_resize_package_validation(const nlohmann::json& validation) {
    constexpr double kMaximumAbsoluteError = 2.0e-5;
    validate_resize_validation_header(validation, kMaximumAbsoluteError);
    const auto& cases = validation.at("cases");
    if (!cases.is_array() || cases.size() != 2)
        throw std::runtime_error("SAM3 hard-mask resize validation is incomplete");
    std::unordered_set<int32_t> batches;
    for (const auto& value : cases)
        validate_resize_validation_case(value, kMaximumAbsoluteError, batches);
}

void register_runtime_packages(void* library, const Sam3TrackerStepRuntimeManifest& manifest,
                               const Sam3TrackerMemoryAotiManifest& memory_manifest,
                               const Sam3HardMaskResizeAotiManifest& resize_manifest,
                               const std::filesystem::path& cache_directory) {
    const auto api = load_native_plugin_api(library);
    validate_native_plugin_api(api, manifest);
    for (const auto& package : memory_manifest.packages) {
        const auto package_path = cache_directory / package.section;
        if (api.register_memory_package(package.package_global.c_str(), package_path.c_str(),
                                        package.sha256.c_str(), package.policy.c_str(),
                                        package.batch_size) != 0)
            throw std::runtime_error("SAM3 tracker-memory AOTI package registration failed");
    }
    for (const auto& package : resize_manifest.packages) {
        const auto package_path = cache_directory / package.section;
        if (api.register_memory_package(package.package_global.c_str(), package_path.c_str(),
                                        package.sha256.c_str(), "resize",
                                        package.batch_size) != 0) {
            throw std::runtime_error("SAM3 hard-mask resize AOTI package registration failed");
        }
    }
    for (const auto& pipeline : manifest.pipelines) {
        const auto& encoder = find_package(manifest, pipeline.batch_size, "encoder");
        const auto& decoder = find_package(manifest, pipeline.batch_size, "decoder");
        const auto encoder_path = cache_directory / encoder.section;
        const auto decoder_path = cache_directory / decoder.section;
        if (api.register_pipeline(pipeline.global_name.c_str(), encoder_path.c_str(),
                                  decoder_path.c_str(), encoder.sha256.c_str(),
                                  decoder.sha256.c_str(), pipeline.batch_size) != 0)
            throw std::runtime_error("SAM3 tracker-step AOTI pipeline registration failed");
    }
}

std::mutex load_mutex;
std::unordered_map<std::string, void*> loaded_plugins_by_manifest_and_device;

} // namespace

std::string sam3_tracker_step_sha256_hex(const std::vector<char>& data) {
    auto state = kSha256InitialState;
    const auto padded = pad_sha256_message(data);
    for (std::size_t offset = 0; offset < padded.size(); offset += 64U)
        compress_sha256_block(state, padded.data() + offset);
    return sha256_state_hex(state);
}

Sam3TrackerStepRuntimeManifest
validate_sam3_tracker_step_runtime_manifest(const BundleFile& bundle) {
    const auto& manifest_bytes = require_section(bundle, kSam3TrackerStepRuntimeManifestSection);
    const auto parsed = nlohmann::json::parse(manifest_bytes.begin(), manifest_bytes.end());
    auto manifest = parse_step_manifest_header(parsed);
    parse_step_packages(bundle, parsed, manifest);
    parse_step_pipelines(parsed, manifest);
    require_sha(require_section(bundle, manifest.plugin_section), manifest.plugin_sha256,
                manifest.plugin_section);
    return manifest;
}

Sam3TrackerMemoryAotiManifest
validate_sam3_tracker_memory_aoti_manifest(const BundleFile& bundle,
                                           const Sam3TrackerStepRuntimeManifest& step_manifest) {
    const auto& manifest_bytes = require_section(bundle, kSam3TrackerMemoryAotiManifestSection);
    const auto parsed = nlohmann::json::parse(manifest_bytes.begin(), manifest_bytes.end());
    validate_memory_manifest_contract(parsed);
    auto manifest = parse_memory_manifest_header(parsed, step_manifest);
    parse_memory_packages(bundle, parsed, manifest);
    validate_memory_package_validation(parsed.at("package_validation"));
    return manifest;
}

Sam3HardMaskResizeAotiManifest
validate_sam3_hard_mask_resize_aoti_manifest(const BundleFile& bundle,
                                             const Sam3TrackerStepRuntimeManifest& step_manifest) {
    const auto& manifest_bytes = require_section(bundle, kSam3HardMaskResizeAotiManifestSection);
    const auto parsed = nlohmann::json::parse(manifest_bytes.begin(), manifest_bytes.end());
    validate_resize_manifest_contract(parsed);
    auto manifest = parse_resize_manifest_header(parsed, step_manifest);
    parse_resize_packages(bundle, parsed, manifest);
    validate_resize_package_validation(parsed.at("package_validation"));
    return manifest;
}

void load_sam3_tracker_step_runtime(const BundleFile& bundle) {
    const auto manifest = validate_sam3_tracker_step_runtime_manifest(bundle);
    const auto memory_manifest = validate_sam3_tracker_memory_aoti_manifest(bundle, manifest);
    const auto resize_manifest = validate_sam3_hard_mask_resize_aoti_manifest(bundle, manifest);
    const auto& manifest_bytes = require_section(bundle, kSam3TrackerStepRuntimeManifestSection);
    const auto& memory_manifest_bytes =
        require_section(bundle, kSam3TrackerMemoryAotiManifestSection);
    const auto& resize_manifest_bytes =
        require_section(bundle, kSam3HardMaskResizeAotiManifestSection);
    std::vector<char> combined_manifest_bytes(manifest_bytes.begin(), manifest_bytes.end());
    combined_manifest_bytes.push_back('\0');
    combined_manifest_bytes.insert(combined_manifest_bytes.end(), memory_manifest_bytes.begin(),
                                   memory_manifest_bytes.end());
    combined_manifest_bytes.push_back('\0');
    combined_manifest_bytes.insert(combined_manifest_bytes.end(), resize_manifest_bytes.begin(),
                                   resize_manifest_bytes.end());
    const std::string manifest_sha = sam3_tracker_step_sha256_hex(combined_manifest_bytes);
    const int32_t device_id = validate_runtime_device(manifest);
    const std::string load_key = manifest_sha + ":cuda:" + std::to_string(device_id);
    std::lock_guard lock(load_mutex);
    if (loaded_plugins_by_manifest_and_device.find(load_key) !=
        loaded_plugins_by_manifest_and_device.end())
        return;

    const auto cache_directory = artifact_cache_directory(manifest_sha);
    materialize_runtime_artifacts(bundle, manifest, memory_manifest, resize_manifest,
                                  cache_directory);
    const auto plugin_path = cache_directory / "libtrtmc_sam3_tracker_step_native_plugin.so";
    void* library = open_native_plugin(plugin_path);
    register_runtime_packages(library, manifest, memory_manifest, resize_manifest, cache_directory);
    loaded_plugins_by_manifest_and_device.emplace(load_key, library);
}

} // namespace trtmc
