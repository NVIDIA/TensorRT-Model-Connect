/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/artifact_contract.h"

#include "bundle/bundle_format.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace trtmc::wan2_2_ti2v {
namespace {

constexpr std::string_view kWan22ArtifactSchemaV4 = "trtmc.wan2_2_ti2v.bundle-artifacts.v4";
constexpr std::array<std::string_view, 7> kWan22RequiredSections = {
    "text_encoder_0_plan", "denoiser_plan",
    "vae_decoder_plan",    "vae_decoder_first_frame_plan",
    "tokenizer.json",      "wan2_2_ti2v_plugins.so",
    "config.json",
};
constexpr std::array<std::string_view, 6> kWan22ManifestSections = {
    "text_encoder_0_plan",          "denoiser_plan",  "vae_decoder_plan",
    "vae_decoder_first_frame_plan", "tokenizer.json", "wan2_2_ti2v_plugins.so",
};

bool json_object_has_exact_keys(const nlohmann::json& object,
                                std::initializer_list<std::string_view> expected) {
    if (!object.is_object() || object.size() != expected.size())
        return false;
    for (const auto key : expected) {
        if (!object.contains(std::string(key)))
            return false;
    }
    return true;
}

bool is_lowercase_sha256(const nlohmann::json& value) {
    if (!value.is_string())
        return false;
    const auto& text = value.get_ref<const std::string&>();
    return text.size() == 64 && std::all_of(text.begin(), text.end(), [](unsigned char character) {
               return std::isdigit(character) != 0 ||
                      (character >= static_cast<unsigned char>('a') &&
                       character <= static_cast<unsigned char>('f'));
           });
}

std::uint64_t require_nonzero_size(const nlohmann::json& value, const std::string& section_name) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        throw std::runtime_error("Wan2.2 artifact size is not an integer for " + section_name);
    }
    if (value.is_number_integer() && value.get<std::int64_t>() <= 0) {
        throw std::runtime_error("Wan2.2 artifact size must be positive for " + section_name);
    }
    const auto result = value.get<std::uint64_t>();
    if (result == 0)
        throw std::runtime_error("Wan2.2 artifact size must be positive for " + section_name);
    return result;
}

std::string hash_bundle_section(BundleSectionReader& reader, const std::string& section_name) {
    detail::Sha256 digest;
    reader.for_each_chunk(section_name, 4U << 20U, [&digest](const char* data, std::size_t size) {
        digest.update(data, size);
    });
    return digest.hex_digest();
}

template <std::size_t Size>
nlohmann::json string_array_json(const std::array<std::string_view, Size>& values) {
    nlohmann::json result = nlohmann::json::array();
    for (const auto value : values)
        result.push_back(value);
    return result;
}

void validate_artifact_provenance(BundleSectionReader& reader, const nlohmann::json& config,
                                  std::size_t materialized_config_size) {
    const auto& info = reader.info();
    if (info.sections.size() != kWan22RequiredSections.size()) {
        throw std::runtime_error("Wan2.2 provenance requires exactly seven bundle sections");
    }

    std::unordered_map<std::string, const BundleSectionInfo*> section_info;
    section_info.reserve(info.sections.size());
    for (const auto& section : info.sections) {
        if (section.name.empty() || !section_info.emplace(section.name, &section).second) {
            throw std::runtime_error("Duplicate or empty Wan2.2 bundle section: " + section.name);
        }
    }
    for (const auto required : kWan22RequiredSections) {
        if (section_info.find(std::string(required)) == section_info.end()) {
            throw std::runtime_error("Wan2.2 provenance is missing bundle section: " +
                                     std::string(required));
        }
    }
    if (section_info.at("config.json")->size != materialized_config_size) {
        throw std::runtime_error("Wan2.2 config.json size disagrees with bundle metadata");
    }

    const auto contract = config.find("runtime_contract");
    if (contract == config.end() || !contract->is_object() ||
        !contract->contains("required_bundle_sections") ||
        (*contract)["required_bundle_sections"] != string_array_json(kWan22RequiredSections)) {
        throw std::runtime_error(
            "Wan2.2 runtime_contract must declare the exact seven-section integrity contract");
    }
    if (contract->value("bundle_trust_model", std::string{}) != "trusted_executable_artifact" ||
        !contract->contains("executable_bundle_sections") ||
        (*contract)["executable_bundle_sections"] !=
            nlohmann::json::array({"wan2_2_ti2v_plugins.so"})) {
        throw std::runtime_error(
            "Wan2.2 runtime_contract must declare its trusted executable bundle section");
    }

    const auto manifest_iterator = config.find("artifact_manifest");
    if (manifest_iterator == config.end() ||
        !json_object_has_exact_keys(*manifest_iterator,
                                    {"schema", "family", "profile", "runtime", "sections"})) {
        throw std::runtime_error("Wan2.2 artifact_manifest is missing or malformed");
    }
    const auto& manifest = *manifest_iterator;
    const std::string schema = manifest.value("schema", std::string{});
    if (schema != kWan22ArtifactSchemaV4 ||
        manifest.value("family", std::string{}) != "wan2_2_ti2v" ||
        manifest.value("runtime", std::string{}) != "native_cpp_cuda_tensorrt") {
        throw std::runtime_error("Wan2.2 artifact_manifest identity is invalid");
    }
    if (contract->value("artifact_integrity", std::string{}) != "sha256_size_v1") {
        throw std::runtime_error(
            "Wan2.2 v4 artifact manifest requires sha256_size_v1 integrity mode");
    }
    const nlohmann::json expected_profile = {
        {"video_width", 1280},
        {"video_height", 704},
        {"video_num_frames", 121},
        {"latent_shape", {1, 48, 31, 44, 80}},
        {"architecture",
         {{"model_type", "ti2v"},
          {"in_channels", 48},
          {"out_channels", 48},
          {"dim", 3072},
          {"ffn_dim", 14336},
          {"freq_dim", 256},
          {"num_heads", 24},
          {"num_layers", 30},
          {"head_dim", 128},
          {"text_dim", 4096},
          {"text_seq_len", 512},
          {"eps", 1e-6},
          {"patch_size", {1, 2, 2}},
          {"z_dim", 48},
          {"scale_factor_temporal", 4},
          {"scale_factor_spatial", 16},
          {"frame_rate", 24},
          {"num_inference_steps", 50},
          {"guidance_scale", 5.0},
          {"flow_shift", 5.0},
          {"train_timesteps", 1000}}},
        {"text_seq_len", 512},
        {"text_encoder_dim", 4096},
        {"text_encoder_numerics",
         {{"shape", {1, 512, 4096}},
          {"num_heads", 64},
          {"epsilon", 1e-6},
          {"source_softmax", true},
          {"source_rmsnorm", true}}},
        {"precision", "bf16"},
    };
    if (manifest["profile"] != expected_profile)
        throw std::runtime_error("Wan2.2 artifact_manifest profile is not the official profile");

    const auto plugin_contract = config.find("_trtmc_wan22_plugin_contract");
    if (plugin_contract == config.end() || !plugin_contract->is_object()) {
        throw std::runtime_error("Wan2.2 config is missing the AOT plugin contract");
    }
    const std::string canonical_plugin_contract = plugin_contract->dump();
    detail::Sha256 plugin_contract_sha256;
    plugin_contract_sha256.update(canonical_plugin_contract.data(),
                                  canonical_plugin_contract.size());
    const std::string expected_plugin_contract_digest = plugin_contract_sha256.hex_digest();

    const auto& sections = manifest["sections"];
    if (!sections.is_object() || sections.size() != kWan22ManifestSections.size()) {
        throw std::runtime_error(
            "Wan2.2 artifact_manifest must describe exactly six model-owned sections");
    }
    for (const auto expected : kWan22ManifestSections) {
        if (!sections.contains(std::string(expected))) {
            throw std::runtime_error("Wan2.2 artifact_manifest is missing section: " +
                                     std::string(expected));
        }
    }
    const auto& plugin_entry = sections["wan2_2_ti2v_plugins.so"];
    if (!plugin_entry.is_object() || !plugin_entry.contains("sha256") ||
        !is_lowercase_sha256(plugin_entry["sha256"])) {
        throw std::runtime_error("Wan2.2 AOT plugin artifact digest is malformed");
    }
    const std::string expected_plugin_elf_digest = plugin_entry["sha256"].get<std::string>();

    for (const auto section_view : kWan22ManifestSections) {
        const std::string section_name(section_view);
        const auto& entry = sections[section_name];
        const bool is_plan = section_name.size() >= 5 &&
                             section_name.compare(section_name.size() - 5, 5, "_plan") == 0;
        const bool exact_keys =
            is_plan ? json_object_has_exact_keys(
                          entry, {"sha256", "size", "source_sha256", "source_inputs"})
                    : json_object_has_exact_keys(entry, {"sha256", "size"});
        if (!exact_keys || !is_lowercase_sha256(entry["sha256"])) {
            throw std::runtime_error("Wan2.2 artifact manifest entry is malformed for " +
                                     section_name);
        }

        const std::uint64_t header_size = section_info.at(section_name)->size;
        if (header_size == 0) {
            throw std::runtime_error("Wan2.2 artifact size must be positive for " + section_name);
        }
        const std::uint64_t expected_size = require_nonzero_size(entry["size"], section_name);
        if (header_size != expected_size) {
            throw std::runtime_error("Wan2.2 artifact size mismatch for " + section_name);
        }
        if (is_plan) {
            if (!is_lowercase_sha256(entry["source_sha256"]) ||
                !entry["source_inputs"].is_array() || entry["source_inputs"].empty()) {
                throw std::runtime_error("Wan2.2 source provenance is malformed for " +
                                         section_name);
            }
            bool has_bound_plugin_contract = false;
            bool has_bound_plugin_elf = false;
            std::unordered_map<std::string, std::string> source_digests;
            for (const auto& source : entry["source_inputs"]) {
                if (!json_object_has_exact_keys(source, {"name", "sha256"}) ||
                    !source["name"].is_string() || source["name"].get<std::string>().empty() ||
                    !is_lowercase_sha256(source["sha256"])) {
                    throw std::runtime_error("Wan2.2 source input is malformed for " +
                                             section_name);
                }
                const auto source_name = source["name"].get<std::string>();
                const auto source_digest = source["sha256"].get<std::string>();
                if (!source_digests.emplace(source_name, source_digest).second) {
                    throw std::runtime_error("Wan2.2 source inputs contain duplicates for " +
                                             section_name);
                }
                if (source_name == "plugin/contract.json") {
                    if (source_digest != expected_plugin_contract_digest) {
                        throw std::runtime_error("Wan2.2 plan is bound to a different AOT plugin "
                                                 "contract: " +
                                                 section_name);
                    }
                    has_bound_plugin_contract = true;
                }
                if (source_name == "plugin/elf") {
                    if (source_digest != expected_plugin_elf_digest) {
                        throw std::runtime_error("Wan2.2 plan is bound to a different AOT plugin "
                                                 "ELF: " +
                                                 section_name);
                    }
                    has_bound_plugin_elf = true;
                }
            }
            if (!has_bound_plugin_contract) {
                throw std::runtime_error(
                    "Wan2.2 plan source identity is missing plugin/contract.json: " + section_name);
            }
            if (!has_bound_plugin_elf) {
                throw std::runtime_error("Wan2.2 plan source identity is missing plugin/elf: " +
                                         section_name);
            }
            const nlohmann::json source_document = {
                {"family", "wan2_2_ti2v"},
                {"component", section_name},
                {"profile", manifest["profile"]},
                {"inputs", entry["source_inputs"]},
            };
            const std::string canonical_source = source_document.dump();
            detail::Sha256 source_digest;
            source_digest.update(canonical_source.data(), canonical_source.size());
            if (source_digest.hex_digest() != entry["source_sha256"].get<std::string>()) {
                throw std::runtime_error("Wan2.2 source identity mismatch for " + section_name);
            }
        }

        // Lazy plans are authenticated by the Wan model plugin after the AOT
        // plugin contract is validated and immediately before TensorRT
        // deserialization. Eager model data remains authenticated here.
        if (!is_plan &&
            hash_bundle_section(reader, section_name) != entry["sha256"].get<std::string>()) {
            throw std::runtime_error("Wan2.2 artifact SHA256 mismatch for " + section_name);
        }
    }
}

} // namespace

void validate_bundle_artifact_provenance(BundleSectionReader& reader,
                                         const std::string& config_json,
                                         std::size_t materialized_config_size) {
    nlohmann::json config;
    try {
        config = nlohmann::json::parse(config_json);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("Invalid Wan2.2 config.json: ") + error.what());
    }
    validate_artifact_provenance(reader, config, materialized_config_size);
}

} // namespace trtmc::wan2_2_ti2v
