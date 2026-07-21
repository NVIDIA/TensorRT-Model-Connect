/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-WAN22-ARTIFACT-CPP-01
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-WAN22-ARTIFACT-01
// Intent:         Validate Wan2.2 bundle provenance, lazy-plan policy, and fail-closed order
// Preconditions:  Synthetic source-bound Wan2.2 bundle fixtures
// Postconditions:  Eager artifacts are authenticated, plans stay lazy, and errors are exact
// =============================================================================

#include "../../test_helpers.h"
#include "bundle/bundle_format.h"
#include "runtime/models/wan2_2_ti2v/artifact_contract.h"
#include "runtime/registry/bundle_materialization.h"
#include "utils/sha256.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void check_equal(const std::string& actual, const std::string& expected, const char* name) {
    if (actual != expected) {
        std::cerr << "FAIL: " << name << "\nactual:   " << actual << "\nexpected: " << expected
                  << '\n';
        ++failures;
    }
}

template <typename Function>
void check_throws_exactly(Function&& function, const std::string& expected, const char* name) {
    try {
        function();
        std::cerr << "FAIL: " << name << " did not throw\n";
        ++failures;
    } catch (const std::runtime_error& error) {
        check_equal(error.what(), expected, name);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << name << " threw the wrong exception: " << error.what() << '\n';
        ++failures;
    }
}

const std::vector<std::string>& required_sections() {
    static const std::vector<std::string> sections = {
        "text_encoder_0_plan", "denoiser_plan",
        "vae_decoder_plan",    "vae_decoder_first_frame_plan",
        "tokenizer.json",      "wan2_2_ti2v_plugins.so",
        "config.json",
    };
    return sections;
}

bool is_plan_section(const std::string& name) {
    return name.size() >= 5 && name.compare(name.size() - 5, 5, "_plan") == 0;
}

std::string sha256(const std::vector<char>& payload) {
    trtmc::detail::Sha256 digest;
    digest.update(payload.data(), payload.size());
    return digest.hex_digest();
}

nlohmann::json official_profile() {
    return {
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
}

nlohmann::json l0_profile() {
    auto profile = official_profile();
    profile["video_width"] = 672;
    profile["video_height"] = 384;
    profile["video_num_frames"] = 5;
    profile["latent_shape"] = {1, 48, 2, 24, 42};
    profile["architecture"]["num_inference_steps"] = 15;
    return profile;
}

struct ProvenanceFixture {
    std::vector<std::string> section_order;
    std::map<std::string, std::vector<char>> payloads;
    nlohmann::json config;
};

ProvenanceFixture make_fixture(nlohmann::json profile = official_profile()) {
    ProvenanceFixture fixture;
    fixture.section_order = required_sections();
    const std::vector<std::string> model_owned(fixture.section_order.begin(),
                                               fixture.section_order.end() - 1);
    for (const auto& name : model_owned) {
        const std::string value = "authenticated:" + name;
        fixture.payloads[name] = std::vector<char>(value.begin(), value.end());
    }

    const nlohmann::json plugin_contract = {
        {"schema", 1},
        {"family", "wan2_2_ti2v"},
        {"semantic_abi", "wan22-ti2v-plugins-v1"},
        {"source_digest", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},
        {"creator_set", "A:1:;B:1:"},
        {"runtime_abi",
         {{"tensorrt_major", 11}, {"tensorrt_minor", 1}, {"cuda_major", 13}, {"cudnn_major", 9}}},
        {"cuda_architectures", {103, 110}},
    };
    const auto plugin_contract_text = plugin_contract.dump();
    const auto plugin_contract_digest =
        sha256(std::vector<char>(plugin_contract_text.begin(), plugin_contract_text.end()));
    const auto plugin_elf_digest = sha256(fixture.payloads.at("wan2_2_ti2v_plugins.so"));

    nlohmann::json manifest_sections = nlohmann::json::object();
    for (const auto& name : model_owned) {
        nlohmann::json entry = {
            {"sha256", sha256(fixture.payloads.at(name))},
            {"size", fixture.payloads.at(name).size()},
        };
        if (is_plan_section(name)) {
            const std::vector<char> source_bytes(name.begin(), name.end());
            nlohmann::json inputs = nlohmann::json::array(
                {{{"name", "source/test"}, {"sha256", sha256(source_bytes)}},
                 {{"name", "plugin/contract.json"}, {"sha256", plugin_contract_digest}},
                 {{"name", "plugin/elf"}, {"sha256", plugin_elf_digest}}});
            const nlohmann::json source_document = {
                {"family", "wan2_2_ti2v"},
                {"component", name},
                {"profile", profile},
                {"inputs", inputs},
            };
            const auto source_text = source_document.dump();
            entry["source_sha256"] =
                sha256(std::vector<char>(source_text.begin(), source_text.end()));
            entry["source_inputs"] = std::move(inputs);
        }
        manifest_sections[name] = std::move(entry);
    }

    fixture.config = {
        {"runtime_strategy", "diffusion_wan2_2_ti2v"},
        {"video_width", profile["video_width"]},
        {"video_height", profile["video_height"]},
        {"video_num_frames", profile["video_num_frames"]},
        {"num_inference_steps", profile["architecture"]["num_inference_steps"]},
        {"guidance_scale", profile["architecture"]["guidance_scale"]},
        {"flow_shift", profile["architecture"]["flow_shift"]},
        {"frame_rate", profile["architecture"]["frame_rate"]},
        {"text_seq_len", profile["text_seq_len"]},
        {"_trtmc_wan22_plugin_contract", plugin_contract},
        {"runtime_contract",
         {{"implementation", "native_cpp_cuda_tensorrt"},
          {"artifact_integrity", "sha256_size_v1"},
          {"bundle_trust_model", "trusted_executable_artifact"},
          {"executable_bundle_sections", {"wan2_2_ti2v_plugins.so"}},
          {"required_bundle_sections", required_sections()}}},
        {"bundle_loading",
         {{"mode", "staged"},
          {"eager_sections", {"tokenizer.json", "config.json"}},
          {"lazy_sections",
           {"wan2_2_ti2v_plugins.so", "text_encoder_0_plan", "denoiser_plan", "vae_decoder_plan",
            "vae_decoder_first_frame_plan"}}}},
        {"artifact_manifest",
         {{"schema", "trtmc.wan2_2_ti2v.bundle-artifacts.v4"},
          {"family", "wan2_2_ti2v"},
          {"profile", profile},
          {"runtime", "native_cpp_cuda_tensorrt"},
          {"sections", manifest_sections}}},
    };
    return fixture;
}

std::string write_fixture(const std::filesystem::path& path, ProvenanceFixture fixture,
                          bool shorten_config_metadata = false) {
    const auto config_text = fixture.config.dump();
    fixture.payloads["config.json"] = std::vector<char>(config_text.begin(), config_text.end());

    nlohmann::json header = {{"model_id", "wan22"}, {"sections", nlohmann::json::object()}};
    std::vector<char> combined;
    std::uint64_t offset = 0;
    for (const auto& name : fixture.section_order) {
        const auto& payload = fixture.payloads.at(name);
        auto metadata_size = payload.size();
        if (shorten_config_metadata && name == "config.json")
            --metadata_size;
        header["sections"][name] = {{"offset", offset}, {"size", metadata_size}};
        combined.insert(combined.end(), payload.begin(), payload.end());
        offset += payload.size();
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Failed to create synthetic Wan2.2 bundle");
    const auto header_text = header.dump();
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    trtmc_test::write_u64_le(output, header_text.size());
    output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    output.write(combined.data(), static_cast<std::streamsize>(combined.size()));
    return config_text;
}

const trtmc::BundleSection* find_materialized(const trtmc::BundleFile& bundle,
                                              const std::string& name) {
    for (const auto& section : bundle.sections) {
        if (section.name == name)
            return &section;
    }
    return nullptr;
}

void validate_direct(const std::filesystem::path& path, const std::string& config_text) {
    trtmc::BundleSectionReader reader(path.string());
    trtmc::wan2_2_ti2v::validate_bundle_artifact_provenance(reader, config_text,
                                                            config_text.size());
}

void expect_fixture_error(const std::filesystem::path& path, ProvenanceFixture fixture,
                          const std::string& expected, const char* name,
                          bool shorten_config_metadata = false) {
    const auto config_text = write_fixture(path, std::move(fixture), shorten_config_metadata);
    check_throws_exactly([&] { validate_direct(path, config_text); }, expected, name);
}

void replace_plugin_elf_binding(nlohmann::json& config, const std::string& digest) {
    auto& inputs = config["artifact_manifest"]["sections"]["denoiser_plan"]["source_inputs"];
    for (auto& input : inputs) {
        if (input["name"] == "plugin/elf")
            input["sha256"] = digest;
    }
}

void test_valid_bundle_and_lazy_plan_policy() {
    check(sha256(std::vector<char>{'a', 'b', 'c'}) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "bundle provenance SHA256 matches the standard abc vector");
    trtmc_test::TempDirGuard temporary;

    const auto valid_path = std::filesystem::path(temporary.path()) / "valid.trtfb";
    const auto valid_config = write_fixture(valid_path, make_fixture());
    trtmc::BundleSectionReader valid_reader(valid_path.string());
    const auto materialized = trtmc::detail::materialize_pipeline_bundle(valid_reader);
    trtmc::wan2_2_ti2v::validate_bundle_artifact_provenance(valid_reader, materialized.config_text,
                                                            materialized.config_text.size());
    check(materialized.config_text == valid_config, "materialized config is byte-exact");
    check(materialized.bundle.sections.size() == 2,
          "authenticated Wan bundle retains staged materialization");
    check(materialized.bundle.info.sections.size() == 7,
          "authenticated Wan bundle has the exact seven-section contract");
    const auto parsed_config = nlohmann::json::parse(materialized.config_text);
    check(parsed_config["artifact_manifest"]["sections"].size() == 6,
          "authenticated Wan manifest has six model-owned sections");
    check(parsed_config["artifact_manifest"]["sections"].contains("wan2_2_ti2v_plugins.so"),
          "authenticated Wan manifest owns the embedded plugin image");
    check(find_materialized(materialized.bundle, "denoiser_plan") == nullptr,
          "authenticated plan remains lazy");
    check(find_materialized(materialized.bundle, "wan2_2_ti2v_plugins.so") == nullptr,
          "preflight-authenticated plugin image is not retained in memory");

    const auto valid_l0_path = std::filesystem::path(temporary.path()) / "valid-l0.trtfb";
    const auto valid_l0_config = write_fixture(valid_l0_path, make_fixture(l0_profile()));
    validate_direct(valid_l0_path, valid_l0_config);
    const auto parsed_l0 = nlohmann::json::parse(valid_l0_config);
    check(parsed_l0["artifact_manifest"]["profile"] == l0_profile(),
          "authenticated Wan bundle accepts the exact L0 artifact profile");

    auto lazy_tamper = make_fixture();
    lazy_tamper.payloads.at("denoiser_plan").front() ^= 0x01;
    const auto lazy_path = std::filesystem::path(temporary.path()) / "lazy-tamper.trtfb";
    const auto lazy_config = write_fixture(lazy_path, std::move(lazy_tamper));
    trtmc::BundleSectionReader lazy_reader(lazy_path.string());
    const auto lazy_materialized = trtmc::detail::materialize_pipeline_bundle(lazy_reader);
    trtmc::wan2_2_ti2v::validate_bundle_artifact_provenance(
        lazy_reader, lazy_materialized.config_text, lazy_materialized.config_text.size());
    check(lazy_materialized.config_text == lazy_config, "tampered lazy bundle keeps exact config");
    check(find_materialized(lazy_materialized.bundle, "denoiser_plan") == nullptr,
          "preflight does not read a tampered lazy plan");
}

void test_exact_payload_and_identity_errors() {
    trtmc_test::TempDirGuard temporary;
    const auto root = std::filesystem::path(temporary.path());

    auto plugin_tamper = make_fixture();
    plugin_tamper.payloads.at("wan2_2_ti2v_plugins.so").front() ^= 0x01;
    expect_fixture_error(root / "plugin.trtfb", std::move(plugin_tamper),
                         "Wan2.2 artifact SHA256 mismatch for wan2_2_ti2v_plugins.so",
                         "embedded plugin payload is authenticated exactly");

    auto binding_tamper = make_fixture();
    replace_plugin_elf_binding(binding_tamper.config, std::string(64, '0'));
    expect_fixture_error(root / "binding.trtfb", std::move(binding_tamper),
                         "Wan2.2 plan is bound to a different AOT plugin ELF: denoiser_plan",
                         "plan-to-plugin binding error is exact");

    auto profile_tamper = make_fixture();
    profile_tamper.config["artifact_manifest"]["profile"]["architecture"]["num_layers"] = 29;
    expect_fixture_error(root / "profile.trtfb", std::move(profile_tamper),
                         "Wan2.2 artifact_manifest profile is not one of the qualified profiles",
                         "qualified profile error is exact");

    auto official_config_l0_manifest = make_fixture();
    official_config_l0_manifest.config["artifact_manifest"]["profile"] = l0_profile();
    expect_fixture_error(
        root / "official-config-l0-manifest.trtfb", std::move(official_config_l0_manifest),
        "Wan2.2 artifact_manifest profile does not match the top-level generation profile",
        "official config rejects an L0 artifact profile");

    auto l0_config_official_manifest = make_fixture(l0_profile());
    l0_config_official_manifest.config["artifact_manifest"]["profile"] = official_profile();
    expect_fixture_error(
        root / "l0-config-official-manifest.trtfb", std::move(l0_config_official_manifest),
        "Wan2.2 artifact_manifest profile does not match the top-level generation profile",
        "L0 config rejects an official artifact profile");

    auto hybrid_top_level = make_fixture();
    hybrid_top_level.config["num_inference_steps"] = 15;
    expect_fixture_error(root / "hybrid-top-level.trtfb", std::move(hybrid_top_level),
                         "Wan2.2 top-level generation profile is not one of the qualified profiles",
                         "top-level config rejects mixed official geometry and L0 steps");

    auto schema_tamper = make_fixture();
    schema_tamper.config["artifact_manifest"]["schema"] = "trtmc.wan2_2_ti2v.bundle-artifacts.v3";
    expect_fixture_error(root / "schema.trtfb", std::move(schema_tamper),
                         "Wan2.2 artifact_manifest identity is invalid",
                         "old artifact schema error is exact");

    auto lazy_and_plugin_tamper = make_fixture();
    lazy_and_plugin_tamper.payloads.at("denoiser_plan").front() ^= 0x01;
    lazy_and_plugin_tamper.payloads.at("wan2_2_ti2v_plugins.so").front() ^= 0x01;
    expect_fixture_error(root / "lazy-and-plugin.trtfb", std::move(lazy_and_plugin_tamper),
                         "Wan2.2 artifact SHA256 mismatch for wan2_2_ti2v_plugins.so",
                         "lazy plan bytes stay deferred when plugin authentication fails");
}

void test_multi_fault_validation_precedence() {
    trtmc_test::TempDirGuard temporary;
    const auto root = std::filesystem::path(temporary.path());

    auto extra_section = make_fixture();
    extra_section.config["runtime_contract"]["required_bundle_sections"] = nlohmann::json::array();
    extra_section.payloads["unexpected.bin"] = {'x'};
    extra_section.section_order.insert(extra_section.section_order.end() - 1, "unexpected.bin");
    expect_fixture_error(root / "section-count-first.trtfb", std::move(extra_section),
                         "Wan2.2 provenance requires exactly seven bundle sections",
                         "bundle section count precedes malformed runtime contract");

    auto size_first = make_fixture();
    size_first.config["runtime_contract"]["required_bundle_sections"] = nlohmann::json::array();
    expect_fixture_error(root / "config-size-first.trtfb", std::move(size_first),
                         "Wan2.2 config.json size disagrees with bundle metadata",
                         "config metadata size precedes malformed runtime contract", true);

    auto runtime_first = make_fixture();
    runtime_first.config["runtime_contract"]["required_bundle_sections"] = nlohmann::json::array();
    runtime_first.config["artifact_manifest"]["schema"] = "trtmc.wan2_2_ti2v.bundle-artifacts.v3";
    expect_fixture_error(
        root / "runtime-first.trtfb", std::move(runtime_first),
        "Wan2.2 runtime_contract must declare the exact seven-section integrity contract",
        "runtime contract precedes manifest identity");

    auto identity_first = make_fixture();
    identity_first.config["artifact_manifest"]["schema"] = "trtmc.wan2_2_ti2v.bundle-artifacts.v3";
    identity_first.config["runtime_contract"]["artifact_integrity"] = "invalid";
    identity_first.config["artifact_manifest"]["profile"]["architecture"]["num_layers"] = 29;
    expect_fixture_error(root / "identity-first.trtfb", std::move(identity_first),
                         "Wan2.2 artifact_manifest identity is invalid",
                         "manifest identity precedes integrity and profile");

    auto integrity_first = make_fixture();
    integrity_first.config["runtime_contract"]["artifact_integrity"] = "invalid";
    integrity_first.config["artifact_manifest"]["profile"]["architecture"]["num_layers"] = 29;
    integrity_first.config.erase("_trtmc_wan22_plugin_contract");
    expect_fixture_error(root / "integrity-first.trtfb", std::move(integrity_first),
                         "Wan2.2 v4 artifact manifest requires sha256_size_v1 integrity mode",
                         "integrity mode precedes profile and plugin contract");

    auto profile_first = make_fixture();
    profile_first.config["artifact_manifest"]["profile"]["architecture"]["num_layers"] = 29;
    profile_first.config.erase("_trtmc_wan22_plugin_contract");
    expect_fixture_error(root / "profile-first.trtfb", std::move(profile_first),
                         "Wan2.2 artifact_manifest profile is not one of the qualified profiles",
                         "profile precedes plugin contract");

    auto plugin_contract_first = make_fixture();
    plugin_contract_first.config.erase("_trtmc_wan22_plugin_contract");
    plugin_contract_first.config["artifact_manifest"]["sections"].erase("tokenizer.json");
    expect_fixture_error(root / "plugin-contract-first.trtfb", std::move(plugin_contract_first),
                         "Wan2.2 config is missing the AOT plugin contract",
                         "plugin contract precedes manifest section set");

    auto plugin_digest_first = make_fixture();
    plugin_digest_first
        .config["artifact_manifest"]["sections"]["wan2_2_ti2v_plugins.so"]["sha256"] = "invalid";
    plugin_digest_first.config["artifact_manifest"]["sections"]["text_encoder_0_plan"].erase(
        "size");
    expect_fixture_error(root / "plugin-digest-first.trtfb", std::move(plugin_digest_first),
                         "Wan2.2 AOT plugin artifact digest is malformed",
                         "plugin digest precedes per-plan entry validation");

    auto first_plan_size = make_fixture();
    first_plan_size.config["artifact_manifest"]["sections"]["text_encoder_0_plan"]["size"] = 1;
    first_plan_size
        .config["artifact_manifest"]["sections"]["text_encoder_0_plan"]["source_inputs"] =
        nlohmann::json::array();
    first_plan_size.config["artifact_manifest"]["sections"]["denoiser_plan"].erase("size");
    expect_fixture_error(root / "first-plan-size.trtfb", std::move(first_plan_size),
                         "Wan2.2 artifact size mismatch for text_encoder_0_plan",
                         "first plan size precedes its source and later entries");

    auto missing_bindings = make_fixture();
    auto& inputs =
        missing_bindings
            .config["artifact_manifest"]["sections"]["text_encoder_0_plan"]["source_inputs"];
    inputs = nlohmann::json::array({inputs.at(0)});
    expect_fixture_error(
        root / "missing-bindings.trtfb", std::move(missing_bindings),
        "Wan2.2 plan source identity is missing plugin/contract.json: text_encoder_0_plan",
        "missing plugin contract binding precedes missing plugin ELF binding");

    auto eager_order = make_fixture();
    eager_order.payloads.at("tokenizer.json").front() ^= 0x01;
    eager_order.payloads.at("wan2_2_ti2v_plugins.so").front() ^= 0x01;
    expect_fixture_error(root / "eager-order.trtfb", std::move(eager_order),
                         "Wan2.2 artifact SHA256 mismatch for tokenizer.json",
                         "tokenizer authentication precedes plugin authentication");
}

} // namespace

int main() {
    test_valid_bundle_and_lazy_plan_policy();
    test_exact_payload_and_identity_errors();
    test_multi_fault_validation_precedence();
    if (failures != 0) {
        std::cerr << failures << " Wan2.2 artifact-contract test(s) failed\n";
        return 1;
    }
    std::cerr << "All Wan2.2 artifact-contract tests passed\n";
    return 0;
}
