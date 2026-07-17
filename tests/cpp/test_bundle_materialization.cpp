/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Regression coverage for pipeline bundle materialization policy.
// Staged Wan bundles must not read multi-GiB TensorRT plans during pipeline
// construction; bundles without the opt-in policy must retain read-all
// behavior for existing plugins.

#include "bundle/bundle_format.h"
#include "runtime/registry/bundle_materialization.h"
#include "test_helpers.h"
#include "utils/sha256.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

std::filesystem::path make_temp_dir() {
    char pattern[] = "/tmp/trtfb_materialization_XXXXXX";
    char* directory = mkdtemp(pattern);
    if (directory == nullptr)
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    return directory;
}

void write_bundle(const std::filesystem::path& path, const std::string& header,
                  const std::vector<char>& payload) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    const std::uint64_t length = header.size();
    unsigned char bytes[8];
    for (int index = 0; index < 8; ++index)
        bytes[index] = static_cast<unsigned char>((length >> (8 * index)) & 0xff);
    output.write(reinterpret_cast<const char*>(bytes), 8);
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

const trtmc::BundleSection* find_materialized(const trtmc::BundleFile& bundle,
                                              const std::string& name) {
    for (const auto& section : bundle.sections) {
        if (section.name == name)
            return &section;
    }
    return nullptr;
}

void test_wan_staged_policy_does_not_eagerly_read_large_plans() {
    const auto temporary = make_temp_dir();
    const auto path = temporary / "wan22-staged.trtfb";
    const std::string config = R"({
      "runtime_strategy": "test_staged_runtime",
      "bundle_loading": {
        "mode": "staged",
        "eager_sections": [
          "wan2_2_umt5_cuda_plugin_so",
          "wan2_2_dit_cuda_plugin_so",
          "wan2_2_vae_cuda_plugin_so",
          "tokenizer.json",
          "config.json"
        ],
        "lazy_sections": [
          "text_encoder_0_plan",
          "denoiser_plan",
          "vae_decoder_plan",
          "vae_decoder_first_frame_plan"
        ]
      }
    })";

    // The physical file ends after the five eager payloads. Each plan claims
    // a 1 TiB range outside the file. A read-all implementation must fail;
    // successful materialization therefore proves no plan payload was read.
    const std::uint64_t config_offset = 5;
    const std::uint64_t lazy_offset = config_offset + config.size();
    const std::uint64_t huge_plan_size = 1ULL << 40;
    std::ostringstream header;
    header << R"({"model_id":"wan22","sections":{)"
           << R"("wan2_2_umt5_cuda_plugin_so":{"offset":0,"size":1},)"
           << R"("wan2_2_dit_cuda_plugin_so":{"offset":1,"size":1},)"
           << R"("wan2_2_vae_cuda_plugin_so":{"offset":2,"size":1},)"
           << R"("tokenizer.json":{"offset":3,"size":2},)"
           << R"("config.json":{"offset":)" << config_offset << R"(,"size":)" << config.size()
           << "},"
           << R"("text_encoder_0_plan":{"offset":)" << lazy_offset << R"(,"size":)"
           << huge_plan_size << "},"
           << R"("denoiser_plan":{"offset":)" << lazy_offset << R"(,"size":)" << huge_plan_size
           << "},"
           << R"("vae_decoder_plan":{"offset":)" << lazy_offset << R"(,"size":)" << huge_plan_size
           << "},"
           << R"("vae_decoder_first_frame_plan":{"offset":)" << lazy_offset << R"(,"size":)"
           << huge_plan_size << "}}}";
    std::vector<char> payload = {'U', 'D', 'V', '{', '}'};
    payload.insert(payload.end(), config.begin(), config.end());
    write_bundle(path, header.str(), payload);

    trtmc::BundleSectionReader reader(path.string());
    const auto materialized = trtmc::detail::materialize_pipeline_bundle(reader);
    check(materialized.bundle.sections.size() == 5,
          "Wan staged policy materializes exactly five eager sections");
    check(find_materialized(materialized.bundle, "config.json") != nullptr,
          "Wan staged policy materializes config");
    check(find_materialized(materialized.bundle, "tokenizer.json") != nullptr,
          "Wan staged policy materializes tokenizer");
    check(find_materialized(materialized.bundle, "denoiser_plan") == nullptr,
          "Wan staged policy leaves denoiser plan lazy");
    check(materialized.bundle.info.sections.size() == 9,
          "Wan staged view retains metadata for all nine sections");

    bool lazy_read_failed = false;
    try {
        (void)reader.read("denoiser_plan");
    } catch (const std::runtime_error& error) {
        lazy_read_failed =
            std::string(error.what()).find("outside file bounds") != std::string::npos;
    }
    check(lazy_read_failed, "deferred plan range is validated only when lazily requested");
    trtmc_test::remove_all_safe(temporary);
}

std::string sha256(const std::vector<char>& payload) {
    trtmc::detail::Sha256 digest;
    digest.update(payload.data(), payload.size());
    return digest.hex_digest();
}

nlohmann::json wan_profile() {
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

void write_provenance_bundle(const std::filesystem::path& path, bool tamper_lazy,
                             bool tamper_profile = false) {
    const std::vector<std::string> required = {
        "wan2_2_umt5_cuda_plugin_so",
        "wan2_2_dit_cuda_plugin_so",
        "wan2_2_vae_cuda_plugin_so",
        "text_encoder_0_plan",
        "denoiser_plan",
        "vae_decoder_plan",
        "vae_decoder_first_frame_plan",
        "tokenizer.json",
        "config.json",
    };
    const std::vector<std::string> model_owned(required.begin(), required.end() - 1);
    std::map<std::string, std::vector<char>> payloads;
    for (const auto& name : model_owned) {
        const std::string value = "authenticated:" + name;
        payloads[name] = std::vector<char>(value.begin(), value.end());
    }

    auto profile = wan_profile();
    if (tamper_profile)
        profile["architecture"]["num_layers"] = 29;

    nlohmann::json manifest_sections = nlohmann::json::object();
    for (const auto& name : model_owned) {
        nlohmann::json entry = {
            {"sha256", sha256(payloads.at(name))},
            {"size", payloads.at(name).size()},
        };
        if (name.size() >= 5 && name.compare(name.size() - 5, 5, "_plan") == 0) {
            const std::vector<char> source_bytes(name.begin(), name.end());
            nlohmann::json inputs = nlohmann::json::array(
                {{{"name", "source/test"}, {"sha256", sha256(source_bytes)}}});
            const nlohmann::json source_document = {
                {"family", "wan2_2_ti2v"},
                {"component", name},
                {"profile", profile},
                {"inputs", inputs},
            };
            const std::string source_text = source_document.dump();
            entry["source_sha256"] =
                sha256(std::vector<char>(source_text.begin(), source_text.end()));
            entry["source_inputs"] = std::move(inputs);
        }
        manifest_sections[name] = std::move(entry);
    }

    const nlohmann::json config = {
        {"runtime_strategy", "diffusion_wan2_2_ti2v"},
        {"runtime_contract",
         {{"implementation", "native_cpp_cuda_tensorrt"},
          {"artifact_integrity", "sha256_size_v1"},
          {"required_bundle_sections", required}}},
        {"bundle_loading",
         {{"mode", "staged"},
          {"eager_sections",
           {"wan2_2_umt5_cuda_plugin_so", "wan2_2_dit_cuda_plugin_so", "wan2_2_vae_cuda_plugin_so",
            "tokenizer.json", "config.json"}},
          {"lazy_sections",
           {"text_encoder_0_plan", "denoiser_plan", "vae_decoder_plan",
            "vae_decoder_first_frame_plan"}}}},
        {"artifact_manifest",
         {{"schema", "trtmc.wan2_2_ti2v.bundle-artifacts.v2"},
          {"family", "wan2_2_ti2v"},
          {"profile", profile},
          {"runtime", "native_cpp_cuda_tensorrt"},
          {"sections", manifest_sections}}},
    };
    const std::string config_text = config.dump();
    payloads["config.json"] = std::vector<char>(config_text.begin(), config_text.end());
    if (tamper_lazy)
        payloads.at("denoiser_plan").front() ^= 0x01;

    nlohmann::json header = {{"model_id", "wan22"}, {"sections", nlohmann::json::object()}};
    std::vector<char> combined;
    std::uint64_t offset = 0;
    for (const auto& name : required) {
        const auto& payload = payloads.at(name);
        header["sections"][name] = {{"offset", offset}, {"size", payload.size()}};
        combined.insert(combined.end(), payload.begin(), payload.end());
        offset += payload.size();
    }
    write_bundle(path, header.dump(), combined);
}

void test_wan_provenance_streams_and_rejects_tampered_lazy_section() {
    check(sha256(std::vector<char>{'a', 'b', 'c'}) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "bundle provenance SHA256 matches the standard abc vector");
    const auto temporary = make_temp_dir();
    const auto valid_path = temporary / "wan22-provenance-valid.trtfb";
    write_provenance_bundle(valid_path, false);
    trtmc::BundleSectionReader valid_reader(valid_path.string());
    const auto materialized = trtmc::detail::materialize_pipeline_bundle(valid_reader);
    check(materialized.bundle.sections.size() == 5,
          "authenticated Wan bundle retains staged materialization");
    check(find_materialized(materialized.bundle, "denoiser_plan") == nullptr,
          "authenticated lazy plan is not retained in memory");

    const auto tampered_path = temporary / "wan22-provenance-tampered.trtfb";
    write_provenance_bundle(tampered_path, true);
    bool rejected = false;
    try {
        trtmc::BundleSectionReader tampered_reader(tampered_path.string());
        (void)trtmc::detail::materialize_pipeline_bundle(tampered_reader);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("artifact SHA256 mismatch for denoiser_plan") !=
                   std::string::npos;
    }
    check(rejected, "Wan provenance rejects a tampered lazy section before plugin loading");

    const auto wrong_profile_path = temporary / "wan22-provenance-wrong-profile.trtfb";
    write_provenance_bundle(wrong_profile_path, false, true);
    rejected = false;
    try {
        trtmc::BundleSectionReader wrong_profile_reader(wrong_profile_path.string());
        (void)trtmc::detail::materialize_pipeline_bundle(wrong_profile_reader);
    } catch (const std::runtime_error& error) {
        rejected =
            std::string(error.what())
                .find("artifact_manifest profile is not the official profile") != std::string::npos;
    }
    check(rejected, "Wan provenance rejects a mutated architecture profile");
    trtmc_test::remove_all_safe(temporary);
}

void test_policy_absence_preserves_existing_read_all_behavior() {
    const auto temporary = make_temp_dir();
    const auto path = temporary / "legacy.trtfb";
    const std::string config = R"({"runtime_strategy":"legacy_decoder"})";
    const std::vector<char> engine = {'P', 'L', 'A', 'N'};
    std::ostringstream header;
    header << R"({"model_id":"legacy","sections":{)"
           << R"("config.json":{"offset":0,"size":)" << config.size() << "},"
           << R"("engine_plan":{"offset":)" << config.size() << R"(,"size":)" << engine.size()
           << "}}}";
    std::vector<char> payload(config.begin(), config.end());
    payload.insert(payload.end(), engine.begin(), engine.end());
    write_bundle(path, header.str(), payload);

    trtmc::BundleSectionReader reader(path.string());
    const auto materialized = trtmc::detail::materialize_pipeline_bundle(reader);
    const auto* engine_section = find_materialized(materialized.bundle, "engine_plan");
    check(materialized.bundle.sections.size() == 2,
          "bundle without staged policy retains read-all section count");
    check(engine_section != nullptr && engine_section->data == engine,
          "bundle without staged policy eagerly reads existing plugin plan");
    trtmc_test::remove_all_safe(temporary);
}

} // namespace

int main() {
    test_wan_staged_policy_does_not_eagerly_read_large_plans();
    test_wan_provenance_streams_and_rejects_tampered_lazy_section();
    test_policy_absence_preserves_existing_read_all_behavior();
    if (failures != 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle materialization tests passed\n";
    return 0;
}
