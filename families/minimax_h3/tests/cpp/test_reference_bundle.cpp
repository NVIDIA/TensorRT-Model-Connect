/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_bundle.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace {
using namespace trtmc::minimax_h3;
using Json = nlohmann::json;

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

Json fixture() {
    Json config = {
        {"schema_version", "minimax_h3.image_ref2va/v1"},
        {"workflow", "image_ref2va"},
        {"weights_variant", "transformer_ref"},
        {"memory_preset", "minimal"},
        {"vae_tile_batch", 1},
        {"height", 1344},
        {"width", 768},
        {"num_frames", 124},
        {"fps", 24},
        {"num_inference_steps", 50},
        {"denoiser_forwards", 49},
        {"latent_frames", 37},
        {"latent_height", 84},
        {"latent_width", 48},
        {"audio_latent_frames", 207},
        {"audio_sampling_rate", 32000},
        {"audio_channels", 2},
        {"max_text_rows", 32768},
        {"max_sequence_rows", 86862},
        {"text_rows", {1, 128, 32768}},
        {"condition_video_rows", {4096, 5504, 16384}},
        {"condition_audio_rows", {0, 0, 0}},
        {"reference_pixels", {16384, 22016, 65536}},
        {"layers", 50},
        {"hidden_size", 5376},
        {"text_width", 5120},
        {"context_parallel_size", 1},
        {"first_block_cache", false},
        {"latents_mean", std::vector<float>(24, 0)},
        {"latents_std", std::vector<float>(24, 1)},
    };
    for (const auto* name : {"transformer_ref", "text_encoder", "vae", "audio_vae"})
        config["config_sha256"][name] = std::string(64, 'a');
    return config;
}

void rejects(const Json& config) {
    try {
        (void)parse_image_reference_config(config);
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("incompatible reference metadata was accepted");
}

void check() {
    auto original = fixture();
    auto parsed = parse_image_reference_config(original);
    require(parsed.memory == ReferenceMemoryPreset::Minimal && parsed.max_text_rows == 32768 &&
                parsed.denoiser.target.height == 84 && parsed.denoiser.target.width == 48 &&
                parsed.denoiser.max_sequence_rows == 86862,
            "original portrait metadata");
    original["memory_preset"] = "original";
    original["vae_tile_batch"] = 28;
    parsed = parse_image_reference_config(original);
    require(parsed.memory == ReferenceMemoryPreset::Original, "resident preset metadata");
    auto master = fixture();
    master["height"] = 1536;
    master["width"] = 864;
    master["latent_height"] = 96;
    master["latent_width"] = 54;
    master["max_sequence_rows"] = 97518;
    parsed = parse_image_reference_config(master);
    require(parsed.denoiser.target.height == 96 && parsed.denoiser.target.width == 54,
            "exact 9:16 master geometry without short-edge reduction");
    auto expanded = fixture();
    expanded["reference_capacity"] = "six_portrait";
    expanded["max_text_rows"] = 65536;
    expanded["text_rows"] = {1, 128, 65536};
    expanded["condition_video_rows"] = {4096, 43776, 49152};
    expanded["max_sequence_rows"] = 152398;
    parsed = parse_image_reference_config(expanded);
    require(parsed.max_text_rows == 65536 && parsed.max_condition_video_rows == 49152 &&
                parsed.denoiser.max_sequence_rows == 152398 &&
                parsed.denoiser.target.height == 84 && parsed.denoiser.target.width == 48,
            "expanded conditioning profiles preserve target geometry and full topology");
    for (const auto& name :
         {"max_text_rows", "text_rows", "condition_video_rows", "max_sequence_rows"}) {
        auto incompatible = expanded;
        incompatible[name] = fixture()[name];
        rejects(incompatible);
    }
    auto unlabeled = expanded;
    unlabeled.erase("reference_capacity");
    rejects(unlabeled); // Existing metadata cannot silently become an expanded profile.
    auto continuation = expanded;
    continuation["schema_version"] = "minimax_h3.image_ref2va_continuation/v1";
    continuation["workflow"] = "image_ref2va_continuation";
    continuation["num_frames"] = 226;
    continuation["condition_video_rows"] = {4096, 50832, 56208};
    continuation["condition_audio_rows"] = {0, 74, 74};
    continuation["max_sequence_rows"] = 159528;
    for (const auto& pair : {std::pair{"continuation_window_frames", 124},
                             {"continuation_overlap_frames", 22},
                             {"continuation_windows", 2},
                             {"decoded_latent_frames", 67},
                             {"decoded_audio_latent_frames", 377},
                             {"total_denoiser_forwards", 98}})
        continuation[pair.first] = pair.second;
    parsed = parse_image_reference_config(continuation);
    require(parsed.output_frames == 226 && parsed.max_condition_audio_rows == 74 &&
                parsed.max_condition_video_rows == 56208 && parsed.denoiser.target.frames == 37 &&
                parsed.denoiser.target.audio_latents == 207,
            "new guide profiles and cumulative decoders retain full single-window DiT geometry");
    for (const auto& name : {"condition_video_rows", "condition_audio_rows", "num_frames",
                             "schema_version", "workflow", "max_sequence_rows"}) {
        auto incompatible = continuation;
        incompatible[name] = expanded[name];
        rejects(incompatible);
    }
    for (const auto* name :
         {"continuation_window_frames", "continuation_overlap_frames", "continuation_windows",
          "decoded_latent_frames", "decoded_audio_latent_frames", "total_denoiser_forwards"}) {
        auto incompatible = continuation;
        incompatible[name] = 1;
        rejects(incompatible);
        incompatible = expanded;
        incompatible[name] = continuation[name];
        rejects(incompatible);
    }
    const std::vector<std::pair<const char*, Json>> changes = {
        {"schema_version", "v2"},
        {"weights_variant", "transformer"},
        {"workflow", "t2va"},
        {"reference_capacity", "auto"},
        {"reference_capacity", "six_portrait"},
        {"reference_capacity", true},
        {"memory_preset", "auto"},
        {"vae_tile_batch", 28},
        {"vae_tile_batch", 1.0},
        {"num_frames", 123},
        {"num_inference_steps", 4},
        {"denoiser_forwards", 4},
        {"layers", 1},
        {"hidden_size", 5376.0},
        {"first_block_cache", true},
        {"height", 1345},
        {"width", 128},
        {"height", uint64_t{1} << 63},
        {"max_sequence_rows", 86861},
        {"latent_height", 83},
        {"latent_width", true},
        {"condition_audio_rows", {0, 0, 1}},
        {"text_rows", {1.0, 128, 32768}},
        {"latents_mean", std::vector<float>(23, 0)},
        {"latents_std", std::vector<float>(24, 0)},
    };
    for (const auto& change : changes) {
        auto bad = fixture();
        bad[change.first] = change.second;
        rejects(bad);
    }
    const auto required = fixture();
    for (const auto& item : required.items()) {
        auto missing = fixture();
        missing.erase(item.key());
        rejects(missing);
    }
    auto bad = fixture();
    bad["config_sha256"]["vae"] = "wrong";
    rejects(bad);
    bad = fixture();
    bad["latents_mean"][0] = true;
    rejects(bad);
    bad = fixture();
    bad["latents_std"][0] = 1e300;
    rejects(bad);
    const auto names = image_reference_plan_names();
    require(names.size() == 8 && std::set<std::string>(names.begin(), names.end()).size() == 8,
            "exact eight unique plan sections");

    std::ifstream contract_file(std::filesystem::path(__FILE__).parent_path() /
                                "../fixtures/omniref-pdd-contract.json");
    Json contract;
    contract_file >> contract;
    for (const bool multi : {false, true}) {
        auto student = multi ? continuation : expanded;
        student["schema_version"] =
            multi ? "minimax_h3.omniref_pdd_continuation/v1" : "minimax_h3.omniref_pdd/v1";
        student["workflow"] = multi ? "omniref_pdd_continuation" : "omniref_pdd";
        student["num_inference_steps"] = student["denoiser_forwards"] = 8;
        if (multi)
            student["total_denoiser_forwards"] = 16;
        student["checkpoint"] = {
            {"hf_id", "FastVideo/FastVideo-FastH3-Omni-8-Step-V1"},
            {"revision", "53e0d26a1691aa473b6262d8196a8feb2c297df0"},
        };
        student["inference_contract"] = contract;
        student["max_vsa_tiles"] = 2048;
        const auto value = parse_omniref_pdd_config(student);
        require(value.denoiser.recipe == ReferenceSamplingRecipe::OmniRefPdd8 &&
                    value.denoiser.max_vsa_tiles == 2048 &&
                    value.output_frames == (multi ? 226 : 124),
                "closed student contract selects eight-forward runtime and VSA ABI");
        rejects(student); // A student cannot masquerade as the original recipe.
        const auto reject_student = [](const Json& config) {
            try {
                (void)parse_omniref_pdd_config(config);
            } catch (const std::exception&) {
                return;
            }
            throw std::runtime_error("incompatible OmniRef recipe was accepted");
        };
        reject_student(multi ? continuation : expanded);
        for (const auto* name : {"checkpoint", "inference_contract", "max_vsa_tiles"}) {
            auto missing = student;
            missing.erase(name);
            reject_student(missing);
        }
        for (const auto& field : contract.items()) {
            auto missing = student;
            missing["inference_contract"].erase(field.key());
            reject_student(missing);
            auto changed = student;
            changed["inference_contract"][field.key()] = nullptr;
            reject_student(changed);
        }
        for (const auto& change :
             std::vector<std::pair<const char*, Json>>{{"max_vsa_tiles", 2047},
                                                       {"max_vsa_tiles", 0},
                                                       {"max_vsa_tiles", 8194},
                                                       {"max_vsa_tiles", true},
                                                       {"denoiser_forwards", 49},
                                                       {"num_inference_steps", 50}}) {
            auto changed = student;
            changed[change.first] = change.second;
            reject_student(changed);
        }
        for (const auto* name : {"hf_id", "revision"}) {
            auto changed = student;
            changed["checkpoint"][name] = "unknown";
            reject_student(changed);
        }
        auto unexpected = student;
        unexpected["inference_contract"]["extra"] = false;
        reject_student(unexpected);
        auto mistyped = student;
        mistyped["inference_contract"]["pdd_steps"] = 32.0;
        reject_student(mistyped);
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && (std::string(argv[1]) == "--validate-stdin" ||
                          std::string(argv[1]) == "--validate-pdd-stdin")) {
            Json input;
            std::cin >> input;
            const auto config = std::string(argv[1]) == "--validate-pdd-stdin"
                                    ? trtmc::minimax_h3::parse_omniref_pdd_config(input)
                                    : trtmc::minimax_h3::parse_image_reference_config(input);
            std::cout << Json({{"max_sequence_rows", config.denoiser.max_sequence_rows},
                               {"latent_height", config.denoiser.target.height},
                               {"latent_width", config.denoiser.target.width}})
                             .dump()
                      << '\n';
        } else {
            check();
            std::cout << "original H3 reference bundle metadata checks passed\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
