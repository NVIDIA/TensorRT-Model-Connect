/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam3/sam3_tracker_step_runtime.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void add_section(trtmc::BundleFile& bundle, std::string name, std::string contents) {
    trtmc::BundleSection section;
    section.name = std::move(name);
    section.data.assign(contents.begin(), contents.end());
    bundle.sections.push_back(std::move(section));
}

std::string sha256(std::string_view contents) {
    return trtmc::sam3_tracker_step_sha256_hex(std::vector<char>(contents.begin(), contents.end()));
}

std::vector<char> decode_hex(std::string_view value) {
    auto nibble = [](char character) -> uint8_t {
        return character <= '9' ? static_cast<uint8_t>(character - '0')
                                : static_cast<uint8_t>(character - 'a' + 10);
    };
    std::vector<char> result;
    result.reserve(value.size() / 2);
    for (std::size_t index = 0; index < value.size(); index += 2)
        result.push_back(
            static_cast<char>((nibble(value[index]) << 4U) | nibble(value[index + 1])));
    return result;
}

std::string pipeline_sha(const std::string& encoder_sha, const std::string& decoder_sha) {
    constexpr std::string_view domain = "trtmc.sam3.tracker_step.split_aoti.v1";
    std::vector<char> payload(domain.begin(), domain.end());
    payload.push_back('\0');
    const auto encoder = decode_hex(encoder_sha);
    const auto decoder = decode_hex(decoder_sha);
    payload.insert(payload.end(), encoder.begin(), encoder.end());
    payload.insert(payload.end(), decoder.begin(), decoder.end());
    return trtmc::sam3_tracker_step_sha256_hex(payload);
}

std::string package_global(std::string_view stage, int32_t batch_size,
                           const std::string& package_sha) {
    std::ostringstream stream;
    if (stage == "encoder")
        stream << "trtmc.sam3.tracker_encoder.b" << batch_size << ".m1_10.p1_19.";
    else
        stream << "trtmc.sam3.tracker_decoder.b" << batch_size << ".static.";
    stream << package_sha.substr(0, 20);
    return stream.str();
}

std::string pipeline_global(int32_t batch_size, const std::string& encoder_sha,
                            const std::string& decoder_sha) {
    return "trtmc.sam3.tracker_step.b" + std::to_string(batch_size) + ".split_aoti." +
           pipeline_sha(encoder_sha, decoder_sha).substr(0, 20);
}

std::string memory_global(std::string_view policy, int32_t batch_size,
                          const std::string& package_sha) {
    return "trtmc.sam3.tracker_memory." + std::string(policy) + ".b" + std::to_string(batch_size) +
           ".fixed." + package_sha.substr(0, 20);
}

std::string resize_global(int32_t batch_size, const std::string& package_sha) {
    return "trtmc.sam3.tracker_memory.resize.b" + std::to_string(batch_size) + ".fixed." +
           package_sha.substr(0, 20);
}

std::string make_resize_manifest(const std::string& b1_sha, const std::string& b2_sha) {
    std::ostringstream stream;
    const auto write_package = [&](int32_t batch_size, const std::string& package_sha) {
        stream << R"({"batch_size":)" << batch_size << R"(,"filename":"sam3_hard_mask_resize_b)"
               << batch_size << '_' << package_sha << R"(.pt2","section":"sam3_hard_mask_resize_b)"
               << batch_size << R"(.pt2","sha256":")" << package_sha << R"(","package_global":")"
               << resize_global(batch_size, package_sha) << R"("})";
    };
    stream
        << R"({"schema_version":1,"scope":"torch_bilinear_288_to_1008_b1_b2",)"
        << R"("artifact_format":"torch.aot_inductor.package.pt2",)"
        << R"("implementation":{"library":"torch","operator":"torch.nn.functional.interpolate",)"
        << R"("mode":"bilinear","align_corners":false,"source_size":288,"target_size":1008},)"
        << R"("producer":{"torch_version":"2.9.0","transformers_version":"5.2.0",)"
        << R"("cuda_version":"12.8","compute_capability":[8,9],"host_architecture":"x86_64",)"
        << R"("torch_cxx11_abi":false,"torch_aoti_abi_version":7},)"
        << R"("host_architecture":"x86_64","exporter_sha256":")" << sha256("resize-exporter")
        << R"(","input_abi":[{"name":"tracker_mask","dtype":"float32","shape":["B",1,288,288]}],)"
        << R"("output_abi":[{"name":"resized_tracker_mask","dtype":"float32","shape":["B",1,1008,1008]}],)"
        << R"("packages":[)";
    write_package(1, b1_sha);
    stream << ',';
    write_package(2, b2_sha);
    stream << R"(],"package_validation":{"reference":"same torch.interpolate eager execution",)"
           << R"("maximum_absolute_error":0.00002,"cases":[)"
           << R"({"batch_size":1,"maximum_absolute_error":0.000001,"passed":true},)"
           << R"({"batch_size":2,"maximum_absolute_error":0.000001,"passed":true}]}})";
    return stream.str();
}

std::string make_memory_manifest(const std::string& soft_b1_sha, const std::string& hard_b1_sha,
                                 const std::string& soft_b2_sha, const std::string& hard_b2_sha) {
    std::ostringstream stream;
    const auto write_package = [&](std::string_view policy, int32_t batch_size,
                                   const std::string& package_sha) {
        stream << R"({"policy":")" << policy << R"(","batch_size":)" << batch_size
               << R"(,"fixed_shape":true,"inputs":[)"
               << R"({"name":"tracker_feature_2","dtype":"float32","shape":[1,256,72,72]},)"
               << R"({"name":")" << (policy == "hard" ? "owned_tracker_mask" : "final_mask")
               << R"(","dtype":"float32","shape":[)" << batch_size << R"(,1,)"
               << (policy == "hard" ? 1008 : 288) << ',' << (policy == "hard" ? 1008 : 288)
               << R"(]},)"
               << R"({"name":"object_score_logits","dtype":"float32","shape":[)" << batch_size
               << R"(,1]},)"
               << R"({"name":"suppress_area_shrinkage","dtype":"int32","shape":[)" << batch_size
               << R"(,1]}],"outputs":[)"
               << R"({"name":"packed_memory_and_position","dtype":"float32","shape":)";
        if (batch_size == 1)
            stream << R"([2,5184,1,64])";
        else
            stream << R"([2,2,5184,64])";
        stream << R"(}],"hard_mask":)" << (policy == "hard" ? "true" : "false")
               << R"(,"filename":"sam3_tracker_memory_)" << policy << "_b" << batch_size << "_"
               << package_sha << R"(.pt2","section":"sam3_tracker_memory_)" << policy << "_b"
               << batch_size << R"(.pt2","sha256":")" << package_sha << R"(","package_global":")"
               << memory_global(policy, batch_size, package_sha) << R"("})";
    };
    const auto write_validation = [&](std::string_view policy, int32_t batch_size) {
        stream
            << R"({"policy":")" << policy << R"(","batch_size":)" << batch_size
            << R"(,"hard_mask":)" << (policy == "hard" ? "true" : "false")
            << R"(,"cosine":1.0,"relative_l2":0.0,"maximum_absolute_error":0.0,)"
            << R"("planes":{"memory":{"cosine":1.0,"relative_l2":0.0,"maximum_absolute_error":0.0},)"
            << R"("position":{"cosine":1.0,"relative_l2":0.0,"maximum_absolute_error":0.0}},)"
            << R"("passed":true})";
    };
    stream
        << R"({"schema_version":2,"scope":"fixed_memory_encoder_soft_hard_b1_b2",)"
        << R"("artifact_format":"torch.aot_inductor.package.pt2",)"
        << R"("implementation":{"library":"transformers","model_class":"Sam3TrackerVideoModel",)"
        << R"("module":"Sam3TrackerVideoMemoryEncoder","license":"Apache-2.0",)"
        << R"("source_import_policy":"transformers-only"},"model_sha256":")" << sha256("model")
        << R"(","exporter_sha256":")" << sha256("exporter")
        << R"(","producer":{"torch_version":"2.9.0","transformers_version":"5.2.0",)"
        << R"("cuda_version":"12.8","compute_capability":[8,9],"host_architecture":"x86_64",)"
        << R"("torch_cxx11_abi":false,"torch_aoti_abi_version":7},"input_abi":[)"
        << R"({"policy":"soft","tensors":[)"
        << R"({"name":"tracker_feature_2","dtype":"float32","shape":[1,256,72,72]},)"
        << R"({"name":"final_mask","dtype":"float32","shape":["B",1,288,288]},)"
        << R"({"name":"object_score_logits","dtype":"float32","shape":["B",1]},)"
        << R"({"name":"suppress_area_shrinkage","dtype":"int32","shape":["B",1]}]},)"
        << R"({"policy":"hard","tensors":[)"
        << R"({"name":"tracker_feature_2","dtype":"float32","shape":[1,256,72,72]},)"
        << R"({"name":"owned_tracker_mask","dtype":"float32","shape":["B",1,1008,1008]},)"
        << R"({"name":"object_score_logits","dtype":"float32","shape":["B",1]},)"
        << R"({"name":"suppress_area_shrinkage","dtype":"int32","shape":["B",1]}]}],)"
        << R"("mask_policy":{"soft":"288 bilinear 1152, clamp rejected rows to <=-10, sigmoid, scale 20, bias -10",)"
        << R"("hard":"globally owned binary FP32 1008, scale 20, bias -10, antialiased bilinear 1152; suppression input ignored",)"
        << R"("b1_layout":[2,5184,1,64],"b2_layout":[2,2,5184,64],)"
        << R"("stored_precision":"bfloat16 rounded then promoted to float32 carrier"},"packages":[)";
    write_package("soft", 1, soft_b1_sha);
    stream << ',';
    write_package("hard", 1, hard_b1_sha);
    stream << ',';
    write_package("soft", 2, soft_b2_sha);
    stream << ',';
    write_package("hard", 2, hard_b2_sha);
    stream
        << R"(],"package_validation":{"reference":"same Transformers module eager execution before cache publication",)"
        << R"("minimum_cosine":0.999,"maximum_relative_l2":0.02,"cases":[)";
    write_validation("soft", 1);
    stream << ',';
    write_validation("hard", 1);
    stream << ',';
    write_validation("soft", 2);
    stream << ',';
    write_validation("hard", 2);
    stream << "]}}";
    return stream.str();
}

std::string
make_manifest(const std::string& plugin_sha, const std::string& encoder_b1_sha,
              const std::string& decoder_b1_sha, const std::string& encoder_b2_sha,
              const std::string& decoder_b2_sha, int32_t encoder_b2_batch = 2,
              const std::string& encoder_b1_section = "sam3_tracker_encoder_b1_dynamic.pt2",
              const std::string& pipeline_b1_encoder_sha = "") {
    const std::string referenced_b1_encoder =
        pipeline_b1_encoder_sha.empty() ? encoder_b1_sha : pipeline_b1_encoder_sha;
    std::ostringstream stream;
    stream << R"({"schema_version":1,"step_scope":"meta_split_dynamic_encoder_static_decoder",)"
              R"("plugin":{"section":")"
           << trtmc::kSam3TrackerStepNativePluginSection << R"(","sha256":")" << plugin_sha
           << R"(","type":"Sam3TrackerStepFfi","version":"2"},"producer":{)"
              R"("torch_version":"2.9.0","transformers_version":"5.2.0",)"
              R"("tvm_ffi_version":"0.1.6","tensorrt_version":"11.2.0",)"
              R"("cuda_version":"12.8","host_architecture":"x86_64",)"
              R"("torch_cxx11_abi":false,"aoti_abi_version":7,)"
              R"("compute_capability":[8,9]},"packages":[)"
           << R"({"stage":"encoder","package_global":")"
           << package_global("encoder", 1, encoder_b1_sha) << R"(","section":")"
           << encoder_b1_section << R"(","sha256":")" << encoder_b1_sha << R"(","batch_size":1},)"
           << R"({"stage":"decoder","package_global":")"
           << package_global("decoder", 1, decoder_b1_sha)
           << R"(","section":"sam3_tracker_decoder_b1_static.pt2","sha256":")" << decoder_b1_sha
           << R"(","batch_size":1},)"
           << R"({"stage":"encoder","package_global":")"
           << package_global("encoder", encoder_b2_batch, encoder_b2_sha)
           << R"(","section":"sam3_tracker_encoder_b2_dynamic.pt2","sha256":")" << encoder_b2_sha
           << R"(","batch_size":)" << encoder_b2_batch << "},"
           << R"({"stage":"decoder","package_global":")"
           << package_global("decoder", 2, decoder_b2_sha)
           << R"(","section":"sam3_tracker_decoder_b2_static.pt2","sha256":")" << decoder_b2_sha
           << R"(","batch_size":2}],"pipelines":[)"
           << R"({"global_name":")" << pipeline_global(1, referenced_b1_encoder, decoder_b1_sha)
           << R"(","encoder_sha256":")" << referenced_b1_encoder << R"(","decoder_sha256":")"
           << decoder_b1_sha << R"(","batch_size":1},)"
           << R"({"global_name":")" << pipeline_global(2, encoder_b2_sha, decoder_b2_sha)
           << R"(","encoder_sha256":")" << encoder_b2_sha << R"(","decoder_sha256":")"
           << decoder_b2_sha << R"(","batch_size":2}]} )";
    return stream.str();
}

trtmc::BundleFile make_valid_bundle() {
    constexpr std::string_view plugin = "native-plugin";
    constexpr std::string_view encoder_b1 = "aoti-encoder-b1";
    constexpr std::string_view decoder_b1 = "aoti-decoder-b1";
    constexpr std::string_view encoder_b2 = "aoti-encoder-b2";
    constexpr std::string_view decoder_b2 = "aoti-decoder-b2";
    constexpr std::string_view memory_soft_b1 = "aoti-memory-soft-b1";
    constexpr std::string_view memory_hard_b1 = "aoti-memory-hard-b1";
    constexpr std::string_view memory_soft_b2 = "aoti-memory-soft-b2";
    constexpr std::string_view memory_hard_b2 = "aoti-memory-hard-b2";
    constexpr std::string_view resize_b1 = "aoti-resize-b1";
    constexpr std::string_view resize_b2 = "aoti-resize-b2";
    trtmc::BundleFile bundle;
    add_section(bundle, trtmc::kSam3TrackerStepNativePluginSection, std::string(plugin));
    add_section(bundle, "sam3_tracker_encoder_b1_dynamic.pt2", std::string(encoder_b1));
    add_section(bundle, "sam3_tracker_decoder_b1_static.pt2", std::string(decoder_b1));
    add_section(bundle, "sam3_tracker_encoder_b2_dynamic.pt2", std::string(encoder_b2));
    add_section(bundle, "sam3_tracker_decoder_b2_static.pt2", std::string(decoder_b2));
    add_section(bundle, "sam3_tracker_memory_soft_b1.pt2", std::string(memory_soft_b1));
    add_section(bundle, "sam3_tracker_memory_hard_b1.pt2", std::string(memory_hard_b1));
    add_section(bundle, "sam3_tracker_memory_soft_b2.pt2", std::string(memory_soft_b2));
    add_section(bundle, "sam3_tracker_memory_hard_b2.pt2", std::string(memory_hard_b2));
    add_section(bundle, "sam3_hard_mask_resize_b1.pt2", std::string(resize_b1));
    add_section(bundle, "sam3_hard_mask_resize_b2.pt2", std::string(resize_b2));
    add_section(bundle, trtmc::kSam3TrackerMemoryAotiManifestSection,
                make_memory_manifest(sha256(memory_soft_b1), sha256(memory_hard_b1),
                                     sha256(memory_soft_b2), sha256(memory_hard_b2)));
    add_section(bundle, trtmc::kSam3HardMaskResizeAotiManifestSection,
                make_resize_manifest(sha256(resize_b1), sha256(resize_b2)));
    add_section(bundle, trtmc::kSam3TrackerStepRuntimeManifestSection,
                make_manifest(sha256(plugin), sha256(encoder_b1), sha256(decoder_b1),
                              sha256(encoder_b2), sha256(decoder_b2)));
    return bundle;
}

template <typename Function>
void check_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    check(false, message);
}

std::string section_contents(const trtmc::BundleFile& bundle, std::size_t index) {
    return std::string(bundle.sections[index].data.begin(), bundle.sections[index].data.end());
}

void replace_manifest(trtmc::BundleFile& bundle, std::string manifest) {
    bundle.sections.back().data.assign(manifest.begin(), manifest.end());
}

std::string section_contents(const trtmc::BundleFile& bundle, std::string_view name) {
    for (const auto& section : bundle.sections) {
        if (section.name == name)
            return std::string(section.data.begin(), section.data.end());
    }
    throw std::runtime_error("synthetic bundle section is missing");
}

void replace_section(trtmc::BundleFile& bundle, std::string_view name, std::string contents) {
    for (auto& section : bundle.sections) {
        if (section.name == name) {
            section.data.assign(contents.begin(), contents.end());
            return;
        }
    }
    throw std::runtime_error("synthetic bundle section is missing");
}

void test_sha256_known_answer() {
    check(sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SAM3 tracker-step SHA-256 matches the standard known answer");
}

void test_valid_manifest() {
    const auto bundle = make_valid_bundle();
    const auto manifest = trtmc::validate_sam3_tracker_step_runtime_manifest(bundle);
    check(manifest.schema_version == 1, "SAM3 tracker-step manifest accepts schema 1");
    check(manifest.step_scope == trtmc::kSam3TrackerStepScope,
          "SAM3 tracker-step manifest requires the Meta split scope");
    check(manifest.plugin_type == "Sam3TrackerStepFfi",
          "SAM3 tracker-step manifest fixes the TensorRT plugin type");
    check(manifest.transformers_version == "5.2.0" && manifest.tvm_ffi_version == "0.1.6" &&
              manifest.tensorrt_version == "11.2.0" && !manifest.torch_cxx11_abi,
          "SAM3 tracker-step manifest binds exporter and native build ABI fields");
    check(manifest.packages.size() == 4 && manifest.pipelines[0].batch_size == 1 &&
              manifest.pipelines[1].batch_size == 2,
          "SAM3 tracker-step manifest requires four packages and two pipelines");
    const auto memory = trtmc::validate_sam3_tracker_memory_aoti_manifest(bundle, manifest);
    check(memory.schema_version == 2 && memory.scope == trtmc::kSam3TrackerMemoryScope &&
              memory.packages.size() == 4,
          "SAM3 tracker-memory manifest requires fixed soft/hard B1/B2 packages");
    check(memory.torch_version == manifest.torch_version &&
              memory.aoti_abi_version == manifest.aoti_abi_version &&
              memory.compute_capability_major == manifest.compute_capability_major,
          "SAM3 tracker-memory producer ABI is bound to the step packages");
    const auto resize = trtmc::validate_sam3_hard_mask_resize_aoti_manifest(bundle, manifest);
    check(resize.schema_version == 1 && resize.scope == trtmc::kSam3HardMaskResizeScope &&
              resize.packages[0].batch_size == 1 && resize.packages[1].batch_size == 2,
          "SAM3 hard-mask resize manifest requires fixed B1/B2 packages");
}

void test_missing_resize_manifest_fails_closed() {
    auto bundle = make_valid_bundle();
    replace_section(bundle, trtmc::kSam3HardMaskResizeAotiManifestSection, "");
    const auto step = trtmc::validate_sam3_tracker_step_runtime_manifest(bundle);
    check_throws([&] { trtmc::validate_sam3_hard_mask_resize_aoti_manifest(bundle, step); },
                 "SAM3 tracker runtime rejects a missing resize manifest");
}

void test_resize_artifact_hash_mismatch_fails_closed() {
    auto bundle = make_valid_bundle();
    replace_section(bundle, "sam3_hard_mask_resize_b1.pt2", "tampered-resize");
    const auto step = trtmc::validate_sam3_tracker_step_runtime_manifest(bundle);
    check_throws([&] { trtmc::validate_sam3_hard_mask_resize_aoti_manifest(bundle, step); },
                 "SAM3 tracker runtime rejects a resize artifact hash mismatch");
}

void test_missing_memory_manifest_fails_closed() {
    auto bundle = make_valid_bundle();
    replace_section(bundle, trtmc::kSam3TrackerMemoryAotiManifestSection, "");
    const auto step = trtmc::validate_sam3_tracker_step_runtime_manifest(bundle);
    check_throws([&] { trtmc::validate_sam3_tracker_memory_aoti_manifest(bundle, step); },
                 "SAM3 tracker runtime rejects a missing memory manifest");
}

void test_memory_artifact_hash_mismatch_fails_closed() {
    auto bundle = make_valid_bundle();
    for (auto& section : bundle.sections) {
        if (section.name == "sam3_tracker_memory_soft_b1.pt2") {
            section.data.front() = 'X';
            break;
        }
    }
    const auto step = trtmc::validate_sam3_tracker_step_runtime_manifest(bundle);
    check_throws([&] { trtmc::validate_sam3_tracker_memory_aoti_manifest(bundle, step); },
                 "SAM3 tracker runtime rejects a memory artifact hash mismatch");
}

void test_memory_global_hash_mismatch_fails_closed() {
    auto bundle = make_valid_bundle();
    std::string manifest = section_contents(bundle, trtmc::kSam3TrackerMemoryAotiManifestSection);
    const auto marker = manifest.find(".fixed.");
    const auto digest = marker + std::string(".fixed.").size();
    check(marker != std::string::npos && digest < manifest.size(),
          "SAM3 synthetic memory manifest contains a global digest");
    manifest[digest] = manifest[digest] == '0' ? '1' : '0';
    replace_section(bundle, trtmc::kSam3TrackerMemoryAotiManifestSection, std::move(manifest));
    const auto step = trtmc::validate_sam3_tracker_step_runtime_manifest(bundle);
    check_throws([&] { trtmc::validate_sam3_tracker_memory_aoti_manifest(bundle, step); },
                 "SAM3 tracker runtime rejects a memory global/hash mismatch");
}

void test_memory_tensor_contract_mismatch_fails_closed() {
    auto bundle = make_valid_bundle();
    std::string manifest = section_contents(bundle, trtmc::kSam3TrackerMemoryAotiManifestSection);
    const auto marker = manifest.find("[1,256,72,72]");
    check(marker != std::string::npos,
          "SAM3 synthetic memory manifest contains the feature contract");
    manifest.replace(marker, std::string("[1,256,72,72]").size(), "[1,128,72,72]");
    replace_section(bundle, trtmc::kSam3TrackerMemoryAotiManifestSection, std::move(manifest));
    const auto step = trtmc::validate_sam3_tracker_step_runtime_manifest(bundle);
    check_throws([&] { trtmc::validate_sam3_tracker_memory_aoti_manifest(bundle, step); },
                 "SAM3 tracker runtime rejects a memory tensor contract mismatch");
}

void test_hard_memory_requires_global_tracker_grid() {
    auto bundle = make_valid_bundle();
    std::string manifest = section_contents(bundle, trtmc::kSam3TrackerMemoryAotiManifestSection);
    const auto marker = manifest.find("[1,1,1008,1008]");
    check(marker != std::string::npos,
          "SAM3 synthetic memory manifest contains the hard tracker-grid contract");
    manifest.replace(marker, std::string("[1,1,1008,1008]").size(), "[1,1,288,288]");
    replace_section(bundle, trtmc::kSam3TrackerMemoryAotiManifestSection, std::move(manifest));
    const auto step = trtmc::validate_sam3_tracker_step_runtime_manifest(bundle);
    check_throws([&] { trtmc::validate_sam3_tracker_memory_aoti_manifest(bundle, step); },
                 "SAM3 tracker runtime rejects a per-chunk hard-mask grid");
}

void test_memory_producer_mismatch_fails_closed() {
    auto bundle = make_valid_bundle();
    std::string manifest = section_contents(bundle, trtmc::kSam3TrackerMemoryAotiManifestSection);
    const auto marker = manifest.find("\"compute_capability\":[8,9]");
    check(marker != std::string::npos,
          "SAM3 synthetic memory manifest contains the producer architecture");
    manifest.replace(marker, std::string("\"compute_capability\":[8,9]").size(),
                     "\"compute_capability\":[12,0]");
    replace_section(bundle, trtmc::kSam3TrackerMemoryAotiManifestSection, std::move(manifest));
    const auto step = trtmc::validate_sam3_tracker_step_runtime_manifest(bundle);
    check_throws([&] { trtmc::validate_sam3_tracker_memory_aoti_manifest(bundle, step); },
                 "SAM3 tracker runtime rejects a memory/step producer mismatch");
}

void test_artifact_hash_mismatch_fails_closed() {
    auto bundle = make_valid_bundle();
    bundle.sections[1].data.front() = 'X';
    check_throws([&] { trtmc::validate_sam3_tracker_step_runtime_manifest(bundle); },
                 "SAM3 tracker-step manifest rejects an artifact hash mismatch");
}

void test_duplicate_stage_batch_fails_closed() {
    auto bundle = make_valid_bundle();
    replace_manifest(bundle, make_manifest(sha256(section_contents(bundle, 0)),
                                           sha256(section_contents(bundle, 1)),
                                           sha256(section_contents(bundle, 2)),
                                           sha256(section_contents(bundle, 3)),
                                           sha256(section_contents(bundle, 4)), 1));
    check_throws([&] { trtmc::validate_sam3_tracker_step_runtime_manifest(bundle); },
                 "SAM3 tracker-step manifest rejects duplicate stage/batch packages");
}

void test_unsafe_section_name_fails_closed() {
    auto bundle = make_valid_bundle();
    replace_manifest(
        bundle,
        make_manifest(sha256(section_contents(bundle, 0)), sha256(section_contents(bundle, 1)),
                      sha256(section_contents(bundle, 2)), sha256(section_contents(bundle, 3)),
                      sha256(section_contents(bundle, 4)), 2, "../tracker-encoder-b1.pt2"));
    check_throws([&] { trtmc::validate_sam3_tracker_step_runtime_manifest(bundle); },
                 "SAM3 tracker-step manifest rejects unsafe artifact names");
}

void test_partial_step_scope_fails_closed() {
    auto bundle = make_valid_bundle();
    std::string manifest = section_contents(bundle, bundle.sections.size() - 1);
    const auto scope = manifest.find(trtmc::kSam3TrackerStepScope);
    check(scope != std::string::npos, "SAM3 synthetic manifest contains the split scope");
    manifest.replace(scope, std::string(trtmc::kSam3TrackerStepScope).size(), "encoder_only");
    replace_manifest(bundle, std::move(manifest));
    check_throws([&] { trtmc::validate_sam3_tracker_step_runtime_manifest(bundle); },
                 "SAM3 tracker-step manifest rejects a partial encoder-only scope");
}

void test_non_content_addressed_package_global_fails_closed() {
    auto bundle = make_valid_bundle();
    std::string manifest = section_contents(bundle, bundle.sections.size() - 1);
    const auto marker = manifest.find(".m1_10.p1_19.");
    check(marker != std::string::npos, "SAM3 synthetic manifest contains an encoder global");
    manifest.replace(marker, std::string(".m1_10.p1_19.").size(), ".aoti.");
    replace_manifest(bundle, std::move(manifest));
    check_throws([&] { trtmc::validate_sam3_tracker_step_runtime_manifest(bundle); },
                 "SAM3 tracker-step manifest rejects a non-content-addressed package global");
}

void test_package_global_hash_mismatch_fails_closed() {
    auto bundle = make_valid_bundle();
    std::string manifest = section_contents(bundle, bundle.sections.size() - 1);
    const auto marker = manifest.find(".m1_10.p1_19.");
    const auto digest = marker + std::string(".m1_10.p1_19.").size();
    check(marker != std::string::npos && digest < manifest.size(),
          "SAM3 synthetic manifest contains a package digest");
    manifest[digest] = manifest[digest] == '0' ? '1' : '0';
    replace_manifest(bundle, std::move(manifest));
    check_throws([&] { trtmc::validate_sam3_tracker_step_runtime_manifest(bundle); },
                 "SAM3 tracker-step manifest rejects a package global/hash mismatch");
}

void test_pipeline_global_hash_mismatch_fails_closed() {
    auto bundle = make_valid_bundle();
    std::string manifest = section_contents(bundle, bundle.sections.size() - 1);
    const auto marker = manifest.find(".split_aoti.");
    const auto digest = marker + std::string(".split_aoti.").size();
    check(marker != std::string::npos && digest < manifest.size(),
          "SAM3 synthetic manifest contains a pipeline digest");
    manifest[digest] = manifest[digest] == '0' ? '1' : '0';
    replace_manifest(bundle, std::move(manifest));
    check_throws([&] { trtmc::validate_sam3_tracker_step_runtime_manifest(bundle); },
                 "SAM3 tracker-step manifest rejects a pipeline global/hash mismatch");
}

void test_pipeline_package_pairing_fails_closed() {
    auto bundle = make_valid_bundle();
    const auto encoder_b2_sha = sha256(section_contents(bundle, 3));
    replace_manifest(bundle, make_manifest(sha256(section_contents(bundle, 0)),
                                           sha256(section_contents(bundle, 1)),
                                           sha256(section_contents(bundle, 2)), encoder_b2_sha,
                                           sha256(section_contents(bundle, 4)), 2,
                                           "sam3_tracker_encoder_b1_dynamic.pt2", encoder_b2_sha));
    check_throws([&] { trtmc::validate_sam3_tracker_step_runtime_manifest(bundle); },
                 "SAM3 tracker-step pipeline rejects a cross-batch package reference");
}

void test_unknown_build_abi_fails_closed() {
    auto bundle = make_valid_bundle();
    std::string manifest = section_contents(bundle, bundle.sections.size() - 1);
    const auto version = manifest.find("0.1.6");
    check(version != std::string::npos, "SAM3 synthetic manifest contains a TVM-FFI version");
    manifest.replace(version, std::string("0.1.6").size(), "unknown");
    replace_manifest(bundle, std::move(manifest));
    check_throws([&] { trtmc::validate_sam3_tracker_step_runtime_manifest(bundle); },
                 "SAM3 tracker-step manifest rejects an unknown build ABI");
}

} // namespace

int main() {
    test_sha256_known_answer();
    test_valid_manifest();
    test_missing_resize_manifest_fails_closed();
    test_resize_artifact_hash_mismatch_fails_closed();
    test_missing_memory_manifest_fails_closed();
    test_memory_artifact_hash_mismatch_fails_closed();
    test_memory_global_hash_mismatch_fails_closed();
    test_memory_tensor_contract_mismatch_fails_closed();
    test_hard_memory_requires_global_tracker_grid();
    test_memory_producer_mismatch_fails_closed();
    test_artifact_hash_mismatch_fails_closed();
    test_duplicate_stage_batch_fails_closed();
    test_unsafe_section_name_fails_closed();
    test_partial_step_scope_fails_closed();
    test_non_content_addressed_package_global_fails_closed();
    test_package_global_hash_mismatch_fails_closed();
    test_pipeline_global_hash_mismatch_fails_closed();
    test_pipeline_package_pairing_fails_closed();
    test_unknown_build_abi_fails_closed();
    std::cout << "PASS: SAM3 split tracker-step runtime manifest\n";
    return 0;
}
