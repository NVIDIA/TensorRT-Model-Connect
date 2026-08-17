/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "tools/sam2_native_builder/sam2_engine_builder.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using trtmc::sam2::native::Sam2CompilationResult;
using trtmc::sam2::native::Sam2EngineBuildError;
using trtmc::sam2::native::Sam2EngineBuildOptions;
using trtmc::sam2::native::Sam2GraphBuildFacts;
using trtmc::sam2::native::Sam2GraphKind;
using trtmc::sam2::native::Sam2RuntimeBuildFacts;
using trtmc::sam2::native::Sam2SerializedPlan;

std::filesystem::path makeTemporaryDirectory() {
    std::array<char, 64> pattern{};
    const std::string value = "/tmp/trtmc_sam2_engine_builder_XXXXXX";
    std::copy(value.begin(), value.end(), pattern.begin());
    char* path = ::mkdtemp(pattern.data());
    if (path == nullptr)
        throw std::runtime_error("mkdtemp failed");
    return path;
}

void check(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

void expectBuildError(const std::function<void()>& function, std::string_view context) {
    try {
        function();
    } catch (const Sam2EngineBuildError&) {
        return;
    }
    throw std::runtime_error("expected Sam2EngineBuildError: " + std::string(context));
}

Sam2EngineBuildOptions validOptions(const std::filesystem::path& directory) {
    Sam2EngineBuildOptions options;
    options.checkpoint_path = directory / "checkpoint.pt";
    options.source_config_path = directory / "config.yaml";
    options.output_path = directory / "sam2.bundle";
    options.created_at_utc = "2026-08-15T12:34:56Z";
    return options;
}

Sam2CompilationResult validCompilation() {
    constexpr std::array<std::int32_t, 6> layer_counts = {1139, 882, 1630, 1652, 1674, 1696};
    constexpr std::array<std::size_t, 6> referenced_counts = {282U, 185U, 291U, 291U, 291U, 291U};
    Sam2CompilationResult result;
    result.runtime = Sam2RuntimeBuildFacts{
        std::string(trtmc::sam2::kTargetTensorRtVersion),
        std::string(trtmc::sam2::kTargetTensorRtAbi),
        "13.3.0",
        "13.0.0",
        std::string(trtmc::sam2::kTargetGpuName),
        0,
        trtmc::sam2::kTargetComputeCapabilityMajor,
        trtmc::sam2::kTargetComputeCapabilityMinor,
        24'000'000'000ULL,
        true,
        false,
    };
    result.plan_profiling_verbosity = std::string(trtmc::sam2::kPlanProfilingVerbosity);
    result.plans.reserve(trtmc::sam2::kRequiredPlanSections.size());
    for (std::size_t index = 0; index < trtmc::sam2::kRequiredPlanSections.size(); ++index) {
        Sam2GraphBuildFacts graph;
        graph.section = trtmc::sam2::kRequiredPlanSections[index];
        graph.kind = index == 0U   ? Sam2GraphKind::kImage
                     : index == 1U ? Sam2GraphKind::kPrompt
                                   : Sam2GraphKind::kRecurrent;
        graph.history_frames = index < 2U ? 0 : static_cast<std::int32_t>(index - 1U);
        graph.input_count = index == 0U ? 1 : (index == 1U ? 4 : 5);
        graph.output_count = index == 0U ? 9 : 3;
        graph.layer_count = layer_counts[index];
        graph.referenced_tensor_count = referenced_counts[index];
        graph.graph_complete = true;
        if (index == 0U) {
            graph.convolution_layer_count = 23;
            graph.activation_layer_count = 28;
            graph.pooling_layer_count = 6;
            graph.element_wise_layer_count = 130;
            graph.shuffle_layer_count = 313;
            graph.constant_layer_count = 216;
            graph.slice_layer_count = 67;
            graph.resize_layer_count = 2;
            graph.normalization_layer_count = 32;
            graph.cast_layer_count = 223;
            graph.matrix_multiply_layer_count = 67;
            graph.softmax_layer_count = 0;
            graph.plugin_v3_layer_count = 0;
            graph.attention_input_layer_count = 16;
            graph.attention_output_layer_count = 16;
        }
        result.plans.push_back(
            Sam2SerializedPlan{std::move(graph), std::vector<std::uint8_t>(index + 1U, 0xa5U)});
    }
    return result;
}

std::string sha256(const void* data, std::size_t size) {
    trtmc::internal::Sha256 hash;
    hash.update(data, size);
    return hash.hex_digest();
}

std::size_t countOccurrences(std::string_view text, std::string_view needle) {
    std::size_t count = 0;
    for (std::size_t offset = 0; (offset = text.find(needle, offset)) != std::string_view::npos;
         offset += needle.size())
        ++count;
    return count;
}

void testEmbeddedConfig() {
    const std::string_view config = trtmc::sam2::native::sam2EmbeddedConfigJson();
    constexpr std::string_view prefix = "{\"schema_version\":1,";
    constexpr std::string_view suffix = "\"runtime_eligible\":false}";
    check(config.size() >= prefix.size() && config.substr(0, prefix.size()) == prefix,
          "embedded config is canonical compact JSON");
    check(config.size() >= suffix.size() &&
              config.substr(config.size() - suffix.size(), suffix.size()) == suffix,
          "embedded config is explicitly runtime-ineligible");
    check(config.find(trtmc::sam2::kCheckpointSha256) != std::string_view::npos,
          "embedded config pins the checkpoint");
    check(config.find(trtmc::sam2::kConfigSha256) != std::string_view::npos,
          "embedded config pins the delivered source config");
    check(config.find(trtmc::sam2::kGoldenManifestSha256) != std::string_view::npos,
          "embedded config pins the reference golden");
    check(config.find("\"engine_contract_version\":5") != std::string_view::npos,
          "embedded config requires the detailed-inspection IAttentionV2 engine contract");
    for (const std::string_view section : trtmc::sam2::kRequiredPlanSections)
        check(countOccurrences(config, section) == 1U,
              "embedded config names every plan section exactly once");
}

void expectTargetMutationRejectedBeforeSerialization(
    const std::filesystem::path& directory, const char* filename,
    const std::function<void(Sam2CompilationResult&)>& mutate) {
    Sam2EngineBuildOptions options = validOptions(directory);
    options.output_path = directory / filename;
    Sam2CompilationResult compilation = validCompilation();
    mutate(compilation);
    check(!std::filesystem::exists(options.output_path),
          "target mutation output unexpectedly existed before the write seam");
    expectBuildError(
        [&] { trtmc::sam2::native::detail::writeCompiledSam2NativeBundle(options, compilation); },
        filename);
    check(!std::filesystem::exists(options.output_path),
          "target mutation reached bundle serialization");
}

void testValidation(const std::filesystem::path& directory) {
    const Sam2EngineBuildOptions options = validOptions(directory);
    const Sam2CompilationResult compilation = validCompilation();
    trtmc::sam2::native::validateSam2EngineBuildOptions(options);
    trtmc::sam2::native::validateSam2Compilation(compilation);

    expectTargetMutationRejectedBeforeSerialization(
        directory, "wrong-tensorrt-version.bundle", [](Sam2CompilationResult& invalid) {
            invalid.runtime.tensorrt_version.append(".mutated");
        });
    expectTargetMutationRejectedBeforeSerialization(
        directory, "wrong-tensorrt-abi.bundle",
        [](Sam2CompilationResult& invalid) { invalid.runtime.tensorrt_abi.append(".mutated"); });
    expectTargetMutationRejectedBeforeSerialization(
        directory, "wrong-gpu-name.bundle",
        [](Sam2CompilationResult& invalid) { invalid.runtime.gpu_name.append(" mutated"); });
    expectTargetMutationRejectedBeforeSerialization(
        directory, "wrong-compute-major.bundle", [](Sam2CompilationResult& invalid) {
            invalid.runtime.gpu_compute_major = trtmc::sam2::kTargetComputeCapabilityMajor + 1;
        });
    expectTargetMutationRejectedBeforeSerialization(
        directory, "wrong-compute-minor.bundle", [](Sam2CompilationResult& invalid) {
            invalid.runtime.gpu_compute_minor = trtmc::sam2::kTargetComputeCapabilityMinor + 1;
        });

    auto invalid_options = options;
    invalid_options.workspace_bytes = 0;
    expectBuildError([&] { trtmc::sam2::native::validateSam2EngineBuildOptions(invalid_options); },
                     "zero workspace");
    invalid_options = options;
    invalid_options.created_at_utc = "2026-08-15 12:34:56";
    expectBuildError([&] { trtmc::sam2::native::validateSam2EngineBuildOptions(invalid_options); },
                     "noncanonical timestamp");

    auto invalid = compilation;
    invalid.runtime.strongly_typed = false;
    expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                     "weakly typed network");
    invalid = compilation;
    invalid.runtime.tf32_enabled = true;
    expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                     "TF32 enabled");
    invalid = compilation;
    invalid.plan_profiling_verbosity = "layer_names_only";
    expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                     "non-detailed plan profiling verbosity");
    invalid = compilation;
    invalid.plans.pop_back();
    expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                     "missing recurrent plan");
    invalid = compilation;
    invalid.plans[0].bytes.clear();
    expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                     "empty image plan");
    invalid = compilation;
    invalid.plans[4].graph.graph_complete = false;
    expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                     "incomplete recurrent graph");
    invalid = compilation;
    std::swap(invalid.plans[2], invalid.plans[3]);
    expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                     "noncanonical plan order");
    invalid = compilation;
    invalid.plans[0].graph.referenced_tensor_count = 281;
    expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                     "image checkpoint coverage drift");
    const std::array<std::function<void(Sam2CompilationResult&)>, 15> inventory_mutations = {
        [](auto& value) { ++value.plans[0].graph.convolution_layer_count; },
        [](auto& value) { ++value.plans[0].graph.activation_layer_count; },
        [](auto& value) { ++value.plans[0].graph.pooling_layer_count; },
        [](auto& value) { ++value.plans[0].graph.element_wise_layer_count; },
        [](auto& value) { ++value.plans[0].graph.shuffle_layer_count; },
        [](auto& value) { ++value.plans[0].graph.constant_layer_count; },
        [](auto& value) { ++value.plans[0].graph.slice_layer_count; },
        [](auto& value) { ++value.plans[0].graph.resize_layer_count; },
        [](auto& value) { ++value.plans[0].graph.normalization_layer_count; },
        [](auto& value) { ++value.plans[0].graph.cast_layer_count; },
        [](auto& value) { ++value.plans[0].graph.matrix_multiply_layer_count; },
        [](auto& value) { ++value.plans[0].graph.softmax_layer_count; },
        [](auto& value) { ++value.plans[0].graph.plugin_v3_layer_count; },
        [](auto& value) { ++value.plans[0].graph.attention_input_layer_count; },
        [](auto& value) { ++value.plans[0].graph.attention_output_layer_count; },
    };
    for (const auto& mutate : inventory_mutations) {
        invalid = compilation;
        mutate(invalid);
        expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                         "image layer-type inventory drift");
    }
    for (std::size_t index = 0; index < compilation.plans.size(); ++index) {
        invalid = compilation;
        ++invalid.plans[index].graph.layer_count;
        expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                         "exact graph layer count drift");
        invalid = compilation;
        ++invalid.plans[index].graph.referenced_tensor_count;
        expectBuildError([&] { trtmc::sam2::native::validateSam2Compilation(invalid); },
                         "exact graph checkpoint-reference count drift");
    }
}

void testReceiptAndAssembly(const std::filesystem::path& directory) {
    const Sam2EngineBuildOptions options = validOptions(directory);
    const Sam2CompilationResult compilation = validCompilation();
    const std::string first = trtmc::sam2::native::makeSam2BuildReceipt(options, compilation);
    const std::string second = trtmc::sam2::native::makeSam2BuildReceipt(options, compilation);
    check(first == second, "receipt generation is deterministic for identical facts");
    constexpr std::string_view receipt_prefix = "{\"schema_version\":2,";
    check(trtmc::sam2::kBuildReceiptSchemaVersion == 2 && first.size() >= receipt_prefix.size() &&
              first.compare(0, receipt_prefix.size(), receipt_prefix) == 0,
          "receipt uses only the mandatory v2 schema");
    check(first.find("\"state\":\"unqualified\"") != std::string::npos, "receipt is unqualified");
    check(first.find("\"runtime_eligible\":false") != std::string::npos,
          "receipt is runtime-ineligible");
    check(first.find("\"golden_parity_verified\":false") != std::string::npos,
          "receipt does not claim golden parity");
    check(first.find("\"tf32_enabled\":false") != std::string::npos,
          "receipt records disabled TF32");
    check(trtmc::sam2::kBuilderOptimizationLevel == 3 &&
              countOccurrences(first, "\"builder_optimization_level\":3") == 1U,
          "receipt records exact TensorRT builder optimization level 3");
    check(first.find("\"plan_profiling_verbosity\":\"detailed\"") != std::string::npos,
          "receipt records detailed plan profiling verbosity");
    check(first.find("\"tensorrt_version\":\"" + std::string(trtmc::sam2::kTargetTensorRtVersion) +
                     "\"") != std::string::npos,
          "receipt records the exact TensorRT patch build");
    check(first.find("\"workspace_bytes\":8589934592") != std::string::npos,
          "receipt records the 8 GiB default workspace");
    check(countOccurrences(first, "\"image_attention\":{") == 1U,
          "receipt has exactly one native image-attention contract");
    check(first.find("\"implementation\":\"tensorrt_iattention_v2\","
                     "\"operator\":\"IAttention\",\"api\":\"addAttentionV2\","
                     "\"block_count\":16,\"head_dimension\":96,"
                     "\"query_form\":\"padded_bhnd\","
                     "\"key_value_form\":\"padded_bhnd\",\"output_form\":\"padded_bhnd\","
                     "\"normalization\":\"softmax\",\"causal_mask\":\"none\","
                     "\"decomposable\":false,\"fused_kernel_intent\":true,"
                     "\"metadata_prefix\":\"trtmc.sam2.iattention.block.\","
                     "\"metadata_index_width\":2,"
                     "\"q_scale_formula\":\"1/sqrt(head_dimension)\","
                     "\"k_scale_formula\":\"none\","
                     "\"effective_score_scale\":\"1/sqrt(head_dimension)\","
                     "\"scale_dtype\":\"bf16\"") != std::string::npos,
          "receipt binds the exact TensorRT IAttentionV2 contract");
    check(countOccurrences(first, "\"graph_complete\":true") == 6U,
          "receipt records all six complete graphs");
    check(first.find("\"layers\":1139,\"convolution_layers\":23,\"activation_layers\":28,"
                     "\"pooling_layers\":6,\"element_wise_layers\":130,"
                     "\"shuffle_layers\":313,\"constant_layers\":216,\"slice_layers\":67,"
                     "\"resize_layers\":2,\"normalization_layers\":32,\"cast_layers\":223,"
                     "\"matrix_multiply_layers\":67,\"softmax_layers\":0,"
                     "\"plugin_v3_layers\":0,\"attention_input_layers\":16,"
                     "\"attention_output_layers\":16,\"referenced_checkpoint_tensors\":282") !=
              std::string::npos,
          "receipt records the exact IAttentionV2 image graph layer inventory");
    check(countOccurrences(first, "\"serialized_sha256\":\"") == 6U,
          "receipt binds every serialized plan by SHA-256");
    for (const auto& plan : compilation.plans) {
        check(first.find(sha256(plan.bytes.data(), plan.bytes.size())) != std::string::npos,
              "receipt contains the exact serialized plan SHA-256");
    }
    const std::string_view embedded_config = trtmc::sam2::native::sam2EmbeddedConfigJson();
    check(first.find(sha256(embedded_config.data(), embedded_config.size())) != std::string::npos,
          "receipt binds the embedded config section by SHA-256");

    auto changed_plan = compilation;
    changed_plan.plans[3].bytes[0] ^= 1U;
    const std::string changed_receipt =
        trtmc::sam2::native::makeSam2BuildReceipt(options, changed_plan);
    check(changed_receipt != first &&
              changed_receipt.find(sha256(changed_plan.plans[3].bytes.data(),
                                          changed_plan.plans[3].bytes.size())) != std::string::npos,
          "receipt changes when serialized plan bytes change");

    const auto build_result =
        trtmc::sam2::native::detail::writeCompiledSam2NativeBundle(options, compilation);
    check(std::filesystem::is_regular_file(options.output_path),
          "CPU seam writes an eight-section bundle");
    std::ifstream input(options.output_path, std::ios::binary);
    const std::string contents((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    check(contents.find(trtmc::sam2::native::sam2EmbeddedConfigJson()) != std::string::npos,
          "bundle embeds the exact canonical config");
    check(contents.find(first) != std::string::npos, "bundle embeds the deterministic receipt");
    check(build_result.build_receipt_json == first,
          "builder returns the exact receipt bytes embedded in the bundle");
    check(build_result.build_receipt_sha256 == sha256(first.data(), first.size()),
          "builder returns the exact embedded receipt digest");
    for (std::size_t index = 0; index < compilation.plans.size(); ++index) {
        check(build_result.plan_sha256[index] == sha256(compilation.plans[index].bytes.data(),
                                                        compilation.plans[index].bytes.size()),
              "builder returns each ordered serialized plan digest");
    }
    struct stat published_status{};
    check(::stat(options.output_path.c_str(), &published_status) == 0,
          "published bundle can be statted");
    check(build_result.bundle.sha256 == sha256(contents.data(), contents.size()) &&
              build_result.bundle.size_bytes == contents.size() &&
              build_result.bundle.filesystem_identity_available &&
              build_result.bundle.device == static_cast<std::uint64_t>(published_status.st_dev) &&
              build_result.bundle.inode == static_cast<std::uint64_t>(published_status.st_ino),
          "builder returns full digest and identity for the exact published bundle");
    expectBuildError([&] { trtmc::sam2::native::validateSam2EngineBuildOptions(options); },
                     "existing destination");

    auto wrong_device = validOptions(directory);
    wrong_device.output_path = directory / "wrong-device.bundle";
    wrong_device.gpu_device = 1;
    expectBuildError(
        [&] { (void)trtmc::sam2::native::makeSam2BuildReceipt(wrong_device, compilation); },
        "runtime facts from a different device");
}

void testConfigAuthentication(const std::filesystem::path& directory,
                              const char* delivered_config) {
    const auto mismatch = directory / "mismatch.yaml";
    {
        std::ofstream output(mismatch, std::ios::binary);
        output << "not the delivered SAM2 config\n";
    }
    expectBuildError([&] { trtmc::sam2::native::verifySam2SourceConfig(mismatch); },
                     "source config SHA mismatch");
    if (delivered_config != nullptr)
        trtmc::sam2::native::verifySam2SourceConfig(delivered_config);
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: " << argv[0] << " [delivered-config]\n";
        return 2;
    }
    try {
        static_assert(trtmc::sam2::native::kDefaultSam2WorkspaceBytes == 8'589'934'592ULL);
        static_assert(trtmc::sam2::native::kMaximumSam2ConfigBytes == 1'048'576ULL);
        const auto directory = makeTemporaryDirectory();
        testEmbeddedConfig();
        testValidation(directory);
        testReceiptAndAssembly(directory);
        testConfigAuthentication(directory, argc == 2 ? argv[1] : nullptr);
        std::filesystem::remove_all(directory);
        std::cout << "SAM2 engine builder contract tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
