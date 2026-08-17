/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_native_bundle_loader.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using Contracts = std::vector<trtmc::sam2::TensorContract>;

constexpr std::string_view kModelId = "sam2.1-hiera-small-bbox";
constexpr std::string_view kCreatedAt = "2026-08-15T00:00:00Z";

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

template <typename Exception, typename Function>
void checkThrows(Function&& function, const char* needle, const char* message) {
    static_assert(std::is_base_of<std::exception, Exception>::value,
                  "test exception must derive from std::exception");
    try {
        function();
    } catch (const Exception& error) {
        if (std::strstr(error.what(), needle) != nullptr)
            return;
        std::cerr << "FAIL: " << message << " (wrong message: " << error.what() << ")\n";
        std::exit(1);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << message << " (wrong exception: " << error.what() << ")\n";
        std::exit(1);
    }
    std::cerr << "FAIL: " << message << " (no exception)\n";
    std::exit(1);
}

std::filesystem::path makeTemporaryDirectory() {
    std::array<char, 64> pattern{};
    const std::string value = "/tmp/trtmc_sam2_bundle_loader_XXXXXX";
    std::copy(value.begin(), value.end(), pattern.begin());
    char* result = ::mkdtemp(pattern.data());
    if (result == nullptr)
        throw std::runtime_error("mkdtemp failed");
    return result;
}

std::string sha256(const void* data, std::size_t size) {
    trtmc::internal::Sha256 hash;
    hash.update(data, size);
    return hash.hex_digest();
}

std::string sha256(const std::vector<char>& data) {
    return sha256(data.data(), data.size());
}

void writeU64LittleEndian(std::ostream& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64U; shift += 8U)
        output.put(static_cast<char>((value >> shift) & 0xffU));
}

std::vector<std::string_view> sectionNames(bool omit_receipt) {
    std::vector<std::string_view> result;
    result.insert(result.end(), trtmc::sam2::kRequiredPlanSections.begin(),
                  trtmc::sam2::kRequiredPlanSections.end());
    result.push_back(trtmc::sam2::kConfigSection);
    if (!omit_receipt)
        result.push_back(trtmc::sam2::kBuildReceiptSection);
    return result;
}

struct ArtifactOptions {
    bool qualified{false};
    std::int32_t engine_contract_version{
        static_cast<std::int32_t>(trtmc::sam2::kEngineContractVersion)};
    bool receipt_qualified{false};
    bool receipt_qualification_override{false};
    bool wrong_header_plan_hash{false};
    bool wrong_receipt_plan_hash{false};
    bool wrong_receipt_config_hash{false};
    bool wrong_graph_facts{false};
    bool omit_image_attention{false};
    bool wrong_attention_implementation{false};
    bool wrong_attention_operator{false};
    bool wrong_attention_api{false};
    bool wrong_attention_block_count{false};
    bool wrong_attention_head_dimension{false};
    bool wrong_attention_query_form{false};
    bool wrong_attention_key_value_form{false};
    bool wrong_attention_output_form{false};
    bool wrong_attention_normalization{false};
    bool wrong_attention_causal_mask{false};
    bool wrong_attention_decomposable{false};
    bool wrong_attention_fused_kernel_intent{false};
    bool wrong_attention_metadata_prefix{false};
    bool wrong_attention_metadata_index_width{false};
    bool wrong_attention_q_scale_formula{false};
    bool wrong_attention_k_scale_formula{false};
    bool wrong_attention_effective_score_scale{false};
    bool wrong_attention_scale_dtype{false};
    bool extra_attention_field{false};
    bool wrong_image_layer_type_facts{false};
    bool wrong_plan_profiling_verbosity{false};
    bool omit_receipt{false};
    bool append_trailing_byte{false};
    std::string header_model_id{std::string(kModelId)};
    std::string trt_version{std::string(trtmc::sam2::kTargetTensorRtVersion)};
    std::string trt_abi{std::string(trtmc::sam2::kTargetTensorRtAbi)};
    std::string gpu_name{std::string(trtmc::sam2::kTargetGpuName)};
    std::string compute_capability{std::string(trtmc::sam2::kTargetComputeCapability)};
};

std::array<std::vector<char>, 6> planPayloads() {
    std::array<std::vector<char>, 6> plans;
    for (std::size_t index = 0; index < plans.size(); ++index) {
        const std::string text = "synthetic-sam2-plan-" + std::to_string(index);
        plans[index] = std::vector<char>(text.begin(), text.end());
    }
    return plans;
}

Json makeConfig(bool qualified, std::int32_t engine_contract_version) {
    Json result;
    result["schema_version"] = 1;
    result["family"] = "sam2";
    result["model_id"] = kModelId;
    result["engine_contract_version"] = engine_contract_version;
    result["runtime_strategy"] = trtmc::sam2::kStrategyName;
    result["precision"] = "mixed_bf16_fp32";
    result["checkpoint_sha256"] = trtmc::sam2::kCheckpointSha256;
    result["source_config_sha256"] = trtmc::sam2::kConfigSha256;
    result["golden_manifest_sha256"] = trtmc::sam2::kGoldenManifestSha256;
    result["frame_count"] = trtmc::sam2::kFrameCount;
    result["selected_object_count"] = trtmc::sam2::kSelectedObjectCount;
    result["model_image_size"] = trtmc::sam2::kModelImageSize;
    result["original_image_height"] = trtmc::sam2::kOriginalImageHeight;
    result["original_image_width"] = trtmc::sam2::kOriginalImageWidth;
    result["plan_sections"] = Json::array();
    for (const auto section : trtmc::sam2::kRequiredPlanSections)
        result["plan_sections"].push_back(section);
    result["qualification"] = qualified ? "qualified" : "unqualified";
    result["runtime_eligible"] = qualified;
    return result;
}

Json makeReceipt(const ArtifactOptions& options, const std::array<std::vector<char>, 6>& plans,
                 std::string_view config_digest) {
    const bool receipt_qualified =
        options.receipt_qualification_override ? options.receipt_qualified : options.qualified;
    Json result;
    result["schema_version"] = 1;
    result["family"] = "sam2";
    result["model_id"] = kModelId;
    result["qualification"] = {
        {"state", receipt_qualified ? "qualified" : "unqualified"},
        {"runtime_eligible", receipt_qualified},
        {"golden_parity_verified", receipt_qualified},
    };
    result["assets"] = {
        {"checkpoint_sha256", trtmc::sam2::kCheckpointSha256},
        {"source_config_sha256", trtmc::sam2::kConfigSha256},
        {"golden_manifest_sha256", trtmc::sam2::kGoldenManifestSha256},
        {"embedded_config_sha256",
         options.wrong_receipt_config_hash ? std::string(64U, '0') : std::string(config_digest)},
    };
    result["build"] = {
        {"created_at_utc", kCreatedAt},
        {"workspace_bytes", UINT64_C(8589934592)},
        {"network_mode", "strongly_typed"},
        {"tf32_enabled", false},
        {"plan_profiling_verbosity",
         options.wrong_plan_profiling_verbosity ? "layer_names_only" : "detailed"},
        {"tensorrt_version", options.trt_version},
        {"tensorrt_abi", options.trt_abi},
        {"cuda_runtime_version", "13.3.0"},
        {"cuda_driver_version", "13.0.0"},
        {"gpu",
         {{"device", 0},
          {"name", options.gpu_name},
          {"compute_capability", options.compute_capability},
          {"global_memory_bytes", UINT64_C(24146608128)}}},
    };
    if (!options.omit_image_attention) {
        result["image_attention"] = {
            {"implementation",
             options.wrong_attention_implementation ? "tensorrt_native" : "tensorrt_iattention_v2"},
            {"operator", options.wrong_attention_operator ? "IMatrixMultiplyLayer" : "IAttention"},
            {"api", options.wrong_attention_api ? "addMatrixMultiply" : "addAttentionV2"},
            {"block_count", options.wrong_attention_block_count ? 15 : 16},
            {"head_dimension", options.wrong_attention_head_dimension ? 64 : 96},
            {"query_form", options.wrong_attention_query_form ? "packed_nhd" : "padded_bhnd"},
            {"key_value_form",
             options.wrong_attention_key_value_form ? "packed_nhd" : "padded_bhnd"},
            {"output_form", options.wrong_attention_output_form ? "packed_nhd" : "padded_bhnd"},
            {"normalization", options.wrong_attention_normalization ? "none" : "softmax"},
            {"causal_mask", options.wrong_attention_causal_mask ? "upper_left" : "none"},
            {"decomposable", options.wrong_attention_decomposable},
            {"fused_kernel_intent", !options.wrong_attention_fused_kernel_intent},
            {"metadata_prefix", options.wrong_attention_metadata_prefix
                                    ? "unreviewed."
                                    : "trtmc.sam2.iattention.block."},
            {"metadata_index_width", options.wrong_attention_metadata_index_width ? 1 : 2},
            {"q_scale_formula",
             options.wrong_attention_q_scale_formula ? "none" : "1/sqrt(head_dimension)"},
            {"k_scale_formula",
             options.wrong_attention_k_scale_formula ? "1/sqrt(head_dimension)" : "none"},
            {"effective_score_scale",
             options.wrong_attention_effective_score_scale ? "none" : "1/sqrt(head_dimension)"},
            {"scale_dtype", options.wrong_attention_scale_dtype ? "fp32" : "bf16"},
        };
        if (options.extra_attention_field)
            result["image_attention"]["unreviewed_option"] = true;
    }
    constexpr std::array<std::string_view, 6> kinds = {"image",     "prompt",    "recurrent",
                                                       "recurrent", "recurrent", "recurrent"};
    constexpr std::array<std::int32_t, 6> histories = {0, 0, 1, 2, 3, 4};
    constexpr std::array<std::int32_t, 6> inputs = {1, 4, 5, 5, 5, 5};
    constexpr std::array<std::int32_t, 6> outputs = {9, 3, 3, 3, 3, 3};
    constexpr std::array<std::uint64_t, 6> layers = {1139U, 882U, 1630U, 1652U, 1674U, 1696U};
    constexpr std::array<std::uint64_t, 6> referenced_tensors = {282U, 185U, 291U,
                                                                 291U, 291U, 291U};
    result["graphs"] = Json::array();
    for (std::size_t index = 0; index < plans.size(); ++index) {
        const std::string digest = options.wrong_receipt_plan_hash && index == 0U
                                       ? std::string(64U, '0')
                                       : sha256(plans[index]);
        Json graph = {
            {"section", trtmc::sam2::kRequiredPlanSections[index]},
            {"kind", kinds[index]},
            {"history_frames", histories[index]},
            {"inputs", inputs[index]},
            {"outputs", outputs[index]},
            {"layers", layers[index] + (options.wrong_graph_facts && index == 0U ? 1U : 0U)},
            {"referenced_checkpoint_tensors", referenced_tensors[index]},
            {"serialized_bytes", plans[index].size()},
            {"serialized_sha256", digest},
            {"graph_complete", true},
        };
        if (index == 0U) {
            graph["convolution_layers"] = options.wrong_image_layer_type_facts ? 22 : 23;
            graph["activation_layers"] = 28;
            graph["pooling_layers"] = 6;
            graph["element_wise_layers"] = 130;
            graph["shuffle_layers"] = 313;
            graph["constant_layers"] = 216;
            graph["slice_layers"] = 67;
            graph["resize_layers"] = 2;
            graph["normalization_layers"] = 32;
            graph["cast_layers"] = options.wrong_image_layer_type_facts ? 222 : 223;
            graph["matrix_multiply_layers"] = options.wrong_image_layer_type_facts ? 66 : 67;
            graph["softmax_layers"] = options.wrong_image_layer_type_facts ? 1 : 0;
            graph["plugin_v3_layers"] = options.wrong_image_layer_type_facts ? 1 : 0;
            graph["attention_input_layers"] = options.wrong_image_layer_type_facts ? 15 : 16;
            graph["attention_output_layers"] = options.wrong_image_layer_type_facts ? 15 : 16;
        }
        result["graphs"].push_back(std::move(graph));
    }
    return result;
}

std::filesystem::path writeArtifact(const std::filesystem::path& path,
                                    const ArtifactOptions& options = {}) {
    const auto plans = planPayloads();
    const std::string config_text =
        makeConfig(options.qualified, options.engine_contract_version).dump();
    const std::string config_digest = sha256(config_text.data(), config_text.size());
    const std::string receipt_text = makeReceipt(options, plans, config_digest).dump();

    std::vector<std::vector<char>> payloads;
    payloads.reserve(8U);
    payloads.insert(payloads.end(), plans.begin(), plans.end());
    payloads.emplace_back(config_text.begin(), config_text.end());
    if (!options.omit_receipt)
        payloads.emplace_back(receipt_text.begin(), receipt_text.end());
    const auto names = sectionNames(options.omit_receipt);

    Json header;
    header["model_id"] = options.header_model_id;
    header["model_type"] = "sam2_video_tracking";
    header["family"] = "sam2";
    header["precision"] = "mixed_bf16_fp32";
    header["trt_version"] = options.trt_version;
    header["trt_abi"] = options.trt_abi;
    header["gpu_name"] = options.gpu_name;
    header["created_at"] = kCreatedAt;
    header["runtime_strategy"] = trtmc::sam2::kStrategyName;
    header["sections"] = Json::object();
    std::uint64_t offset = 0;
    for (std::size_t index = 0; index < payloads.size(); ++index) {
        std::string digest = sha256(payloads[index]);
        if (options.wrong_header_plan_hash && index == 0U)
            digest.assign(64U, '0');
        header["sections"][std::string(names[index])] = {
            {"offset", offset}, {"size", payloads[index].size()}, {"sha256", digest}};
        offset += payloads[index].size();
    }
    const std::string header_text = header.dump();

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to create test bundle");
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic),
                 static_cast<std::streamsize>(sizeof(trtmc::kBundleMagic)));
    writeU64LittleEndian(output, header_text.size());
    output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const auto& payload : payloads)
        output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (options.append_trailing_byte)
        output.put('X');
    if (!output)
        throw std::runtime_error("failed to write test bundle");
    return path;
}

std::string sha256File(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to open test bundle for full-file hashing");
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
    return sha256(bytes);
}

trtmc::sam2::NativeQualificationRecord
makeQualificationRecord(const std::filesystem::path& bundle_path,
                        const ArtifactOptions& options = {}) {
    const auto plans = planPayloads();
    const std::string config_text =
        makeConfig(options.qualified, options.engine_contract_version).dump();
    const std::string receipt_text =
        makeReceipt(options, plans, sha256(config_text.data(), config_text.size())).dump();

    trtmc::sam2::NativeQualificationRecord record;
    record.authority_id = "sam2-l4-trt11.1-contract5-0001";
    record.authority_serial = 17U;
    record.self_authorizing = false;
    record.scope = {
        "sam2",
        std::string(kModelId),
        trtmc::sam2::kEngineContractVersion,
        std::string(trtmc::sam2::kStrategyName),
        "mixed_bf16_fp32",
        std::string(trtmc::sam2::kTargetGpuName),
        std::string(trtmc::sam2::kTargetComputeCapability),
        std::string(trtmc::sam2::kTargetTensorRtVersion),
        std::string(trtmc::sam2::kTargetTensorRtAbi),
    };
    record.bundle.sha256 = sha256File(bundle_path);
    record.bundle.size_bytes = std::filesystem::file_size(bundle_path);
    record.bundle.embedded_config_sha256 = sha256(config_text.data(), config_text.size());
    record.bundle.build_receipt_sha256 = sha256(receipt_text.data(), receipt_text.size());
    for (std::size_t index = 0; index < plans.size(); ++index) {
        record.bundle.plans[index] = {std::string(trtmc::sam2::kRequiredPlanSections[index]),
                                      sha256(plans[index])};
    }
    auto& evidence = record.accuracy_evidence;
    evidence.receipt_sha256 = std::string(64U, '1');
    evidence.receipt_size_bytes = 4096U;
    evidence.regular_receipt_sha256 = std::string(64U, '5');
    evidence.regular_receipt_size_bytes = 8192U;
    evidence.mode = "accuracy_only";
    evidence.policy_id = std::string(trtmc::sam2::kNativeSemanticAccuracyPolicyId);
    evidence.replay_count = 3U;
    evidence.frames_per_replay = 5U;
    evidence.reset_before_each_replay = true;
    evidence.all_semantic_gates_passed = true;
    evidence.timing_performed = false;
    evidence.golden_manifest_sha256 = std::string(trtmc::sam2::kGoldenManifestSha256);
    evidence.golden_masks_sha256 = std::string(trtmc::sam2::kGoldenMasksSha256);
    evidence.benchmark_executable_sha256 = std::string(64U, '2');
    evidence.benchmark_source_manifest_sha256 = std::string(64U, '3');
    evidence.benchmark_source_closure_sha256 = std::string(64U, '4');
    record.generated_at_utc = "2026-08-16T00:00:00Z";
    return record;
}

void writeTextFile(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to create test record");
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output)
        throw std::runtime_error("failed to write test record");
}

std::filesystem::path
writeQualificationRecord(const std::filesystem::path& path,
                         const trtmc::sam2::NativeQualificationRecord& record) {
    writeTextFile(path, trtmc::sam2::makeCanonicalNativeQualificationRecord(record));
    return path;
}

trtmc::sam2::qualification_internal::NativeQualificationTestPin
testPin(const std::filesystem::path& record_path, std::uint64_t minimum_serial = 17U,
        std::string authority_id = "sam2-l4-trt11.1-contract5-0001") {
    return {std::move(authority_id), minimum_serial, sha256File(record_path)};
}

trtmc::DType dtype(trtmc::sam2::TensorDataType value) {
    return value == trtmc::sam2::TensorDataType::kFloat32 ? trtmc::DType::kFloat32
                                                          : trtmc::DType::kBFloat16;
}

std::vector<int64_t> shape(const trtmc::sam2::TensorContract& contract) {
    std::vector<int64_t> result;
    for (std::uint8_t index = 0; index < contract.rank; ++index)
        result.push_back(contract.dimensions[index]);
    return result;
}

const trtmc::sam2::TensorContract* find(const Contracts& contracts, const std::string& name) {
    const auto found = std::find_if(contracts.begin(), contracts.end(),
                                    [&](const auto& contract) { return contract.name == name; });
    return found == contracts.end() ? nullptr : &*found;
}

class ContractModule final : public trtmc::ITrtModule {
  public:
    ContractModule(Contracts inputs, Contracts outputs)
        : inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}

    trtmc::TensorMap forward(const trtmc::TensorMap&) override { return {}; }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return info(inputs_, true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return info(outputs_, false); }
    bool has_input(const std::string& name) const override {
        return find(inputs_, name) != nullptr;
    }
    bool has_output(const std::string& name) const override {
        return find(outputs_, name) != nullptr;
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        const auto* contract = find(inputs_, name);
        if (contract == nullptr)
            contract = find(outputs_, name);
        return contract == nullptr ? trtmc::DType::kFloat32 : dtype(contract->data_type);
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        const auto* contract = find(inputs_, name);
        if (contract == nullptr)
            contract = find(outputs_, name);
        return contract == nullptr ? std::vector<int64_t>{} : shape(*contract);
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    static std::vector<trtmc::TensorInfo> info(const Contracts& contracts, bool input) {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& contract : contracts)
            result.push_back(
                {std::string(contract.name), shape(contract), dtype(contract.data_type), input});
        return result;
    }

    Contracts inputs_;
    Contracts outputs_;
};

Contracts imageInputs() {
    return {trtmc::sam2::kPixelValues};
}

Contracts imageOutputs() {
    Contracts result(trtmc::sam2::kTrackerFpn.begin(), trtmc::sam2::kTrackerFpn.end());
    result.insert(result.end(), trtmc::sam2::kBboxMaps.begin(), trtmc::sam2::kBboxMaps.end());
    return result;
}

Contracts promptInputs() {
    Contracts result(trtmc::sam2::kTrackerFpn.begin(), trtmc::sam2::kTrackerFpn.end());
    result.push_back(trtmc::sam2::kBoxPrompt);
    return result;
}

Contracts trackerOutputs() {
    return {trtmc::sam2::kMaskLogits256, trtmc::sam2::kObjectPointer, trtmc::sam2::kMemoryFeatures};
}

Contracts recurrentInputs(std::int32_t history) {
    Contracts result(trtmc::sam2::kTrackerFpn.begin(), trtmc::sam2::kTrackerFpn.end());
    result.push_back(trtmc::sam2::historyMemoryFeatures(history));
    result.push_back(trtmc::sam2::historyObjectPointers(history));
    return result;
}

using FactoryObserver =
    std::function<void(std::size_t, std::string_view, const void*, std::size_t)>;

trtmc::sam2::NativePlanModuleFactory makeFactory(std::vector<std::string>& calls,
                                                 bool bad_image_abi = false,
                                                 bool return_null = false,
                                                 FactoryObserver observer = {}) {
    return [&calls, bad_image_abi, return_null, observer = std::move(observer)](
               std::string_view section, const void* data,
               std::size_t size) -> std::unique_ptr<trtmc::ITrtModule> {
        check(data != nullptr && size != 0U, "factory received an empty plan");
        if (observer)
            observer(calls.size(), section, data, size);
        calls.emplace_back(section);
        if (return_null)
            return nullptr;
        if (section == trtmc::sam2::kImagePlanSection) {
            auto outputs = imageOutputs();
            if (bad_image_abi)
                outputs.pop_back();
            return std::make_unique<ContractModule>(imageInputs(), std::move(outputs));
        }
        if (section == trtmc::sam2::kPromptPlanSection)
            return std::make_unique<ContractModule>(promptInputs(), trackerOutputs());
        for (std::size_t index = 0; index < trtmc::sam2::kRecurrentPlanSections.size(); ++index) {
            if (section == trtmc::sam2::kRecurrentPlanSections[index]) {
                return std::make_unique<ContractModule>(
                    recurrentInputs(static_cast<std::int32_t>(index + 1U)), trackerOutputs());
            }
        }
        throw std::runtime_error("unexpected plan section");
    };
}

trtmc::sam2::NativeBundleRuntimeTarget runtimeTarget() {
    return {std::string(trtmc::sam2::kTargetTensorRtVersion),
            std::string(trtmc::sam2::kTargetTensorRtAbi), std::string(trtmc::sam2::kTargetGpuName),
            std::string(trtmc::sam2::kTargetComputeCapability)};
}

template <typename Mutation>
std::filesystem::path writeMutatedQualificationRecord(const std::filesystem::path& path,
                                                      const std::string& canonical,
                                                      Mutation&& mutation) {
    Json value = Json::parse(canonical);
    mutation(value);
    writeTextFile(path, value.dump() + '\n');
    return path;
}

void expectProductionQualificationRejected(
    const std::filesystem::path& bundle_path, const std::filesystem::path& record_path,
    const trtmc::sam2::qualification_internal::NativeQualificationTestPin& pin, const char* needle,
    const char* message, std::vector<std::string>& calls) {
    calls.clear();
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadProductionQualifiedNativeVideoEngineSetFromBundleForTest(
                bundle_path.string(), record_path.string(), runtimeTarget(), makeFactory(calls),
                pin);
        },
        needle, message);
    check(calls.empty(), "factory ran before production qualification rejection");
}

void expectArtifactTargetMutationRejected(const std::filesystem::path& directory,
                                          const char* filename, const ArtifactOptions& options,
                                          const char* authenticated_field,
                                          std::vector<std::string>& calls) {
    const auto artifact = writeArtifact(directory / filename, options);
    auto caller_controlled_target = runtimeTarget();
    caller_controlled_target.tensorrt_version = options.trt_version;
    caller_controlled_target.tensorrt_abi = options.trt_abi;
    caller_controlled_target.gpu_name = options.gpu_name;
    caller_controlled_target.compute_capability = options.compute_capability;

    calls.clear();
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                artifact.string(), caller_controlled_target, makeFactory(calls));
        },
        "pinned TRT 11.1 L4 target", "caller-controlled target mutation was accepted");
    check(calls.empty(), "factory ran for a caller-controlled target mutation");

    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                artifact.string(), runtimeTarget(), makeFactory(calls));
        },
        authenticated_field, "authenticated target mutation was accepted");
    check(calls.empty(), "factory ran for an authenticated target mutation");
}

} // namespace

int main() {
    const auto directory = makeTemporaryDirectory();

    std::vector<std::string> calls;
    const auto unqualified = writeArtifact(directory / "unqualified.bundle");

    const auto valid_record_value = makeQualificationRecord(unqualified);
    const std::string canonical_record =
        trtmc::sam2::makeCanonicalNativeQualificationRecord(valid_record_value);
    check(canonical_record ==
                  trtmc::sam2::makeCanonicalNativeQualificationRecord(valid_record_value) &&
              !canonical_record.empty() && canonical_record.back() == '\n' &&
              canonical_record[canonical_record.size() - 2U] != '\n',
          "qualification record generator is not deterministic and newline canonical");
    for (const std::string_view private_provenance :
         {"qualification_environment", "gpu_uuid", "pci_bus_id", "cuda_runtime_version",
          "cuda_driver_version", "hostname"}) {
        check(canonical_record.find(private_provenance) == std::string::npos,
              "public qualification record leaked infrastructure provenance");
    }
    const auto valid_record = directory / "valid-qualification.json";
    writeTextFile(valid_record, canonical_record);

    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadProductionQualifiedNativeVideoEngineSetFromBundle(
                (directory / "must-not-be-opened.bundle").string(), valid_record.string(),
                runtimeTarget(), makeFactory(calls));
        },
        "no active compiled production authority pin",
        "empty production authority registry did not fail closed before bundle open");
    check(calls.empty(), "factory ran with an empty production authority registry");

    auto production_engines =
        trtmc::sam2::loadProductionQualifiedNativeVideoEngineSetFromBundleForTest(
            unqualified.string(), valid_record.string(), runtimeTarget(), makeFactory(calls),
            testPin(valid_record));
    check(production_engines.image != nullptr && production_engines.prompt != nullptr &&
              calls.size() == trtmc::sam2::kRequiredPlanSections.size(),
          "test-only injected pin did not exercise the production loader path");
    calls.clear();

    auto mismatched_pin = testPin(valid_record);
    mismatched_pin.record_sha256.assign(64U, '0');
    expectProductionQualificationRejected(unqualified, valid_record, mismatched_pin,
                                          "compiled authority pin",
                                          "record digest mismatch was accepted", calls);

    std::string duplicate_record = canonical_record;
    const std::string schema_field = "\"schema_version\":2";
    const auto schema_offset = duplicate_record.find(schema_field);
    check(schema_offset != std::string::npos, "canonical record omitted schema_version");
    duplicate_record.replace(schema_offset, schema_field.size(), schema_field + "," + schema_field);
    const auto duplicate_path = directory / "duplicate-key-qualification.json";
    writeTextFile(duplicate_path, duplicate_record);
    expectProductionQualificationRejected(unqualified, duplicate_path, testPin(duplicate_path),
                                          "duplicate key",
                                          "duplicate qualification record key was accepted", calls);

    const auto extra_path = writeMutatedQualificationRecord(
        directory / "extra-field-qualification.json", canonical_record,
        [](Json& value) { value["unreviewed"] = true; });
    expectProductionQualificationRejected(unqualified, extra_path, testPin(extra_path),
                                          "field set drifted",
                                          "extra qualification record field was accepted", calls);

    nlohmann::json private_environment_record = nlohmann::json::parse(canonical_record);
    private_environment_record["accuracy_evidence"]["qualification_environment"] = {
        {"gpu_uuid", "GPU-private"},        {"pci_bus_id", "0000:01:00.0"},
        {"cuda_runtime_version", "13.3.0"}, {"cuda_driver_version", "13.0.0"},
        {"hostname", "qualification-host"},
    };
    const auto private_environment_path = directory / "private-environment-qualification.json";
    writeTextFile(private_environment_path, private_environment_record.dump() + '\n');
    expectProductionQualificationRejected(
        unqualified, private_environment_path, testPin(private_environment_path),
        "accuracy_evidence field set drifted",
        "private qualification environment was accepted in a public record", calls);

    const auto noncanonical_path = directory / "noncanonical-qualification.json";
    writeTextFile(noncanonical_path, " " + canonical_record);
    expectProductionQualificationRejected(unqualified, noncanonical_path,
                                          testPin(noncanonical_path), "is not canonical",
                                          "noncanonical qualification record was accepted", calls);

    const auto old_schema_path = writeMutatedQualificationRecord(
        directory / "old-schema-qualification.json", canonical_record,
        [](Json& value) { value["schema_version"] = 1; });
    expectProductionQualificationRejected(unqualified, old_schema_path, testPin(old_schema_path),
                                          "schema_version is not supported",
                                          "old qualification schema was accepted", calls);

    const auto old_serial_path = writeMutatedQualificationRecord(
        directory / "old-serial-qualification.json", canonical_record,
        [](Json& value) { value["authority_serial"] = 16; });
    expectProductionQualificationRejected(unqualified, old_serial_path,
                                          testPin(old_serial_path, 17U),
                                          "authority_serial is below the compiled minimum",
                                          "old authority serial was accepted", calls);

    const auto wrong_authority_path = writeMutatedQualificationRecord(
        directory / "wrong-authority-qualification.json", canonical_record,
        [](Json& value) { value["authority_id"] = "different-reviewed-authority"; });
    expectProductionQualificationRejected(unqualified, wrong_authority_path,
                                          testPin(wrong_authority_path),
                                          "authority_id does not match the compiled authority pin",
                                          "wrong qualification authority was accepted", calls);

    const auto self_authorizing_path = writeMutatedQualificationRecord(
        directory / "self-authorizing-qualification.json", canonical_record,
        [](Json& value) { value["self_authorizing"] = true; });
    expectProductionQualificationRejected(
        unqualified, self_authorizing_path, testPin(self_authorizing_path),
        "must not be self-authorizing", "self-authorizing record was accepted", calls);

    const auto wrong_bundle_hash_path = writeMutatedQualificationRecord(
        directory / "wrong-bundle-hash-qualification.json", canonical_record,
        [](Json& value) { value["bundle"]["sha256"] = std::string(64U, '0'); });
    expectProductionQualificationRejected(unqualified, wrong_bundle_hash_path,
                                          testPin(wrong_bundle_hash_path),
                                          "sealed snapshot full SHA-256 mismatch",
                                          "wrong qualification bundle hash was accepted", calls);

    const auto wrong_bundle_size_path = writeMutatedQualificationRecord(
        directory / "wrong-bundle-size-qualification.json", canonical_record, [](Json& value) {
            value["bundle"]["size_bytes"] = value["bundle"]["size_bytes"].get<std::uint64_t>() + 1U;
        });
    expectProductionQualificationRejected(
        unqualified, wrong_bundle_size_path, testPin(wrong_bundle_size_path),
        "bundle size binding mismatch", "wrong qualification bundle size was accepted", calls);

    const auto wrong_config_hash_path = writeMutatedQualificationRecord(
        directory / "wrong-config-hash-qualification.json", canonical_record,
        [](Json& value) { value["bundle"]["embedded_config_sha256"] = std::string(64U, '0'); });
    expectProductionQualificationRejected(unqualified, wrong_config_hash_path,
                                          testPin(wrong_config_hash_path),
                                          "embedded config SHA-256 binding mismatch",
                                          "wrong qualification config hash was accepted", calls);

    const auto wrong_build_receipt_hash_path = writeMutatedQualificationRecord(
        directory / "wrong-build-receipt-hash-qualification.json", canonical_record,
        [](Json& value) { value["bundle"]["build_receipt_sha256"] = std::string(64U, '0'); });
    expectProductionQualificationRejected(
        unqualified, wrong_build_receipt_hash_path, testPin(wrong_build_receipt_hash_path),
        "build receipt SHA-256 binding mismatch",
        "wrong qualification build-receipt hash was accepted", calls);

    const auto wrong_plan_hash_path = writeMutatedQualificationRecord(
        directory / "wrong-plan-hash-qualification.json", canonical_record,
        [](Json& value) { value["bundle"]["plans"][0]["sha256"] = std::string(64U, '0'); });
    expectProductionQualificationRejected(
        unqualified, wrong_plan_hash_path, testPin(wrong_plan_hash_path),
        "six-plan binding mismatch", "wrong qualification plan hash was accepted", calls);

    const auto wrong_target_path = writeMutatedQualificationRecord(
        directory / "wrong-target-qualification.json", canonical_record,
        [](Json& value) { value["scope"]["compute_capability"] = "9.0"; });
    expectProductionQualificationRejected(
        unqualified, wrong_target_path, testPin(wrong_target_path),
        "compiled SAM2 TRT 11.1 L4 contract", "wrong qualification target was accepted", calls);

    const auto wrong_policy_path = writeMutatedQualificationRecord(
        directory / "wrong-policy-qualification.json", canonical_record,
        [](Json& value) { value["accuracy_evidence"]["replay_count"] = 2; });
    expectProductionQualificationRejected(unqualified, wrong_policy_path,
                                          testPin(wrong_policy_path),
                                          "semantic policy sam2_semantic_accuracy_v1",
                                          "wrong qualification policy was accepted", calls);

    const auto record_symlink = directory / "qualification-symlink.json";
    std::filesystem::create_symlink(valid_record.filename(), record_symlink);
    expectProductionQualificationRejected(unqualified, record_symlink, testPin(valid_record),
                                          "no-follow snapshot source",
                                          "qualification record symlink was followed", calls);

    for (const std::int32_t old_version : {3, 4}) {
        ArtifactOptions old_engine_contract;
        old_engine_contract.engine_contract_version = old_version;
        const auto old_engine_contract_path = writeArtifact(
            directory / ("old-engine-contract-v" + std::to_string(old_version) + ".bundle"),
            old_engine_contract);
        checkThrows<trtmc::sam2::NativeBundleLoadError>(
            [&] {
                (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                    old_engine_contract_path.string(), runtimeTarget(), makeFactory(calls));
            },
            "engine_contract_version", "old SAM2 engine contract was accepted");
        check(calls.empty(), "factory ran for an old SAM2 engine contract");
    }

    auto diagnostic_engines = trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
        unqualified.string(), runtimeTarget(), makeFactory(calls));
    check(diagnostic_engines.image != nullptr && diagnostic_engines.prompt != nullptr,
          "diagnostic load did not return image and prompt engines");
    check(std::all_of(diagnostic_engines.recurrent.begin(), diagnostic_engines.recurrent.end(),
                      [](const auto& module) { return module != nullptr; }),
          "diagnostic load did not return all recurrent engines");
    check(calls.size() == trtmc::sam2::kRequiredPlanSections.size(),
          "diagnostic load did not create exactly six modules");
    for (std::size_t index = 0; index < calls.size(); ++index)
        check(calls[index] == trtmc::sam2::kRequiredPlanSections[index],
              "module creation order drifted");

    calls.clear();
    const std::string unqualified_full_sha256 = sha256File(unqualified);
    auto digest_bound_engines =
        trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundleWithExpectedSha256(
            unqualified.string(), unqualified_full_sha256, runtimeTarget(), makeFactory(calls));
    check(digest_bound_engines.image != nullptr && digest_bound_engines.prompt != nullptr &&
              calls.size() == trtmc::sam2::kRequiredPlanSections.size(),
          "digest-bound loader rejected its exact sealed snapshot");

    calls.clear();
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundleWithExpectedSha256(
                unqualified.string(), std::string(64U, '0'), runtimeTarget(), makeFactory(calls));
        },
        "sealed snapshot full SHA-256 mismatch",
        "digest-bound loader accepted a mismatched full-bundle digest");
    check(calls.empty(), "factory ran before the sealed snapshot digest was verified");

    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundleWithExpectedSha256(
                unqualified.string(), "not-a-sha256", runtimeTarget(), makeFactory(calls));
        },
        "expected full SHA-256 is invalid", "malformed expected full-bundle digest was accepted");
    check(calls.empty(), "factory ran for a malformed expected full-bundle digest");

    calls.clear();
    ArtifactOptions qualified_options;
    qualified_options.qualified = true;
    const auto qualified = writeArtifact(directory / "qualified.bundle", qualified_options);
    const auto qualified_record =
        writeQualificationRecord(directory / "self-qualified-bundle-record.json",
                                 makeQualificationRecord(qualified, qualified_options));
    expectProductionQualificationRejected(qualified, qualified_record, testPin(qualified_record),
                                          "requires exact unqualified golden config facts",
                                          "externally pinned self-qualified bundle was accepted",
                                          calls);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                qualified.string(), runtimeTarget(), makeFactory(calls));
        },
        "requires exact unqualified golden config facts",
        "self-promoted qualified diagnostic bundle was accepted");
    check(calls.empty(), "factory ran for a self-promoted bundle");

    ArtifactOptions wrong_trt_version;
    wrong_trt_version.trt_version.append(".mutated");
    expectArtifactTargetMutationRejected(directory, "wrong-trt-version.bundle", wrong_trt_version,
                                         "trt_version", calls);

    ArtifactOptions wrong_trt_abi;
    wrong_trt_abi.trt_abi.append(".mutated");
    expectArtifactTargetMutationRejected(directory, "wrong-trt-abi.bundle", wrong_trt_abi,
                                         "trt_abi", calls);

    ArtifactOptions wrong_gpu_name;
    wrong_gpu_name.gpu_name.append(" mutated");
    expectArtifactTargetMutationRejected(directory, "wrong-gpu-name.bundle", wrong_gpu_name,
                                         "gpu_name", calls);

    ArtifactOptions wrong_compute_capability;
    wrong_compute_capability.compute_capability.append(".mutated");
    expectArtifactTargetMutationRejected(directory, "wrong-compute-capability.bundle",
                                         wrong_compute_capability, "compute_capability", calls);

    calls.clear();
    ArtifactOptions wrong_header_hash;
    wrong_header_hash.wrong_header_plan_hash = true;
    const auto header_hash_path =
        writeArtifact(directory / "wrong-header-hash.bundle", wrong_header_hash);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                header_hash_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "section SHA-256 mismatch", "header plan hash mismatch was accepted");
    check(calls.empty(), "factory ran before header hash validation");

    ArtifactOptions wrong_receipt_plan;
    wrong_receipt_plan.wrong_receipt_plan_hash = true;
    const auto receipt_plan_path =
        writeArtifact(directory / "wrong-receipt-plan.bundle", wrong_receipt_plan);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                receipt_plan_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "serialized_sha256", "receipt plan hash mismatch was accepted");
    check(calls.empty(), "factory ran before receipt plan hash validation");

    ArtifactOptions wrong_receipt_config;
    wrong_receipt_config.wrong_receipt_config_hash = true;
    const auto receipt_config_path =
        writeArtifact(directory / "wrong-receipt-config.bundle", wrong_receipt_config);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                receipt_config_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "embedded_config_sha256", "receipt config hash mismatch was accepted");
    check(calls.empty(), "factory ran before receipt config hash validation");

    ArtifactOptions wrong_plan_profiling_verbosity;
    wrong_plan_profiling_verbosity.wrong_plan_profiling_verbosity = true;
    const auto wrong_plan_profiling_verbosity_path = writeArtifact(
        directory / "wrong-plan-profiling-verbosity.bundle", wrong_plan_profiling_verbosity);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                wrong_plan_profiling_verbosity_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "plan_profiling_verbosity", "non-detailed plan profiling verbosity was accepted");
    check(calls.empty(), "factory ran for non-detailed plan profiling verbosity");

    ArtifactOptions qualification_disagreement;
    qualification_disagreement.receipt_qualification_override = true;
    qualification_disagreement.receipt_qualified = true;
    const auto disagreement_path =
        writeArtifact(directory / "qualification-disagreement.bundle", qualification_disagreement);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                disagreement_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "qualification facts disagree", "config and receipt qualification mismatch was accepted");

    ArtifactOptions missing_receipt;
    missing_receipt.omit_receipt = true;
    const auto missing_path = writeArtifact(directory / "missing.bundle", missing_receipt);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                missing_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "exactly six plans", "bundle missing receipt was accepted");

    ArtifactOptions trailing;
    trailing.append_trailing_byte = true;
    const auto trailing_path = writeArtifact(directory / "trailing.bundle", trailing);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                trailing_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "trailing", "unbound trailing payload was accepted");

    ArtifactOptions wrong_model;
    wrong_model.header_model_id = "different-model";
    const auto wrong_model_path = writeArtifact(directory / "wrong-model.bundle", wrong_model);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                wrong_model_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "model_id", "wrong model identity was accepted");

    ArtifactOptions missing_attention;
    missing_attention.omit_image_attention = true;
    const auto missing_attention_path =
        writeArtifact(directory / "missing-image-attention.bundle", missing_attention);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                missing_attention_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "field set drifted", "receipt without native image-attention identity was accepted");

    ArtifactOptions wrong_attention_implementation;
    wrong_attention_implementation.wrong_attention_implementation = true;
    const auto wrong_attention_implementation_path = writeArtifact(
        directory / "wrong-attention-implementation.bundle", wrong_attention_implementation);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                wrong_attention_implementation_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "implementation", "non-native image attention was accepted");

    ArtifactOptions wrong_attention_blocks;
    wrong_attention_blocks.wrong_attention_block_count = true;
    const auto wrong_attention_blocks_path =
        writeArtifact(directory / "wrong-attention-blocks.bundle", wrong_attention_blocks);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                wrong_attention_blocks_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "block_count", "wrong native attention block count was accepted");

    ArtifactOptions wrong_attention_head;
    wrong_attention_head.wrong_attention_head_dimension = true;
    const auto wrong_attention_head_path =
        writeArtifact(directory / "wrong-attention-head.bundle", wrong_attention_head);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                wrong_attention_head_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "head_dimension", "wrong native attention head dimension was accepted");

    const auto expectAttentionFactRejected =
        [&](const char* filename, bool ArtifactOptions::* mutation, const char* field) {
            ArtifactOptions options;
            options.*mutation = true;
            const auto path = writeArtifact(directory / filename, options);
            checkThrows<trtmc::sam2::NativeBundleLoadError>(
                [&] {
                    (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                        path.string(), runtimeTarget(), makeFactory(calls));
                },
                field, "wrong TensorRT IAttentionV2 fact was accepted");
        };
    expectAttentionFactRejected("wrong-attention-operator.bundle",
                                &ArtifactOptions::wrong_attention_operator, "operator");
    expectAttentionFactRejected("wrong-attention-api.bundle", &ArtifactOptions::wrong_attention_api,
                                "api");
    expectAttentionFactRejected("wrong-attention-query-form.bundle",
                                &ArtifactOptions::wrong_attention_query_form, "query_form");
    expectAttentionFactRejected("wrong-attention-key-value-form.bundle",
                                &ArtifactOptions::wrong_attention_key_value_form, "key_value_form");
    expectAttentionFactRejected("wrong-attention-output-form.bundle",
                                &ArtifactOptions::wrong_attention_output_form, "output_form");
    expectAttentionFactRejected("wrong-attention-normalization.bundle",
                                &ArtifactOptions::wrong_attention_normalization, "normalization");
    expectAttentionFactRejected("wrong-attention-causal-mask.bundle",
                                &ArtifactOptions::wrong_attention_causal_mask, "causal_mask");
    expectAttentionFactRejected("wrong-attention-decomposable.bundle",
                                &ArtifactOptions::wrong_attention_decomposable, "decomposable");
    expectAttentionFactRejected("wrong-attention-fused-intent.bundle",
                                &ArtifactOptions::wrong_attention_fused_kernel_intent,
                                "fused-kernel intent");
    expectAttentionFactRejected("wrong-attention-metadata-prefix.bundle",
                                &ArtifactOptions::wrong_attention_metadata_prefix,
                                "metadata_prefix");
    expectAttentionFactRejected("wrong-attention-metadata-index-width.bundle",
                                &ArtifactOptions::wrong_attention_metadata_index_width,
                                "metadata_index_width");
    expectAttentionFactRejected("wrong-attention-q-scale.bundle",
                                &ArtifactOptions::wrong_attention_q_scale_formula,
                                "q_scale_formula");
    expectAttentionFactRejected("wrong-attention-k-scale.bundle",
                                &ArtifactOptions::wrong_attention_k_scale_formula,
                                "k_scale_formula");
    expectAttentionFactRejected("wrong-attention-effective-scale.bundle",
                                &ArtifactOptions::wrong_attention_effective_score_scale,
                                "effective_score_scale");
    expectAttentionFactRejected("wrong-attention-dtype.bundle",
                                &ArtifactOptions::wrong_attention_scale_dtype, "scale_dtype");

    ArtifactOptions extra_attention_field;
    extra_attention_field.extra_attention_field = true;
    const auto extra_attention_field_path =
        writeArtifact(directory / "extra-attention-field.bundle", extra_attention_field);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                extra_attention_field_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "field set drifted", "extra native-attention receipt option was accepted");

    ArtifactOptions wrong_graph_facts;
    wrong_graph_facts.wrong_graph_facts = true;
    const auto wrong_graph_path =
        writeArtifact(directory / "wrong-graph-facts.bundle", wrong_graph_facts);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                wrong_graph_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "exact graph construction facts drifted", "wrong exact graph counts were accepted");

    ArtifactOptions wrong_image_layer_type_facts;
    wrong_image_layer_type_facts.wrong_image_layer_type_facts = true;
    const auto wrong_image_layer_type_path = writeArtifact(
        directory / "wrong-image-layer-type-facts.bundle", wrong_image_layer_type_facts);
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                wrong_image_layer_type_path.string(), runtimeTarget(), makeFactory(calls));
        },
        "exact image layer-type facts drifted", "wrong native image layer types were accepted");

    calls.clear();
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                unqualified.string(), runtimeTarget(), makeFactory(calls, true));
        },
        "tensor count drifted", "engine I/O ABI drift was accepted");
    check(calls.size() == 1U, "loader continued after image ABI drift");

    calls.clear();
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
                unqualified.string(), runtimeTarget(), makeFactory(calls, false, true));
        },
        "module creation failed", "null module was accepted");
    check(calls.size() == 1U, "loader continued after null image module");

    const auto substituted_path = writeArtifact(directory / "post-build-substitution.bundle");
    const std::string expected_pre_substitution_sha256 = sha256File(substituted_path);
    ArtifactOptions valid_substitute_options;
    valid_substitute_options.header_model_id = "substituted-model";
    const auto valid_substitute_path =
        writeArtifact(directory / "post-build-substitute.bundle", valid_substitute_options);
    std::filesystem::rename(valid_substitute_path, substituted_path);
    check(sha256File(substituted_path) != expected_pre_substitution_sha256,
          "post-build substitution fixture did not change the full bundle digest");
    calls.clear();
    checkThrows<trtmc::sam2::NativeBundleLoadError>(
        [&] {
            (void)trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundleWithExpectedSha256(
                substituted_path.string(), expected_pre_substitution_sha256, runtimeTarget(),
                makeFactory(calls));
        },
        "sealed snapshot full SHA-256 mismatch",
        "post-build pathname substitution bypassed sealed-snapshot digest binding");
    check(calls.empty(), "factory ran for a post-build substituted bundle");

    const auto expected_plans = planPayloads();
    const auto swap_path = writeArtifact(directory / "swap-source.bundle");
    ArtifactOptions replacement_options;
    replacement_options.header_model_id = "swapped-model";
    const auto replacement_path =
        writeArtifact(directory / "swap-replacement.bundle", replacement_options);
    bool swapped = false;
    calls.clear();
    FactoryObserver swap_observer = [&](std::size_t ordinal, std::string_view, const void* data,
                                        std::size_t size) {
        check(ordinal < expected_plans.size(), "swap observer ordinal overflowed");
        check(size == expected_plans[ordinal].size() &&
                  std::memcmp(data, expected_plans[ordinal].data(), size) == 0,
              "path swap changed a snapshotted plan");
        if (ordinal == 0U) {
            std::filesystem::rename(replacement_path, swap_path);
            swapped = true;
        }
    };
    auto swap_engines = trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
        swap_path.string(), runtimeTarget(), makeFactory(calls, false, false, swap_observer));
    check(swapped && swap_engines.image != nullptr && calls.size() == 6U,
          "sealed snapshot did not survive a source path swap");

    const auto mutation_path = writeArtifact(directory / "mutation-source.bundle");
    bool mutated = false;
    calls.clear();
    FactoryObserver mutation_observer = [&](std::size_t ordinal, std::string_view, const void* data,
                                            std::size_t size) {
        check(ordinal < expected_plans.size(), "mutation observer ordinal overflowed");
        check(size == expected_plans[ordinal].size() &&
                  std::memcmp(data, expected_plans[ordinal].data(), size) == 0,
              "source mutation changed a snapshotted plan");
        if (ordinal == 0U) {
            std::fstream output(mutation_path, std::ios::binary | std::ios::in | std::ios::out);
            output.seekp(-1, std::ios::end);
            output.put('X');
            output.flush();
            check(static_cast<bool>(output), "failed to mutate original bundle after snapshot");
            mutated = true;
        }
    };
    auto mutation_engines = trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundle(
        mutation_path.string(), runtimeTarget(),
        makeFactory(calls, false, false, mutation_observer));
    check(mutated && mutation_engines.image != nullptr && calls.size() == 6U,
          "sealed snapshot did not survive source mutation");

    const auto record_swap_path = writeQualificationRecord(directory / "record-swap-source.json",
                                                           makeQualificationRecord(unqualified));
    const auto record_swap_pin = testPin(record_swap_path);
    const auto record_replacement_path = directory / "record-swap-replacement.json";
    writeTextFile(record_replacement_path, "{}\n");
    bool record_swapped = false;
    calls.clear();
    FactoryObserver record_swap_observer = [&](std::size_t ordinal, std::string_view, const void*,
                                               std::size_t) {
        if (ordinal == 0U) {
            std::filesystem::rename(record_replacement_path, record_swap_path);
            record_swapped = true;
        }
    };
    auto record_swap_engines =
        trtmc::sam2::loadProductionQualifiedNativeVideoEngineSetFromBundleForTest(
            unqualified.string(), record_swap_path.string(), runtimeTarget(),
            makeFactory(calls, false, false, record_swap_observer), record_swap_pin);
    check(record_swapped && record_swap_engines.image != nullptr && calls.size() == 6U,
          "qualification record path swap escaped the immutable pre-deserialization snapshot");

    std::filesystem::remove_all(directory);
    std::cout << "SAM2 native bundle loader tests passed\n";
    return 0;
}
