/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_engine_builder.h"

#include "bundle_writer.h"
#include "checkpoint_reader.h"
#include "runtime/models/sam2/sam2_engine_contract.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <utility>

namespace trtmc::sam2::native {

namespace {

struct ExactGraphFacts {
    std::int32_t layer_count;
    std::size_t referenced_tensor_count;
};

constexpr std::array<ExactGraphFacts, 6> kExactGraphFacts = {{
    {1139, 282U},
    {882, 185U},
    {1630, 291U},
    {1652, 291U},
    {1674, 291U},
    {1696, 291U},
}};
constexpr std::string_view kEmbeddedConfig =
    R"({"schema_version":1,"family":"sam2","model_id":"sam2.1-hiera-small-bbox","engine_contract_version":5,"runtime_strategy":"sam2_bbox_video_tracking","precision":"mixed_bf16_fp32","checkpoint_sha256":"89fd676560809c8504411b574cea305c86db1f65bda790ec7fe16cedc6c6ff73","source_config_sha256":"59488bb78c7cc48aaaebd966ea9d054014f683459d062b7a959a4aa501342656","golden_manifest_sha256":"c25251ee27da05afd75adc3c6869cbc2944b80c05c5d6e703b6ebbbba697a4f0","frame_count":5,"selected_object_count":1,"model_image_size":1024,"original_image_height":1280,"original_image_width":1088,"plan_sections":["sam2_image_engine_plan","sam2_prompt_engine_plan","sam2_recurrent_h1_engine_plan","sam2_recurrent_h2_engine_plan","sam2_recurrent_h3_engine_plan","sam2_recurrent_h4_engine_plan"],"qualification":"unqualified","runtime_eligible":false})";

std::string escapeJson(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    constexpr char kHex[] = "0123456789abcdef";
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (byte < 0x20U) {
                result += "\\u00";
                result.push_back(kHex[byte >> 4U]);
                result.push_back(kHex[byte & 0x0fU]);
            } else {
                result.push_back(static_cast<char>(byte));
            }
        }
    }
    return result;
}

bool isDigitAt(std::string_view value, std::size_t index) {
    return index < value.size() && std::isdigit(static_cast<unsigned char>(value[index])) != 0;
}

std::int32_t twoDigits(std::string_view value, std::size_t index) {
    return static_cast<std::int32_t>((value[index] - '0') * 10 + (value[index + 1U] - '0'));
}

bool isCanonicalUtcTimestamp(std::string_view value) {
    if (value.size() != 20U || value[4] != '-' || value[7] != '-' || value[10] != 'T' ||
        value[13] != ':' || value[16] != ':' || value[19] != 'Z')
        return false;
    constexpr std::array<std::size_t, 14> digit_positions = {0, 1,  2,  3,  5,  6,  8,
                                                             9, 11, 12, 14, 15, 17, 18};
    if (!std::all_of(digit_positions.begin(), digit_positions.end(),
                     [value](std::size_t index) { return isDigitAt(value, index); }))
        return false;
    const auto month = twoDigits(value, 5);
    const auto day = twoDigits(value, 8);
    const auto hour = twoDigits(value, 11);
    const auto minute = twoDigits(value, 14);
    const auto second = twoDigits(value, 17);
    return month >= 1 && month <= 12 && day >= 1 && day <= 31 && hour <= 23 && minute <= 59 &&
           second <= 60;
}

const char* graphKindName(Sam2GraphKind kind) {
    switch (kind) {
    case Sam2GraphKind::kImage:
        return "image";
    case Sam2GraphKind::kPrompt:
        return "prompt";
    case Sam2GraphKind::kRecurrent:
        return "recurrent";
    }
    throw Sam2EngineBuildError("SAM2 graph has an unsupported kind");
}

void requireNonempty(std::string_view value, const char* field) {
    if (value.empty())
        throw Sam2EngineBuildError(std::string("SAM2 build fact is empty: ") + field);
}

std::string sha256Hex(const void* data, std::size_t size) {
    ::trtmc::internal::Sha256 hash;
    hash.update(data, size);
    return hash.hex_digest();
}

void validateGraph(const Sam2SerializedPlan& plan, std::size_t index) {
    const auto& graph = plan.graph;
    const auto expected_section = trtmc::sam2::kRequiredPlanSections[index];
    if (graph.section != expected_section)
        throw Sam2EngineBuildError("SAM2 plans are missing or not in canonical section order");
    if (plan.bytes.empty())
        throw Sam2EngineBuildError("SAM2 serialized plan is empty: " + graph.section);
    if (!graph.graph_complete)
        throw Sam2EngineBuildError("SAM2 graph is not complete: " + graph.section);
    const auto& exact = kExactGraphFacts[index];
    if (graph.layer_count != exact.layer_count ||
        graph.referenced_tensor_count != exact.referenced_tensor_count) {
        throw Sam2EngineBuildError("SAM2 graph construction facts drifted: " + graph.section);
    }

    if (index == 0U) {
        if (graph.kind != Sam2GraphKind::kImage || graph.history_frames != 0 ||
            graph.input_count != 1 || graph.output_count != 9 ||
            graph.convolution_layer_count != 23 || graph.activation_layer_count != 28 ||
            graph.pooling_layer_count != 6 || graph.element_wise_layer_count != 130 ||
            graph.shuffle_layer_count != 313 || graph.constant_layer_count != 216 ||
            graph.slice_layer_count != 67 || graph.resize_layer_count != 2 ||
            graph.normalization_layer_count != 32 || graph.cast_layer_count != 223 ||
            graph.matrix_multiply_layer_count != 67 || graph.softmax_layer_count != 0 ||
            graph.plugin_v3_layer_count != 0 || graph.attention_input_layer_count != 16 ||
            graph.attention_output_layer_count != 16)
            throw Sam2EngineBuildError("SAM2 image graph contract drifted");
        return;
    }
    if (graph.convolution_layer_count != -1 || graph.activation_layer_count != -1 ||
        graph.pooling_layer_count != -1 || graph.element_wise_layer_count != -1 ||
        graph.shuffle_layer_count != -1 || graph.constant_layer_count != -1 ||
        graph.slice_layer_count != -1 || graph.resize_layer_count != -1 ||
        graph.normalization_layer_count != -1 || graph.cast_layer_count != -1 ||
        graph.matrix_multiply_layer_count != -1 || graph.softmax_layer_count != -1 ||
        graph.plugin_v3_layer_count != -1 || graph.attention_input_layer_count != -1 ||
        graph.attention_output_layer_count != -1) {
        throw Sam2EngineBuildError("SAM2 tracker graph contains image-only layer facts");
    }
    if (index == 1U) {
        if (graph.kind != Sam2GraphKind::kPrompt || graph.history_frames != 0 ||
            graph.input_count != 4 || graph.output_count != 3)
            throw Sam2EngineBuildError("SAM2 prompt graph contract drifted");
        return;
    }
    const auto expected_history = static_cast<std::int32_t>(index - 1U);
    if (graph.kind != Sam2GraphKind::kRecurrent || graph.history_frames != expected_history ||
        graph.input_count != 5 || graph.output_count != 3)
        throw Sam2EngineBuildError("SAM2 recurrent graph contract drifted");
}

} // namespace

std::string_view sam2EmbeddedConfigJson() noexcept {
    return kEmbeddedConfig;
}

void verifySam2SourceConfig(const std::filesystem::path& path) {
    if (path.empty())
        throw Sam2EngineBuildError("SAM2 source config path must not be empty");
    try {
        const std::string actual =
            CheckpointReader::checkpointSha256(path, kMaximumSam2ConfigBytes);
        if (actual != trtmc::sam2::kConfigSha256) {
            throw Sam2EngineBuildError("SAM2 source config SHA-256 mismatch: expected " +
                                       std::string(trtmc::sam2::kConfigSha256) + ", got " + actual);
        }
    } catch (const CheckpointError& error) {
        throw Sam2EngineBuildError(std::string("failed to authenticate SAM2 source config: ") +
                                   error.what());
    }
}

void validateSam2EngineBuildOptions(const Sam2EngineBuildOptions& options) {
    if (options.checkpoint_path.empty())
        throw Sam2EngineBuildError("SAM2 checkpoint path must not be empty");
    if (options.source_config_path.empty())
        throw Sam2EngineBuildError("SAM2 source config path must not be empty");
    if (options.output_path.empty() || !options.output_path.has_filename())
        throw Sam2EngineBuildError("SAM2 output path must name a file");
    if (options.workspace_bytes == 0U ||
        options.workspace_bytes >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw Sam2EngineBuildError("SAM2 workspace size is outside the native size_t range");
    if (options.gpu_device < 0)
        throw Sam2EngineBuildError("SAM2 GPU device must be nonnegative");
    if (!isCanonicalUtcTimestamp(options.created_at_utc))
        throw Sam2EngineBuildError("SAM2 created-at must use canonical YYYY-MM-DDTHH:MM:SSZ UTC");
    if (std::filesystem::exists(options.output_path))
        throw Sam2EngineBuildError("SAM2 output path already exists");
}

void validateSam2RuntimeBuildFacts(const Sam2RuntimeBuildFacts& runtime) {
    requireNonempty(runtime.tensorrt_version, "tensorrt_version");
    requireNonempty(runtime.tensorrt_abi, "tensorrt_abi");
    requireNonempty(runtime.cuda_runtime_version, "cuda_runtime_version");
    requireNonempty(runtime.cuda_driver_version, "cuda_driver_version");
    requireNonempty(runtime.gpu_name, "gpu_name");
    if (runtime.gpu_device < 0 || runtime.gpu_compute_major <= 0 || runtime.gpu_compute_minor < 0 ||
        runtime.gpu_global_memory_bytes == 0U)
        throw Sam2EngineBuildError("SAM2 GPU build facts are invalid");
    if (runtime.tensorrt_version != trtmc::sam2::kTargetTensorRtVersion) {
        throw Sam2EngineBuildError("SAM2 compilation requires exact TensorRT 11.1.0.106");
    }
    if (runtime.tensorrt_abi != trtmc::sam2::kTargetTensorRtAbi) {
        throw Sam2EngineBuildError("SAM2 compilation requires exact TensorRT ABI 11.1");
    }
    if (runtime.gpu_name != trtmc::sam2::kTargetGpuName) {
        throw Sam2EngineBuildError("SAM2 compilation requires exact NVIDIA L4 GPU identity");
    }
    if (runtime.gpu_compute_major != trtmc::sam2::kTargetComputeCapabilityMajor ||
        runtime.gpu_compute_minor != trtmc::sam2::kTargetComputeCapabilityMinor) {
        throw Sam2EngineBuildError("SAM2 compilation requires exact GPU compute capability 8.9");
    }
    if (!runtime.strongly_typed)
        throw Sam2EngineBuildError("SAM2 plans require strongly typed TensorRT networks");
    if (runtime.tf32_enabled)
        throw Sam2EngineBuildError("SAM2 plans require TF32 to be explicitly disabled");
}

void validateSam2Compilation(const Sam2CompilationResult& compilation) {
    validateSam2RuntimeBuildFacts(compilation.runtime);
    if (compilation.plan_profiling_verbosity != trtmc::sam2::kPlanProfilingVerbosity) {
        throw Sam2EngineBuildError("SAM2 compilation requires detailed plan profiling verbosity");
    }
    if (compilation.plans.size() != trtmc::sam2::kRequiredPlanSections.size())
        throw Sam2EngineBuildError("SAM2 compilation requires exactly six serialized plans");
    for (std::size_t index = 0; index < compilation.plans.size(); ++index)
        validateGraph(compilation.plans[index], index);
}

std::string makeSam2BuildReceipt(const Sam2EngineBuildOptions& options,
                                 const Sam2CompilationResult& compilation) {
    validateSam2EngineBuildOptions(options);
    validateSam2Compilation(compilation);
    const auto& runtime = compilation.runtime;
    if (runtime.gpu_device != options.gpu_device)
        throw Sam2EngineBuildError("SAM2 runtime facts do not match the requested GPU device");
    std::ostringstream output;
    output.imbue(std::locale::classic());
    const std::string_view embedded_config = sam2EmbeddedConfigJson();
    output << "{\"schema_version\":" << trtmc::sam2::kBuildReceiptSchemaVersion
           << ",\"family\":\"sam2\",\"model_id\":\"" << kSam2ModelId
           << "\",\"qualification\":{\"state\":\"unqualified\""
              ",\"runtime_eligible\":false,\"golden_parity_verified\":false},\"assets\":{"
              "\"checkpoint_sha256\":\""
           << trtmc::sam2::kCheckpointSha256 << "\",\"source_config_sha256\":\""
           << trtmc::sam2::kConfigSha256 << "\",\"golden_manifest_sha256\":\""
           << trtmc::sam2::kGoldenManifestSha256 << "\",\"embedded_config_sha256\":\""
           << sha256Hex(embedded_config.data(), embedded_config.size())
           << "\"},\"build\":{\"created_at_utc\":\"" << escapeJson(options.created_at_utc)
           << "\",\"workspace_bytes\":" << options.workspace_bytes
           << ",\"network_mode\":\"strongly_typed\",\"tf32_enabled\":false"
              ",\"builder_optimization_level\":"
           << trtmc::sam2::kBuilderOptimizationLevel << ",\"plan_profiling_verbosity\":\""
           << escapeJson(compilation.plan_profiling_verbosity) << "\",\"tensorrt_version\":\""
           << escapeJson(runtime.tensorrt_version) << "\",\"tensorrt_abi\":\""
           << escapeJson(runtime.tensorrt_abi) << "\",\"cuda_runtime_version\":\""
           << escapeJson(runtime.cuda_runtime_version) << "\",\"cuda_driver_version\":\""
           << escapeJson(runtime.cuda_driver_version)
           << "\",\"gpu\":{\"device\":" << runtime.gpu_device << ",\"name\":\""
           << escapeJson(runtime.gpu_name) << "\",\"compute_capability\":\""
           << runtime.gpu_compute_major << '.' << runtime.gpu_compute_minor
           << "\",\"global_memory_bytes\":" << runtime.gpu_global_memory_bytes
           << "}},\"image_attention\":{\"implementation\":\"tensorrt_iattention_v2\""
              ",\"operator\":\"IAttention\",\"api\":\"addAttentionV2\""
              ",\"block_count\":16,\"head_dimension\":96"
              ",\"query_form\":\"padded_bhnd\""
              ",\"key_value_form\":\"padded_bhnd\""
              ",\"output_form\":\"padded_bhnd\""
              ",\"normalization\":\"softmax\",\"causal_mask\":\"none\""
              ",\"decomposable\":false,\"fused_kernel_intent\":true"
              ",\"metadata_prefix\":\""
           << trtmc::sam2::kImageAttentionMetadataPrefix
           << "\",\"metadata_index_width\":" << trtmc::sam2::kImageAttentionMetadataIndexWidth
           << ",\"q_scale_formula\":\"1/sqrt(head_dimension)\""
              ",\"k_scale_formula\":\"none\""
              ",\"effective_score_scale\":\"1/sqrt(head_dimension)\""
              ",\"scale_dtype\":\"bf16\"},\"graphs\":[";

    for (std::size_t index = 0; index < compilation.plans.size(); ++index) {
        if (index != 0U)
            output << ',';
        const auto& plan = compilation.plans[index];
        output << "{\"section\":\"" << escapeJson(plan.graph.section) << "\",\"kind\":\""
               << graphKindName(plan.graph.kind)
               << "\",\"history_frames\":" << plan.graph.history_frames
               << ",\"inputs\":" << plan.graph.input_count
               << ",\"outputs\":" << plan.graph.output_count
               << ",\"layers\":" << plan.graph.layer_count;
        if (plan.graph.kind == Sam2GraphKind::kImage) {
            output << ",\"convolution_layers\":" << plan.graph.convolution_layer_count
                   << ",\"activation_layers\":" << plan.graph.activation_layer_count
                   << ",\"pooling_layers\":" << plan.graph.pooling_layer_count
                   << ",\"element_wise_layers\":" << plan.graph.element_wise_layer_count
                   << ",\"shuffle_layers\":" << plan.graph.shuffle_layer_count
                   << ",\"constant_layers\":" << plan.graph.constant_layer_count
                   << ",\"slice_layers\":" << plan.graph.slice_layer_count
                   << ",\"resize_layers\":" << plan.graph.resize_layer_count
                   << ",\"normalization_layers\":" << plan.graph.normalization_layer_count
                   << ",\"cast_layers\":" << plan.graph.cast_layer_count
                   << ",\"matrix_multiply_layers\":" << plan.graph.matrix_multiply_layer_count
                   << ",\"softmax_layers\":" << plan.graph.softmax_layer_count
                   << ",\"plugin_v3_layers\":" << plan.graph.plugin_v3_layer_count
                   << ",\"attention_input_layers\":" << plan.graph.attention_input_layer_count
                   << ",\"attention_output_layers\":" << plan.graph.attention_output_layer_count;
        }
        output << ",\"referenced_checkpoint_tensors\":" << plan.graph.referenced_tensor_count
               << ",\"serialized_bytes\":" << plan.bytes.size() << ",\"serialized_sha256\":\""
               << sha256Hex(plan.bytes.data(), plan.bytes.size()) << "\",\"graph_complete\":true}";
    }
    output << "]}";
    return output.str();
}

namespace detail {

Sam2NativeBundleBuildResult
writeCompiledSam2NativeBundle(const Sam2EngineBuildOptions& options,
                              const Sam2CompilationResult& compilation) {
    validateSam2EngineBuildOptions(options);
    validateSam2Compilation(compilation);
    const std::string receipt = makeSam2BuildReceipt(options, compilation);
    const std::string_view config = sam2EmbeddedConfigJson();

    std::vector<BundleSectionView> sections;
    sections.reserve(trtmc::sam2::kRequiredPlanSections.size() + 2U);
    for (const auto& plan : compilation.plans)
        sections.push_back({plan.graph.section, plan.bytes.data(), plan.bytes.size()});
    sections.push_back({trtmc::sam2::kConfigSection, config.data(), config.size()});
    sections.push_back({trtmc::sam2::kBuildReceiptSection, receipt.data(), receipt.size()});

    const BundleMetadata metadata{std::string(kSam2ModelId), compilation.runtime.tensorrt_version,
                                  compilation.runtime.tensorrt_abi, compilation.runtime.gpu_name,
                                  options.created_at_utc};
    Sam2NativeBundleBuildResult result;
    result.build_receipt_json = receipt;
    result.build_receipt_sha256 = sha256Hex(receipt.data(), receipt.size());
    for (std::size_t index = 0; index < compilation.plans.size(); ++index) {
        const auto& bytes = compilation.plans[index].bytes;
        result.plan_sha256[index] = sha256Hex(bytes.data(), bytes.size());
    }
    result.bundle = writeSam2NativeBundle(options.output_path, metadata, sections);
    return result;
}

} // namespace detail

} // namespace trtmc::sam2::native
