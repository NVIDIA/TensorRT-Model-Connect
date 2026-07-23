/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/tool/qualification_diagnostics.h"

#include "utils/sha256.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace trtmc::openpi::tool {
namespace {

using Json = nlohmann::json;

std::string_view dtype_name(DiagnosticTensorType dtype) {
    switch (dtype) {
    case DiagnosticTensorType::kBool:
        return "bool";
    case DiagnosticTensorType::kInt32:
        return "int32";
    case DiagnosticTensorType::kBFloat16:
        return "bfloat16";
    case DiagnosticTensorType::kFloat32:
        return "float32";
    }
    throw std::invalid_argument("qualification diagnostic has an unknown dtype");
}

std::size_t dtype_width(DiagnosticTensorType dtype) {
    switch (dtype) {
    case DiagnosticTensorType::kBool:
        return 1U;
    case DiagnosticTensorType::kBFloat16:
        return 2U;
    case DiagnosticTensorType::kInt32:
    case DiagnosticTensorType::kFloat32:
        return 4U;
    }
    throw std::invalid_argument("qualification diagnostic has an unknown dtype");
}

std::string_view stage_name(DiagnosticStage stage) {
    switch (stage) {
    case DiagnosticStage::kPreprocess:
        return "preprocess";
    case DiagnosticStage::kVision:
        return "vision";
    case DiagnosticStage::kPrefix:
        return "prefix";
    case DiagnosticStage::kFlow:
        return "flow";
    case DiagnosticStage::kPostprocess:
        return "postprocess";
    }
    throw std::invalid_argument("qualification diagnostic has an unknown stage");
}

std::string_view role_name(DiagnosticRole role) {
    switch (role) {
    case DiagnosticRole::kInput:
        return "input";
    case DiagnosticRole::kIntermediate:
        return "intermediate";
    case DiagnosticRole::kOutput:
        return "output";
    }
    throw std::invalid_argument("qualification diagnostic has an unknown role");
}

std::size_t expected_bytes(const DiagnosticTensor& tensor) {
    if (tensor.shape.empty())
        throw std::invalid_argument("qualification diagnostic tensor shape must not be empty");
    std::size_t elements = 1U;
    for (int64_t dimension : tensor.shape) {
        if (dimension <= 0 || elements > std::numeric_limits<std::size_t>::max() /
                                             static_cast<std::size_t>(dimension)) {
            throw std::invalid_argument("qualification diagnostic tensor shape is invalid");
        }
        elements *= static_cast<std::size_t>(dimension);
    }
    const std::size_t width = dtype_width(tensor.dtype);
    if (elements > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error("qualification diagnostic tensor size overflow");
    return elements * width;
}

bool safe_tensor_name(std::string_view name) {
    const auto accepted = [](const char character) {
        const bool lowercase = character >= 'a' && character <= 'z';
        const bool uppercase = character >= 'A' && character <= 'Z';
        const bool digit = character >= '0' && character <= '9';
        return lowercase || uppercase || digit || character == '_' || character == '.' ||
               character == '-';
    };
    return !name.empty() && std::all_of(name.begin(), name.end(), accepted);
}

std::string sha256_bytes(const std::vector<uint8_t>& bytes) {
    internal::Sha256 digest;
    digest.update(bytes.data(), bytes.size());
    return digest.hex_digest();
}

void write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to create qualification tensor: " + path.string());
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output)
        throw std::runtime_error("failed to write qualification tensor: " + path.string());
}

void validate_diagnostic_output(const ActionDiagnosticResult& diagnostics,
                                const std::filesystem::path& output_directory) {
    constexpr uint16_t endian_marker = 1U;
    if (*reinterpret_cast<const uint8_t*>(&endian_marker) != 1U)
        throw std::runtime_error("qualification diagnostics require a little-endian host");
    if (diagnostics.tensors.empty())
        throw std::invalid_argument("qualification diagnostics contain no tensors");
    if (output_directory.empty())
        throw std::invalid_argument("qualification diagnostics directory must not be empty");
    if (std::filesystem::exists(output_directory)) {
        throw std::invalid_argument("qualification diagnostics directory already exists: " +
                                    output_directory.string());
    }
}

std::filesystem::path make_staging_directory(const std::filesystem::path& output_directory) {
    const auto parent = output_directory.parent_path().empty() ? std::filesystem::current_path()
                                                               : output_directory.parent_path();
    std::filesystem::create_directories(parent);
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto staging =
        parent / ("." + output_directory.filename().string() + ".tmp-" + std::to_string(nonce));
    if (std::filesystem::exists(staging))
        throw std::runtime_error("qualification diagnostics staging directory already exists");
    return staging;
}

void validate_tensor_name(const DiagnosticTensor& tensor, std::unordered_set<std::string>& names) {
    if (!safe_tensor_name(tensor.name) || !names.insert(tensor.name).second) {
        throw std::invalid_argument("qualification diagnostic tensor name is unsafe or "
                                    "duplicated: " +
                                    tensor.name);
    }
}

Json tensor_descriptor(const DiagnosticTensor& tensor,
                       const std::filesystem::path& tensor_directory,
                       std::unordered_set<std::string>& names) {
    validate_tensor_name(tensor, names);
    const std::size_t byte_count = expected_bytes(tensor);
    if (tensor.bytes.size() != byte_count) {
        throw std::invalid_argument("qualification diagnostic tensor byte count mismatch: " +
                                    tensor.name);
    }
    const auto payload = tensor_directory / (tensor.name + ".bin");
    write_bytes(payload, tensor.bytes);
    return {
        {"path", "tensors/" + tensor.name + ".bin"},
        {"stage", stage_name(tensor.stage)},
        {"role", role_name(tensor.role)},
        {"dtype", dtype_name(tensor.dtype)},
        {"shape", tensor.shape},
        {"byte_length", byte_count},
        {"sha256", sha256_bytes(tensor.bytes)},
    };
}

Json write_tensor_payloads(const ActionDiagnosticResult& diagnostics,
                           const std::filesystem::path& staging) {
    const auto tensor_directory = staging / "tensors";
    std::filesystem::create_directories(tensor_directory);
    Json descriptors = Json::object();
    std::unordered_set<std::string> names;
    for (const auto& tensor : diagnostics.tensors)
        descriptors[tensor.name] = tensor_descriptor(tensor, tensor_directory, names);
    return descriptors;
}

void write_manifest(const std::filesystem::path& staging, std::string_view model_id,
                    Json tensor_descriptors) {
    const Json manifest = {
        {"schema_version", 1},
        {"artifact_type", "trtmc_action_qualification_diagnostics"},
        {"runtime_contract", "native_cpp_tensorrt"},
        {"model_id", std::string(model_id)},
        {"tensors", std::move(tensor_descriptors)},
    };
    const auto manifest_path = staging / "manifest.json";
    std::ofstream output(manifest_path, std::ios::out | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to create qualification diagnostics manifest");
    output << manifest.dump(2) << '\n';
    if (!output)
        throw std::runtime_error("failed to write qualification diagnostics manifest");
}

} // namespace

std::filesystem::path write_qualification_diagnostics(const ActionDiagnosticResult& diagnostics,
                                                      const std::filesystem::path& output_directory,
                                                      std::string_view model_id) {
    validate_diagnostic_output(diagnostics, output_directory);
    const auto staging = make_staging_directory(output_directory);

    try {
        write_manifest(staging, model_id, write_tensor_payloads(diagnostics, staging));
        std::filesystem::rename(staging, output_directory);
        return output_directory / "manifest.json";
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove_all(staging, cleanup_error);
        throw;
    }
}

} // namespace trtmc::openpi::tool
