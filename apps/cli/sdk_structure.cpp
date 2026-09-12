/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cli/sdk_dispatch.h"
#include "trtmc/structure.hpp"

#include <nlohmann/json.hpp>

namespace trtmc::cli {
namespace {
using detail::has_option;
using detail::require_option;

std::string input_encoding(const Command& command, const std::filesystem::path& input) {
    if (has_option(command, "--input-encoding"))
        return require_option(command, "--input-encoding");
    const auto extension = input.extension().string();
    if (extension == ".yaml" || extension == ".yml")
        return "yaml";
    if (extension == ".json")
        return "json";
    if (extension == ".b2rq")
        return "b2rq";
    throw std::invalid_argument(
        "unknown structure request extension; specify --input-encoding explicitly");
}

Config structure_config(const Command& command, const std::vector<ConfigField>& fields) {
    // Preserve the existing CLI flag's sampling-step meaning without supplying
    // a default. As with other Task commands, the family must declare the key.
    Command normalized = command;
    const auto steps = normalized.options.find("--num-steps");
    if (steps != normalized.options.end()) {
        normalized.config_entries.emplace_back("sampling_steps", steps->second);
        normalized.options.erase(steps);
    }
    return detail::task_config(normalized, fields,
                               {"--input", "--input-encoding", "--output", "--output-json"});
}

nlohmann::json confidence_json(const std::optional<MolecularStructureConfidenceView>& value) {
    if (!value)
        return nullptr;
    auto plddt = nlohmann::json::array();
    for (std::uint64_t i = 0; i < value->plddt.size; ++i)
        plddt.push_back(value->plddt.data[i]);
    return {{"confidence_score", value->confidence_score},
            {"ptm", value->ptm},
            {"iptm", value->iptm},
            {"ligand_iptm", value->ligand_iptm},
            {"protein_iptm", value->protein_iptm},
            {"complex_plddt", value->complex_plddt},
            {"complex_iplddt", value->complex_iplddt},
            {"plddt", std::move(plddt)}};
}

void write_document(const std::filesystem::path& path, std::string_view contents) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    detail::write_binary(path, contents);
}
} // namespace

std::string_view structure_task_for_command(const Command& command) {
    return command.kind == CommandKind::kPredictStructure ? MolecularDocumentToStructure::kTask
                                                          : std::string_view{};
}

bool dispatch_sdk_structure(const Command& command, const Model& model, std::string_view id,
                            std::ostream& output) {
    if (command.kind != CommandKind::kPredictStructure || id != MolecularDocumentToStructure::kTask)
        return false;
    const auto input_path = require_option(command, "--input");
    const auto encoding = input_encoding(command, input_path);
    const auto document = detail::read_structure_document(input_path);
    const auto task = model.task<MolecularDocumentToStructure>();
    const auto config = structure_config(command, task.config_fields());
    const auto result =
        task.run({{reinterpret_cast<const std::uint8_t*>(document.data()), document.size()},
                  encoding,
                  input_path},
                 config);
    const bool pdb = result.format() == MolecularStructureFormat::Pdb;
    const std::filesystem::path structure_path = has_option(command, "--output")
                                                     ? require_option(command, "--output")
                                                     : (pdb ? "prediction.pdb" : "prediction.cif");
    const std::filesystem::path metadata_path = has_option(command, "--output-json")
                                                    ? require_option(command, "--output-json")
                                                    : structure_path.string() + ".metadata.json";
    write_document(structure_path, result.structure());
    write_document(metadata_path, result.metadata_json());
    const auto confidence = confidence_json(result.confidence());
    nlohmann::json summary{{"structure_path", structure_path.string()},
                           {"metadata_path", metadata_path.string()},
                           {"format", pdb ? "pdb" : "mmcif"},
                           {"input_encoding", encoding},
                           {"confidence", confidence}};
    // Retain the existing command's top-level summary fields, with explicit
    // absence rather than fabricated zero scores for confidence-free families.
    for (const char* key : {"confidence_score", "complex_plddt", "ptm"})
        summary[key] = confidence.is_null() ? nlohmann::json(nullptr) : confidence.at(key);
    detail::write_json(output, summary);
    return true;
}
} // namespace trtmc::cli
