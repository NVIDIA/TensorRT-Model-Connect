/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/plugin_helpers.h"

#include "bundle/bundle_view.h"
#include "utils/sha256.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace trtmc::openpi {
namespace {

const std::vector<char>* require_unique_section(const BundleFile& bundle, const char* name) {
    const std::vector<char>* result = nullptr;
    for (const auto& section : bundle.sections) {
        if (section.name != name) {
            continue;
        }
        if (result != nullptr) {
            throw std::runtime_error(std::string("OpenPI bundle contains duplicate section '") +
                                     name + "'");
        }
        result = &section.data;
    }
    if (result == nullptr || result->empty()) {
        throw std::runtime_error(std::string("OpenPI bundle is missing required section '") + name +
                                 "'");
    }
    return result;
}

std::string_view section_text(const std::vector<char>& section) {
    return std::string_view(section.data(), section.size());
}

std::string sha256(std::string_view bytes) {
    internal::Sha256 hash;
    hash.update(bytes);
    return hash.hex_digest();
}

void require_matching_digest(std::string_view section_name, std::string_view bytes,
                             const std::string& expected) {
    if (sha256(bytes) != expected) {
        throw std::runtime_error("OpenPI bundle section '" + std::string(section_name) +
                                 "' does not match config.json SHA-256");
    }
}

} // namespace

VerifiedOpenPIBundle verify_openpi_bundle_integrity(const BundleFile& bundle) {
    std::unordered_set<std::string_view> section_names;
    for (const auto& section : bundle.sections) {
        if (!section_names.emplace(section.name).second) {
            throw std::runtime_error("OpenPI bundle contains duplicate section '" + section.name +
                                     "'");
        }
    }
    std::unordered_set<std::string_view> expected_section_names;
    for (const auto name : kRequiredIntegritySectionNames) {
        expected_section_names.emplace(name);
    }
    if (section_names != expected_section_names) {
        throw std::runtime_error("OpenPI bundle contains missing or unexpected physical sections");
    }

    const auto* config_section = require_unique_section(bundle, "config.json");
    const auto* prefill_plan = require_unique_section(bundle, "engine_plan");
    const auto* action_plan = require_unique_section(bundle, "openpi_action_step_engine_plan");
    const auto* tokenizer_section = require_unique_section(bundle, "tokenizer.model");
    const auto* normalization_section = require_unique_section(bundle, "preprocessor_config.json");

    VerifiedOpenPIBundle verified;
    verified.config_json = section_text(*config_section);
    verified.tokenizer_bytes = section_text(*tokenizer_section);
    verified.normalization_bytes = section_text(*normalization_section);
    verified.prefill_plan = prefill_plan;
    verified.action_plan = action_plan;
    verified.config = parse_openpi_config(verified.config_json);
    verified.normalization =
        parse_openpi_normalization(verified.normalization_bytes, verified.config);

    require_matching_digest("engine_plan", section_text(*prefill_plan),
                            verified.config.prefill_engine_sha256);
    require_matching_digest("openpi_action_step_engine_plan", section_text(*action_plan),
                            verified.config.action_engine_sha256);
    require_matching_digest("tokenizer.model", verified.tokenizer_bytes,
                            verified.config.tokenizer_sha256);
    require_matching_digest("preprocessor_config.json", verified.normalization_bytes,
                            verified.config.normalization_sha256);
    return verified;
}

std::unique_ptr<ITrtModule> load_openpi_module(IBackend* backend, const std::vector<char>* plan,
                                               const char* section_name, const char* timing_label,
                                               const ModuleCreateOptions& options) {
    if (backend == nullptr) {
        throw std::runtime_error("OpenPI cannot load without a TensorRT backend");
    }
    const char* backend_name = backend->name();
    if (backend_name == nullptr) {
        throw std::runtime_error("OpenPI requires TensorRT backend 'trt'; backend name is null");
    }
    if (std::string_view(backend_name) != "trt") {
        throw std::runtime_error(std::string("OpenPI requires TensorRT backend 'trt'; got '") +
                                 backend_name + "'");
    }
    if (plan == nullptr || plan->empty()) {
        throw std::runtime_error(std::string("OpenPI bundle is missing required section '") +
                                 section_name + "'");
    }
    auto module = backend->create_module(plan->data(), plan->size(), options);
    if (!module || !module->ok()) {
        throw std::runtime_error(std::string("OpenPI failed to deserialize '") + section_name +
                                 "'");
    }
    module->set_timing_label(timing_label);
    return module;
}

} // namespace trtmc::openpi
