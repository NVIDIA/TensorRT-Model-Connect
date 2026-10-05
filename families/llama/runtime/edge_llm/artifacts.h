/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "contract.h"
#include "trtmc/bundle.h"

#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <vector>

namespace trtmc::llama::edge_llm {
namespace fs = std::filesystem;
/// Own extracted engine/checkpoint files until after the Edge runtime is destroyed.
class Artifacts {
  public:
    explicit Artifacts(const BundleReader& bundle, const nlohmann::json& marker) {
        std::set<std::string> names;
        for (const auto& entry : marker.at("artifacts")) {
            const auto name = entry.get<std::string>();
            if (!safe_artifact_path(name) || !names.insert(name).second ||
                !bundle.find_section(name))
                throw std::runtime_error("Invalid Llama3 Edge artifact: " + name);
        }
        std::vector<std::string> required_files{
            "edge_llm/engine/tokenizer.json", "edge_llm/engine/tokenizer_config.json",
            (marker.value("version", 1) == 2 && marker.at("provider").at("version") == "0.11.0"
                 ? "edge_llm/engine/chat_template.jinja"
                 : "edge_llm/engine/processed_chat_template.json"),
            "edge_llm/checkpoint/config.json"};
        if (marker.value("execution_variant", "autoregressive") == "eagle3") {
            for (const auto* name :
                 {"spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json"})
                required_files.push_back(std::string("edge_llm/engine/") + name);
            required_files.push_back("edge_llm/checkpoint/draft/config.json");
        } else {
            required_files.push_back("edge_llm/engine/llm.engine");
            required_files.push_back("edge_llm/engine/config.json");
        }
        for (const auto& required : required_files)
            if (!names.count(required) || bundle.find_section(required)->length == 0)
                throw std::runtime_error(std::string("Required Llama3 Edge artifact missing: ") +
                                         required);
        std::string pattern = (fs::temp_directory_path() / "trtmc-llama-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create Llama3 Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section(name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract Llama3 Edge artifact: " + name);
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~Artifacts() { cleanup(); }
    Artifacts(const Artifacts&) = delete;
    Artifacts& operator=(const Artifacts&) = delete;
    std::string engine() const { return (root_ / "edge_llm/engine").string(); }
    std::string checkpoint() const { return (root_ / "edge_llm/checkpoint").string(); }

  private:
    void cleanup() noexcept {
        std::error_code ignored;
        fs::remove_all(root_, ignored);
    }
    fs::path root_;
};

} // namespace trtmc::llama::edge_llm
