/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
/// A routing test must fail before attempting any native engine load.
class RejectBackend final : public trtmc::IBackend {
  public:
    std::unique_ptr<trtmc::ITrtModule> create_module(const void*, size_t,
                                                     const trtmc::ModuleCreateOptions&) override {
        throw std::logic_error("unexpected engine load");
    }
    std::unique_ptr<trtmc::ITrtModule>
    create_module_prebound(const void*, size_t, const trtmc::ModuleCreateOptions&,
                           const std::vector<trtmc::ModuleExternalBinding>&) override {
        throw std::logic_error("unexpected prebound engine load");
    }
    trtmc::BackendDualProfileModules
    create_dual_profile_modules(const void*, size_t, const trtmc::ModuleCreateOptions&) override {
        throw std::logic_error("unexpected dual engine load");
    }
    const char* name() const override { return "trt"; }
};

/// Write a deliberately incomplete bundle to observe which factory owns its error.
void write_bundle(const std::filesystem::path& path, bool edge) {
    nlohmann::json sections = nlohmann::json::object();
    if (edge)
        sections["edge_llm.json"] = {{"offset", 0}, {"length", 2}};
    const auto header = nlohmann::json{
        {"format", 1},
        {"family", "qwen3_8"},
        {"task", "text_generation"},
        {"backend", "trt"},
        {"sections",
         sections}}.dump();
    std::ofstream out(path, std::ios::binary);
    out.write("BUNDLE\x01\x00", 8);
    const std::uint64_t size = header.size();
    for (int shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((size >> shift) & 0xff));
    out << header;
    if (edge)
        out << "{}";
    if (!out)
        throw std::runtime_error("Cannot write factory test bundle");
}
} // namespace

int main() {
    char pattern[] = "/tmp/trtmc_qwen3_8_factory_XXXXXX";
    const auto* directory = mkdtemp(pattern);
    if (!directory)
        return 1;
    const auto path = std::filesystem::path(directory) / "test.bundle";
    RejectBackend backend;
    int failures = 0;
    for (const bool edge : {false, true}) {
        write_bundle(path, edge);
        trtmc::BundleReader bundle(path.string());
        try {
            std::unique_ptr<trtmc::ITask> task(trtmc_create_family({bundle, backend, 0}));
            ++failures;
        } catch (const std::exception& error) {
            const std::string message = error.what();
            if (!edge) {
                if (message.find("runtime.json") == std::string::npos)
                    ++failures;
            } else {
#ifdef TRTMC_HAS_EDGE_LLM
                // The malformed Edge marker must not reach native metadata parsing.
                if (message.find("runtime.json") != std::string::npos ||
                    message.find("unexpected") != std::string::npos)
                    ++failures;
#else
                if (message.find("TRTMC_ENABLE_EDGELLM=ON") == std::string::npos)
                    ++failures;
#endif
            }
        }
    }
    std::filesystem::remove_all(directory);
    if (failures)
        std::cerr << "Family Edge/native factory routing failed" << std::endl;
    return failures == 0 ? 0 : 1;
}
