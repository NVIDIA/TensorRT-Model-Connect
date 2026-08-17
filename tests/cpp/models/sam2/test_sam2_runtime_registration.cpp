/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_pipeline.h"
#include "trtmc/models/sam2_video.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace trtmc {
void register_sam2_plugin(PipelineRegistry& registry);
}

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Function>
std::string capture_error(Function&& function) {
    try {
        function();
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

class TemporaryRecord final {
  public:
    TemporaryRecord() {
        std::array<char, 64> pattern{};
        const std::string value = "/tmp/trtmc-sam2-runtime-record-XXXXXX";
        std::copy(value.begin(), value.end(), pattern.begin());
        descriptor_ = ::mkstemp(pattern.data());
        if (descriptor_ < 0)
            throw std::runtime_error("failed to create temporary SAM2 record");
        path_ = pattern.data();
        static constexpr std::array<char, 2> kContents{'x', '\n'};
        if (::write(descriptor_, kContents.data(), kContents.size()) !=
            static_cast<ssize_t>(kContents.size())) {
            throw std::runtime_error("failed to write temporary SAM2 record");
        }
        if (::close(descriptor_) != 0)
            throw std::runtime_error("failed to close temporary SAM2 record");
        descriptor_ = -1;
    }

    ~TemporaryRecord() {
        if (descriptor_ >= 0)
            ::close(descriptor_);
        if (!path_.empty())
            ::unlink(path_.c_str());
    }

    const std::string& path() const noexcept { return path_; }

  private:
    int descriptor_{-1};
    std::string path_;
};

void test_manifest_registers_own_family() {
    auto root = std::filesystem::path(__FILE__).parent_path();
    for (int index = 0; index < 4; ++index)
        root = root.parent_path();
    const auto manifest = root / "src/runtime/models/sam2/MODEL.toml";
    std::ifstream input(manifest);
    const std::string text((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    check(input.good() || input.eof(), "SAM2 runtime manifest is readable");
    check(text.find("id = \"sam2\"") != std::string::npos,
          "SAM2 runtime manifest owns the sam2 family");
    check(text.find("plugin.cpp|register_sam2_plugin") != std::string::npos,
          "SAM2 runtime manifest owns its plugin registration");
    check(text.find("sam2_bbox_video_tracking") != std::string::npos,
          "SAM2 runtime manifest owns its runtime strategy");
}

void test_plugin_requires_explicit_external_record() {
    auto& registry = trtmc::PipelineRegistry::instance();
    trtmc::register_sam2_plugin(registry);
    auto* plugin = registry.lookup("sam2_bbox_video_tracking");
    check(plugin != nullptr, "SAM2 plugin registers its strategy");
    if (plugin == nullptr)
        return;

    trtmc::BundleFile bundle;
    bundle.info.model_id = "sam2.1-hiera-small-bbox";
    trtmc::BaseConfig base;
    const std::string empty;
    const std::string bundle_path = "/not/opened.bundle";
    trtmc::PipelineContext context{bundle, base,  empty, empty,   bundle_path, nullptr,
                                   empty,  false, 0U,    nullptr, empty};
    const auto error = capture_error([&] { (void)plugin->create(context); });
    check(error.find("explicit qualification-record path") != std::string::npos,
          "SAM2 plugin fails closed without an external record path");
}

void test_pipeline_rejects_unpinned_record_before_module_factory() {
    TemporaryRecord record;
    trtmc::sam2::NativeBundleRuntimeTarget target{
        std::string(trtmc::sam2::kTargetTensorRtVersion),
        std::string(trtmc::sam2::kTargetTensorRtAbi),
        std::string(trtmc::sam2::kTargetGpuName),
        std::string(trtmc::sam2::kTargetComputeCapability),
    };
    int factory_calls = 0;
    trtmc::sam2::NativePlanModuleFactory factory =
        [&](std::string_view, const void*, std::size_t) -> std::unique_ptr<trtmc::ITrtModule> {
        ++factory_calls;
        return nullptr;
    };
    const auto error = capture_error([&] {
        (void)trtmc::sam2::Sam2Pipeline::createProductionQualified(
            "/bundle-must-not-be-opened-before-authority.bundle", record.path(), target, factory,
            "sam2.1-hiera-small-bbox");
    });
    check(error.find("no active compiled production authority pin") != std::string::npos,
          "SAM2 production pipeline rejects an unpinned record");
    check(factory_calls == 0,
          "SAM2 production pipeline rejects an unpinned record before module factory calls");
}

void test_c_api_separates_qualified_and_legacy_constructors() {
    auto* qualified =
        trtmc_sam2_video_create_from_qualified_bundle_v1(nullptr, "record", "plugin", "backend");
    check(qualified == nullptr, "SAM2 qualified C API rejects missing explicit arguments");
    check(std::string(trtmc_sam2_video_last_error()).find("qualification-record") !=
              std::string::npos,
          "SAM2 qualified C API reports its explicit record contract");

    auto* legacy = trtmc_sam2_video_create_from_bundle_v1("bundle", "plugin", "backend");
    check(legacy == nullptr, "SAM2 legacy C API remains fail-closed");
    check(std::string(trtmc_sam2_video_last_error()).find("unavailable") != std::string::npos,
          "SAM2 legacy C API retains its fail-closed diagnostic");
}

} // namespace

int main() {
    test_manifest_registers_own_family();
    test_plugin_requires_explicit_external_record();
    test_pipeline_rejects_unpinned_record_before_module_factory();
    test_c_api_separates_qualified_and_legacy_constructors();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
