/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Wan-owned C ABI regression for the legacy "diffusion" strategy alias.

#include "../../test_helpers.h"
#include "trtmc/pipeline.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

void write_u64_le(std::ofstream& out, uint64_t value) {
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xFFU);
    }
    out.write(reinterpret_cast<const char*>(bytes), 8);
}

struct BundleSectionSpec {
    std::string name;
    std::string data;
};

std::string build_bundle_header_json(const std::vector<BundleSectionSpec>& sections) {
    std::string sections_json;
    std::size_t offset = 0;
    for (std::size_t i = 0; i < sections.size(); ++i) {
        const auto& section = sections[i];
        if (i != 0)
            sections_json += ",\n";
        sections_json += "    \"" + section.name + "\": {\"offset\": " + std::to_string(offset) +
                         ", \"size\": " + std::to_string(section.data.size()) + "}";
        offset += section.data.size();
    }

    return std::string(R"({
  "model_id": "wan-cabi-regression-test",
  "model_type": "unit-test",
  "family": "unit",
  "hidden_size": 64,
  "num_layers": 1,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "max_cache_length": 32,
  "sections": {
)") + sections_json +
           R"(
  }
})";
}

void write_bundle_with_sections(const std::filesystem::path& path,
                                const std::vector<BundleSectionSpec>& sections) {
    static constexpr unsigned char kBundleMagic[8] = {'T', 'R', 'T', 'F', 'B', '\0', '\x01', '\0'};

    const std::string header = build_bundle_header_json(sections);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(kBundleMagic), sizeof(kBundleMagic));
    write_u64_le(out, static_cast<uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& section : sections) {
        out.write(section.data.data(), static_cast<std::streamsize>(section.data.size()));
    }
}

bool message_contains_any(const std::string& msg, std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (msg.find(needle) != std::string::npos)
            return true;
    }
    return false;
}

void test_legacy_diffusion_alias_reports_wan_section_guard() {
    trtmc_test::TempDirGuard dir;
    const std::filesystem::path bundle_path =
        std::filesystem::path(dir.path()) / "wan_legacy_alias_missing_sections.trtfb";

    const std::string config = R"({
  "runtime_strategy": "diffusion",
  "num_text_encoders": 1,
  "scheduler": "flow_match_euler"
})";
    write_bundle_with_sections(bundle_path, {BundleSectionSpec{"config.json", config}});

    auto* pipeline = trtmc_create_pipeline(bundle_path.string().c_str(), 0);
    check(pipeline == nullptr, "wan legacy alias bundle without plans returns nullptr");

    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "trtmc_last_error set for wan guard path");
    if (err != nullptr) {
        check(message_contains_any(err, {"New runtime build failed", "denoiser_plan",
                                         "multi-engine bundles", "Diffusion pipeline",
                                         "Bundle missing", "Backend \"trt\" not available",
                                         "Could not load libtrtmc_backend_trt.so",
                                         "No compatible backend DSO available"}),
              "wan legacy alias reports missing diffusion section guard");
    }
}

} // namespace

int main() {
    test_legacy_diffusion_alias_reports_wan_section_guard();

    if (failures > 0) {
        std::cerr << failures << " wan C ABI runtime regression test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All wan C ABI runtime regression tests passed\n";
    return 0;
}
