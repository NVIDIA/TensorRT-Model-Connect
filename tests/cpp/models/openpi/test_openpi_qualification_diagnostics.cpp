/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/tool/qualification_diagnostics.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

template <typename Function>
void check_throws(Function&& function, const char* name) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, name);
}

void test_atomic_manifest_and_payload() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() /
                      ("trtmc_qualification_diagnostics_test_" + std::to_string(nonce));

    trtmc::openpi::ActionDiagnosticResult capture;
    trtmc::openpi::DiagnosticTensor tensor;
    tensor.name = "normalized_actions";
    tensor.stage = trtmc::openpi::DiagnosticStage::kPostprocess;
    tensor.role = trtmc::openpi::DiagnosticRole::kOutput;
    tensor.dtype = trtmc::openpi::DiagnosticTensorType::kFloat32;
    tensor.shape = {1, 1, 1};
    tensor.bytes.resize(sizeof(float));
    const float value = 1.5F;
    std::memcpy(tensor.bytes.data(), &value, sizeof(value));
    capture.tensors.push_back(tensor);

    const auto manifest =
        trtmc::openpi::tool::write_qualification_diagnostics(capture, root, "openpi-test");
    check(manifest == root / "manifest.json" && std::filesystem::is_regular_file(manifest),
          "qualification manifest is atomically published");
    std::ifstream manifest_stream(manifest);
    const auto document = nlohmann::json::parse(manifest_stream);
    const auto& descriptor = document.at("tensors").at("normalized_actions");
    check(document.at("artifact_type") == "trtmc_action_qualification_diagnostics" &&
              descriptor.at("path") == "tensors/normalized_actions.bin" &&
              descriptor.at("stage") == "postprocess" && descriptor.at("role") == "output" &&
              descriptor.at("dtype") == "float32" && descriptor.at("shape") == tensor.shape &&
              descriptor.at("byte_length") == sizeof(float) &&
              descriptor.at("sha256").get<std::string>().size() == 64U,
          "qualification manifest uses reference-compatible descriptors");
    std::ifstream payload(root / "tensors" / "normalized_actions.bin", std::ios::binary);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(payload),
                                     std::istreambuf_iterator<char>()};
    check(bytes == tensor.bytes, "qualification payload preserves raw little-endian bytes");
    check_throws(
        [&] {
            (void)trtmc::openpi::tool::write_qualification_diagnostics(capture, root,
                                                                       "openpi-test");
        },
        "qualification writer rejects an existing target");

    std::filesystem::remove_all(root);
}

void test_invalid_payload_fails_closed() {
    trtmc::openpi::ActionDiagnosticResult capture;
    capture.tensors.push_back(
        trtmc::openpi::DiagnosticTensor{"bad/name",
                                        trtmc::openpi::DiagnosticStage::kFlow,
                                        trtmc::openpi::DiagnosticRole::kIntermediate,
                                        trtmc::openpi::DiagnosticTensorType::kFloat32,
                                        {1},
                                        std::vector<uint8_t>(sizeof(float))});
    check_throws(
        [&] {
            (void)trtmc::openpi::tool::write_qualification_diagnostics(
                capture, std::filesystem::temp_directory_path() / "trtmc_never_publish_bad_name",
                "openpi-test");
        },
        "qualification writer rejects unsafe tensor names");
}

} // namespace

int main() {
    test_atomic_manifest_and_payload();
    test_invalid_payload_fails_closed();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cerr << "All qualification diagnostics tests passed.\n";
    return 0;
}
