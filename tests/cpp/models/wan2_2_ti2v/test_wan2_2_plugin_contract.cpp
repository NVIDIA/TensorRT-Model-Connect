/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/plugin_contract.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using trtmc::wan2_2_ti2v::PluginContract;

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

constexpr const char* kDigest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

std::string contract_json(const char* semantic_abi = "wan22-ti2v-plugins-v1",
                          const char* digest = kDigest,
                          const char* creator_set = "A:1:;B:1:", int trt_minor = 1,
                          const char* architectures = "[103,110]") {
    return std::string{"{"} + R"("schema":1,"family":"wan2_2_ti2v","semantic_abi":")" +
           semantic_abi + R"(","source_digest":")" + digest + R"(","creator_set":")" + creator_set +
           R"(","runtime_abi":{"tensorrt_major":11,"tensorrt_minor":)" + std::to_string(trt_minor) +
           R"(,"cuda_major":13,"cudnn_major":9},"cuda_architectures":)" + architectures + "}";
}

std::string bundle_config(const std::string& contract) {
    return std::string{"{"} + R"("model_type":"wan2.2","_trtmc_wan22_plugin_contract":)" +
           contract + "}";
}

template <typename Function>
bool rejects(Function&& function, const char* message_fragment) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(message_fragment) != std::string::npos;
    }
    return false;
}

template <typename Function>
bool rejects_exactly(Function&& function, const std::string& expected_message) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        return error.what() == expected_message;
    }
    return false;
}

void test_parse_and_validate_exact_contract() {
    const auto manifest = contract_json();
    const auto expected = trtmc::wan2_2_ti2v::parse_bundle_plugin_contract(bundle_config(manifest));
    const auto installed = trtmc::wan2_2_ti2v::parse_companion_plugin_contract(manifest);
    check(expected == installed, "bundle and companion contracts parse identically");
    check(trtmc::wan2_2_ti2v::canonical_runtime_abi(installed.runtime_abi) ==
              "tensorrt=11.1;cuda=13;cudnn=9",
          "runtime ABI canonical encoding is stable");
    trtmc::wan2_2_ti2v::validate_plugin_contract(expected, installed,
                                                 "tensorrt=11.1;cuda=13;cudnn=9", 110);

    const auto trt_11_0 = trtmc::wan2_2_ti2v::parse_companion_plugin_contract(
        contract_json("wan22-ti2v-plugins-v1", kDigest, "A:1:;B:1:", 0));
    check(trtmc::wan2_2_ti2v::canonical_runtime_abi(trt_11_0.runtime_abi) ==
              "tensorrt=11.0;cuda=13;cudnn=9",
          "TensorRT 11.0 is a valid runtime ABI");
}

void test_missing_or_malformed_contract_fails_closed() {
    check(rejects([]() { (void)trtmc::wan2_2_ti2v::parse_bundle_plugin_contract("{}"); },
                  "missing object field"),
          "missing bundle contract is rejected");
    check(rejects(
              []() {
                  (void)trtmc::wan2_2_ti2v::parse_companion_plugin_contract(
                      contract_json("wan22-ti2v-plugins-v1", "ABC"));
              },
              "lowercase SHA-256"),
          "malformed source digest is rejected");
    check(rejects(
              []() {
                  (void)trtmc::wan2_2_ti2v::parse_companion_plugin_contract(
                      contract_json("wan22-ti2v-plugins-v1", kDigest, "A:1:;B:1:", 1, "[110,103]"));
              },
              "sorted and unique"),
          "non-canonical architecture list is rejected");
    check(rejects(
              []() {
                  (void)trtmc::wan2_2_ti2v::parse_companion_plugin_contract(
                      contract_json("wan22-ti2v-plugins-v1", kDigest, "A:1:;B:1:", 1, "[110]"));
              },
              "exactly [103,110]"),
          "single-platform architecture list is rejected");
    check(rejects(
              []() {
                  (void)trtmc::wan2_2_ti2v::parse_companion_plugin_contract(
                      contract_json("wan22-ti2v-plugins-v1", kDigest, "B:1:;A:1:"));
              },
              "creator_set must be sorted and unique"),
          "unsorted creator set is rejected");
    check(rejects(
              []() {
                  (void)trtmc::wan2_2_ti2v::parse_companion_plugin_contract(
                      contract_json("wan22-ti2v-plugins-v1", kDigest, "A:1:;;B:1:"));
              },
              "canonical name:version:namespace"),
          "empty creator set entry is rejected");
}

void test_every_provenance_dimension_fails_closed() {
    const auto installed = trtmc::wan2_2_ti2v::parse_companion_plugin_contract(contract_json());
    const auto runtime_abi = "tensorrt=11.1;cuda=13;cudnn=9";

    auto semantic_mismatch =
        trtmc::wan2_2_ti2v::parse_companion_plugin_contract(contract_json("other-v1"));
    check(rejects(
              [&]() {
                  trtmc::wan2_2_ti2v::validate_plugin_contract(semantic_mismatch, installed,
                                                               runtime_abi, 110);
              },
              "semantic ABI mismatch"),
          "semantic ABI mismatch is rejected");

    auto digest_mismatch = installed;
    digest_mismatch.source_digest =
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    check(rejects(
              [&]() {
                  trtmc::wan2_2_ti2v::validate_plugin_contract(digest_mismatch, installed,
                                                               runtime_abi, 110);
              },
              "source digest mismatch"),
          "source digest mismatch is rejected");

    auto creator_mismatch = installed;
    creator_mismatch.creator_set = "A:1:";
    check(rejects(
              [&]() {
                  trtmc::wan2_2_ti2v::validate_plugin_contract(creator_mismatch, installed,
                                                               runtime_abi, 110);
              },
              "creator set mismatch"),
          "creator set mismatch is rejected");

    auto abi_mismatch = installed;
    abi_mismatch.runtime_abi.tensorrt_minor = 2;
    check(rejects(
              [&]() {
                  trtmc::wan2_2_ti2v::validate_plugin_contract(abi_mismatch, installed, runtime_abi,
                                                               110);
              },
              "declared runtime ABI mismatch"),
          "declared TRT ABI mismatch is rejected");

    check(rejects(
              [&]() {
                  trtmc::wan2_2_ti2v::validate_plugin_contract(
                      installed, installed, "tensorrt=11.1;cuda=12;cudnn=9", 110);
              },
              "loaded runtime ABI mismatch"),
          "loaded CUDA ABI mismatch is rejected");

    check(rejects(
              [&]() {
                  trtmc::wan2_2_ti2v::validate_plugin_contract(installed, installed, runtime_abi,
                                                               120);
              },
              "does not contain current GPU sm_120"),
          "missing fatbin architecture is rejected");
}

void test_validation_error_precedence_is_stable() {
    const auto installed = trtmc::wan2_2_ti2v::parse_companion_plugin_contract(contract_json());
    const auto runtime_abi = "tensorrt=11.1;cuda=13;cudnn=9";

    auto identity_mismatch = installed;
    identity_mismatch.schema = 2;
    identity_mismatch.family = "other-family";
    identity_mismatch.semantic_abi = "other-abi";
    check(rejects_exactly(
              [&]() {
                  trtmc::wan2_2_ti2v::validate_plugin_contract(identity_mismatch, installed,
                                                               runtime_abi, 110);
              },
              "Wan2.2 AOT plugin contract schema mismatch"),
          "schema mismatch precedes later identity mismatches");

    identity_mismatch.schema = installed.schema;
    check(rejects_exactly(
              [&]() {
                  trtmc::wan2_2_ti2v::validate_plugin_contract(identity_mismatch, installed,
                                                               runtime_abi, 110);
              },
              "Wan2.2 AOT plugin family mismatch: bundle='other-family', "
              "installed='wan2_2_ti2v'"),
          "family mismatch precedes semantic ABI mismatch");

    auto runtime_mismatch = installed;
    runtime_mismatch.runtime_abi.tensorrt_minor = 2;
    check(rejects_exactly(
              [&]() {
                  trtmc::wan2_2_ti2v::validate_plugin_contract(
                      runtime_mismatch, installed, "tensorrt=11.1;cuda=12;cudnn=9", 120);
              },
              "Wan2.2 AOT plugin declared runtime ABI mismatch: "
              "bundle='tensorrt=11.2;cuda=13;cudnn=9', "
              "installed='tensorrt=11.1;cuda=13;cudnn=9'"),
          "declared runtime mismatch precedes loaded runtime and architecture mismatches");

    auto missing_current_sm = installed;
    missing_current_sm.cuda_architectures = {103};
    check(rejects_exactly(
              [&]() {
                  trtmc::wan2_2_ti2v::validate_plugin_contract(installed, missing_current_sm,
                                                               runtime_abi, 110);
              },
              "Wan2.2 AOT plugin does not contain current GPU sm_110"),
          "missing current SM precedes architecture-set mismatch");
}

} // namespace

int main() {
    test_parse_and_validate_exact_contract();
    test_missing_or_malformed_contract_fails_closed();
    test_every_provenance_dimension_fails_closed();
    test_validation_error_precedence_is_stable();
    if (failures != 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All Wan2.2 AOT plugin contract tests passed\n";
    return 0;
}
