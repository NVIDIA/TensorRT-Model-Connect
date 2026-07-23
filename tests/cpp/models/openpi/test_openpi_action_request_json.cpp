/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CLI-CPP-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-CABI-01
// Intent:         Strict native action-request JSON parsing and result encoding
// Preconditions:  None
// Postconditions: Malformed policy requests fail closed and valid typed results
//                 serialize with a stable machine-readable schema
// =============================================================================

#include "runtime/models/openpi/tool/action_request_json.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

template <typename Function>
void check_throws(Function&& function, const char* needle, const char* name) {
    try {
        function();
        check(false, name);
    } catch (const std::exception& error) {
        check(std::string(error.what()).find(needle) != std::string::npos, name);
    }
}

const char* kValidRequest = R"json(
{
  "prompt": "pick up the red block",
  "cameras": {
    "base_0_rgb": {"path": "base.png", "valid": true},
    "left_wrist_0_rgb": {"path": "left.png", "valid": true},
    "right_wrist_0_rgb": {"path": "right.png", "valid": false}
  },
  "state": [0, 1.5, -2, 3, 4, 5, 6, 7],
  "initial_noise": [0.25, -0.5],
  "seed": 7,
  "denoise_steps": 10
}
)json";

void test_valid_request() {
    const auto request = trtmc::openpi::tool::parse_action_request_json(kValidRequest);
    check(request.prompt == "pick up the red block", "valid prompt");
    check(request.cameras[0].name == "base_0_rgb" && request.cameras[0].path == "base.png" &&
              request.cameras[0].valid,
          "base camera");
    check(request.cameras[1].name == "left_wrist_0_rgb" && request.cameras[1].path == "left.png" &&
              request.cameras[1].valid,
          "left camera");
    check(request.cameras[2].name == "right_wrist_0_rgb" &&
              request.cameras[2].path == "right.png" && !request.cameras[2].valid,
          "right camera mask");
    check(request.state.size() == 8U && request.state[1] == 1.5F, "state array");
    check(request.initial_noise.size() == 2U && request.initial_noise[1] == -0.5F,
          "initial noise array");
    check(request.seed == 7, "seed");
    check(request.denoise_steps == 10, "denoise steps");
}

void test_optional_fields() {
    const auto request = trtmc::openpi::tool::parse_action_request_json(R"json({
      "prompt": "move",
      "cameras": {
        "base_0_rgb": {"path": "a.png", "valid": true},
        "left_wrist_0_rgb": {"path": "b.png", "valid": true},
        "right_wrist_0_rgb": {"path": "c.png", "valid": false}
      },
      "state": [0]
    })json");
    check(request.initial_noise.empty(), "optional initial noise absent");
    check(request.seed == -1, "optional seed absent");
    check(request.denoise_steps == -1, "optional denoise steps absent");
}

void test_strict_keys_and_types() {
    check_throws(
        [] {
            trtmc::openpi::tool::parse_action_request_json(
                R"json({"prompt":"a","prompt":"b","cameras":{},"state":[0]})json");
        },
        "duplicate JSON object key", "duplicate root key rejected");
    check_throws(
        [] {
            trtmc::openpi::tool::parse_action_request_json(R"json({
              "prompt":"a",
              "cameras":{
                "base_0_rgb":{"path":"a","path":"b","valid":true},
                "left_wrist_0_rgb":{"path":"b","valid":true},
                "right_wrist_0_rgb":{"path":"c","valid":false}},
              "state":[0]})json");
        },
        "duplicate JSON object key", "duplicate camera key rejected");
    check_throws(
        [] {
            trtmc::openpi::tool::parse_action_request_json(R"json({
              "prompt":"a","cameras":{
                "base_0_rgb":{"path":"a","valid":true},
                "left_wrist_0_rgb":{"path":"b","valid":true},
                "right_wrist_0_rgb":{"path":"c","valid":false}},
              "state":[0],"unknown":1})json");
        },
        "unexpected field 'unknown'", "unknown root field rejected");
    check_throws(
        [] {
            trtmc::openpi::tool::parse_action_request_json(R"json({
              "prompt":"a","cameras":{
                "base_0_rgb":{"path":"a","valid":1},
                "left_wrist_0_rgb":{"path":"b","valid":true},
                "right_wrist_0_rgb":{"path":"c","valid":false}},
              "state":[0]})json");
        },
        "must be a boolean", "integer camera mask rejected");
    check_throws(
        [] {
            trtmc::openpi::tool::parse_action_request_json(R"json({
              "prompt":"a","cameras":{
                "base_0_rgb":{"path":"a","valid":true},
                "left_wrist_0_rgb":{"path":"b","valid":true},
                "right_wrist_0_rgb":{"path":"c","valid":false}},
              "state":[0],"seed":-1})json");
        },
        "must be non-negative", "negative seed rejected");
    check_throws(
        [] {
            trtmc::openpi::tool::parse_action_request_json(R"json({
              "prompt":"a","cameras":{
                "base_0_rgb":{"path":"a","valid":true},
                "left_wrist_0_rgb":{"path":"b","valid":true},
                "right_wrist_0_rgb":{"path":"c","valid":false}},
              "state":[0],"denoise_steps":0})json");
        },
        "positive int32", "zero denoise steps rejected");
    check_throws(
        [] {
            trtmc::openpi::tool::parse_action_request_json(R"json({
              "prompt":"a","cameras":{
                "base_0_rgb":{"path":"a","valid":true},
                "left_wrist_0_rgb":{"path":"b","valid":true},
                "right_wrist_0_rgb":{"path":"c","valid":false}},
              "state":[1e40]})json");
        },
        "out-of-range", "out-of-range float rejected");
}

void test_result_serialization() {
    trtmc::openpi::ActionResult result;
    result.actions = {0.25F, -0.5F, 1.0F, 2.0F};
    result.horizon = 2;
    result.action_dim = 2;
    result.timings.preprocess_ms = 0.1;
    result.timings.prefill_ms = 1.2;
    result.timings.denoise_ms = 3.4;
    result.timings.postprocess_ms = 0.2;

    const auto document =
        nlohmann::json::parse(trtmc::openpi::tool::serialize_action_result_json(result));
    check(document.size() == 4U, "result exact root field count");
    check(document.at("horizon") == 2, "result horizon");
    check(document.at("action_dim") == 2, "result action dim");
    check(document.at("actions").size() == 4U && document.at("actions")[1] == -0.5F,
          "result actions");
    check(document.at("timings").size() == 4U && document.at("timings").at("denoise_ms") == 3.4,
          "result timings");

    result.actions.pop_back();
    check_throws([&] { trtmc::openpi::tool::serialize_action_result_json(result); },
                 "does not match horizon", "result shape rejected");
    result.actions = {0.25F, -0.5F, 1.0F, std::numeric_limits<float>::infinity()};
    check_throws([&] { trtmc::openpi::tool::serialize_action_result_json(result); },
                 "non-finite action", "non-finite result rejected");
    result.actions.back() = 2.0F;
    result.timings.prefill_ms = -1.0;
    check_throws([&] { trtmc::openpi::tool::serialize_action_result_json(result); },
                 "must be finite and non-negative", "negative timing rejected");
}

} // namespace

int main() {
    test_valid_request();
    test_optional_fields();
    test_strict_keys_and_types();
    test_result_serialization();

    if (failures != 0) {
        std::cerr << failures << " action request JSON tests failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All action request JSON tests passed\n";
    return EXIT_SUCCESS;
}
