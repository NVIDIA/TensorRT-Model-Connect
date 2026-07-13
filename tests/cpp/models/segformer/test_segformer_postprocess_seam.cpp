/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-02-SEGFORMER
// Architecture:   ARCH-MODPLUG-001
// Unit Design:    UD-SEG-01
// Intent:         SegFormer-owned postprocess argmax: classification, tie-breaking, invalid shapes
// Preconditions:  Logit tensors with known class distributions
// Postconditions: Argmax selects correct class, ties select first, invalid shapes rejected
// =============================================================================

#include "runtime/models/segformer/segformer_postprocess_seam.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++g_failures;
    }
}

void check_eq(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected,
              const char* test_name) {
    if (actual != expected) {
        std::cerr << "FAIL: " << test_name << " size(actual)=" << actual.size()
                  << " size(expected)=" << expected.size() << '\n';
        ++g_failures;
    }
}

void test_segformer_postprocess_argmax_basic() {
    const trtmc::SegformerLogitsShape shape{
        3, // classes
        2, // output_h
        2  // output_w
    };

    // Logits are [C, H, W] flattened by class plane.
    const std::vector<float> logits = {
        // class 0
        0.1F,
        0.2F,
        0.9F,
        0.1F,
        // class 1
        0.5F,
        0.3F,
        0.8F,
        0.6F,
        // class 2
        0.4F,
        0.7F,
        0.2F,
        0.6F,
    };

    std::vector<int32_t> class_map;
    const auto status = trtmc::compute_segformer_class_map_from_logits(logits, shape, class_map);

    check(status == trtmc::SegformerPostprocessStatus::kOk, "postprocess basic: status ok");
    check_eq(class_map, std::vector<int32_t>{1, 2, 0, 1}, "postprocess basic: expected class map");
}

void test_segformer_postprocess_tie_selects_first_class() {
    const trtmc::SegformerLogitsShape shape{2, 1, 1};
    const std::vector<float> logits = {
        5.0F, // class 0
        5.0F  // class 1 (tie, should keep class 0)
    };

    std::vector<int32_t> class_map;
    const auto status = trtmc::compute_segformer_class_map_from_logits(logits, shape, class_map);

    check(status == trtmc::SegformerPostprocessStatus::kOk, "postprocess tie: status ok");
    check(class_map.size() == 1, "postprocess tie: output size");
    if (class_map.size() == 1) {
        check(class_map[0] == 0, "postprocess tie: first class selected");
    }
}

void test_segformer_postprocess_invalid_shape() {
    const trtmc::SegformerLogitsShape shape{
        0, // invalid classes
        2,
        2,
    };
    const std::vector<float> logits = {1.0F, 2.0F, 3.0F, 4.0F};

    std::vector<int32_t> class_map = {9, 9};
    const auto status = trtmc::compute_segformer_class_map_from_logits(logits, shape, class_map);

    check(status == trtmc::SegformerPostprocessStatus::kInvalidShape,
          "postprocess invalid shape: status");
    check(class_map.empty(), "postprocess invalid shape: class_map cleared");
}

void test_segformer_postprocess_logits_size_mismatch() {
    const trtmc::SegformerLogitsShape shape{2, 2, 2}; // expected logits size = 8
    const std::vector<float> logits = {
        0.1F, 0.2F, 0.3F, 0.4F, // class 0
        1.1F, 1.2F, 1.3F        // class 1 (one element missing)
    };

    std::vector<int32_t> class_map = {7};
    const auto status = trtmc::compute_segformer_class_map_from_logits(logits, shape, class_map);

    check(status == trtmc::SegformerPostprocessStatus::kLogitsSizeMismatch,
          "postprocess logits mismatch: status");
    check(class_map.empty(), "postprocess logits mismatch: class_map cleared");
}

} // namespace

int main() {
    test_segformer_postprocess_argmax_basic();
    test_segformer_postprocess_tie_selects_first_class();
    test_segformer_postprocess_invalid_shape();
    test_segformer_postprocess_logits_size_mismatch();
    if (g_failures != 0) {
        std::cerr << g_failures << " SegFormer postprocess seam test(s) failed" << '\n';
        return 1;
    }

    std::cout << "SegFormer postprocess seam tests passed" << '\n';
    return 0;
}
