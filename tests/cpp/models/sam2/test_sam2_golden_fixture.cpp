/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_golden_fixture.h"

#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path root =
        argc == 2 ? argv[1] : "tests/cpp/models/sam2/data/golden/compatible_source_pytorch_bf16";
    const auto golden = trtmc::sam2::test::loadGoldenFixture(root);
    require(golden.foreground_pixels[0] == 3600, "golden foreground count drifted");
    require(trtmc::sam2::test::compareMasks(golden.masks, golden).passes(),
            "golden self-mask comparison failed");
    require(trtmc::sam2::test::compareBbox(golden.bbox, golden).passes(),
            "golden self-bbox comparison failed");

    auto changed = golden.masks;
    changed[0] ^= 1U;
    const auto accuracy = trtmc::sam2::test::compareMasks(changed, golden);
    require(accuracy.frame_iou[0] < 1.0, "one-bit mask mutation was not detected");

    bool rejected_nonbinary = false;
    changed[0] = 2;
    try {
        (void)trtmc::sam2::test::compareMasks(changed, golden);
    } catch (const std::invalid_argument&) {
        rejected_nonbinary = true;
    }
    require(rejected_nonbinary, "non-binary candidate mask was accepted");
    return 0;
}
