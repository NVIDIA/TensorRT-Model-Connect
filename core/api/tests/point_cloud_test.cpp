/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/point_cloud.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
template <class Function>
void rejects(Function function, trtmc_status expected, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const trtmc::Error& error) {
        rejected = error.code() == expected;
    }
    check(rejected, message);
}
void bundle(const std::filesystem::path& path, const std::string& mode) {
    const char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header = "{\"format\":1,\"family\":\"point_cloud_fixture\",\"task\":\"" +
                               mode + "\",\"backend\":\"fake\",\"sections\":{}}";
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.write(magic, sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        auto load = [&](const char* mode) {
            const auto path = root / (std::string("point-cloud-cpp-") + mode + ".bundle");
            bundle(path, mode);
            return trtmc::Model::load(path.string(), options);
        };
        // Two XYZ points: six floats, so the float count differs from num_points.
        const float points[6]{1, 2, 2, 3, 2, 2};
        auto model = load("points_to_semantic_segmentation");
        auto task = model.task<trtmc::PointsToSemanticSegmentation>();
        auto result = task.run({{points}, 2, 3});
        const auto view = result.view();
        check(view.num_points == 2 && view.labels[0] == 1 && view.labels[1] == 3,
              "wrapper sends num_points, not the float count, across the C ABI");
        check(view.num_classes == 4 && view.class_score_count == 8 &&
                  view.class_scores[0 * 4 + 1] == 1.0F && view.class_scores[1 * 4 + 3] == 1.0F,
              "per-point [num_points, num_classes] scores are preserved");
        auto pairs_result = task.run({{points}, 3, 2});
        const auto pairs = pairs_result.view();
        check(pairs.num_points == 3 && pairs.labels[0] == 1 && pairs.labels[1] == 2 &&
                  pairs.labels[2] == 2,
              "the same floats split by input_dim into a different point count");
        rejects([&] { (void)task.run({{points}, 3, 3}); }, TRTMC_INVALID_ARGUMENT,
                "num_points * input_dim beyond the caller span is rejected before the call");
        rejects([&] { (void)task.run({{points}, 1, 3}); }, TRTMC_INVALID_ARGUMENT,
                "a span longer than num_points * input_dim is rejected");
        rejects(
            [&] {
                (void)load("bad_labels")
                    .task<trtmc::PointsToSemanticSegmentation>()
                    .run({{points}, 2, 3});
            },
            TRTMC_INTERNAL_ERROR,
            "a family label count that disagrees with num_points is rejected");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return failures ? 1 : 0;
}
