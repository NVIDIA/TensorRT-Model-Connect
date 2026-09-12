/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/perception.hpp"

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
    const std::string header = "{\"format\":1,\"family\":\"image_boxes_fixture\",\"task\":\"" +
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
            const auto path = root / (std::string("image-boxes-cpp-") + mode + ".bundle");
            bundle(path, mode);
            return trtmc::Model::load(path.string(), options);
        };
        float pixels[36]{0.5F};
        const trtmc::ImageToBoxesRequest request{trtmc::ImageInput({pixels}, 3, 4)};
        auto model = load("image_to_boxes");
        check(model.tasks().size() == 1 && model.supports<trtmc::ImageToBoxes>() &&
                  !model.supports<trtmc::ImageTextToBoxes>(),
              "image-only detection is not text grounding");
        auto task = model.task<trtmc::ImageToBoxes>();
        check(task.config_fields().size() == 2, "Config metadata is declared by the family");
        auto result = task.run(request);
        const auto view = result.view();
        check(view.count == 2 && view.image_height == 3 && view.image_width == 4 &&
                  view.boxes[0].class_id == 42 && view.boxes[1].class_id == 7 &&
                  view.boxes[0].score == 0.75F && view.boxes[1].score == 0.0F &&
                  view.boxes[0].box.x_min == -2.0F && view.boxes[0].box.x_max == 7.0F &&
                  view.boxes[0].box.y_min == 0.5F && view.boxes[0].box.y_max == 3.0F,
              "class IDs, scores, dimensions and unmodified pixel coordinates survive all layers");
        auto zero = task.run(request, {{"score", 0.0}, {"empty", false}});
        check(zero.view().count == 2 && zero.view().boxes[0].score == 0,
              "explicit zero and false are not replaced by defaults");
        auto empty = task.run(request, {{"empty", true}});
        check(empty.view().count == 0 && empty.view().image_height == 3 &&
                  empty.view().image_width == 4,
              "empty detections preserve original image dimensions");
        auto guarded = load("must_not_run").task<trtmc::ImageToBoxes>();
        rejects([&] { (void)guarded.run(request, {{"unknown", true}}); }, TRTMC_INVALID_CONFIG,
                "Core rejects unknown Config before invoking the family");
        rejects([&] { (void)guarded.run(request, {{"score", true}}); }, TRTMC_INVALID_CONFIG,
                "Core rejects wrong Config type before invoking the family");
        rejects([&] { (void)guarded.run(request, {{"score", 0.1}, {"score", 0.2}}); },
                TRTMC_INVALID_CONFIG, "Core rejects duplicate Config before invoking the family");
        rejects([&] { (void)guarded.run(request); }, TRTMC_INTERNAL_ERROR,
                "guarded fixture really throws when execution is reached");
        rejects([&] { (void)task.run(request, {{"score", 2.0}}); }, TRTMC_INVALID_CONFIG,
                "family numeric range validation is preserved");
        rejects([&] { (void)load("disabled").task<trtmc::ImageToBoxes>(); }, TRTMC_UNSUPPORTED,
                "an unavailable Task is not granted by its C++ class");
        for (const char* mode :
             {"bad_dimensions", "wrong_image_dimensions", "bad_box", "bad_score"})
            rejects([&] { (void)load(mode).task<trtmc::ImageToBoxes>().run(request); },
                    TRTMC_INTERNAL_ERROR, "malformed family output is rejected");
        auto malformed = request;
        malformed.image.wire.byte_size = 0;
        rejects([&] { (void)task.run(malformed); }, TRTMC_INVALID_ARGUMENT,
                "malformed image storage is rejected");
        auto retained = [&] {
            auto temporary = load("image_to_boxes");
            return temporary.task<trtmc::ImageToBoxes>().run(request);
        }();
        auto moved = std::move(retained);
        check(moved.view().count == 2 && moved.view().boxes[0].class_id == 42,
              "result ownership survives model scope and result move");
        std::cout << "Image-only detection: classes 42 and 7; input 4x3 pixels; output 2 boxes\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return failures ? 1 : 0;
}
