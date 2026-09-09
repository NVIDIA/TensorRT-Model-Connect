/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/image.hpp"

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
void write_bundle(const std::filesystem::path& path, const std::string& mode = "all_images") {
    const char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header = "{\"format\":1,\"family\":\"image_fixture\",\"task\":\"" + mode +
                               "\","
                               "\"backend\":\"fake\",\"sections\":{}}";
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(magic, sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const auto path = std::filesystem::path(argv[1]) / "cpp-image.bundle";
    try {
        write_bundle(path);
        trtmc::LoadOptions options;
        options.runtime_root = argv[1];
        auto model = trtmc::Model::load(path.string(), options);
        const auto worker_path = path.parent_path() / "cpp-image-worker.bundle";
        write_bundle(worker_path, "worker");
        auto worker_model = trtmc::Model::load(worker_path.string(), options);
        auto worker = worker_model.task<trtmc::TextToImage>().run({"participate"});
        check(worker.is_worker() && worker.pixels().empty(),
              "explicit non-output participant completion is not an image failure");
        write_bundle(worker_path, "bad_empty");
        auto bad_model = trtmc::Model::load(worker_path.string(), options);
        bool bad_empty = false;
        try {
            (void)bad_model.task<trtmc::TextToImage>().run({"bad"});
        } catch (const trtmc::Error& error) {
            bad_empty = error.code() == TRTMC_INTERNAL_ERROR;
        }
        check(bad_empty, "malformed empty image is not accepted as a worker");
        std::filesystem::remove(worker_path);
        auto generated = model.task<trtmc::TextToImage>().run({"red"});
        check(generated.height() == 1 && generated.width() == 1 && generated.channels() == 3 &&
                  generated.pixels()[0] == 0.25F,
              "typed image output and family default");
        const float first[] = {0.2F, 0.3F, 0.4F};
        const std::uint8_t second[] = {255, 32, 64};
        const trtmc::ImageInput first_image{trtmc::Span<const float>{first}, 1, 1};
        const trtmc::ImageInput second_image{trtmc::Span<const std::uint8_t>{second}, 1, 1};
        if (sizeof(std::size_t) >= sizeof(std::uint64_t)) {
            bool overflow_rejected = false;
            try {
                (void)trtmc::ImageInput{{first, std::numeric_limits<std::size_t>::max()}, 1, 1};
            } catch (const std::overflow_error&) {
                overflow_rejected = true;
            }
            check(overflow_rejected, "pixel byte count cannot wrap during C++ to C conversion");
        }
        auto edited = model.task<trtmc::ImagesTextToImageEdit>().run(
            {{first_image, second_image}, "combine"}, {{"level", 0.5}});
        check(edited.pixels()[0] == 0.5F && edited.pixels()[1] == first[0] &&
                  edited.pixels()[2] == 1,
              "ordered source inputs retain their types and family config");
        const float replay_values[] = {0, -0.25F, 0.5F};
        auto replayed = model.task<trtmc::TextToImage>().run({"red", {replay_values}});
        check(replayed.pixels()[0] == 0.25F && replayed.pixels()[1] == 3.0F / 32 - 0.25F &&
                  replayed.pixels()[2] == 0.625F,
              "all float32 replay values including zero and negative reach the family");
        auto replay_edit = model.task<trtmc::ImagesTextToImageEdit>().run(
            {{first_image, second_image}, "combine", {replay_values}}, {{"level", 0.5}});
        check(replay_edit.pixels()[0] == 0.5F && replay_edit.pixels()[1] == first[0] - 0.25F &&
                  replay_edit.pixels()[2] == 1.5F,
              "image edit retains both its required source images and optional replay");
        auto replay_batch = model.task<trtmc::BatchTextToImage>().run(
            {{{"one", {replay_values}}, {{"level", 0.5}}}});
        check(replay_batch.size() == 1 && replay_batch[0].pixels[0] == 0.5F &&
                  replay_batch[0].pixels[2] == 1.375F,
              "one-item native batch transports its own replay operand");
        for (const bool invalid_count : {true, false}) {
            bool replay_rejected = false;
            try {
                if (invalid_count)
                    (void)model.task<trtmc::TextToImage>().run({"red", {replay_values, 2}});
                else
                    (void)model.task<trtmc::BatchTextToImage>().run(
                        {{{"one", {replay_values}}, {}}, {{"two"}, {}}});
            } catch (const trtmc::Error& error) {
                replay_rejected = error.code() == TRTMC_INVALID_ARGUMENT;
            }
            check(replay_rejected, "family validates its replay layout and batch restriction");
        }
        auto retained_replay = [&] {
            auto local_model = trtmc::Model::load(path.string(), options);
            const std::vector<float> local_values{0, -0.25F, 0.5F};
            return local_model.task<trtmc::TextToImage>().run(
                {"red", {local_values.data(), local_values.size()}});
        }();
        check(retained_replay.pixels()[2] == 0.625F,
              "result owns its data after replay buffer and user model leave scope");
        const float mask[] = {0};
        auto masked = model.task<trtmc::MaskedImageTextToImage>().run(
            {first_image, {mask, 1, 1}, "preserve"});
        check(masked.pixels()[0] == first[0], "mask is typed input with preserved polarity");
        auto batch = [&] {
            auto local = trtmc::Model::load(path.string(), options);
            return local.task<trtmc::BatchTextToImage>().run(
                {{{"first"}, {{"level", 0.5}}}, {{"second"}, {{"level", 0.75}}}});
        }();
        check(batch.size() == 2 && batch[0].pixels[0] == 0.5F && batch[1].pixels[0] == 0.75F &&
                  batch[1].pixels[2] == 0.875F,
              "independent configs use native batch and owned results");
        bool rejected = false;
        try {
            (void)batch.at(2);
        } catch (const trtmc::Error& e) {
            rejected = e.code() == TRTMC_INVALID_ARGUMENT;
        }
        check(rejected, "out-of-range batch view rejects clearly");
        rejected = false;
        try {
            (void)model.task<trtmc::TextToImage>().run({"red"}, {{"level", "bad"}});
        } catch (const trtmc::Error& e) {
            rejected = e.code() == TRTMC_INVALID_CONFIG;
        }
        check(rejected, "image config errors cross all three layers");
        auto moved = std::move(generated);
        check(moved.pixels().size() == 3 && generated.pixels().empty(),
              "move clears borrowed source views");
        const auto fields = model.task<trtmc::TextToImage>().config_fields();
        check(fields.size() == 1 && fields[0].default_value->get<double>() == 0.25,
              "image Task config discovery is model-owned");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        ++failures;
    }
    std::filesystem::remove(path);
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
