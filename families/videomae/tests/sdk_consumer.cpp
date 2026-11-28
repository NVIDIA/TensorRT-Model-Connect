/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++ SDK consumer for the family-owned end-to-end test.
//
// Usage: BUNDLE RUNTIME_ROOT FRAMES u8|f32 FRAME_COUNT HEIGHT WIDTH MASK OUTPUT
//
// Same inputs as sdk_consumer.c. This consumer passes a timeline (frame i at
// i / 30 s), checks four requests the family must refuse, writes the THWC
// float32 result to OUTPUT and prints one JSON line.

#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <trtmc/video.hpp>
#include <vector>

namespace {

std::uint32_t positive(const char* argument) {
    const std::string text(argument);
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (consumed != text.size() || !value || value > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("clip dimensions must be positive int32 values");
    return static_cast<std::uint32_t>(value);
}

template <class Value>
std::vector<Value> read_exact(const char* path, std::size_t count) {
    std::vector<Value> values(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(count * sizeof(Value)));
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error(std::string(path) + " does not hold exactly the stated clip");
    return values;
}

void expect_refused(const std::string& name, const std::function<void()>& call) {
    try {
        call();
    } catch (const trtmc::Error& error) {
        if (error.code() == TRTMC_INVALID_ARGUMENT)
            return;
        throw std::runtime_error("request '" + name + "' failed with status " +
                                 std::to_string(static_cast<int>(error.code())) + ": " +
                                 error.what());
    }
    throw std::runtime_error("request '" + name + "' was not refused");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 10) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT FRAMES u8|f32 FRAME_COUNT HEIGHT WIDTH MASK OUTPUT\n";
        return 2;
    }
    try {
        const std::string format = argv[4];
        if (format != "u8" && format != "f32")
            throw std::invalid_argument("format must be u8 or f32");
        const auto frames = positive(argv[5]), height = positive(argv[6]),
                   width = positive(argv[7]);
        if (frames < 2)
            throw std::invalid_argument("a clip needs at least two frames");
        // Bound every product by division before forming it; reads take a streamsize.
        const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max());
        if (std::uint64_t{width} > limit / height ||
            std::uint64_t{height} * width > limit / 3 / sizeof(float) / frames)
            throw std::invalid_argument("clip size overflows host storage");
        const std::size_t plane = std::size_t{height} * width;
        const std::size_t values = std::size_t{frames} * plane * 3;
        std::vector<std::uint8_t> bytes;
        std::vector<float> floats;
        if (format == "u8")
            bytes = read_exact<std::uint8_t>(argv[3], values);
        else
            floats = read_exact<float>(argv[3], values);
        auto mask = read_exact<float>(argv[8], std::size_t{frames} * plane);
        const std::vector<float> empty(mask.size(), 0.0F);

        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<trtmc::MaskedVideoTextToVideo>();
        if (!task.config_fields().empty())
            throw std::runtime_error("videomae must expose no runtime Config fields");

        trtmc::MaskedVideoTextToVideoRequest request;
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            const auto offset = std::size_t{frame} * plane * 3;
            if (format == "u8")
                request.source.frames.emplace_back(
                    trtmc::Span<const std::uint8_t>(bytes.data() + offset, plane * 3), height,
                    width);
            else
                request.source.frames.emplace_back(
                    trtmc::Span<const float>(floats.data() + offset, plane * 3), height, width);
            request.source.timestamps_seconds.push_back(frame / 30.0);
        }
        request.mask = {{mask.data(), mask.size()}, frames, height, width};

        request.prompt = "a red car";
        expect_refused("prompt", [&] { (void)task.run(request); });
        request.prompt.clear();
        mask[0] = 1.0F - mask[0]; // the first tube becomes mixed
        expect_refused("mixed_tube", [&] { (void)task.run(request); });
        mask[0] = 1.0F - mask[0];
        request.mask.values = {empty.data(), empty.size()};
        expect_refused("empty_mask", [&] { (void)task.run(request); });
        request.mask.values = {mask.data(), mask.size()};
        auto shorter = request;
        shorter.source.frames.pop_back();
        shorter.source.timestamps_seconds.pop_back();
        shorter.mask = {{mask.data(), mask.size() - plane}, frames - 1, height, width};
        expect_refused("frame_count", [&] { (void)task.run(shorter); });

        const auto result = task.run(request);
        const auto output = result.frames();
        const auto times = result.timestamps_seconds();
        if (output.size() != frames || result.conditioned_prefix_frames() != 0 ||
            times.size() != frames)
            throw std::runtime_error("result must hold one frame and one time per source frame");
        std::ofstream file(argv[9], std::ios::binary);
        for (std::size_t frame = 0; frame < output.size(); ++frame) {
            const auto& image = output[frame];
            if (image.height != height || image.width != width || image.channels != 3 ||
                image.pixel_count != plane * 3)
                throw std::runtime_error("a result frame has another shape");
            if (times[frame] != request.source.timestamps_seconds[frame])
                throw std::runtime_error("result frames must keep the source timeline");
            for (std::size_t i = 0; i < image.pixel_count; ++i)
                if (!(image.pixels[i] >= 0.0F && image.pixels[i] <= 1.0F))
                    throw std::runtime_error("a result frame leaves [0, 1]");
            file.write(reinterpret_cast<const char*>(image.pixels),
                       static_cast<std::streamsize>(image.pixel_count * sizeof(float)));
        }
        file.close();
        if (!file)
            throw std::runtime_error(std::string("cannot write ") + argv[9]);

        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << "{\"task\":\"masked_video_text_to_video\",\"format\":\"" << format
                  << "\",\"frames\":" << output.size() << ",\"height\":" << height
                  << ",\"width\":" << width << ",\"channels\":3,\"conditioned_prefix_frames\":"
                  << result.conditioned_prefix_frames() << ",\"timestamps_seconds\":[";
        for (std::size_t frame = 0; frame < times.size(); ++frame)
            std::cout << (frame ? "," : "") << times[frame];
        std::cout << "],\"refused\":[\"prompt\",\"mixed_tube\",\"empty_mask\",\"frame_count\"],"
                  << "\"setup_ms\":" << result.setup_ms()
                  << ",\"inference_ms\":" << result.inference_ms() << "}\n";
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
