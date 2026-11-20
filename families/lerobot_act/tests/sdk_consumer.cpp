/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C++ SDK consumer for the family-owned end-to-end test. It predicts one
 * chunk from a Float32 image, replays it through the action queue, refills and
 * resets the queue, and prints the chunk after releasing the model. */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <trtmc/action.hpp>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "../../../third_party/stb/stb_image.h"

namespace {

struct Image {
    std::vector<float> pixels;
    std::uint32_t height{0};
    std::uint32_t width{0};
};

Image read_image(const char* path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(path, &width, &height, &channels, 3);
    if (raw == nullptr)
        throw std::runtime_error(std::string("unable to decode the recorded image: ") + path);
    Image image;
    image.height = static_cast<std::uint32_t>(height);
    image.width = static_cast<std::uint32_t>(width);
    image.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);
    for (std::size_t index = 0; index < image.pixels.size(); ++index)
        image.pixels[index] = static_cast<float>(raw[index]) / 255.0F;
    stbi_image_free(raw);
    return image;
}

std::vector<float> read_floats(const char* path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto bytes = input ? static_cast<std::streamoff>(input.tellg()) : -1;
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
        throw std::runtime_error(std::string("unable to read the recorded state: ") + path);
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!input)
        throw std::runtime_error(std::string("unable to read the recorded state: ") + path);
    return values;
}

void json_string(std::string_view value) {
    std::cout << '"';
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\')
            std::cout << '\\' << static_cast<char>(byte);
        else if (byte < 32)
            std::cout << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                      << static_cast<unsigned>(byte) << std::dec << std::setfill(' ');
        else
            std::cout << static_cast<char>(byte);
    }
    std::cout << '"';
}

std::string_view text(trtmc_string_view value) {
    return {value.data, value.size};
}

// Exact equality: every queued step replays the chunk the same engine produced.
bool same_row(const trtmc::FloatMatrixView& chunk, std::uint64_t row,
              trtmc::Span<const float> step) {
    const auto* begin = chunk.values.data() + row * chunk.columns;
    return step.size() == chunk.columns && std::equal(begin, begin + chunk.columns, step.data());
}

bool same_matrix(const trtmc::FloatMatrixView& left, const trtmc::FloatMatrixView& right) {
    return left.rows == right.rows && left.columns == right.columns &&
           left.values.size() == right.values.size() &&
           std::equal(left.values.begin(), left.values.end(), right.values.begin());
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " BUNDLE RUNTIME_ROOT IMAGE STATE\n";
        return 2;
    }
    try {
        const auto image = read_image(argv[3]);
        const auto state = read_floats(argv[4]);
        std::uint64_t queue_steps = 0;
        auto chunk = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::ImageStateToActionChunk>();
            const auto queue = model.task<trtmc::ImageStateActionQueue>();
            require(task.config_fields().empty() && queue.config_fields().empty(),
                    "LeRobot ACT must expose no runtime Config fields");
            const trtmc::ImageStateObservation observation{
                trtmc::ImageInput({image.pixels.data(), image.pixels.size()}, image.height,
                                  image.width),
                {state.data(), state.size()}};
            auto result = task.run({observation});
            const auto actions = result.actions();
            const auto schema = result.view().actions.schema;
            require(actions.rows > 0 && actions.columns > 0 &&
                        actions.values.size() == actions.rows * actions.columns,
                    "action chunk shape is incomplete");
            require(std::all_of(actions.values.begin(), actions.values.end(),
                                [](float value) { return std::isfinite(value); }),
                    "action chunk contains a non-finite value");
            require(result.inference_ms() > 0 && text(schema.domain) == "lerobot.act" &&
                        text(schema.normalization) == "unnormalized",
                    "action chunk is mislabeled");

            // Every chunk step, one refill after the chunk is spent, and one step after reset.
            auto session = queue.create();
            for (std::uint64_t index = 0; index <= actions.rows + 1; ++index) {
                const bool started = index == 0 || index >= actions.rows;
                if (index == actions.rows + 1)
                    session.reset();
                const auto step = session.act(observation);
                require(step.started_new_chunk() == started &&
                            same_row(actions, index < actions.rows ? index : 0, step.values()) &&
                            (started ? step.inference_ms() > 0 : step.inference_ms() == 0),
                        "a queued step does not replay the action chunk");
                ++queue_steps;
                if (index == 0) {
                    // An independent chunk neither consumes nor replaces the live queue.
                    const auto again = task.run({observation});
                    require(same_matrix(actions, again.actions()), "repeated action chunk differs");
                }
            }
            return result;
        }();

        // The chunk result stays readable after the session and model are released.
        const auto actions = chunk.actions();
        const auto view = chunk.view().actions;
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"task\":\"image_state_to_action_chunk\",\"image_format\":\"float32\""
                  << ",\"rows\":" << actions.rows << ",\"columns\":" << actions.columns
                  << ",\"within_training_bounds\":" << std::boolalpha
                  << chunk.within_training_bounds() << ",\"queue_steps\":" << queue_steps
                  << ",\"schema\":{\"domain\":";
        json_string(text(view.schema.domain));
        std::cout << ",\"normalization\":";
        json_string(text(view.schema.normalization));
        std::cout << ",\"component_names\":" << view.schema.component_names.size
                  << ",\"units\":" << view.schema.units.size
                  << "},\"timestamps\":" << view.timestamps_seconds.size
                  << ",\"frame_spans\":" << view.frame_span_count << ",\"actions\":[";
        for (std::size_t index = 0; index < actions.values.size(); ++index) {
            if (index)
                std::cout << ',';
            std::cout << actions.values[index];
        }
        std::cout << "]}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write the action chunk");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
