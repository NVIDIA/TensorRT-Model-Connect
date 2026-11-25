/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/video.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        auto task = model.task<trtmc::TextToAudioVideo>();
        nlohmann::json requests;
        std::cin >> requests;
        auto results = nlohmann::json::array();
        for (const auto& request : requests) {
            trtmc::Config config;
            config.add("seed", request.at("seed").get<int64_t>());
            const auto result = task.run({request.at("prompt").get<std::string>()}, config);
            std::ofstream video(request.at("video").get<std::string>(), std::ios::binary);
            std::ofstream wave(request.at("audio").get<std::string>(), std::ios::binary);
            for (const auto& frame : result.frames()) {
                if (frame.height != 480 || frame.width != 832 || frame.channels != 3 ||
                    frame.pixel_count != 480 * 832 * 3)
                    throw std::runtime_error("Unexpected Pika frame geometry");
                video.write(reinterpret_cast<const char*>(frame.pixels),
                            frame.pixel_count * sizeof(float));
            }
            const auto audio = result.audio();
            wave.write(reinterpret_cast<const char*>(audio.samples.data()),
                       audio.samples.size() * sizeof(float));
            video.close();
            wave.close();
            if (!video)
                throw std::runtime_error("Could not write Pika video: " +
                                         request.at("video").get<std::string>());
            if (!wave)
                throw std::runtime_error("Could not write Pika audio: " +
                                         request.at("audio").get<std::string>());
            const auto timestamps = result.timestamps_seconds();
            results.push_back(
                {{"frames", result.frames().size()},
                 {"height", 480},
                 {"width", 832},
                 {"sample_rate", audio.sample_rate.value_or(0)},
                 {"channels", audio.channels},
                 {"audio_count", audio.samples.size()},
                 {"audio_start_seconds", result.audio_start_seconds()},
                 {"conditioned_prefix_frames", result.video_view().conditioned_prefix_frames},
                 {"timestamps", std::vector<double>(timestamps.begin(), timestamps.end())}});
        }
        std::cout << results.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
