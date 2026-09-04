/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<float> read_floats(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("cannot open float32 input: " + path);
    const auto bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
        throw std::runtime_error("invalid float32 input size: " + path);
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!input)
        throw std::runtime_error("cannot read float32 input: " + path);
    return values;
}

void write_floats(const std::string& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot create action output: " + path);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output)
        throw std::runtime_error("cannot write action output: " + path);
}

struct Image {
    std::vector<float> pixels;
    int width{0};
    int height{0};
};

Image read_image(const std::string& path) {
    Image image;
    int channels = 0;
    unsigned char* raw = stbi_load(path.c_str(), &image.width, &image.height, &channels, 3);
    if (raw == nullptr)
        throw std::runtime_error("cannot decode image: " + path);
    const auto count = static_cast<std::size_t>(image.width) * image.height * 3U;
    image.pixels.resize(count);
    for (std::size_t index = 0; index < count; ++index)
        image.pixels[index] = static_cast<float>(raw[index]) / 255.0F;
    stbi_image_free(raw);
    return image;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double peak_resident_memory_mib() {
    std::ifstream status("/proc/self/status");
    std::string label;
    while (status >> label) {
        if (label == "VmHWM:") {
            double kib = 0.0;
            std::string unit;
            status >> kib >> unit;
            return unit == "kB" ? kib / 1024.0 : 0.0;
        }
        std::string remainder;
        std::getline(status, remainder);
    }
    return 0.0;
}

std::pair<std::size_t, std::size_t> gpu_memory() {
    std::size_t free = 0;
    std::size_t total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess)
        throw std::runtime_error("cudaMemGetInfo failed");
    return {free, total};
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::invalid_argument(
                "usage: lerobot_act_qualification BUNDLE RUNTIME_ROOT IMAGE STATE ACTIONS");
        }
        const auto image = read_image(argv[3]);
        const auto state = read_floats(argv[4]);
        if (image.width != 640 || image.height != 480 || state.size() != 14)
            throw std::runtime_error("recorded ACT observation has the wrong shape");

        using Clock = std::chrono::steady_clock;
        const auto memory_before = gpu_memory();
        const auto load_started = Clock::now();
        auto task = trtmc::load_task(argv[1], argv[2]);
        const double startup_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - load_started).count();
        auto* control = dynamic_cast<trtmc::IRobotControl*>(task.get());
        if (control == nullptr)
            throw std::runtime_error("bundle does not implement IRobotControl");

        const trtmc::RobotObservation observation{{image.pixels.data(), image.pixels.size()},
                                                  image.height,
                                                  image.width,
                                                  3,
                                                  {state.data(), state.size()}};
        auto chunk = control->predict_action_chunk(observation);
        if (chunk.num_actions != 100 || chunk.action_dim != 14 || chunk.actions.size() != 1400)
            throw std::runtime_error("ACT result must be one 100x14 action chunk");
        write_floats(argv[5], chunk.actions);

        for (int index = 0; index < 2; ++index)
            (void)control->predict_action_chunk(observation);
        std::vector<double> inference_ms;
        for (int index = 0; index < 10; ++index)
            inference_ms.push_back(control->predict_action_chunk(observation).inference_ms);

        constexpr double control_hz = 50.0;
        const auto period = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(1.0 / control_hz));
        const auto control_start = Clock::now() + std::chrono::milliseconds(10);
        std::vector<Clock::time_point> emitted;
        int missed_deadlines = 0;
        volatile float checksum = 0.0F;
        for (int32_t step = 0; step < chunk.num_actions; ++step) {
            const auto target = control_start + step * period;
            std::this_thread::sleep_until(target);
            const auto actual = Clock::now();
            emitted.push_back(actual);
            if (actual > target + period)
                ++missed_deadlines;
            checksum += chunk.actions[static_cast<std::size_t>(step) * chunk.action_dim];
        }
        (void)checksum;

        std::vector<double> jitter_ms;
        for (std::size_t index = 1; index < emitted.size(); ++index) {
            const double interval =
                std::chrono::duration<double, std::milli>(emitted[index] - emitted[index - 1])
                    .count();
            jitter_ms.push_back(std::abs(interval - 1000.0 / control_hz));
        }
        const double elapsed =
            std::chrono::duration<double>(emitted.back() - emitted.front()).count();
        const double effective_hz = static_cast<double>(emitted.size() - 1) / elapsed;
        const auto memory_after = gpu_memory();
        const double gpu_delta =
            memory_before.first >= memory_after.first
                ? static_cast<double>(memory_before.first - memory_after.first) / (1024.0 * 1024.0)
                : 0.0;
        const double p50 = percentile(inference_ms, 0.50);
        const double p95 = percentile(inference_ms, 0.95);
        std::cout << nlohmann::json(
                         {
                             {"num_actions", chunk.num_actions},
                             {"action_dim", chunk.action_dim},
                             {"within_training_bounds", chunk.within_training_bounds},
                             {"chunk_inference_p50_ms", p50},
                             {"chunk_inference_p95_ms", p95},
                             {"action_step_capacity_hz", 1000.0 * chunk.num_actions / p50},
                             {"control_effective_hz", effective_hz},
                             {"control_p99_abs_jitter_ms", percentile(jitter_ms, 0.99)},
                             {"control_missed_deadlines", missed_deadlines},
                             {"gpu_memory_delta_mib", gpu_delta},
                             {"gpu_memory_total_mib",
                              static_cast<double>(memory_after.second) / (1024.0 * 1024.0)},
                             {"peak_resident_memory_mib", peak_resident_memory_mib()},
                             {"startup_ms", startup_ms},
                         })
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
