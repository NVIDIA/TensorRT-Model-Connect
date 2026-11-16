/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++17 SDK consumer for the family-owned Nemotron 3 Diarization E2E.
// Reads mono float32 PCM, runs audio_to_timed_labels with an explicit
// threshold, releases the model, then prints the owned result in the CLI's
// detect --audio JSON layout.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <trtmc/audio.hpp>
#include <vector>

namespace {

std::uint32_t positive(const char* argument) {
    const std::string text(argument);
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (consumed != text.size() || !value || value > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("the sample rate must be a positive uint32 value");
    return static_cast<std::uint32_t>(value);
}

std::vector<float> read_pcm(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = file ? static_cast<std::streamoff>(file.tellg()) : std::streamoff{0};
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
        throw std::runtime_error("input must be nonempty mono float32 PCM");
    std::vector<float> samples(static_cast<std::size_t>(bytes) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(samples.data()), bytes);
    if (!file)
        throw std::runtime_error("failed to read the float32 PCM input");
    return samples;
}

void check(const trtmc::TimedLabelsResult& result) {
    const auto labels = result.labels();
    const auto scores = result.frame_scores();
    if (labels.empty() || scores.columns != labels.size() ||
        scores.rows * scores.columns != scores.values.size() ||
        result.score_kind() != TRTMC_SCORE_LOGIT || !(result.frame_seconds() > 0))
        throw std::runtime_error("timed labels result breaks its contract");
    for (const auto& interval : result.intervals())
        if (!(interval.end_seconds > interval.start_seconds) || interval.label >= labels.size())
            throw std::runtime_error("timed label interval breaks its contract");
    for (const float value : scores.values)
        if (!std::isfinite(value))
            throw std::runtime_error("timed label frame score is not finite");
}

void print(const trtmc::TimedLabelsResult& result) {
    auto& out = std::cout;
    out << "{\"task\":\"audio_to_timed_labels\",\"labels\":[";
    const auto labels = result.labels();
    for (std::size_t i = 0; i < labels.size(); ++i)
        out << (i ? "," : "") << '"' << labels[i] << '"';
    out << "],\"vocabulary_id\":\"" << result.vocabulary_id() << "\",\"intervals\":[";
    out.precision(std::numeric_limits<double>::max_digits10);
    const auto intervals = result.intervals();
    for (std::size_t i = 0; i < intervals.size(); ++i)
        out << (i ? "," : "") << "{\"start_seconds\":" << intervals[i].start_seconds
            << ",\"end_seconds\":" << intervals[i].end_seconds
            << ",\"label\":" << intervals[i].label << '}';
    out << "],\"frame_seconds\":" << result.frame_seconds()
        << ",\"score_kind\":\"logit\",\"frame_scores\":[";
    out.precision(std::numeric_limits<float>::max_digits10);
    const auto scores = result.frame_scores();
    for (std::uint64_t row = 0; row < scores.rows; ++row) {
        out << (row ? ",[" : "[");
        for (std::uint64_t column = 0; column < scores.columns; ++column)
            out << (column ? "," : "") << scores.values[row * scores.columns + column];
        out << ']';
    }
    out << "]}\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " BUNDLE RUNTIME_ROOT PCM_F32 SAMPLE_RATE\n";
        return 2;
    }
    try {
        static_assert(sizeof(float) == 4, "input uses float32");
        const auto sample_rate = positive(argv[4]);
        auto samples = read_pcm(argv[3]);

        // The model and its input go out of scope before the result is read.
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::AudioToTimedLabels>();
            const auto fields = task.config_fields();
            if (fields.size() != 1 || fields[0].name != "threshold")
                throw std::runtime_error(
                    "Nemotron 3 Diarization must expose exactly its threshold Config");
            // The family default, supplied explicitly through the typed Config.
            return task.run({{{samples.data(), samples.size()}, sample_rate, 1}},
                            {{"threshold", 0.5}});
        }();
        samples.clear();
        samples.shrink_to_fit();

        check(result);
        print(result);
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write timed labels output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
