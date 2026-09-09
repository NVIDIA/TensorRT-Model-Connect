/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cli/cli.h"
#include "cli/io.h"
#include "trtmc/core.hpp"
#include "trtmc/image.hpp"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <nlohmann/json_fwd.hpp>
#include <stdexcept>
#include <string_view>

namespace trtmc::cli::detail {

// Existing CLI parsing and formatting, shared by typed command groups.
std::string require_option(const Command&, const std::string&);
bool has_option(const Command&, const std::string&);
bool parse_bool(const std::string&, const std::string&);
std::int32_t int_option(const Command&, const std::string&, std::int32_t, std::int32_t);
float float_option(const Command&, const std::string&, float);
io::LoadedImage read_image(const std::string&);
std::vector<std::string> image_paths(const Command&);
std::vector<float> read_float32_file(const std::string&);
std::vector<std::string> read_lines(const std::string&);
std::vector<std::uint32_t> parse_seeds(const std::string&);
void require_finite(const std::vector<float>&, const std::string&);
void write_json(std::ostream&, const nlohmann::json&);
nlohmann::json text_json(const TextContinuationResult&);
nlohmann::json text_json(const TextResultView&);
nlohmann::json write_image(const ImageResultView&, const std::string&);
Config task_config(const Command&, const std::vector<ConfigField>&,
                   std::initializer_list<std::string_view> input_options);

template <typename Values>
void write_binary(const std::filesystem::path& path, const Values& values) {
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) /
                            sizeof(*values.data()))
        throw std::runtime_error("output is too large: " + path.string());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to create output: " + path.string());
    if (!values.empty()) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(*values.data())));
    }
    output.close();
    if (!output)
        throw std::runtime_error("failed to write output: " + path.string());
}

} // namespace trtmc::cli::detail

namespace trtmc::cli {

bool dispatch_sdk_features(const Command&, const Model&, std::string_view, std::ostream&);
bool dispatch_sdk_action(const Command&, const Model&, std::string_view, std::ostream&);
std::string_view replay_task_for_inputs(const Command&);
bool dispatch_sdk_replay(const Command&, const Model&, std::string_view, std::ostream&);
std::string_view audio_task_for_command(const Command&, const Model&);
bool dispatch_sdk_audio(const Command&, const Model&, std::string_view, std::ostream&);
std::string_view image_task_for_command(const Command&, const Model&);
bool dispatch_sdk_image(const Command&, const Model&, std::string_view, std::ostream&);
std::string_view video_task_for_command(const Command&, const Model&);
bool dispatch_sdk_video(const Command&, const Model&, std::string_view, std::ostream&);
std::string_view numeric_task_for_command(const Command&, const Model&);
bool dispatch_sdk_numeric(const Command&, const Model&, std::string_view, std::ostream&);
std::string_view perception_task_for_command(const Command&, const Model&);
bool dispatch_sdk_perception(const Command&, const Model&, std::string_view, std::ostream&);
std::string_view language_task_for_command(const Command&, const Model&);
bool dispatch_sdk_language(const Command&, const Model&, std::string_view, std::ostream&);

} // namespace trtmc::cli
