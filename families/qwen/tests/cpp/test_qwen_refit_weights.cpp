/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/refit_weights.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unistd.h>

namespace {
using json = nlohmann::json;

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void write_bundle(const std::filesystem::path& path, const json& metadata, bool payload = true) {
    const auto text = metadata.dump();
    json header{{"format", 1},
                {"family", "qwen"},
                {"task", "text_continuation"},
                {"backend", "trt"},
                {"sections", {{"refit.json", {{"offset", 0}, {"length", text.size()}}}}}};
    if (payload)
        header["sections"]["refit_weights"] = {{"offset", text.size()}, {"length", 8}};
    const auto encoded = header.dump();
    std::ofstream output(path, std::ios::binary);
    output.write("BUNDLE\1\0", 8);
    for (int shift = 0; shift != 64; shift += 8)
        output.put(static_cast<char>((encoded.size() >> shift) & 255));
    output << encoded << text;
    if (payload) {
        const float data[] = {2, 3};
        output.write(reinterpret_cast<const char*>(data), sizeof(data));
    }
}
} // namespace

int main() {
    char directory[] = "/tmp/qwen_refit_XXXXXX";
    if (!mkdtemp(directory))
        return 1;
    const auto path = std::filesystem::path(directory) / "model.bundle";
    try {
        json valid{
            {"version", 1},
            {"engines", {{"engine.plan", {"w"}}, {"prefill.plan", {"w"}}}},
            {"weights", {{"w", {{"dtype", "F32"}, {"shape", {2}}, {"offset", 0}, {"nbytes", 8}}}}}};
        write_bundle(path, valid);
        {
            trtmc::qwen::QwenRefitSource source{trtmc::BundleReader(path.string())};
            auto moved = std::move(source);
            const auto* weights = moved.weights_for("engine.plan");
            check(weights && weights->size() == 1, "exact engine bindings");
            float values[2];
            std::memcpy(values, weights->at("w").data, sizeof(values));
            check(values[0] == 2 && values[1] == 3 && weights->at("w").count == 2,
                  "mapped weight values survive move and reader destruction");
            check(moved.weights_for("prefill.plan")->at("w").data == weights->at("w").data,
                  "both engines borrow the same payload");
        }
        auto rejects = [&](const json& manifest, bool payload = true) {
            write_bundle(path, manifest, payload);
            try {
                trtmc::qwen::QwenRefitSource invalid{trtmc::BundleReader(path.string())};
            } catch (const std::exception&) {
                return;
            }
            throw std::runtime_error("invalid refit metadata was accepted");
        };
        auto bad = valid;
        bad["version"] = 2;
        rejects(bad);
        bad = valid;
        bad["weights"]["w"]["shape"] = {std::numeric_limits<std::int64_t>::max(), 2};
        rejects(bad);
        bad = valid;
        bad["weights"]["w"]["shape"] = {-1};
        rejects(bad);
        bad = valid;
        bad["weights"]["w"]["offset"] = 1;
        rejects(bad);
        bad = valid;
        bad["engines"]["engine.plan"] = {"missing"};
        rejects(bad);
        bad = valid;
        bad["engines"]["engine.plan"] = {"w", "w"};
        rejects(bad);
        rejects(valid, false);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::filesystem::remove_all(directory);
        return 1;
    }
    std::filesystem::remove_all(directory);
    return 0;
}
