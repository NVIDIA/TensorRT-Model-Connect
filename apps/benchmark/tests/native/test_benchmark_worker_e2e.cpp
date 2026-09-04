/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace {

using Json = nlohmann::json;

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void write_bundle(const std::filesystem::path& path) {
    static constexpr unsigned char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', '\x01', '\0'};
    const std::string header =
        R"({"format":1,"family":"fake","task":"time_series_forecast","backend":"fake","sections":{"runtime.json":{"offset":0,"length":2},"engine.plan":{"offset":2,"length":4}}})";
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    const std::uint64_t length = header.size();
    for (int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((length >> shift) & 0xffU));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("{}PLAN", 6);
    if (!output)
        throw std::runtime_error("failed to write fake bundle");
}

std::string shell_quote(const std::string& value) {
    std::string result{"'"};
    for (const char character : value)
        result += character == '\'' ? "'\\''" : std::string(1, character);
    return result + "'";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_benchmark_worker_e2e WORKER RUNTIME_ROOT\n";
        return 2;
    }

    const std::filesystem::path runtime_root(argv[2]);
    const auto bundle_path = runtime_root / "benchmark_fake.bundle";
    const auto request_path = runtime_root / "benchmark_fake_request.json";
    const auto output_path = runtime_root / "benchmark_fake_result.json";
    std::filesystem::remove(bundle_path);
    std::filesystem::remove(request_path);
    std::filesystem::remove(output_path);

    try {
        write_bundle(bundle_path);
        const Json request = {
            {"schema_version", 2},
            {"case_name", "fake-forecast"},
            {"bundle", bundle_path.string()},
            {"runtime_root", runtime_root.string()},
            {"operation", "solve"},
            {"request", {{"past_values", {1.0F, 2.0F, 3.0F}}}},
            {"measurement",
             {{"warmup", 1}, {"iterations", 2}, {"timing_scope", "public_task_call_wall"}}},
        };
        {
            std::ofstream request_file(request_path);
            request_file << request << '\n';
            if (!request_file)
                throw std::runtime_error("failed to write worker request");
        }

        const std::string command = shell_quote(argv[1]) + " --request " +
                                    shell_quote(request_path.string()) + " --output " +
                                    shell_quote(output_path.string());
        check(std::system(command.c_str()) == 0, "worker process completed");

        std::ifstream output_file(output_path);
        Json result;
        output_file >> result;
        if (!output_file)
            throw std::runtime_error("failed to read worker result");

        check(result.at("schema_version") == "trtmc.benchmark-worker-result/v2", "result schema");
        check(result.at("status") == "completed", "result status");
        check(result.at("case_name") == "fake-forecast", "case identity");
        check(result.at("operation") == "solve", "public task operation");
        check(result.at("timing_scope") == "public_task_call_wall", "timing scope");
        check(result.at("warmup") == 1, "warmup count");
        check(result.at("iterations") == 2, "iteration count");
        check(result.at("load_ms").is_number() && result.at("load_ms").get<double>() >= 0.0,
              "task load measured");

        const auto& observations = result.at("observations");
        check(observations.is_array() && observations.size() == 2, "two observations returned");
        for (const auto& observation : observations) {
            check(observation.at("windows") == 1, "forecast window returned");
            check(observation.at("forecast_elements") == 3, "forecast values returned");
            check(observation.at("shape") == Json::array({1, 3}), "forecast shape returned");
            check(observation.at("runtime_e2e_wall_ms").is_number() &&
                      observation.at("runtime_e2e_wall_ms").get<double>() >= 0.0,
                  "task call measured");
        }
        const auto& summary = result.at("output_summary");
        check(summary.at("windows") == 1, "summary window returned");
        check(summary.at("forecast_elements") == 3, "summary values returned");
        check(summary.at("shape") == Json::array({1, 3}), "summary shape returned");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }

    std::filesystem::remove(bundle_path);
    std::filesystem::remove(request_path);
    std::filesystem::remove(output_path);
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
