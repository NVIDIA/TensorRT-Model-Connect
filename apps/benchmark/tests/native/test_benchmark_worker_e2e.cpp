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

void write_bundle(const std::filesystem::path& path,
                  const std::string& task = "time_series_forecast",
                  const std::string& family = "fake") {
    static constexpr unsigned char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', '\x01', '\0'};
    const std::string header = Json{{"format", 1},
                                    {"family", family},
                                    {"task", task},
                                    {"backend", "fake"},
                                    {"sections",
                                     {{"runtime.json", {{"offset", 0}, {"length", 2}}},
                                      {"engine.plan", {{"offset", 2}, {"length", 4}}}}}}
                                   .dump();
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
    if (argc != 4) {
        std::cerr
            << "usage: test_benchmark_worker_e2e WORKER EXISTING_RUNTIME_ROOT SDK_RUNTIME_ROOT\n";
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
        check(result.at("observation_serialization_included") == false,
              "receipt identifies the corrected observation timing boundary");
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

        Json sdk = request;
        sdk["runtime_root"] = argv[3];
        sdk["operation"] = "generate";
        sdk["request"] = {{"prompt", "Hello"}, {"config", {{"suffix", "done"}}}};
        auto invoke = [&](bool expected_success = true) {
            {
                std::ofstream file(request_path);
                file << sdk;
            }
            const int status = std::system(command.c_str());
            check(expected_success ? status == 0 : status != 0, "SDK worker process status");
            std::ifstream file(output_path);
            Json value;
            file >> value;
            check(value.at("status") == (expected_success ? "completed" : "failed"),
                  "SDK success/failure receipt");
            return value;
        };
        for (const auto* id : {"text_continuation", "conditional_text_generation",
                               "corrupted_text_reconstruction", "text_summarization"}) {
            write_bundle(bundle_path, id, "text_fixture");
            const auto value = invoke();
            check(value.at("task") == id && value.at("observations").size() == 2,
                  "typed text route and count");
            const auto text = value.at("output_summary").at("text").get<std::string>();
            check(text.find("Hello") != std::string::npos && text.find("done") != std::string::npos,
                  "typed prompt and explicit family config reach benchmark output");
        }
        write_bundle(bundle_path, "text_continuation", "api_fixture");
        sdk["request"] = {{"prompt", "Hello"}};
        auto value = invoke();
        check(value.at("output_summary").at("decode_ms") == 4 &&
                  value.at("output_summary").at("prefill_ms") == 0.75,
              "worker does not invent unspecified generation controls");
        sdk["request"]["config"] = {{"max_new_tokens", 0},
                                    {"temperature", 0.0},
                                    {"emit_eos", false},
                                    {"suffix", ""},
                                    {"token_biases", Json::array({5, 0})},
                                    {"schedule", Json::array({0.25, 0.0})},
                                    {"labels", Json::array({"a", ""})}};
        value = invoke();
        check(value.at("output_summary").at("decode_ms") == 0 &&
                  value.at("output_summary").at("prefill_ms") == 0 &&
                  value.at("output_summary").at("text") == "Hello|a||5",
              "seven typed config kinds preserved");
        sdk["request"]["temperature"] = 1.0;
        invoke(false); // Flat and nested keys must not overwrite one another.
        sdk["request"] = {{"prompt", "Hello"}, {"config", {{"unknown", 1}}}};
        invoke(false);
        sdk["request"] = {{"prompt", "Hello"}, {"config", {{"temperature", "0.8"}}}};
        invoke(false);

        const auto image_path = runtime_root / "benchmark_input.ppm";
        {
            std::ofstream image(image_path, std::ios::binary);
            image << "P6\n1 1\n255\n";
            image.put('A');
            image.put('B');
            image.put('C');
        }
        write_bundle(bundle_path, "images_text_to_text", "language_fixture");
        sdk["request"] = {{"prompt", "Describe"}, {"image_path", image_path.string()}};
        for (bool include_assets : {false, true}) {
            sdk["measurement"]["asset_loading_included"] = include_assets;
            value = invoke();
            check(value.at("asset_loading_included") == include_assets &&
                      value.at("output_summary").at("text").get<std::string>().find("Describe") !=
                          std::string::npos,
                  "VLM input and asset timing policy preserved");
        }
        std::filesystem::remove(image_path);

        sdk["operation"] = "solve";
        sdk["request"] = {{"past_values", {1.0F, 2.0F, 3.0F}}, {"frequency", 2}};
        for (const auto* id : {"series_to_point_forecast", "series_to_quantile_forecast",
                               "series_to_point_and_quantile_forecast"}) {
            write_bundle(bundle_path, id, "numeric_fixture");
            value = invoke();
            const auto& item = value.at("output_summary");
            check(item.at("windows") == 1 && item.at("forecast_elements").get<int>() > 0,
                  "one forecast request remains one window");
            if (std::string(id) == "series_to_point_forecast")
                check(item.at("shape") == Json::array({2, 1}) && item.at("values").at(0) == 201,
                      "point output preserves horizon/channel axes and explicit frequency");
            else if (std::string(id) == "series_to_quantile_forecast")
                check(item.at("shape") == Json::array({2, 2, 1}) &&
                          item.at("quantile_levels") == Json::array({0.1, 0.9}),
                      "quantile axis is not a batch");
            else
                check(item.contains("point") && item.contains("quantiles"),
                      "joint forecast keeps both outputs");
        }
        sdk["request"]["observed_mask"] = {1, 2, 0};
        invoke(false);
        sdk["request"].erase("observed_mask");
        sdk["request"]["shape"] = {3, 1};
        value = invoke();
        check(value.at("output_summary").at("point").at("shape") == Json::array({2, 1}),
              "scalar typed history shape is an operand, not a config key");
        const auto scalar_request = sdk["request"];
        const auto scalar_measurement = sdk["measurement"];
        sdk["request"] = {{"past_values", {1.0, 2.0, 3.0, 4.0}}, {"shape", {2, 2}}};
        value = invoke();
        check(
            value.at("output_summary").at("point").at("shape") == Json::array({2, 2}),
            "explicit scalar channel axis is consumed instead of silently treating input as flat");
        sdk["request"]["shape"] = {3, 2};
        invoke(false);
        const Json batch_request = {{"items", Json::array({{{"past_values", {1.0, 2.0, 3.0}},
                                                            {"shape", {3, 1}},
                                                            {"observed_mask", {1, 1, 1}}},
                                                           {{"past_values", {8.0, 9.0}},
                                                            {"shape", {2, 1}},
                                                            {"config", {{"frequency", 2}}}}})}};
        sdk["measurement"]["warmup"] = 0;
        sdk["measurement"]["iterations"] = 1;
        for (const std::string id :
             {"batch_series_to_point_forecast", "batch_series_to_quantile_forecast",
              "batch_series_to_point_and_quantile_forecast"}) {
            write_bundle(bundle_path, id, "numeric_fixture");
            sdk["request"] = batch_request;
            value = invoke();
            const auto& output = value.at("output_summary");
            check(value.at("observations").size() == 1 && output.at("windows") == 2 &&
                      output.at("items").size() == 2,
                  "native forecast batch uses one invocation and keeps two ordered results");
            for (size_t i = 0; i < 2; ++i) {
                const auto& item = output.at("items").at(i);
                const bool joint = id == "batch_series_to_point_and_quantile_forecast";
                const auto& point = joint ? item.at("point") : item;
                const auto& quantile = joint ? item.at("quantiles") : item;
                const float base = i == 0 ? 1001 : 1208;
                const auto& axes = joint ? point : item;
                check(axes.at("horizon_steps") == Json::array({1, 3}) &&
                          axes.at("channel_names").empty() && axes.at("channel_units").empty(),
                      "batch forecast keeps real horizon steps and unspecified channel metadata");
                if (id != "batch_series_to_quantile_forecast")
                    check(point.at("shape") == Json::array({2, 1}) &&
                              point.at("values").at(0) == base,
                          "batch point result preserves per-item values and family frequency");
                if (id != "batch_series_to_point_forecast")
                    check(quantile.at("shape") == Json::array({3, 2, 1}) &&
                              quantile.at("quantile_levels") == Json::array({0.1, 0.5, 0.9}) &&
                              quantile.at("values").at(2) == base + 5,
                          "batch quantiles retain Q,H,C and do not replace the independent point");
            }
        }
        sdk["request"]["items"][0]["past_values"][0] = nullptr;
        sdk["request"]["items"][0]["observed_mask"][0] = 0;
        value = invoke();
        check(value.at("output_summary").at("items").at(0).at("point").at("values").at(0) == 990,
              "masked missing input crosses JSON without becoming an observed zero");
        sdk["request"]["items"][1]["config"]["frequency"] = 0.5;
        value = invoke(false);
        check(value.at("error").get<std::string>().find("batch item[1]") != std::string::npos &&
                  !value.contains("output_summary") && !value.contains("observations"),
              "later item config failure identifies the item and returns no successful output");
        sdk["request"] = batch_request;
        write_bundle(bundle_path, "series_to_point_forecast", "numeric_fixture");
        invoke(false);
        sdk["request"] = scalar_request;
        sdk["measurement"] = scalar_measurement;
        write_bundle(bundle_path, "series_to_point_and_quantile_forecast", "numeric_fixture");
        if (std::filesystem::exists("/dev/full")) {
            sdk["request"].erase("observed_mask");
            {
                std::ofstream file(request_path);
                file << sdk;
            }
            const auto failed_write = shell_quote(argv[1]) + " --request " +
                                      shell_quote(request_path.string()) + " --output /dev/full";
            check(std::system(failed_write.c_str()) != 0,
                  "output flush failure cannot report success");
        }
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
