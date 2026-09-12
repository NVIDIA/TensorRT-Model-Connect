/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

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
        write_bundle(bundle_path, "unconditional_text_generation", "text_fixture");
        sdk["request"] = Json::object();
        const auto unconditional = invoke().at("output_summary");
        check(unconditional.at("text") == "unconditional!" &&
                  unconditional.at("token_ids") == Json::array({13}) &&
                  unconditional.at("output_tokens") == 1,
              "unconditional generation invokes the no-request Task and counts real result tokens");
        check(unconditional.at("segments").at(0).at("token_ids") == Json::array({13, 99}) &&
                  unconditional.at("segments").at(0).at("end_seconds") == 0.5,
              "text result segments are preserved independently of the actual output-token count");
        sdk["request"] = {{"config", {{"suffix", ""}}}};
        check(invoke().at("output_summary").at("text") == "unconditional",
              "unconditional generation preserves explicit empty Config");
        sdk["request"]["prompt"] = "";
        invoke(false);

        write_bundle(bundle_path, "text_translation", "text_fixture");
        sdk["operation"] = "translate";
        sdk["expected_family"] = "text_fixture";
        sdk["expected_task"] = "text_translation";
        sdk["request"] = {{"source_text", "Bonjour"}};
        auto translated = invoke();
        check(translated.at("task") == "text_translation" &&
                  translated.at("operation") == "translate" &&
                  translated.at("output_summary").at("text") ==
                      "translation:fixed-src->en:Bonjour!",
              "translation leaves absent source and target languages to the family");
        for (const auto& observation : translated.at("observations")) {
            check(observation.at("token_ids") == Json::array({14}) &&
                      observation.at("output_tokens") == 1 && observation.at("prefill_ms") == 2 &&
                      observation.at("decode_ms") == 3 &&
                      observation.at("runtime_e2e_wall_ms").get<double>() >= 0,
                  "translation preserves text tokens, family stages and Task-call timing");
        }
        const std::string source_text("Bon\0jour", 8);
        sdk["request"] = {{"source_text", source_text},
                          {"source_language", "fr"},
                          {"target_language", "de"},
                          {"config", {{"suffix", "done"}}}};
        translated = invoke();
        check(translated.at("output_summary").at("text") ==
                  "translation:fr->de:" + source_text + "done",
              "translation preserves explicit typed languages, embedded NUL and family Config");
        for (const auto* language : {"source_language", "target_language"}) {
            for (const auto& invalid : Json::array({nullptr, "", false, 3})) {
                sdk["request"] = {{"source_text", "Bonjour"}, {language, invalid}};
                check(invoke(false).at("error").get<std::string>().find("non-empty string") !=
                          std::string::npos,
                      "present invalid language is rejected instead of becoming absent");
            }
        }
        sdk["request"] = {{"source_text", "Bonjour"}, {"config", {{"target_language", "de"}}}};
        invoke(false); // Language operands cannot be smuggled into family Config.
        sdk["request"] = {{"source_text", "Bonjour"}};
        sdk["runtime_root"] = (runtime_root / "missing-runtime").string();
        sdk["expected_family"] = "other_family";
        check(invoke(false).at("error").get<std::string>().find("bundle identity mismatch") !=
                  std::string::npos,
              "wrong family is rejected before any DSO load");
        sdk["expected_family"] = "text_fixture";
        sdk["expected_task"] = "text_generation";
        check(invoke(false).at("error").get<std::string>().find("bundle identity mismatch") !=
                  std::string::npos,
              "old primary Task cache cannot stand in for a different requested Task");
        sdk.erase("expected_task");
        check(invoke(false).at("error").get<std::string>().find("requires both") !=
                  std::string::npos,
              "partial expected bundle identity is invalid");
        sdk.erase("expected_family");
        sdk["runtime_root"] = argv[3];
        sdk["operation"] = "generate";
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

        write_bundle(bundle_path, "series_to_regression_distribution", "numeric_fixture");
        sdk["operation"] = "regress";
        sdk["request"] = {{"past_values", {1.0, nullptr, 3.0, 4.0}},
                          {"shape", {2, 2}},
                          {"observed_mask", {1, 0, 1, 1}}};
        for (const auto* distribution : {"normal", "student_t", "negative_binomial"}) {
            sdk["request"]["config"] = {{"distribution", distribution}};
            value = invoke();
            const auto& output = value.at("output_summary");
            const auto parameter_count = std::string(distribution) == "student_t" ? 3 : 2;
            check(output.at("distribution") == distribution && output.at("target_count") == 2 &&
                      output.at("axes") == Json::array({"target"}) &&
                      output.at("regression_targets") == 2 &&
                      output.at("parameter_elements") == parameter_count * 2 &&
                      output.at("parameters").size() == static_cast<std::size_t>(parameter_count) &&
                      output.at("target_names").empty() && output.at("target_units").empty() &&
                      !output.contains("forecast_elements") && !output.contains("horizon_steps"),
                  "regression distribution remains named per-target parameters without invented "
                  "forecast axes");
            const auto& first = output.at("parameters").at(0);
            if (std::string(distribution) == "normal")
                check(first.at("name") == "scale" && first.at("values") == Json::array({0.5, 1.0}),
                      "normal parameter order and raw scale values preserved");
            else if (std::string(distribution) == "student_t")
                check(first.at("name") == "degrees_of_freedom" &&
                          first.at("values") == Json::array({3, 4}),
                      "Student-t degrees of freedom are not dropped");
            else
                check(first.at("name") == "total_count" &&
                          output.at("parameters").at(1).at("values") == Json::array({-1, 1}),
                      "negative-binomial logits are not normalized");
        }
        sdk["request"]["observed_mask"] = {1, 1, 1, 1};
        invoke(false); // Null history must be explicitly masked out.
        sdk["request"]["observed_mask"] = {1, 0, 1, 1};
        sdk["request"]["shape"] = {3, 2};
        invoke(false);
        sdk["request"]["shape"] = {2, 2};
        sdk["request"]["config"] = {{"distribution", "unknown"}};
        invoke(false);

        std::vector<std::filesystem::path> latent_files;
        auto floats = [&](const char* name, std::initializer_list<float> values) {
            const auto path = runtime_root / name;
            std::ofstream file(path, std::ios::binary);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file.write(reinterpret_cast<const char*>(values.begin()),
                       static_cast<std::streamsize>(values.size() * sizeof(float)));
            file.close();
            latent_files.push_back(path);
            return path.string();
        };
        const auto condition = floats("benchmark_condition.f32", {1, 2, 3, 4});
        const auto mask = floats("benchmark_condition_mask.f32", {1, 0.5F});
        const auto initial = floats("benchmark_initial.f32", {9, 8, 7, 6});
        const auto noise = floats("benchmark_noise.f32", {10, 11, 12, 13, 14, 15, 16, 17});
        const auto steps = floats("benchmark_steps.f32", {0, 0.25F, 0.75F, 1});
        const auto branch = floats("benchmark_branch.f32", {1, 2, 9, 8, 3, 4, 7, 6});
        const auto denoise_trunk = floats("benchmark_denoise_trunk.f32", {0.25F, 3, 0});
        const auto logits_trunk = floats("benchmark_logits_trunk.f32", {0.25F, 3, 0.8F});
        write_bundle(bundle_path, "latent_conditioned_text_generation", "numeric_fixture");
        sdk["operation"] = "generate";
        const Json conditioned{{"condition_latents_path", condition}, {"condition_mask_path", mask},
                               {"initial_latents_path", initial},     {"sde_noises_path", noise},
                               {"sampling_steps_path", steps},        {"prompt", "kept"}};
        for (const bool include_assets : {false, true}) {
            sdk["measurement"]["asset_loading_included"] = include_assets;
            sdk["request"] = conditioned;
            value = invoke();
            check(value.at("output_summary").at("text") == "conditioned" &&
                      value.at("output_summary").at("token_ids") ==
                          Json::array({1, 50, 9, 17, 25, 4}) &&
                      value.at("output_summary").at("output_tokens") == 6 &&
                      value.at("asset_loading_included") == include_assets,
                  "condition, float mask, initial state, SDE noise, schedule and prompt reach one "
                  "typed generation call");
        }
        sdk["request"]["config"] = {{"sampling_steps", {0.0, 0.5, 1.0}}};
        invoke(false); // The raw schedule cannot overwrite a declared Config entry.
        sdk["request"] = conditioned;
        sdk["request"].erase("condition_mask_path");
        invoke(false);
        sdk["request"] = {{"condition_latents_path", condition}, {"condition_mask_path", mask}};
        check(
            invoke().at("output_summary").at("token_ids") == Json::array({1, 50, -1, -1, 50, 0}),
            "missing optional replay operands retain family initialization and schedule defaults");

        write_bundle(bundle_path, "latent_replay_to_text", "numeric_fixture");
        sdk["request"] = {{"initial_latents_path", initial}};
        value = invoke();
        check(value.at("output_summary").at("text") == "|replayed" &&
                  value.at("output_summary").at("token_ids") == Json::array({9, -1, 50}),
              "initial-only replay does not invent prompt, noise or schedule");
        sdk["request"] = {
            {"sde_noises_path", noise}, {"sampling_steps_path", steps}, {"prompt", "kept"}};
        check(invoke().at("output_summary").at("token_ids") == Json::array({-1, 17, 25}),
              "SDE-only replay passes through without synthesizing initial latents");
        sdk["request"].erase("sampling_steps_path");
        invoke(false); // Family validates SDE count against its resolved schedule.
        sdk["request"] = {{"prompt", "kept"}};
        invoke(false);
        sdk["request"] = conditioned;
        invoke(false); // Conditioned input cannot silently select another Task.

        for (const bool decoder : {false, true}) {
            write_bundle(bundle_path, decoder ? "latent_to_token_logits" : "latent_denoising_step",
                         "numeric_fixture");
            sdk["operation"] = decoder ? "decode_logits" : "denoise";
            const auto& trunk = decoder ? logits_trunk : denoise_trunk;
            for (const bool include_assets : {false, true}) {
                sdk["measurement"]["asset_loading_included"] = include_assets;
                sdk["request"] = {{"branch_path", branch}, {"trunk_path", trunk}};
                value = invoke();
                const auto& output = value.at("output_summary");
                check(output.at("shape") == (decoder ? Json::array({2, 3}) : Json::array({2, 2})) &&
                          output.at("values").at(0) == 12.25 && !output.contains("output_tokens") &&
                          output.at(decoder ? "logit_elements" : "latent_elements") ==
                              (decoder ? 6 : 4),
                      "native-packed latent call preserves guidance and output width without "
                      "inventing token counts");
                if (decoder)
                    check(output.at("vocabulary_id") == "fixture-vocabulary" &&
                              output.at("axes") == Json::array({"position", "vocabulary_id"}),
                          "decoder logits retain vocabulary identity and axes");
            }
            sdk["request"] = {{"latents_path", condition},
                              {"shape", {2, 2}},
                              {"self_condition_path", initial},
                              {"timestep", 0.25},
                              {"config", {{"self_cond_cfg_scale", 3.0}}}};
            check(invoke().at("output_summary").at("values").at(0) == 12.25,
                  "logical latents with separate self-condition match native-packed call");
            sdk["request"]["shape"] = {3, 2};
            invoke(false);
            sdk["request"] = {{"branch_path", branch},
                              {"trunk_path", decoder ? denoise_trunk : logits_trunk}};
            invoke(false);
            sdk["request"] = {{"branch_path", branch},
                              {"trunk_path", trunk},
                              {"config", {{"self_cond_cfg_scale", 3.0}}}};
            invoke(false); // Raw guidance cannot overwrite explicit Config.
            sdk["request"] = {
                {"branch_path", branch}, {"trunk_path", trunk}, {"self_condition_path", initial}};
            invoke(false);
        }
        for (const auto& path : latent_files)
            std::filesystem::remove(path);

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
