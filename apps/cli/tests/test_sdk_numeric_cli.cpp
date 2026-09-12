/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#include "trtmc/numeric.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>

namespace {
using nlohmann::json;
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void bundle(const std::filesystem::path& path, const std::string& mode) {
    const auto header = json{
        {"format", 1},
        {"family", "numeric_fixture"},
        {"task", mode},
        {"backend", "fake"},
        {"sections",
         json::object()}}.dump();
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        file.put(static_cast<char>((static_cast<uint64_t>(header.size()) >> shift) & 255U));
    file.write(header.data(), header.size());
}
void floats(const std::filesystem::path& path, std::initializer_list<float> values) {
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    for (const auto value : values)
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
struct Run {
    int status;
    std::string output, error;
};
Run run(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args)
        argv.push_back(arg.data());
    std::ostringstream output, error;
    const auto status = trtmc::cli::run(static_cast<int>(argv.size()), argv.data(), output, error);
    return {status, output.str(), error.str()};
}
void exercise(const std::filesystem::path& root) {
    const auto model = root / "numeric-cli-point.bundle", input = root / "numeric-cli-input.f32",
               mask = root / "numeric-cli-mask.f32";
    bundle(model, std::string(trtmc::SeriesToPointForecast::kTask));
    floats(input, {1, 2, 3, 4});
    floats(mask, {0, 1, 1, 1});
    const std::vector<std::string> base = {"trtmc",          "forecast",    model.string(),
                                           "--runtime-root", root.string(), "--input",
                                           input.string()};
    auto invoke = [&](std::initializer_list<std::string> extra) {
        auto args = base;
        args.insert(args.end(), extra.begin(), extra.end());
        return run(std::move(args));
    };
    const auto defaults = invoke({});
    check(defaults.status == 0,
          "forecast defaults to the family's primary Task without shape flags");
    if (defaults.status != 0)
        throw std::runtime_error(defaults.error);
    const auto value = json::parse(defaults.output);
    check(value.at("kind") == "point" && value.at("values") == json::array({1, 3}) &&
              value.at("shape") == json::array({2, 1}) &&
              value.at("axes") == json::array({"horizon", "channel"}) &&
              value.at("horizon_steps") == json::array({1, 3}),
          "point output retains values/shape and adds explicit canonical time/channel meaning");
    const auto configured = invoke({"--mask", mask.string(), "--frequency", "2"});
    check(
        configured.status == 0 && json::parse(configured.output).at("values")[0] == 190,
        "existing float mask file and frequency option cross the typed API through family config");
    const auto duplicate = invoke({"--frequency", "1", "--set", "frequency=2"});
    check(duplicate.status != 0 && duplicate.output.empty(),
          "frequency options do not overwrite duplicate family entries");
    const auto wrong_task = invoke({"--task", "latent_to_token_logits"});
    check(wrong_task.status != 0 && wrong_task.output.empty(),
          "forecast cannot silently execute a different Task kind");
    const auto bad_mask = root / "numeric-cli-bad-mask.f32";
    floats(bad_mask, {1, 0});
    const auto invalid = invoke({"--mask", bad_mask.string()});
    check(invalid.status != 0 && invalid.output.empty(),
          "observed mask size is checked against flat element count");
    const auto fractional_mask = root / "numeric-cli-fractional-mask.f32";
    floats(fractional_mask, {1, 0.5F, 1, 1});
    const auto fractional = invoke({"--mask", fractional_mask.string()});
    check(fractional.status != 0 && fractional.output.empty(),
          "observed flags are not silently thresholded from arbitrary weights");
    const auto quantiles = invoke({"--task", std::string(trtmc::SeriesToQuantileForecast::kTask)});
    check(quantiles.status == 0, "forecast supports an explicitly selected quantile Task");
    if (quantiles.status == 0) {
        const auto q = json::parse(quantiles.output);
        check(q.at("kind") == "quantiles" && q.at("shape") == json::array({2, 2, 1}) &&
                  q.at("quantile_levels") == json::array({0.1, 0.9}) &&
                  q.at("axes") == json::array({"quantile", "horizon", "channel"}),
              "quantile forecast preserves levels and never disguises Q as batch");
    }
    const auto joint =
        invoke({"--task", std::string(trtmc::SeriesToPointAndQuantileForecast::kTask)});
    check(joint.status == 0, "forecast supports the joint Task");
    if (joint.status == 0) {
        const auto j = json::parse(joint.output);
        check(j.at("kind") == "point_and_quantiles" && j.at("point").at("values")[0] == 1001 &&
                  j.at("quantiles").at("values")[2] == 1006 && !j.contains("values"),
              "joint JSON has separately interpretable point and quantile outputs");
    }
    const auto distribution =
        invoke({"--task", std::string(trtmc::SeriesToRegressionDistribution::kTask), "--set",
                "distribution=student_t"});
    check(distribution.status == 0,
          "existing forecast command can expose a regression checkpoint's real result");
    if (distribution.status == 0) {
        const auto r = json::parse(distribution.output);
        check(r.at("kind") == "regression_distribution" && r.at("distribution") == "student_t" &&
                  r.at("parameters").at("degrees_of_freedom") == json::array({3, 4}) &&
                  r.at("target_count") == 2 && !r.contains("shape") && !r.contains("horizon_steps"),
              "regression target parameters are named rather than relabeled as a future forecast");
    }
    const auto channels = root / "numeric-cli-two-channels.bundle";
    const auto targets = root / "numeric-cli-targets.bundle";
    bundle(targets, std::string(trtmc::SeriesToRegressionValues::kTask));
    auto target_args = base;
    target_args[2] = targets.string();
    target_args.insert(target_args.end(), {"--mask", mask.string(), "--set", "scale=2.0"});
    const auto target_output = run(target_args);
    check(target_output.status == 0,
          "forecast command executes the explicit target-regression Task");
    if (target_output.status == 0) {
        const auto value = json::parse(target_output.output);
        check(value.at("kind") == "regression_values" &&
                  value.at("values") == json::array({18, 1}) && value.at("target_count") == 2 &&
                  value.at("axes") == json::array({"target"}) && value.at("target_names").empty() &&
                  value.at("target_units").empty() && !value.contains("distribution") &&
                  !value.contains("horizon_steps"),
              "target regression preserves all values without invented forecast/distribution "
              "metadata");
    }
    target_args.insert(target_args.end(), {"--set", "distribution=normal"});
    const auto unsupported_distribution = run(target_args);
    check(unsupported_distribution.status != 0 && unsupported_distribution.output.empty(),
          "a deterministic regression provider cannot silently accept a distribution selector");
    bundle(channels, "flat_channels_two");
    auto channel_args = base;
    channel_args[2] = channels.string();
    channel_args.insert(channel_args.end(),
                        {"--task", std::string(trtmc::SeriesToPointForecast::kTask)});
    const auto multivariate = run(channel_args);
    check(multivariate.status == 0 &&
              json::parse(multivariate.output).at("shape") == json::array({2, 2}),
          "CLI does not guess single-channel shape for flat history");
    const auto default_one = root / "numeric-cli-default-one.bundle";
    bundle(default_one, "frequency_default_one");
    auto one_args = base;
    one_args[2] = default_one.string();
    one_args.insert(one_args.end(), {"--task", std::string(trtmc::SeriesToPointForecast::kTask)});
    const auto omitted = run(one_args);
    one_args.insert(one_args.end(), {"--frequency", "0"});
    const auto zero = run(one_args);
    check(omitted.status == 0 && zero.status == 0 &&
              json::parse(omitted.output).at("values")[0] == 101 &&
              json::parse(zero.output).at("values")[0] == 1,
          "CLI missing frequency uses family default while explicit zero overrides it");
    const auto batch = invoke({"--task", std::string(trtmc::BatchSeriesToPointForecast::kTask)});
    check(batch.status != 0 && batch.output.empty(),
          "one input file is not a fabricated native batch");

    const auto branch = root / "numeric-cli-packed.f32",
               denoise_trunk = root / "numeric-cli-denoise.f32",
               decode_trunk = root / "numeric-cli-decode.f32";
    floats(branch, {1, 2, 9, 8, 3, 4, 7, 6});
    floats(denoise_trunk, {0.25F, 3, 0});
    floats(decode_trunk, {0.25F, 3, 0.8F});
    const std::vector<std::string> solve = {
        "trtmc",    "solve",         model.string(), "--runtime-root",      root.string(),
        "--branch", branch.string(), "--trunk",      denoise_trunk.string()};
    const auto denoised = run(solve);
    check(denoised.status == 0,
          "existing solve files work without shape flags through native packed input");
    if (denoised.status == 0) {
        const auto d = json::parse(denoised.output);
        check(d.at("task") == trtmc::LatentDenoisingStep::kTask && d.at("dim") == 2 &&
                  d.at("values") == json::array({12.25, 2, 3, 4}) &&
                  d.at("shape") == json::array({2, 2}),
              "solve preserves packed input, timestep, guidance and denoised row width");
    }
    auto decode = solve;
    decode.back() = decode_trunk.string();
    const auto decoded = run(decode);
    check(decoded.status == 0, "legacy decoder selector threshold chooses the token-logit Task");
    if (decoded.status == 0) {
        const auto d = json::parse(decoded.output);
        check(d.at("task") == trtmc::LatentToTokenLogits::kTask && d.at("dim") == 3 &&
                  d.at("shape") == json::array({2, 3}) && d.at("values")[0] == 12.25 &&
                  d.at("vocabulary_id") == "fixture-vocabulary",
              "decode output width differs from latent width without bundle config guessing");
    }
    auto conflicting = decode;
    conflicting.insert(conflicting.end(),
                       {"--task", std::string(trtmc::LatentDenoisingStep::kTask)});
    const auto conflict = run(conflicting);
    check(conflict.status != 0 && conflict.output.empty(),
          "explicit solve Task cannot ignore the existing decoder selector");
    auto duplicate_guide = solve;
    duplicate_guide.insert(duplicate_guide.end(), {"--set", "self_cond_cfg_scale=2"});
    const auto duplicate_control = run(duplicate_guide);
    check(duplicate_control.status != 0 && duplicate_control.output.empty(),
          "trunk guidance does not overwrite a duplicate family config");
    const auto short_branch = root / "numeric-cli-short-packed.f32";
    floats(short_branch, {1, 2, 3, 4});
    auto wrong_length = solve;
    wrong_length[6] = short_branch.string();
    const auto short_input = run(wrong_length);
    check(short_input.status != 0 && short_input.output.empty(),
          "family rejects native packed length rather than guessing rows or adding self-condition");
    const auto bad_trunk = root / "numeric-cli-short-trunk.f32";
    floats(bad_trunk, {0.25F, 3});
    auto bad_control = solve;
    bad_control.back() = bad_trunk.string();
    check(run(bad_control).status != 0, "solve requires all three existing control values");
    const auto nan_trunk = root / "numeric-cli-nan-trunk.f32";
    floats(nan_trunk, {0.25F, 3, std::numeric_limits<float>::quiet_NaN()});
    bad_control.back() = nan_trunk.string();
    check(run(bad_control).status != 0,
          "non-finite selector is not silently interpreted as a denoising mode");
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        exercise(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    return failures ? 1 : 0;
}
