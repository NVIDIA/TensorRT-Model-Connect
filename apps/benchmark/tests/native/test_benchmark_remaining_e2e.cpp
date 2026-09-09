/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace {
using Json = nlohmann::json;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
std::string quote(const std::string& value) {
    std::string output{"'"};
    for (char c : value)
        output += c == '\'' ? "'\\''" : std::string(1, c);
    return output + "'";
}
void bundle(const std::filesystem::path& path, const std::string& task, const std::string& family) {
    const auto header = Json{
        {"format", 1},
        {"family", family},
        {"task", task},
        {"backend", "fake"},
        {"sections",
         Json::object()}}.dump();
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    output << header;
}
void image(const std::filesystem::path& path, int width, unsigned char red) {
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << "P6\n" << width << " 1\n255\n";
    for (int i = 0; i < width; ++i) {
        output.put(static_cast<char>(red));
        output.put(0);
        output.put(0);
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_benchmark_remaining_e2e WORKER SDK_RUNTIME_ROOT\n";
        return 2;
    }
    try {
        const std::filesystem::path runtime(argv[2]);
        const auto model = runtime / "benchmark_remaining.bundle";
        const auto input = runtime / "benchmark_remaining_request.json";
        const auto output = runtime / "benchmark_remaining_result.json";
        auto artifact = output;
        artifact.replace_extension(".disparity.f32");
        const auto left = runtime / "benchmark_remaining_left.ppm";
        const auto right = runtime / "benchmark_remaining_right.ppm";
        image(left, 2, 255);
        image(right, 2, 0);
        Json request{
            {"schema_version", 2},
            {"case_name", "remaining-sdk"},
            {"bundle", model.string()},
            {"runtime_root", runtime.string()},
            {"operation", "disparity"},
            {"request", {{"left_image_path", left.string()}, {"right_image_path", right.string()}}},
            {"measurement",
             {{"warmup", 1}, {"iterations", 2}, {"timing_scope", "public_task_call_wall"}}}};
        const auto command = quote(argv[1]) + " --request " + quote(input.string()) + " --output " +
                             quote(output.string());
        auto run = [&](bool success = true) {
            {
                std::ofstream file(input);
                file << request;
            }
            const int status = std::system(command.c_str());
            check(success ? status == 0 : status != 0, "worker process status");
            std::ifstream file(output);
            Json result;
            file >> result;
            check(result.at("status") == (success ? "completed" : "failed"), "receipt status");
            if (!success)
                check(!result.contains("observations"), "failure has no completed observations");
            return result;
        };
        bundle(model, "stereo_images_to_disparity", "perception_fixture");
        for (bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            const auto result = run();
            const auto& summary = result.at("output_summary");
            check(result.at("observations").size() == 2 &&
                      result.at("observation_serialization_included") == false &&
                      summary.at("disparity_pixels") == 2 && summary.at("height") == 1 &&
                      summary.at("width") == 2 && summary.at("element_count") == 2 &&
                      summary.at("units") == "pixels" &&
                      summary.at("convention") == "x_left_minus_x_right",
                  "stereo metadata, call count and timing boundary preserved");
            std::ifstream file(artifact, std::ios::binary);
            float values[2]{};
            file.read(reinterpret_cast<char*>(values), sizeof(values));
            check(file.good() && values[0] == 1 && values[1] == 1 &&
                      std::filesystem::file_size(artifact) == sizeof(values),
                  "final actual disparity map written exactly once as F32");
        }
        if (std::filesystem::exists("/dev/full")) {
            std::filesystem::remove(artifact);
            std::filesystem::create_symlink("/dev/full", artifact);
            run(false);
            std::filesystem::remove(artifact);
        }
        std::filesystem::create_directory(artifact);
        run(false);
        std::filesystem::remove(artifact);
        request["request"]["config"] = {{"unknown", 1}};
        run(false);
        request["request"].erase("config");
        image(right, 1, 0);
        run(false);
        image(right, 2, 0);
        auto select = [&](const char* task, const char* family, const char* operation, Json args) {
            bundle(model, task, family);
            request["operation"] = operation;
            request["request"] = std::move(args);
        };
        select("image_to_class_scores", "features_fixture", "classify",
               {{"image_path", left.string()}});
        auto result = run();
        auto summary = result.at("output_summary");
        check(summary.at("scores") == Json::array({18, 1}) && summary.at("score_kind") == "logit" &&
                  summary.at("top_class") == 0 && summary.at("top_score") == 18 &&
                  summary.at("labels") == Json::array({"left", "right"}),
              "classification retains raw labeled logits without softmax");
        request["request"] = {{"image_path", right.string()}, {"config", {{"scale", 0.0}}}};
        check(run().at("output_summary").at("top_class") == 0,
              "classification ties choose first index");
        request["request"]["batch_size"] = 2;
        run(false);

        select("image_to_token_and_pooled_features", "features_fixture", "extract_features",
               {{"image_path", left.string()}, {"config", {{"scale", 0.5}}}});
        for (bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            summary = run().at("output_summary");
            check(summary.at("last_hidden_state") == Json::array({10, 3}) &&
                      summary.at("pooler_output") == Json::array({10, 3, 1}) &&
                      summary.at("feature_elements") == 5 &&
                      summary.at("last_hidden_state_shape") == Json::array({1, 1, 2}) &&
                      summary.at("pooler_output_shape") == Json::array({1, 3}),
                  "one joint feature call returns both arrays with a matching invocation marker");
        }
        for (const auto* task :
             {"image_to_token_features", "image_to_pooled_features", "image_to_spatial_features"}) {
            select(task, "features_fixture", "extract_features", {{"image_path", left.string()}});
            summary = run().at("output_summary");
            check(summary.at("processed_images") == 1 &&
                      summary.at("feature_elements").get<int>() > 0,
                  "distinct image feature representation has actual element count");
            if (std::string(task) == "image_to_token_features")
                check(summary.at("tokens").size() == 3 &&
                          summary.at("tokens").at(2).at("role") == "patch",
                      "image token roles and grid metadata retained");
            else if (std::string(task) == "image_to_spatial_features")
                check(summary.at("maps").at(0).at("shape") == Json::array({2, 1, 1}) &&
                          summary.at("source_to_processed").at("offset_x") == -3,
                      "spatial maps preserve their axes and source-image transform");
        }
        select("text_to_pooled_features", "features_fixture", "encode", {{"prompt", "Hello"}});
        summary = run().at("output_summary");
        check(summary.at("values") == Json::array({3, 5}) && summary.at("dim") == 2 &&
                  summary.at("feature_kind") == "pooled" && summary.at("embedding_vectors") == 1,
              "pooled encoding stays a pooled result rather than token features");
        select("text_to_token_features", "features_fixture", "encode", {{"prompt", "Hello"}});
        summary = run().at("output_summary");
        check(summary.at("shape") == Json::array({1, 2}) && summary.at("dim") == 2 &&
                  summary.at("feature_kind") == "token" && summary.at("tokens").size() == 1,
              "explicit token features retain token metadata and feature dimension");
        select("text_to_embedding", "features_fixture", "embed",
               {{"prompt", "Hello"}, {"role", "document"}});
        summary = run().at("output_summary");
        check(summary.at("values") == Json::array({4, 2}) && summary.contains("embedding_space"),
              "embedding role is typed input and embedding space is not discarded");
        request["request"]["role"] = "unsupported";
        run(false);
        request["request"] = {{"prompt", "Hello"}, {"scale", 2.0}, {"config", {{"scale", 3.0}}}};
        run(false);
        select("text_query_documents_to_relevance", "features_fixture", "rerank",
               {{"query", "q"}, {"documents", {"ab", "c"}}});
        summary = run().at("output_summary");
        check(summary.at("documents") == 2 && summary.at("scores") == Json::array({3102, 3111}) &&
                  summary.at("order") == "input_documents",
              "rerank list is one family call and preserves document order without claiming native "
              "batching");
        request["request"]["documents"] = Json::array();
        check(run().at("output_summary").at("scores").empty(), "empty document list stays empty");
        request["request"]["documents"] = Json::array({17});
        run(false);

        select("image_to_semantic_segmentation", "perception_fixture", "segment",
               {{"image_path", left.string()}});
        summary = run().at("output_summary");
        check(summary.at("mask") == Json::array({255, 5}) && summary.at("ignore_label") == 255 &&
                  summary.at("class_ids") == Json::array({0, 5}) &&
                  summary.at("class_scores").size() == 4,
              "semantic segmentation preserves labels, vocabulary and separate class-score grid");
        select("image_points_to_masks", "perception_fixture", "segment",
               {{"image_path", left.string()}, {"config", {{"benchmark_masks", "first_vs_best"}}}});
        summary = run().at("output_summary");
        check(summary.at("mask") == Json::array({0, 1}) && summary.at("num_masks") == 1 &&
                  summary.at("returned_mask_count") == 2 && summary.at("mask_pixels") == 2 &&
                  summary.at("selected_mask_index") == 0 &&
                  summary.at("iou_scores").at(0) < summary.at("iou_scores").at(1) &&
                  summary.at("masks").at(2) == 1 && summary.at("point").at("x") == 1,
              "center helper takes first family mask, not higher-IoU mask, and thresholds at zero");
        request["request"]["config"]["benchmark_masks"] = "empty";
        summary = run().at("output_summary");
        check(summary.at("mask").empty() && summary.at("num_masks") == 0 &&
                  summary.at("selected_mask_index").is_null() && summary.at("mask_pixels") == 0,
              "empty masks are not replaced by zero images");
        request["request"]["point_x"] = 0.5;
        run(false); // Fixed center helper does not silently ignore explicit point controls.
        request["operation"] = "segment_prompted";
        request["request"] = {{"image_path", left.string()},
                              {"point_x", 0.75},
                              {"point_y", 0.25},
                              {"is_foreground", false}};
        summary = run().at("output_summary");
        check(summary.at("generated_masks") == 2 && summary.at("mask_pixels") == 4 &&
                  summary.at("point").at("x") == 1 && summary.at("point").at("y") == 0 &&
                  summary.at("low_res_logits").at(0) == -7,
              "prompted masks retain all outputs, quantized original-pixel point and foreground "
              "flag");
        request["request"]["is_foreground"] = "false";
        run(false);
        request["request"] = {{"image_path", left.string()}, {"point_x", "0.5"}};
        run(false);
        select("image_text_to_instance_masks", "perception_fixture", "segment_prompted",
               {{"image_path", left.string()}, {"prompt", "object"}});
        summary = run().at("output_summary");
        check(summary.at("confidence").size() == 2 && summary.at("iou_scores").empty() &&
                  summary.at("object_ids") == Json::array({100, 101}),
              "instance confidence is not relabeled as IoU and object identities survive");

        const auto one = runtime / "benchmark_remaining_one.ppm";
        const auto zero = runtime / "benchmark_remaining_zero.ppm";
        image(one, 1, 255);
        image(zero, 1, 0);
        const auto state = runtime / "benchmark_remaining_state.f32";
        {
            const float values[]{1, 2};
            std::ofstream file(state, std::ios::binary);
            file.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        select("image_state_to_action_chunk", "action_fixture", "control",
               {{"image_path", one.string()},
                {"state_path", state.string()},
                {"config", {{"tag", "bench"}}}});
        summary = run().at("output_summary");
        check(summary.at("action_steps") == 2 && summary.at("action_dim") == 2 &&
                  summary.at("actions") == Json::array({2, -3, 2, 8}) &&
                  summary.at("within_training_bounds") == false &&
                  summary.at("inference_ms") == 103 &&
                  summary.at("schema").at("domain") == "fixture.bench" &&
                  summary.at("schema").at("normalization") == "unnormalized",
              "stateless action chunk retains values, schema, bounds and fresh-call timing");
        bundle(model, "image_state_action_queue", "action_fixture");
        run(false);

        select("text_to_image", "image_fixture", "generate_image", {{"prompt", "Hello"}});
        summary = run().at("output_summary");
        check(summary.at("generated_images") == 1 && summary.at("generated_frames") == 1 &&
                  summary.at("output_elements") == 3 && summary.at("media_type") == "image",
              "scalar image counts an actual image");
        request["request"]["prompt"] = "benchmark-worker";
        run(false);
        request["request"] = {{"prompt", "Hello"}, {"media_type", "video"}};
        run(false);
        const auto latents = runtime / "benchmark_remaining_latents.f32";
        {
            const float values[]{0.1F, 0.2F, 0.3F};
            std::ofstream file(latents, std::ios::binary);
            file.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        request["request"] = {{"prompt", "Hello"}, {"initial_latents_path", latents.string()}};
        run();
        select("images_text_to_image_edit", "image_fixture", "generate_image",
               {{"prompt", "Edit"}, {"image_path", one.string()}});
        run();
        request["request"] = {{"prompt", "Edit"},
                              {"image_paths", {one.string(), zero.string()}},
                              {"initial_latents_path", latents.string()}};
        run();
        request["request"]["image_path"] = left.string();
        run(false);
        select("batch_text_to_image", "image_fixture", "generate_image",
               {{"prompt", {"first", "benchmark-wide"}},
                {"seeds", {0, 17}},
                {"batch_size", 2},
                {"item_configs", Json::array({Json::object(), Json{{"level", 0.5}}})}});
        summary = run().at("output_summary");
        check(summary.at("generated_images") == 2 && summary.at("generated_frames") == 2 &&
                  summary.at("output_elements") == 9 &&
                  summary.at("images").at(0).at("width") == 1 &&
                  summary.at("images").at(1).at("width") == 2,
              "one native batch transports exact per-item seed/config and unequal output shapes");
        request["request"]["seed"] = 0;
        run(false);
        request["request"].erase("seed");
        request["request"]["config"] = {{"level", 0.25}};
        run(false);
        request["request"].erase("config");
        request["request"]["seeds"] = {0};
        run(false);
        request["request"] = {{"prompt", {"first", "benchmark-worker"}}};
        run(false);
        request["request"] = {{"prompt", {"first"}}, {"initial_latents_path", latents.string()}};
        run(false);
        request["request"] = {{"prompt", {"first"}}, {"item_configs", Json::array({17})}};
        run(false);

        {
            const float values[]{0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
            std::ofstream file(latents, std::ios::binary);
            file.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        select("text_to_video", "video_fixture", "generate_image",
               {{"prompt", "Hello"},
                {"media_type", "video"},
                {"initial_latents_path", latents.string()}});
        summary = run().at("output_summary");
        check(summary.at("generated_images") == 1 && summary.at("generated_frames") == 3 &&
                  summary.at("output_elements") == 9 &&
                  summary.at("timestamps_seconds") == Json::array({0, 0.1, 0.3}) &&
                  summary.at("media_type") == "video",
              "video retains actual nonuniform timeline, frame count and clip-count metric");
        request["request"] = {{"prompt", "benchmark-worker"}, {"media_type", "video"}};
        run(false);
        select("image_text_action_to_video", "video_fixture", "generate_image",
               {{"prompt", "Drive"},
                {"image_path", one.string()},
                {"action", "forward"},
                {"camera_intrinsics", {100, 100, 0.5, 0.5}},
                {"media_type", "video"}});
        run();
        request["request"]["camera_intrinsics"] = Json::array({100, 0, 0.5, 0, 100, 0.5, 0, 0, 1});
        run();
        request["request"]["camera_intrinsics"] = Json::array({1, 2, 3});
        run(false);
        request["request"]["camera_intrinsics"] = Json::array({true, 100, 0.5, 0.5});
        run(false);
        request["request"]["camera_intrinsics"] = Json::array({100, 100, 0.5, 0.5});
        request["request"]["action"] = "";
        run(false);
        std::cout << (failures ? "FAILED\n" : "ALL PASSED\n");
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
