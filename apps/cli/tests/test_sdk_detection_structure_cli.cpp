/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cli/cli.h"
#include "cli/io.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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
void write_file(const std::filesystem::path& path, std::string_view data) {
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
}
std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("missing CLI output: " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
void bundle(const std::filesystem::path& path, const char* family, const char* task) {
    const auto header = json{
        {"format", 1},
        {"family", family},
        {"task", task},
        {"backend", "fake"},
        {"sections",
         json::object()}}.dump();
    std::string bytes("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        bytes.push_back(
            static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255));
    bytes += header;
    write_file(path, bytes);
}
struct Run {
    int status;
    std::string output, error;
};
Run run(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& value : arguments)
        argv.push_back(value.data());
    std::ostringstream output, error;
    const auto status = trtmc::cli::run(static_cast<int>(argv.size()), argv.data(), output, error);
    return {status, output.str(), error.str()};
}
trtmc::cli::Command parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& value : arguments)
        argv.push_back(value.data());
    return trtmc::cli::parse_args(static_cast<int>(argv.size()), argv.data());
}
struct WorkingDirectory {
    explicit WorkingDirectory(const std::filesystem::path& next)
        : previous(std::filesystem::current_path()) {
        std::filesystem::current_path(next);
    }
    ~WorkingDirectory() { std::filesystem::current_path(previous); }
    std::filesystem::path previous;
};

void detection(const std::filesystem::path& root) {
    const auto model = root / "cli-detected-boxes.bundle";
    const auto disabled = root / "cli-detected-boxes-disabled.bundle";
    const auto image = root / "cli-detected-boxes.png";
    bundle(model, "image_boxes_fixture", "image_to_boxes");
    bundle(disabled, "image_boxes_fixture", "disabled");
    trtmc::cli::io::save_png(image.string(), std::vector<float>(36, 0.5F), 4, 3);
    auto invoke = [&](const std::filesystem::path& selected, std::vector<std::string> extra = {}) {
        std::vector<std::string> arguments{"trtmc",          "detect",      selected.string(),
                                           "--runtime-root", root.string(), "--image",
                                           image.string()};
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        return run(std::move(arguments));
    };
    const auto basic = invoke(model);
    check(basic.status == 0, "detect selects image-only Task without --prompt or --task");
    if (basic.status)
        throw std::runtime_error(basic.error);
    const auto output = json::parse(basic.output);
    check(output.at("classes") == json::array({42, 7}) && output.at("image_height") == 3 &&
              output.at("image_width") == 4 && output.at("boxes").size() == 8 &&
              output.at("boxes").at(0) == -2 && output.at("boxes").at(2) == 7 &&
              output.at("scores").at(0).get<float>() == 0.75F && output.at("scores").at(1) == 0 &&
              output.at("coordinate_space") == "original_image_pixels" &&
              output.at("box_format") == "xyxy",
          "detect preserves full numeric classes, pixel boxes, scores and image dimensions");
    const auto zero = invoke(model, {"--set", "score=0", "--set", "empty=false"});
    check(zero.status == 0 && json::parse(zero.output).at("scores").at(0) == 0,
          "detect preserves family-declared explicit zero and false");
    const auto empty = invoke(model, {"--set", "empty=true"});
    check(empty.status == 0 && json::parse(empty.output).at("boxes").empty() &&
              json::parse(empty.output).at("classes").empty() &&
              json::parse(empty.output).at("image_width") == 4,
          "detect preserves genuinely empty outputs and original dimensions");
    const auto unsupported = invoke(disabled);
    check(unsupported.status != 0 && unsupported.output.empty(),
          "missing detection Task has no fallback");
    const auto prompt = invoke(model, {"--task", "image_to_boxes", "--prompt", "bird"});
    check(prompt.status != 0 && prompt.error.find("does not accept --prompt") != std::string::npos,
          "image-only Task does not silently ignore text input");
}

class ExistingStructure final : public trtmc::IStructurePrediction {
  public:
    trtmc::StructurePredictionRequest seen;
    trtmc::StructurePredictionResult
    predict_structure(const trtmc::StructurePredictionRequest& request) override {
        seen = request;
        trtmc::StructurePredictionResult result;
        result.structure = "data_existing\n";
        result.metadata_json = "{\"existing\":true}";
        return result;
    }
};

void structure(const std::filesystem::path& root) {
    const auto model = root / "cli-structure.bundle";
    const auto absent = root / "cli-structure-absent.bundle";
    const auto disabled = root / "cli-structure-disabled.bundle";
    bundle(model, "structure_fixture", "molecular_document_to_structure");
    bundle(absent, "structure_fixture", "without_confidence");
    bundle(disabled, "structure_fixture", "disabled");
    const std::string binary("B2RQ\0\xff\x01", 7);
    const auto prepared = root / "cli-request.b2rq", unknown = root / "cli-request.bin";
    const auto yaml = root / "cli-request.yaml", source_json = root / "cli-request.json";
    write_file(prepared, binary);
    write_file(unknown, binary);
    write_file(yaml, "a: A\n");
    write_file(source_json, "{\"a\":1}");
    auto invoke = [&](const std::filesystem::path& selected, const std::filesystem::path& input,
                      std::vector<std::string> extra = {}) {
        std::vector<std::string> arguments{"trtmc",          "predict-structure", selected.string(),
                                           "--runtime-root", root.string(),       "--input",
                                           input.string()};
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        return run(std::move(arguments));
    };
    const auto directory = root / "sdk-structure-cli-output";
    std::filesystem::create_directories(directory);
    const WorkingDirectory cwd(directory);
    const auto basic = invoke(model, prepared);
    check(basic.status == 0, "structure command uses extension and output defaults");
    if (basic.status)
        throw std::runtime_error(basic.error);
    const auto output = json::parse(basic.output);
    const auto metadata_bytes = read_file("prediction.cif.metadata.json");
    const auto metadata = json::parse(metadata_bytes);
    check(output.at("structure_path") == "prediction.cif" && output.at("format") == "mmcif" &&
              output.at("input_encoding") == "b2rq" &&
              read_file("prediction.cif") == "data_fixture\n# b2rq:4232525100ff01\n" &&
              metadata.at("source_path") == prepared.string() && metadata.at("seed") == 42 &&
              metadata_bytes == "{\"encoding\":\"b2rq\",\"source_path\":\"" + prepared.string() +
                                    "\",\"seed\":42,\"sampling_steps\":200}",
          "binary bytes and source path survive; structure/metadata are written verbatim");
    const auto& confidence = output.at("confidence");
    check(confidence.at("confidence_score").get<float>() == 0.1F &&
              confidence.at("ptm").get<float>() == 0.2F &&
              confidence.at("iptm").get<float>() == 0.3F &&
              confidence.at("ligand_iptm").get<float>() == 0.4F &&
              confidence.at("protein_iptm").get<float>() == 0.5F &&
              confidence.at("complex_plddt") == 66 && confidence.at("complex_iplddt") == 77 &&
              confidence.at("plddt") == json::array({88, 0, 99}),
          "CLI exposes every confidence value and PLDDT element");
    const auto yaml_run = invoke(model, yaml, {"--output", "yaml.cif"});
    const auto json_run = invoke(model, source_json, {"--output", "json.cif"});
    check(yaml_run.status == 0 && json_run.status == 0 &&
              read_file("yaml.cif") == "data_fixture\n# yaml:613a20410a\n" &&
              read_file("json.cif") == "data_fixture\n# json:7b2261223a317d\n",
          "YAML and JSON extensions map explicitly without content rewriting");
    const auto zero = invoke(model, prepared,
                             {"--seed", "0", "--set", "include_confidence=false", "--set",
                              "output_format=pdb", "--output-json", "metadata/custom.json"});
    check(zero.status == 0 && json::parse(zero.output).at("format") == "pdb" &&
              json::parse(zero.output).at("structure_path") == "prediction.pdb" &&
              json::parse(zero.output).at("confidence").is_null() &&
              json::parse(zero.output).at("confidence_score").is_null() &&
              read_file("prediction.pdb") == "HEADER fixture\nREMARK b2rq:4232525100ff01\n" &&
              json::parse(read_file("metadata/custom.json")).at("seed") == 0,
          "explicit Config and metadata path are honored; returned format chooses default suffix");
    const auto missing = invoke(absent, prepared, {"--output", "absent.cif"});
    check(missing.status == 0 && json::parse(missing.output).at("confidence").is_null() &&
              json::parse(read_file("absent.cif.metadata.json")).at("sampling_steps") == 7,
          "family owns absent confidence and a different default; CLI does not inject defaults");
    const auto unknown_extension = invoke(model, unknown, {"--output", "unknown.cif"});
    check(unknown_extension.status != 0 && unknown_extension.output.empty() &&
              unknown_extension.error.find("--input-encoding") != std::string::npos &&
              !std::filesystem::exists("unknown.cif"),
          "unknown extension asks for explicit encoding before output writes");
    const auto explicit_encoding =
        invoke(model, unknown, {"--input-encoding", "b2rq", "--output", "explicit.cif"});
    check(explicit_encoding.status == 0 && read_file("explicit.cif") == read_file("prediction.cif"),
          "explicit encoding supports prepared files with arbitrary extensions");
    const auto mismatch =
        invoke(model, source_json, {"--input-encoding", "b2rq", "--output", "bad.cif"});
    check(mismatch.status != 0 && mismatch.output.empty() && !std::filesystem::exists("bad.cif"),
          "explicit encoding failure is not retried using the file extension");
    const auto unsupported = invoke(disabled, prepared, {"--output", "unsupported.cif"});
    check(unsupported.status != 0 && unsupported.output.empty() &&
              !std::filesystem::exists("unsupported.cif"),
          "unavailable molecular Task has no old-runtime fallback");
    const auto steps =
        invoke(model, prepared, {"--num-steps", "12", "--seed", "0", "--output", "steps.cif"});
    check(
        steps.status == 0 &&
            json::parse(read_file("steps.cif.metadata.json")).at("sampling_steps") == 12 &&
            json::parse(read_file("steps.cif.metadata.json")).at("seed") == 0,
        "legacy step flag and explicit seed reach the new SDK as family sampling_steps=12/seed=0");
    const auto duplicates = invoke(
        model, prepared,
        {"--num-steps", "12", "--set", "sampling_steps=13", "--output", "duplicate-steps.cif"});
    check(duplicates.status != 0 && duplicates.output.empty() &&
              !std::filesystem::exists("duplicate-steps.cif"),
          "step flag and generic Config cannot silently override duplicate keys");
    ExistingStructure existing;
    auto command = parse({"trtmc", "predict-structure", "unused.bundle", "--input",
                          prepared.string(), "--output", "existing.cif", "--output-json",
                          "existing.json", "--num-steps", "12", "--seed", "0"});
    std::ostringstream existing_output;
    check(
        trtmc::cli::dispatch(command, existing, existing_output) == 0 &&
            existing.seen.document == binary && existing.seen.source_path == prepared.string() &&
            existing.seen.config.sampling_steps == 12 && existing.seen.config.seed == 0 &&
            read_file("existing.cif") == "data_existing\n" &&
            read_file("existing.json") == "{\"existing\":true}",
        "unmigrated structure command preserves every existing explicit argument and binary input");
    command.options["--input-encoding"] = "b2rq";
    bool rejected = false;
    try {
        (void)trtmc::cli::dispatch(command, existing, existing_output);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "legacy structure interface does not silently ignore a new encoding argument");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const auto root = std::filesystem::absolute(argv[1]);
        detection(root);
        structure(root);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (!failures)
        std::cout << "Detection and structure CLI contracts passed\n";
    return failures ? 1 : 0;
}
