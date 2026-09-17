/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/structure.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
template <class Function>
void rejects(Function function, trtmc_status status, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const trtmc::Error& error) {
        rejected = error.code() == status;
    }
    check(rejected, message);
}
void bundle(const std::filesystem::path& path, const std::string& mode) {
    const char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header = "{\"format\":1,\"family\":\"structure_fixture\",\"task\":\"" + mode +
                               "\",\"backend\":\"fake\",\"sections\":{}}";
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.write(magic, sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        auto load = [&](const char* mode) {
            const auto path = root / (std::string("structure-cpp-") + mode + ".bundle");
            bundle(path, mode);
            return trtmc::Model::load(path.string(), options);
        };
        const std::uint8_t prepared[]{'B', '2', 'R', 'Q', 0, 255, 1};
        const std::string path = "not-opened/relative/assets/request.yaml";
        const trtmc::MolecularDocumentToStructureRequest request{{prepared}, "b2rq", path};
        auto model = load("molecular_document_to_structure");
        auto task = model.task<trtmc::MolecularDocumentToStructure>();
        check(model.tasks().size() == 1 && task.config_fields().size() == 4,
              "molecular Task and family-owned Config are discoverable");
        auto result = task.run(request);
        check(result.structure() == "data_fixture\n# b2rq:4232525100ff01\n" &&
                  result.format() == trtmc::MolecularStructureFormat::Mmcif &&
                  result.metadata_json() == "{\"encoding\":\"b2rq\",\"source_path\":\"" + path +
                                                "\",\"seed\":42,\"sampling_steps\":200}",
              "binary NUL/FF bytes, encoding and source path reach the family unchanged");
        const auto confidence = result.confidence();
        check(confidence && confidence->confidence_score == 0.1F && confidence->ptm == 0.2F &&
                  confidence->iptm == 0.3F && confidence->ligand_iptm == 0.4F &&
                  confidence->protein_iptm == 0.5F && confidence->complex_plddt == 66.0F &&
                  confidence->complex_iplddt == 77.0F && confidence->plddt.size == 3 &&
                  confidence->plddt.data[0] == 88.0F && confidence->plddt.data[1] == 0.0F &&
                  confidence->plddt.data[2] == 99.0F,
              "all confidence fields and ordered PLDDT values retain family units");
        const std::uint8_t yaml[]{'a', ':', ' ', 'A', '\n'};
        const std::uint8_t json[]{'{', '"', 'a', '"', ':', '1', '}'};
        auto yaml_result = task.run({{yaml}, "yaml", "source.yaml"});
        auto json_result = task.run({{json}, "json", "source.json"});
        check(yaml_result.structure() == "data_fixture\n# yaml:613a20410a\n" &&
                  json_result.structure() == "data_fixture\n# json:7b2261223a317d\n",
              "YAML and JSON encoding/payload are not inferred or rewritten");
        auto zero = task.run(
            request,
            {{"seed", std::int64_t{0}}, {"include_confidence", false}, {"output_format", "pdb"}});
        check(!zero.confidence() && zero.format() == trtmc::MolecularStructureFormat::Pdb &&
                  zero.structure() == "HEADER fixture\nREMARK b2rq:4232525100ff01\n" &&
                  zero.metadata_json().find("\"seed\":0") != std::string_view::npos,
              "zero seed, absent confidence and output format are family Config");
        auto absent =
            load("without_confidence").task<trtmc::MolecularDocumentToStructure>().run(request);
        check(!absent.confidence() && absent.view().has_confidence == 0,
              "a family without confidence reports absence, not zero-valued scores");
        check(absent.metadata_json().find("\"sampling_steps\":7") != std::string_view::npos,
              "different loaded families keep their own sampling defaults");
        auto explicit_steps =
            task.run(request, {{"sampling_steps", std::int64_t{12}}, {"seed", std::int64_t{0}}});
        check(explicit_steps.metadata_json().find("\"seed\":0,\"sampling_steps\":12") !=
                  std::string_view::npos,
              "explicit sampling steps and seed pass through existing Config");
        auto guarded = load("must_not_run").task<trtmc::MolecularDocumentToStructure>();
        rejects([&] { (void)guarded.run(request, {{"unknown", true}}); }, TRTMC_INVALID_CONFIG,
                "Core rejects unknown Config before family execution");
        rejects([&] { (void)guarded.run(request, {{"seed", true}}); }, TRTMC_INVALID_CONFIG,
                "Core rejects wrong Config types before family execution");
        rejects(
            [&] {
                (void)guarded.run(request, {{"seed", std::int64_t{1}}, {"seed", std::int64_t{2}}});
            },
            TRTMC_INVALID_CONFIG, "Core rejects duplicate Config before family execution");
        rejects([&] { (void)guarded.run(request); }, TRTMC_INTERNAL_ERROR,
                "guarded fixture throws if execution is actually reached");
        rejects([&] { (void)task.run(request, {{"seed", std::int64_t{-1}}}); },
                TRTMC_INVALID_CONFIG, "family still owns numeric Config constraints");
        rejects([&] { (void)task.run({{prepared}, "csv", path}); }, TRTMC_UNSUPPORTED,
                "family rejects unsupported encoding without fallback");
        rejects([&] { (void)task.run({{json}, "b2rq", path}); }, TRTMC_INVALID_ARGUMENT,
                "invalid prepared content is not retried as JSON");
        rejects([&] { (void)task.run({{}, "json", {}}); }, TRTMC_INVALID_ARGUMENT,
                "empty document is rejected");
        rejects([&] { (void)task.run({{prepared}, {}, path}); }, TRTMC_INVALID_ARGUMENT,
                "missing encoding is not guessed");
        rejects([&] { (void)load("disabled").task<trtmc::MolecularDocumentToStructure>(); },
                TRTMC_UNSUPPORTED, "family binding controls Task availability");
        for (const char* mode : {"empty_structure", "bad_format", "bad_confidence", "bad_plddt"})
            rejects(
                [&] { (void)load(mode).task<trtmc::MolecularDocumentToStructure>().run(request); },
                TRTMC_INTERNAL_ERROR, "malformed structure result is rejected");
        auto retained = [&] {
            const std::uint8_t temporary_bytes[]{'B', '2', 'R', 'Q', 0, 255, 1};
            auto temporary = load("molecular_document_to_structure");
            return temporary.task<trtmc::MolecularDocumentToStructure>().run(
                {{temporary_bytes}, "b2rq", path});
        }();
        auto moved = std::move(retained);
        check(moved.structure() == result.structure() && moved.confidence()->plddt.data[2] == 99,
              "owned structure/metadata/confidence survive input and model scope and result move");
        std::cout << result.structure() << result.metadata_json() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return failures ? 1 : 0;
}
