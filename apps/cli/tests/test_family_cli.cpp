/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/cli.h"

#include <nlohmann/json.hpp>
#include <string>

#ifdef TRTMC_FAMILY_CLI_FIXTURE

extern "C" int trtmc_family_cli_v1(const char* handler, const char* values_json,
                                   const char* runtime_root, void* context,
                                   trtmc_cli_write_v1 output, trtmc_cli_write_v1) {
    const auto text = nlohmann::json{{"handler", handler},
                                     {"values", nlohmann::json::parse(values_json)},
                                     {"runtime_root", runtime_root}}
                          .dump() +
                      '\n';
    output(context, text.data(), text.size());
    return 0;
}

#else

#include "cli/family_cli.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <unistd.h>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void write_json(const fs::path& path, const Json& document) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << document.dump();
}

Json declaration() {
    return {{"version", 1},
            {"commands",
             Json::array(
                 {{{"name", "echo"},
                   {"executor", "native"},
                   {"handler", "owner_echo"},
                   {"help", "Echo typed fixture arguments"},
                   {"arguments",
                    Json::array({{{"name", "item"}, {"type", "string"}},
                                 {{"name", "count"},
                                  {"flags", {"--count", "-n"}},
                                  {"type", "int"},
                                  {"default", 7}},
                                 {{"name", "ratio"}, {"flags", {"--ratio"}}, {"type", "float"}},
                                 {{"name", "enabled"}, {"flags", {"--enabled"}}, {"type", "bool"}},
                                 {{"name", "switch"},
                                  {"flags", {"--switch"}},
                                  {"type", "bool"},
                                  {"action", "store_true"}},
                                 {{"name", "tags"},
                                  {"flags", {"--tag"}},
                                  {"type", "string"},
                                  {"action", "append"},
                                  {"default", {"base"}}},
                                 {{"name", "choice"},
                                  {"flags", {"--choice"}},
                                  {"type", "string"},
                                  {"choices", {"one", "two"}}}})}},
                  {{"name", "build"},
                   {"executor", "python"},
                   {"handler", "cli:build"},
                   {"help", "Build through an owner"},
                   {"arguments", Json::array()}}})}};
}

struct Result {
    std::optional<int> status;
    std::string output, error;
};
Result run(const fs::path& executable, std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& item : arguments)
        argv.push_back(item.data());
    std::ostringstream output, error;
    const auto result = trtmc::cli::run_family_cli(static_cast<int>(argv.size()), argv.data(),
                                                   output, error, executable);
    return {result, output.str(), error.str()};
}

void descriptor_contract(const fs::path& root, const fs::path& fixture) {
    const auto executable = root / "trtmc";
    const auto path = root / "families/fixture/cli.json";
    const auto valid = declaration();
    write_json(path, valid);
    const auto offline_help = run(executable, {"trtmc", "fixture", "echo", "--help"});
    check(offline_help.status == 0 && offline_help.output.find("[choices:") != std::string::npos &&
              offline_help.output.find("one") != std::string::npos,
          "help reads choices from the declaration with no family DSO installed");
    check(run(executable, {"trtmc", "fixture", "build", "--help"}).status == 0,
          "Python command help needs no Python implementation");
    check(!run(executable, {"trtmc", "run", "old.bundle"}).status,
          "undeclared legacy command remains available to the existing dispatcher");
    const auto global = run(executable, {"trtmc", "--help"});
    check(!global.status && global.output.find("fixture") != std::string::npos,
          "root help adds families and leaves existing SDK usage to the caller");
    check(run(executable, {"trtmc", "fixture", "unknown"}).status == 2,
          "unknown declared-family command never falls back");
    check(run(executable, {"trtmc", "fixture", "echo", "value"}).status == 1,
          "execution alone requires the family DSO");
    fs::copy_file(fixture, root / "libtrtmc_cli_fixture.so");
    const auto typed =
        run(executable, {"trtmc", "fixture", "echo", "value", "--count=-3", "--ratio", "0.25",
                         "--enabled", "false", "--switch", "--tag", "one", "--tag=two"});
    check(typed.status == 0, "native owner is loaded lazily and invoked");
    if (typed.status == 0) {
        const auto payload = Json::parse(typed.output);
        const auto& values = payload.at("values");
        check(payload.at("handler") == "owner_echo" && values.at("count") == -3 &&
                  values.at("ratio") == 0.25 && values.at("enabled") == false &&
                  values.at("switch") == true &&
                  values.at("tags") == Json({"base", "one", "two"}) && !values.contains("choice"),
              "types, omissions, append defaults, and owner handler identity survive transport");
    }
    const auto defaults = run(executable, {"trtmc", "fixture", "echo", "--", "--help"});
    check(defaults.status == 0 &&
              Json::parse(defaults.output).at("values").at("item") == "--help" &&
              !Json::parse(defaults.output).at("values").contains("switch"),
          "option terminator preserves literal help-looking positional input");
    for (const char* finite : {"1e-320", "1e-400"}) {
        const auto result =
            run(executable, {"trtmc", "fixture", "echo", "value", "--ratio", finite});
        check(result.status == 0,
              "finite subnormal and underflow-to-zero values match Python float");
    }
    const auto exponent = run(executable, {"trtmc", "fixture", "echo", "value", "--ratio", "-1e2"});
    check(exponent.status == 0 && Json::parse(exponent.output).at("values").at("ratio") == -100.0,
          "negative exponent values are accepted by a declared float option");
    auto real_default = valid;
    real_default["commands"][0]["arguments"][2]["default"] = 1;
    write_json(path, real_default);
    const auto real = run(executable, {"trtmc", "fixture", "echo", "value"});
    check(real.status == 0 && Json::parse(real.output).at("values").at("ratio").is_number_float(),
          "float defaults have float transport values even when the declaration uses an integer");
    write_json(path, valid);
    const auto negative = run(executable, {"trtmc", "fixture", "echo", "value", "--count", "-3",
                                           "--ratio", "-0.125", "--tag=--switch"});
    check(negative.status == 0 && Json::parse(negative.output).at("values").at("count") == -3 &&
              Json::parse(negative.output).at("values").at("ratio") == -0.125 &&
              Json::parse(negative.output).at("values").at("tags") == Json({"base", "--switch"}),
          "negative scalar values and explicitly attached leading-dash text remain valid");
    for (const auto& extra :
         std::vector<std::vector<std::string>>{{"--count", "9223372036854775808"},
                                               {"--count", "-9223372036854775809"},
                                               {"--ratio", "nan"},
                                               {"--ratio", "1e400"},
                                               {"--ratio", "0x1p2"},
                                               {"--ratio", " 1.0"},
                                               {"--count", "1_0"},
                                               {"--enabled", "1"},
                                               {"--choice", "other"},
                                               {"--count", "1", "-n", "2"},
                                               {"--unknown", "x"},
                                               {"--tag", "--switch"},
                                               {"--tag", "--typo"},
                                               {"--tag", "-1"},
                                               {"--count", "-2x"},
                                               {"--switch", "--switch"}}) {
        std::vector<std::string> arguments{"trtmc", "fixture", "echo", "value"};
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        check(run(executable, arguments).status == 2,
              "invalid scalar, choice, or duplicate arguments are rejected");
    }
    std::vector<Json> malformed;
    for (const Json& version : {Json(true), Json(1.0), Json(2)}) {
        auto document = valid;
        document["version"] = version;
        malformed.push_back(document);
    }
    auto changed = valid;
    changed["unknown"] = true;
    malformed.push_back(changed);
    changed = valid;
    changed["commands"] = Json::array();
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["unexpected"] = true;
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["arguments"][0].erase("type");
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["arguments"][0]["required"] = 1;
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["arguments"][0]["unexpected"] = true;
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["arguments"][1]["default"] = nullptr;
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["arguments"][1]["flags"] = {"--help"};
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["arguments"][1]["default"] = UINT64_C(9223372036854775808);
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["arguments"][1]["default"] = true;
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["arguments"][6]["choices"] = {"one", "one"};
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["arguments"][6]["choices"] = {1};
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["executor"] = Json::array();
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["handler"] = "../escape";
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][1]["handler"] = "../other:build";
    malformed.push_back(changed);
    changed = valid;
    changed["commands"].push_back(changed["commands"][0]);
    malformed.push_back(changed);
    changed = valid;
    changed["commands"][0]["arguments"][0]["required"] = false;
    changed["commands"][0]["arguments"].push_back({{"name", "later"}, {"type", "string"}});
    malformed.push_back(changed);
    for (const auto& document : malformed) {
        write_json(path, document);
        check(run(executable, {"trtmc", "fixture", "--help"}).status == 2,
              "malformed declared families fail without loading or falling back");
    }
    write_json(path, valid);
}

void installation_contract(const fs::path& root, const fs::path& fixture) {
    for (const std::string libdir : {"lib", "lib64"}) {
        const auto prefix = root / libdir;
        const auto executable = prefix / "bin/trtmc";
        fs::create_directories(executable.parent_path());
        write_json(prefix / "share/trtmc/families/fixture/cli.json", declaration());
        fs::create_directories(prefix / libdir);
        fs::copy_file(fixture, prefix / libdir / "libtrtmc_cli_fixture.so");
        check(run(executable, {"trtmc", "fixture", "echo", "installed"}).status == 0,
              "standard installation finds share declarations and the installed family library");
        write_json(prefix / "bin/families/fixture/cli.json", declaration());
        check(run(executable, {"trtmc", "fixture", "--help"}).status == 0,
              "identical installed declarations can be deduplicated");
        auto conflicting = declaration();
        conflicting["commands"][0]["help"] = "different owner declaration";
        write_json(prefix / "bin/families/fixture/cli.json", conflicting);
        check(run(executable, {"trtmc", "fixture", "--help"}).status == 2,
              "conflicting owner declarations are rejected explicitly");
    }
    const auto site = root / "site-packages";
    fs::create_directories(site / "tensorrt_model_connect/bin");
    write_json(site / "families/fixture/cli.json", declaration());
    check(run(site / "tensorrt_model_connect/bin/trtmc", {"trtmc", "fixture", "--help"}).status ==
              0,
          "wheel layout finds family declarations without importing Python");
}

void independent_owner_contract(const fs::path& root, const fs::path& fixture) {
    const auto executable = root / "trtmc";
    const Json owner = {
        {"version", 1},
        {"commands", Json::array({{{"name", "compose"},
                                   {"executor", "native"},
                                   {"handler", "combine"},
                                   {"arguments", Json::array({{{"name", "policy"},
                                                               {"flags", {"--owner-policy"}},
                                                               {"type", "string"},
                                                               {"choices", {"exact", "fast"}},
                                                               {"default", "exact"}}})}}})}};
    write_json(root / "families/another_owner/cli.json", owner);
    fs::copy_file(fixture, root / "libtrtmc_cli_another_owner.so");
    const auto defaults = run(executable, {"trtmc", "another_owner", "compose"});
    const auto selected =
        run(executable, {"trtmc", "another_owner", "compose", "--owner-policy", "fast"});
    check(defaults.status == 0 && selected.status == 0,
          "new family commands only require an owner description and handler library");
    if (defaults.status == 0 && selected.status == 0) {
        check(Json::parse(defaults.output).at("values") == Json({{"policy", "exact"}}) &&
                  Json::parse(selected.output).at("values") == Json({{"policy", "fast"}}) &&
                  Json::parse(selected.output).at("handler") == "combine",
              "the host forwards the new owner's handler, defaults and opaque argument names");
    }
    check(run(executable, {"trtmc", "another_owner", "compose", "--count", "1"}).status == 2,
          "one owner's options never leak into another owner's command");
    check(run(executable, {"trtmc", "fixture", "echo", "original"}).status == 0,
          "adding an owner preserves the original owner's command");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_family_cli FIXTURE_DSO\n";
        return 2;
    }
    const auto root = fs::temp_directory_path() / ("trtmc-family-cli-" + std::to_string(getpid()));
    fs::create_directories(root);
    try {
        descriptor_contract(root, fs::absolute(argv[1]));
        independent_owner_contract(root, fs::absolute(argv[1]));
        installation_contract(root / "layouts", fs::absolute(argv[1]));
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    fs::remove_all(root);
    return failures == 0 ? 0 : 1;
}
#endif
