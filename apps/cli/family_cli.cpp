/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/family_cli.h"

#include "trtmc/internal/cli.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <ostream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace trtmc::cli {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;

void keys(const Json& value, std::initializer_list<const char*> allowed,
          std::initializer_list<const char*> required) {
    if (!value.is_object())
        throw std::invalid_argument("declaration must be an object");
    for (const auto& item : value.items()) {
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
            throw std::invalid_argument("unknown declaration field: " + item.key());
    }
    for (const auto* name : required) {
        if (!value.contains(name))
            throw std::invalid_argument("missing declaration field: " + std::string(name));
    }
}

bool identifier(const std::string& value) {
    return !value.empty() && value.front() >= 'a' && value.front() <= 'z' &&
           std::all_of(value.begin(), value.end(), [](unsigned char c) {
               return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
           });
}

std::vector<fs::path> roots(const fs::path& executable) {
    const auto directory = executable.parent_path();
    return {directory / "families", directory / "../share/trtmc/families",
            directory / "../../families"};
}

Json scalar(const std::string& text, const std::string& type) {
    if (type == "string" || type == "path")
        return text;
    if (type == "bool") {
        if (text == "true")
            return true;
        if (text == "false")
            return false;
        throw std::invalid_argument("expected true or false");
    }
    std::size_t end = 0;
    static const std::regex integer_pattern("[+-]?[0-9]+");
    static const std::regex float_pattern("[+-]?([0-9]+(\\.[0-9]*)?|\\.[0-9]+)([eE][+-]?[0-9]+)?");
    if ((type == "int" && !std::regex_match(text, integer_pattern)) ||
        (type == "float" && !std::regex_match(text, float_pattern)))
        throw std::invalid_argument("invalid " + type + " value: " + text);
    try {
        if (type == "int") {
            const auto value = std::stoll(text, &end);
            if (end == text.size())
                return value;
        } else if (type == "float") {
            // Match Python's finite subnormal and underflow-to-zero behavior.
            char* tail = nullptr;
            const auto value = std::strtod(text.c_str(), &tail);
            if (tail == text.c_str() + text.size() && std::isfinite(value))
                return value;
        }
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid " + type + " value: " + text);
    }
    throw std::invalid_argument("invalid " + type + " value: " + text);
}

bool correct_type(const Json& value, const std::string& type) {
    if (type == "string" || type == "path")
        return value.is_string();
    if (type == "bool")
        return value.is_boolean();
    if (type == "int")
        return value.is_number_integer() &&
               (!value.is_number_unsigned() ||
                value.get<std::uint64_t>() <= static_cast<std::uint64_t>(INT64_MAX));
    return type == "float" && value.is_number() && std::isfinite(value.get<double>());
}

void validate_value(const Json& argument, const Json& value) {
    if (!correct_type(value, argument.at("type").get<std::string>()))
        throw std::invalid_argument("incorrect type for argument " +
                                    argument.at("name").get<std::string>());
    if (argument.contains("choices") &&
        std::find(argument.at("choices").begin(), argument.at("choices").end(), value) ==
            argument.at("choices").end())
        throw std::invalid_argument("invalid choice for argument " +
                                    argument.at("name").get<std::string>());
}

Json default_value(const Json& argument) {
    auto value = argument.at("default");
    if (argument.at("type") == "float") {
        if (argument.value("action", std::string{}) == "append") {
            for (auto& item : value)
                item = item.get<double>();
        } else {
            value = value.get<double>();
        }
    }
    return value;
}

Json read_descriptor(const fs::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::invalid_argument("cannot read family CLI declaration: " + path.string());
    Json descriptor;
    try {
        descriptor = Json::parse(input);
        keys(descriptor, {"version", "commands"}, {"version", "commands"});
        if (!descriptor.at("version").is_number_integer() || descriptor.at("version") != 1 ||
            !descriptor.at("commands").is_array() || descriptor.at("commands").empty())
            throw std::invalid_argument("expected version 1 and commands array");
        std::set<std::string> commands;
        for (const auto& command : descriptor.at("commands")) {
            keys(command, {"name", "help", "executor", "handler", "arguments"},
                 {"name", "executor", "handler", "arguments"});
            const auto name = command.at("name").get<std::string>();
            const auto executor = command.at("executor").get<std::string>();
            const auto handler = command.at("handler").get<std::string>();
            const std::regex handler_pattern(
                executor == "python" ? "[a-z][a-z0-9_]*(\\.[a-z][a-z0-9_]*)*:[a-z][a-z0-9_]*"
                                     : "[a-z][a-z0-9_]*");
            if (!std::regex_match(name, std::regex("[a-z][a-z0-9_-]*")) ||
                !commands.insert(name).second || !std::regex_match(handler, handler_pattern) ||
                (executor != "native" && executor != "python") ||
                !command.at("arguments").is_array())
                throw std::invalid_argument("invalid or duplicate command");
            (void)command.value("help", std::string{});
            std::set<std::string> names, flags;
            bool optional_positional_seen = false;
            for (const auto& argument : command.at("arguments")) {
                keys(argument,
                     {"name", "flags", "type", "action", "required", "default", "choices", "help"},
                     {"name", "type"});
                const auto argument_name = argument.at("name").get<std::string>();
                const auto type = argument.at("type").get<std::string>();
                const auto action = argument.value("action", std::string{});
                if (!identifier(argument_name) || !names.insert(argument_name).second ||
                    (type != "string" && type != "path" && type != "int" && type != "float" &&
                     type != "bool") ||
                    (argument.contains("action") && action != "store_true" && action != "append"))
                    throw std::invalid_argument("invalid or duplicate argument");
                (void)argument.value("help", std::string{});
                if (argument.contains("required") && !argument.at("required").is_boolean())
                    throw std::invalid_argument("required must be a bool");
                if (argument.contains("flags")) {
                    if (!argument.at("flags").is_array() || argument.at("flags").empty())
                        throw std::invalid_argument("flags must be a non-empty array");
                    for (const auto& raw : argument.at("flags")) {
                        const auto flag = raw.get<std::string>();
                        if (!std::regex_match(flag, std::regex("--?[a-zA-Z][a-zA-Z0-9_-]*")) ||
                            flag == "--help" || flag == "-h" || !flags.insert(flag).second)
                            throw std::invalid_argument("invalid or duplicate flag");
                    }
                } else {
                    if (!action.empty())
                        throw std::invalid_argument("positional arguments cannot have an action");
                    if (argument.value("required", true)) {
                        if (optional_positional_seen)
                            throw std::invalid_argument(
                                "required positional follows an optional positional");
                    } else {
                        optional_positional_seen = true;
                    }
                }
                if (action == "store_true" && type != "bool")
                    throw std::invalid_argument("store_true requires bool type");
                if (argument.contains("choices")) {
                    if (!argument.at("choices").is_array() || argument.at("choices").empty())
                        throw std::invalid_argument("choices must be a non-empty array");
                    std::vector<Json> choices;
                    for (const auto& choice : argument.at("choices")) {
                        validate_value(argument, choice);
                        if (std::find(choices.begin(), choices.end(), choice) != choices.end())
                            throw std::invalid_argument("duplicate argument choice");
                        choices.push_back(choice);
                    }
                }
                if (argument.contains("default")) {
                    if (action == "append") {
                        if (!argument.at("default").is_array())
                            throw std::invalid_argument("append default must be an array");
                        for (const auto& item : argument.at("default"))
                            validate_value(argument, item);
                    } else {
                        validate_value(argument, argument.at("default"));
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        throw std::invalid_argument("invalid family CLI declaration " + path.string() + ": " +
                                    error.what());
    }
    return descriptor;
}

void help(const std::string& family, const Json& descriptor, const Json* selected,
          std::ostream& output) {
    output << "Usage: trtmc " << family << ' ';
    if (selected == nullptr) {
        output << "COMMAND [ARGUMENTS]\n\n";
        for (const auto& command : descriptor.at("commands"))
            output << "  " << command.at("name").get<std::string>() << "  "
                   << command.value("help", std::string{}) << '\n';
        return;
    }
    output << selected->at("name").get<std::string>() << " [ARGUMENTS]\n\n"
           << selected->value("help", std::string{}) << '\n';
    for (const auto& argument : selected->at("arguments")) {
        output << "  ";
        if (argument.contains("flags")) {
            for (const auto& flag : argument.at("flags"))
                output << flag.get<std::string>() << ' ';
        } else {
            output << argument.at("name").get<std::string>() << ' ';
        }
        output << '(' << argument.at("type").get<std::string>() << ") "
               << argument.value("help", std::string{});
        if (argument.value("required", !argument.contains("flags")))
            output << " [required]";
        if (argument.contains("default"))
            output << " [default: " << argument.at("default").dump() << ']';
        if (argument.contains("choices"))
            output << " [choices: " << argument.at("choices").dump() << ']';
        output << '\n';
    }
}

Json parse_values(const Json& command, int argc, char** argv) {
    std::map<std::string, const Json*> flags;
    std::vector<const Json*> positional;
    for (const auto& argument : command.at("arguments")) {
        if (argument.contains("flags")) {
            for (const auto& flag : argument.at("flags"))
                flags.emplace(flag.get<std::string>(), &argument);
        } else {
            positional.push_back(&argument);
        }
    }
    Json values = Json::object();
    std::vector<std::string> positional_values;
    bool options = true;
    const std::regex negative_number("-([0-9]+|[0-9]*\\.[0-9]+)");
    for (int i = 3; i < argc; ++i) {
        std::string token = argv[i], text;
        if (options && token == "--") {
            options = false;
            continue;
        }
        const Json* argument = nullptr;
        if (options && token.size() > 1 && token.front() == '-' &&
            !std::regex_match(token, negative_number)) {
            const auto equals = token.find('=');
            const auto flag = token.substr(0, equals);
            const auto found = flags.find(flag);
            if (found == flags.end())
                throw std::invalid_argument("unknown argument: " + flag);
            argument = found->second;
            if (argument->value("action", std::string{}) == "store_true") {
                if (equals != std::string::npos)
                    throw std::invalid_argument(flag + " does not take a value");
                text = "true";
            } else if (equals != std::string::npos) {
                text = token.substr(equals + 1);
            } else {
                if (++i == argc)
                    throw std::invalid_argument(flag + " requires a value");
                text = argv[i];
                if (text.size() > 1 && text.front() == '-') {
                    const auto type = argument->at("type").get<std::string>();
                    if (type != "int" && type != "float")
                        throw std::invalid_argument(flag + " requires a value");
                    (void)scalar(text, type); // A numeric flag may consume a negative literal.
                }
            }
        } else {
            positional_values.push_back(std::move(token));
            continue;
        }
        const auto name = argument->at("name").get<std::string>();
        const auto value = scalar(text, argument->at("type").get<std::string>());
        validate_value(*argument, value);
        if (argument->value("action", std::string{}) == "append") {
            if (!values.contains(name))
                values[name] =
                    argument->contains("default") ? default_value(*argument) : Json::array();
            values[name].push_back(value);
        } else {
            if (values.contains(name))
                throw std::invalid_argument("duplicate argument: " + name);
            values[name] = value;
        }
    }
    std::size_t position = 0;
    for (std::size_t i = 0; i < positional.size(); ++i) {
        const auto& argument = *positional[i];
        const auto required_after =
            std::count_if(positional.begin() + i + 1, positional.end(),
                          [](const Json* spec) { return spec->value("required", true); });
        if (position == positional_values.size() ||
            (!argument.value("required", true) &&
             positional_values.size() - position <= static_cast<std::size_t>(required_after)))
            continue;
        const auto value =
            scalar(positional_values[position++], argument.at("type").get<std::string>());
        validate_value(argument, value);
        values[argument.at("name").get<std::string>()] = value;
    }
    if (position != positional_values.size())
        throw std::invalid_argument("unexpected positional argument: " +
                                    positional_values[position]);
    for (const auto& argument : command.at("arguments")) {
        const auto name = argument.at("name").get<std::string>();
        if (values.contains(name))
            continue;
        if (argument.value("required", !argument.contains("flags")))
            throw std::invalid_argument("missing required argument: " + name);
        if (argument.contains("default"))
            values[name] = default_value(argument);
    }
    return values;
}

struct Sinks {
    std::ostream& output;
    std::ostream& error;
    bool failed{false};
};
void write_output(void* context, const char* data, std::size_t size) {
    auto& sinks = *static_cast<Sinks*>(context);
    try {
        sinks.output.write(data, static_cast<std::streamsize>(size));
    } catch (...) {
        sinks.failed = true;
    }
}
void write_error(void* context, const char* data, std::size_t size) {
    auto& sinks = *static_cast<Sinks*>(context);
    try {
        sinks.error.write(data, static_cast<std::streamsize>(size));
    } catch (...) {
        sinks.failed = true;
    }
}

int invoke(const fs::path& executable, const std::string& family, const Json& command,
           const Json& values, int argc, char** argv, std::ostream& output, std::ostream& error) {
    if (command.at("executor") == "python") {
        std::vector<char*> arguments{const_cast<char*>("python3"), const_cast<char*>("-m"),
                                     const_cast<char*>("tensorrt_model_connect")};
        for (int i = 1; i < argc; ++i)
            arguments.push_back(argv[i]);
        arguments.push_back(nullptr);
        execvp(arguments.front(), arguments.data());
        throw std::runtime_error("cannot execute Python family command: " +
                                 std::string(std::strerror(errno)));
    }
    const auto directory = executable.parent_path();
    fs::path library;
    for (const auto& root : {directory, directory / "../lib", directory / "../lib64"}) {
        auto candidate = root / ("libtrtmc_cli_" + family + ".so");
        if (fs::is_regular_file(candidate)) {
            library = fs::absolute(candidate).lexically_normal();
            break;
        }
    }
    if (library.empty())
        throw std::runtime_error("family CLI library is not installed: " + family);
    void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr)
        throw std::runtime_error("cannot load family CLI: " + std::string(dlerror()));
    struct Close {
        void* handle;
        ~Close() { dlclose(handle); }
    } close{handle};
    dlerror();
    auto dispatch = reinterpret_cast<trtmc_family_cli_fn_v1>(dlsym(handle, "trtmc_family_cli_v1"));
    if (const char* reason = dlerror(); reason != nullptr || dispatch == nullptr)
        throw std::runtime_error("family does not provide trtmc_family_cli_v1: " + family);
    Sinks sinks{output, error};
    const auto result =
        dispatch(command.at("handler").get<std::string>().c_str(), values.dump().c_str(),
                 library.parent_path().c_str(), &sinks, write_output, write_error);
    if (sinks.failed || !output || !error)
        throw std::runtime_error("failed to write family CLI output");
    return result;
}
} // namespace

std::optional<int> run_family_cli(int argc, char** argv, std::ostream& output, std::ostream& error,
                                  const fs::path& executable_override) {
    try {
        const auto executable = executable_override.empty() ? fs::read_symlink("/proc/self/exe")
                                                            : fs::absolute(executable_override);
        const std::string family = argc < 2 ? "--help" : argv[1];
        if (family == "--help" || family == "-h" || family == "help") {
            std::map<std::string, Json> declarations;
            for (const auto& root : roots(executable)) {
                if (!fs::is_directory(root))
                    continue;
                std::map<std::string, fs::path> paths;
                for (const auto& entry : fs::directory_iterator(root)) {
                    const auto name = entry.path().filename().string();
                    if (!fs::exists(entry.path() / "cli.json"))
                        continue;
                    if (!identifier(name))
                        throw std::invalid_argument("invalid family identifier: " + name);
                    paths.emplace(name, entry.path() / "cli.json");
                }
                for (const auto& [name, path] : paths) {
                    const auto descriptor = read_descriptor(path);
                    const auto previous = declarations.find(name);
                    if (previous != declarations.end() && previous->second != descriptor)
                        throw std::invalid_argument("conflicting family CLI declarations: " + name);
                    declarations.emplace(name, descriptor);
                }
            }
            if (!declarations.empty()) {
                output << "Family commands: trtmc FAMILY COMMAND [ARGUMENTS]\n\n";
                for (const auto& [name, descriptor] : declarations)
                    help(name, descriptor, nullptr, output);
                output << '\n';
            }
            return std::nullopt; // The caller also prints the existing SDK/legacy usage.
        }
        if (!identifier(family))
            return std::nullopt;
        std::optional<Json> found_descriptor;
        for (const auto& root : roots(executable)) {
            const auto path = root / family / "cli.json";
            if (!fs::exists(path))
                continue;
            auto descriptor = read_descriptor(path);
            if (found_descriptor && *found_descriptor != descriptor)
                throw std::invalid_argument("conflicting family CLI declarations: " + family);
            found_descriptor = std::move(descriptor);
        }
        if (!found_descriptor)
            return std::nullopt;
        const auto& descriptor = *found_descriptor;
        if (argc == 2 || std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h") {
            help(family, descriptor, nullptr, output);
            return 0;
        }
        const Json* command = nullptr;
        for (const auto& candidate : descriptor.at("commands")) {
            if (candidate.at("name") == argv[2])
                command = &candidate;
        }
        if (command == nullptr)
            throw std::invalid_argument("unknown command for family " + family + ": " + argv[2]);
        for (int i = 3; i < argc; ++i) {
            if (std::string(argv[i]) == "--")
                break;
            if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
                help(family, descriptor, command, output);
                return 0;
            }
        }
        const auto values = parse_values(*command, argc, argv);
        return invoke(executable, family, *command, values, argc, argv, output, error);
    } catch (const std::invalid_argument& exception) {
        error << "Error: " << exception.what() << '\n';
        return 2;
    } catch (const std::exception& exception) {
        error << "Error: " << exception.what() << '\n';
        return 1;
    }
}
} // namespace trtmc::cli
