/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "installer/windows_h3_installer.h"

#ifndef _WIN32
#error "MiniMaxH3Setup is Windows-only"
#endif

#define WIN32_LEAN_AND_MEAN
// clang-format off
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
// clang-format on

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <filesystem>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr char kMarkerName[] = ".minimax-h3-install-id";
constexpr char kMarkerContents[] = "trtmc-minimax-h3-native-install-v1\n";
constexpr wchar_t kUninstallKey[] =
    L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\ModelConnectMiniMaxH3";
constexpr wchar_t kAppPathsKey[] =
    L"Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\trtmc.exe";

struct SetupArguments {
    fs::path install_root;
    fs::path payload_root;
    bool uninstall{false};
    bool quiet{false};
    bool modify_path{true};
    bool verify_only{false};
    bool show_help{false};
};

std::runtime_error windows_error(const std::string& operation, LONG code = GetLastError()) {
    return std::runtime_error(operation + " failed with Windows error " + std::to_string(code));
}

fs::path executable_path() {
    std::vector<wchar_t> buffer(32768);
    const DWORD length =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size())
        throw windows_error("GetModuleFileNameW");
    return fs::path(std::wstring(buffer.data(), length));
}

fs::path default_install_root() {
    PWSTR raw = nullptr;
    const HRESULT status =
        SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_CREATE, nullptr, &raw);
    if (FAILED(status) || raw == nullptr)
        throw std::runtime_error("Unable to locate the current user's LocalAppData directory");
    const fs::path result = fs::path(raw) / L"Programs" / L"ModelConnect" / L"MiniMax-H3";
    CoTaskMemFree(raw);
    return result;
}

SetupArguments parse_arguments(const fs::path& exe) {
    SetupArguments result;
    result.install_root = _wcsicmp(exe.filename().c_str(), L"UninstallMiniMaxH3.exe") == 0
                              ? exe.parent_path()
                              : default_install_root();
    result.payload_root = exe.parent_path() / L"payload";
    int count = 0;
    LPWSTR* values = CommandLineToArgvW(GetCommandLineW(), &count);
    if (values == nullptr)
        throw windows_error("CommandLineToArgvW");
    const auto release = [&] { LocalFree(values); };
    try {
        for (int index = 1; index < count; ++index) {
            const std::wstring value(values[index]);
            const auto require_value = [&](const wchar_t* name) -> fs::path {
                if (++index >= count)
                    throw std::runtime_error("Missing value for setup option");
                (void)name;
                return fs::absolute(fs::path(values[index]));
            };
            if (value == L"--install-dir") {
                result.install_root = require_value(L"--install-dir");
            } else if (value == L"--payload-dir") {
                result.payload_root = require_value(L"--payload-dir");
            } else if (value == L"--uninstall") {
                result.uninstall = true;
            } else if (value == L"--quiet") {
                result.quiet = true;
            } else if (value == L"--no-path") {
                result.modify_path = false;
            } else if (value == L"--verify-only") {
                result.verify_only = true;
            } else if (value == L"--help" || value == L"-h" || value == L"/?") {
                result.show_help = true;
            } else {
                throw std::runtime_error("Unknown MiniMaxH3Setup option");
            }
        }
    } catch (...) {
        release();
        throw;
    }
    release();
    result.install_root = fs::absolute(result.install_root).lexically_normal();
    result.payload_root = fs::absolute(result.payload_root).lexically_normal();
    return result;
}

std::wstring quote_argument(const fs::path& path) {
    const auto value = path.wstring();
    if (value.find(L'"') != std::wstring::npos)
        throw std::runtime_error("Installer path contains an unsupported quote character");
    return L"\"" + value + L"\"";
}

void set_registry_string(HKEY root, const std::wstring& key_path, const wchar_t* value_name,
                         const std::wstring& value, DWORD type = REG_SZ) {
    HKEY key = nullptr;
    const LONG opened = RegCreateKeyExW(root, key_path.c_str(), 0, nullptr, 0, KEY_SET_VALUE,
                                        nullptr, &key, nullptr);
    if (opened != ERROR_SUCCESS)
        throw windows_error("RegCreateKeyExW", opened);
    const DWORD bytes = static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t));
    const LONG written = RegSetValueExW(key, value_name, 0, type,
                                        reinterpret_cast<const BYTE*>(value.c_str()), bytes);
    RegCloseKey(key);
    if (written != ERROR_SUCCESS)
        throw windows_error("RegSetValueExW", written);
}

void set_registry_dword(HKEY root, const std::wstring& key_path, const wchar_t* value_name,
                        DWORD value) {
    HKEY key = nullptr;
    const LONG opened = RegCreateKeyExW(root, key_path.c_str(), 0, nullptr, 0, KEY_SET_VALUE,
                                        nullptr, &key, nullptr);
    if (opened != ERROR_SUCCESS)
        throw windows_error("RegCreateKeyExW", opened);
    const LONG written = RegSetValueExW(key, value_name, 0, REG_DWORD,
                                        reinterpret_cast<const BYTE*>(&value), sizeof(value));
    RegCloseKey(key);
    if (written != ERROR_SUCCESS)
        throw windows_error("RegSetValueExW", written);
}

std::wstring query_user_path() {
    HKEY key = nullptr;
    const LONG opened = RegOpenKeyExW(HKEY_CURRENT_USER, L"Environment", 0, KEY_QUERY_VALUE, &key);
    if (opened == ERROR_FILE_NOT_FOUND)
        return {};
    if (opened != ERROR_SUCCESS)
        throw windows_error("RegOpenKeyExW", opened);
    DWORD type = 0;
    DWORD bytes = 0;
    LONG status = RegQueryValueExW(key, L"Path", nullptr, &type, nullptr, &bytes);
    if (status == ERROR_FILE_NOT_FOUND) {
        RegCloseKey(key);
        return {};
    }
    if (status != ERROR_SUCCESS || (type != REG_SZ && type != REG_EXPAND_SZ)) {
        RegCloseKey(key);
        throw windows_error("RegQueryValueExW", status);
    }
    std::vector<wchar_t> buffer(bytes / sizeof(wchar_t) + 1, L'\0');
    status = RegQueryValueExW(key, L"Path", nullptr, &type, reinterpret_cast<BYTE*>(buffer.data()),
                              &bytes);
    RegCloseKey(key);
    if (status != ERROR_SUCCESS)
        throw windows_error("RegQueryValueExW", status);
    return std::wstring(buffer.data());
}

std::wstring normalized_path_token(std::wstring value) {
    while (value.size() > 3 && (value.back() == L'\\' || value.back() == L'/'))
        value.pop_back();
    std::transform(value.begin(), value.end(), value.begin(), [](wchar_t character) {
        return static_cast<wchar_t>(std::towlower(character));
    });
    return value;
}

std::vector<std::wstring> split_path(const std::wstring& value) {
    std::vector<std::wstring> result;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const auto end = value.find(L';', begin);
        auto token =
            value.substr(begin, end == std::wstring::npos ? std::wstring::npos : end - begin);
        if (!token.empty())
            result.push_back(std::move(token));
        if (end == std::wstring::npos)
            break;
        begin = end + 1;
    }
    return result;
}

void update_user_path(const fs::path& bin_path, bool add) {
    auto tokens = split_path(query_user_path());
    const auto target = normalized_path_token(bin_path.wstring());
    tokens.erase(std::remove_if(tokens.begin(), tokens.end(),
                                [&](const std::wstring& token) {
                                    return normalized_path_token(token) == target;
                                }),
                 tokens.end());
    if (add)
        tokens.push_back(bin_path.wstring());
    std::wstring joined;
    for (const auto& token : tokens) {
        if (!joined.empty())
            joined += L';';
        joined += token;
    }
    set_registry_string(HKEY_CURRENT_USER, L"Environment", L"Path", joined, REG_EXPAND_SZ);
    DWORD_PTR ignored = 0;
    SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0,
                        reinterpret_cast<LPARAM>(L"Environment"), SMTO_ABORTIFHUNG, 5000, &ignored);
}

void register_installation(const fs::path& install_root, bool modify_path) {
    const auto bin = install_root / L"bin";
    const auto cli = bin / L"trtmc.exe";
    const auto uninstaller = install_root / L"UninstallMiniMaxH3.exe";
    set_registry_string(HKEY_CURRENT_USER, kAppPathsKey, nullptr, cli.wstring());
    set_registry_string(HKEY_CURRENT_USER, kAppPathsKey, L"Path", bin.wstring());
    set_registry_string(HKEY_CURRENT_USER, kUninstallKey, L"DisplayName",
                        L"ModelConnect MiniMax-H3 Native Runtime");
    set_registry_string(HKEY_CURRENT_USER, kUninstallKey, L"DisplayVersion", L"0.1.0");
    set_registry_string(HKEY_CURRENT_USER, kUninstallKey, L"Publisher", L"NVIDIA");
    set_registry_string(HKEY_CURRENT_USER, kUninstallKey, L"InstallLocation",
                        install_root.wstring());
    set_registry_string(HKEY_CURRENT_USER, kUninstallKey, L"DisplayIcon", cli.wstring());
    set_registry_string(HKEY_CURRENT_USER, kUninstallKey, L"UninstallString",
                        quote_argument(uninstaller) + L" --uninstall");
    set_registry_dword(HKEY_CURRENT_USER, kUninstallKey, L"NoModify", 1);
    set_registry_dword(HKEY_CURRENT_USER, kUninstallKey, L"NoRepair", 1);
    if (modify_path)
        update_user_path(bin, true);
}

void unregister_installation(const fs::path& install_root) {
    update_user_path(install_root / L"bin", false);
    const LONG app_status = RegDeleteTreeW(HKEY_CURRENT_USER, kAppPathsKey);
    if (app_status != ERROR_SUCCESS && app_status != ERROR_FILE_NOT_FOUND)
        throw windows_error("RegDeleteTreeW(App Paths)", app_status);
    const LONG uninstall_status = RegDeleteTreeW(HKEY_CURRENT_USER, kUninstallKey);
    if (uninstall_status != ERROR_SUCCESS && uninstall_status != ERROR_FILE_NOT_FOUND)
        throw windows_error("RegDeleteTreeW(Uninstall)", uninstall_status);
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

void validate_runtime_payload(const std::vector<trtmc::installer::PayloadEntry>& entries) {
    std::set<std::string> paths;
    for (const auto& entry : entries)
        paths.insert(lowercase(entry.relative_path.generic_u8string()));
    for (const char* required :
         {"bin/trtmc.exe", "bin/trtmc_core.dll", "bin/trtmc_backend_trt_rtx.dll",
          "bin/trtmc/models/minimax_h3/trtmc_model_minimax_h3.dll", "models/MiniMax-H3.bundle",
          "UninstallMiniMaxH3.exe", kMarkerName}) {
        if (paths.count(lowercase(required)) == 0)
            throw std::runtime_error(std::string("Payload is missing required file: ") + required);
    }
    const bool has_rtx = std::any_of(paths.begin(), paths.end(), [](const std::string& path) {
        return path.rfind("bin/tensorrt_rtx_", 0) == 0 && path.size() > 4 &&
               path.compare(path.size() - 4, 4, ".dll") == 0;
    });
    if (!has_rtx)
        throw std::runtime_error("Payload is missing the TensorRT-RTX runtime DLL");
}

void uninstall_files(const fs::path& install_root, const fs::path& running_executable) {
    const auto canonical_root = fs::weakly_canonical(install_root);
    const auto canonical_exe = fs::weakly_canonical(running_executable);
    const bool running_inside = canonical_exe.parent_path() == canonical_root;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(canonical_root)) {
        const auto child = fs::weakly_canonical(entry.path());
        if (running_inside && child == canonical_exe)
            continue;
        fs::remove_all(entry.path(), error);
        if (error)
            throw std::runtime_error("Unable to remove installed file: " + error.message());
    }
    if (running_inside) {
        if (!MoveFileExW(canonical_exe.c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT))
            throw windows_error("MoveFileExW(uninstaller)");
        MoveFileExW(canonical_root.c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
    } else {
        fs::remove(canonical_root, error);
        if (error)
            throw std::runtime_error("Unable to remove installation directory: " + error.message());
    }
}

void show_message(bool quiet, const std::wstring& text, UINT flags) {
    if (!quiet)
        MessageBoxW(nullptr, text.c_str(), L"ModelConnect MiniMax-H3 Setup", flags);
}

} // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    bool quiet = false;
    try {
        const auto exe = executable_path();
        const auto args = parse_arguments(exe);
        quiet = args.quiet;
        if (args.show_help) {
            show_message(false,
                         L"Double-click to install. Optional command-line flags:\n\n"
                         L"--install-dir PATH   Select a per-user destination\n"
                         L"--payload-dir PATH   Select the SHA-256-verified layout payload\n"
                         L"--no-path            Do not add trtmc.exe to the user PATH\n"
                         L"--verify-only        Verify all payload SHA-256 records\n"
                         L"--quiet              Suppress setup dialogs\n"
                         L"--uninstall          Remove the installed runtime",
                         MB_OK | MB_ICONINFORMATION);
            return 0;
        }
        if (args.uninstall) {
            if (!trtmc::installer::installation_marker_matches(args.install_root, kMarkerName,
                                                               kMarkerContents)) {
                throw std::runtime_error(
                    "Refusing to uninstall a directory without the ModelConnect H3 marker");
            }
            unregister_installation(args.install_root);
            uninstall_files(args.install_root, exe);
            show_message(args.quiet, L"ModelConnect MiniMax-H3 was uninstalled.",
                         MB_OK | MB_ICONINFORMATION);
            return 0;
        }

        const auto manifest = args.payload_root.parent_path() / L"payload.manifest";
        const auto entries = trtmc::installer::read_payload_manifest(manifest);
        validate_runtime_payload(entries);
        if (args.verify_only) {
            trtmc::installer::verify_payload(args.payload_root, entries);
            show_message(args.quiet, L"The MiniMax-H3 payload passed SHA-256 verification.",
                         MB_OK | MB_ICONINFORMATION);
            return 0;
        }
        trtmc::installer::install_payload_transactional(args.payload_root, args.install_root,
                                                        entries, kMarkerName, kMarkerContents);
        register_installation(args.install_root, args.modify_path);
        const auto bundle = args.install_root / L"models" / L"MiniMax-H3.bundle";
        show_message(args.quiet,
                     L"ModelConnect MiniMax-H3 was installed successfully.\n\nBundle:\n" +
                         bundle.wstring() + L"\n\nNew terminals can invoke trtmc.exe directly.",
                     MB_OK | MB_ICONINFORMATION);
        return 0;
    } catch (const std::exception& error) {
        std::wstring message = L"Setup failed.\n\n";
        const auto* begin = error.what();
        const auto length = static_cast<int>(std::strlen(begin));
        const int required = MultiByteToWideChar(CP_UTF8, 0, begin, length, nullptr, 0);
        if (required > 0) {
            std::wstring converted(static_cast<std::size_t>(required), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, begin, length, converted.data(), required);
            message += converted;
        } else {
            message += L"Unknown error";
        }
        show_message(quiet, message, MB_OK | MB_ICONERROR);
        return 1;
    }
}
