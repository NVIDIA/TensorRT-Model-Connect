/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "installer/windows_h3_installer.h"

#define WIN32_LEAN_AND_MEAN
#include <aclapi.h>
#include <cstdlib>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Callable>
void check_throws(Callable&& callable, const char* label) {
    try {
        callable();
        check(false, label);
    } catch (const std::runtime_error&) {
    }
}

template <typename Callable>
std::string thrown_message(Callable&& callable, const char* label) {
    try {
        callable();
        check(false, label);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

void write_bytes(const fs::path& path, const std::string& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output)
        throw std::runtime_error("Unable to write installer test file");
}

void write_file_with_size(const fs::path& path, std::uint64_t size) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (size != 0) {
        output.seekp(static_cast<std::streamoff>(size - 1));
        output.put('\0');
    }
    if (!output)
        throw std::runtime_error("Unable to size installer test file");
}

std::string read_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Unable to read installer test file");
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

fs::path current_executable() {
    std::vector<wchar_t> buffer(32768);
    const DWORD size =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (size == 0 || size >= buffer.size())
        throw std::runtime_error("Unable to locate installer test executable");
    return fs::path(std::wstring(buffer.data(), size));
}

HMODULE load_manifest_anchor(const fs::path& destination, const std::string& manifest_bytes) {
    if (!CopyFileW(current_executable().c_str(), destination.c_str(), TRUE))
        throw std::runtime_error("Unable to copy installer test executable");
    HANDLE update = BeginUpdateResourceW(destination.c_str(), FALSE);
    if (update == nullptr)
        throw std::runtime_error("Unable to open installer test resource update");
    std::vector<char> mutable_bytes(manifest_bytes.begin(), manifest_bytes.end());
    if (!UpdateResourceW(update, MAKEINTRESOURCEW(10),
                         MAKEINTRESOURCEW(trtmc::installer::kPayloadManifestResourceId), 0,
                         mutable_bytes.data(), static_cast<DWORD>(mutable_bytes.size()))) {
        EndUpdateResourceW(update, TRUE);
        throw std::runtime_error("Unable to stamp installer test manifest resource");
    }
    if (!EndUpdateResourceW(update, FALSE))
        throw std::runtime_error("Unable to commit installer test manifest resource");
    HMODULE module = LoadLibraryExW(destination.c_str(), nullptr,
                                    LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
    if (module == nullptr)
        throw std::runtime_error("Unable to load installer test manifest resource");
    return module;
}

constexpr DWORD kMutexChildTimedOut = 73;
constexpr DWORD kMutexChildFailed = 74;

std::wstring quote_child_argument(const std::wstring& value) {
    if (value.find(L'"') != std::wstring::npos)
        throw std::runtime_error("Installer test child argument contains a quote");
    return L"\"" + value + L"\"";
}

PROCESS_INFORMATION launch_mutex_child(const wchar_t* mode, const fs::path& install_root,
                                       DWORD timeout_ms) {
    const auto executable = current_executable();
    std::wstring command_line = quote_child_argument(executable.wstring()) + L" " + mode + L" " +
                                quote_child_argument(install_root.wstring()) + L" " +
                                std::to_wstring(timeout_ms);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION child{};
    if (!CreateProcessW(executable.c_str(), command_line.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &child)) {
        throw std::runtime_error("Unable to launch installer mutex test child");
    }
    CloseHandle(child.hThread);
    child.hThread = nullptr;
    return child;
}

DWORD wait_mutex_child(PROCESS_INFORMATION& child) {
    const DWORD waited = WaitForSingleObject(child.hProcess, 15000);
    if (waited != WAIT_OBJECT_0) {
        (void)TerminateProcess(child.hProcess, kMutexChildFailed);
        (void)WaitForSingleObject(child.hProcess, 5000);
        CloseHandle(child.hProcess);
        child.hProcess = nullptr;
        throw std::runtime_error("Installer mutex test child did not exit in time");
    }
    DWORD exit_code = kMutexChildFailed;
    if (!GetExitCodeProcess(child.hProcess, &exit_code)) {
        CloseHandle(child.hProcess);
        child.hProcess = nullptr;
        throw std::runtime_error("Unable to read installer mutex test child exit code");
    }
    CloseHandle(child.hProcess);
    child.hProcess = nullptr;
    return exit_code;
}

DWORD run_mutex_child(const wchar_t* mode, const fs::path& install_root, DWORD timeout_ms) {
    auto child = launch_mutex_child(mode, install_root, timeout_ms);
    return wait_mutex_child(child);
}

void check_mutex_user_only_dacl(const std::wstring& name) {
    HANDLE mutex = OpenMutexW(READ_CONTROL | SYNCHRONIZE, FALSE, name.c_str());
    if (mutex == nullptr)
        throw std::runtime_error("Unable to open installer mutex security descriptor");
    PACL dacl = nullptr;
    PSID owner = nullptr;
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    const DWORD security_status = GetSecurityInfo(
        mutex, SE_KERNEL_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, &owner,
        nullptr, &dacl, nullptr, &descriptor);
    CloseHandle(mutex);
    if (security_status != ERROR_SUCCESS || dacl == nullptr || descriptor == nullptr) {
        if (descriptor != nullptr)
            LocalFree(descriptor);
        throw std::runtime_error("Unable to query installer mutex DACL");
    }

    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
        LocalFree(descriptor);
        throw std::runtime_error("Unable to open installer test process token");
    }
    DWORD required = 0;
    (void)GetTokenInformation(token, TokenUser, nullptr, 0, &required);
    std::vector<std::uint8_t> token_user(required);
    if (required == 0 ||
        !GetTokenInformation(token, TokenUser, token_user.data(), required, &required)) {
        CloseHandle(token);
        LocalFree(descriptor);
        throw std::runtime_error("Unable to query installer test user SID");
    }
    CloseHandle(token);
    const auto* user = reinterpret_cast<const TOKEN_USER*>(token_user.data());

    check(owner != nullptr && EqualSid(owner, user->User.Sid) != FALSE,
          "installer mutex owner is the current user SID");
    check(dacl->AceCount == 1, "installer mutex DACL has exactly one user ACE");
    if (dacl->AceCount == 1) {
        void* raw_ace = nullptr;
        if (!GetAce(dacl, 0, &raw_ace)) {
            LocalFree(descriptor);
            throw std::runtime_error("Unable to query installer mutex DACL entry");
        }
        const auto* ace = static_cast<const ACCESS_ALLOWED_ACE*>(raw_ace);
        const bool is_allowed = ace->Header.AceType == ACCESS_ALLOWED_ACE_TYPE;
        check(is_allowed, "installer mutex DACL entry is allow-only");
        if (is_allowed) {
            check(EqualSid(user->User.Sid, const_cast<DWORD*>(&ace->SidStart)) != FALSE,
                  "installer mutex DACL is scoped to the current user SID");
            check(ace->Mask == (SYNCHRONIZE | MUTEX_MODIFY_STATE | READ_CONTROL),
                  "installer mutex DACL grants only required transaction rights");
        }
    }
    LocalFree(descriptor);
}

int run_mutex_child_mode(int argc, wchar_t** argv) {
    if (argc != 4)
        return static_cast<int>(kMutexChildFailed);
    wchar_t* end = nullptr;
    const unsigned long parsed_timeout = std::wcstoul(argv[3], &end, 10);
    if (end == argv[3] || *end != L'\0' || parsed_timeout > MAXDWORD)
        return static_cast<int>(kMutexChildFailed);
    try {
        auto lock = trtmc::installer::acquire_install_transaction_lock(
            fs::path(argv[2]), static_cast<DWORD>(parsed_timeout));
        (void)lock;
        if (std::wcscmp(argv[1], L"--mutex-abandon") == 0)
            ExitProcess(0);
        if (std::wcscmp(argv[1], L"--mutex-probe") != 0)
            return static_cast<int>(kMutexChildFailed);
        return 0;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("Timed out after ") != std::string::npos)
            return static_cast<int>(kMutexChildTimedOut);
        std::cerr << "Mutex child failure: " << error.what() << '\n';
        return static_cast<int>(kMutexChildFailed);
    }
}

std::string manifest_row(const fs::path& path, const std::string& relative) {
    return trtmc::installer::sha256_hex(trtmc::installer::sha256_file(path)) + "\t" +
           std::to_string(fs::file_size(path)) + "\t" + relative + "\n";
}

trtmc::installer::PayloadEntry manifest_entry(const std::string& relative) {
    trtmc::installer::PayloadEntry entry;
    entry.relative_path = fs::u8path(relative);
    return entry;
}

std::vector<trtmc::installer::PayloadEntry> valid_runtime_entries() {
    std::vector<trtmc::installer::PayloadEntry> entries;
    for (const char* path : {
             ".minimax-h3-install-id",
             "UninstallMiniMaxH3.exe",
             "bin/trtmc.exe",
             "bin/trtmc_core.dll",
             "bin/trtmc_backend_trt_rtx.dll",
             "bin/trtmc/models/minimax_h3/trtmc_model_minimax_h3.dll",
             "bin/tensorrt_rtx_1_6.dll",
             "models/MiniMax-H3.bundle",
             "licenses/LICENSE",
         }) {
        entries.push_back(manifest_entry(path));
    }
    return entries;
}

} // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc > 1 && (std::wcscmp(argv[1], L"--mutex-probe") == 0 ||
                     std::wcscmp(argv[1], L"--mutex-abandon") == 0)) {
        return run_mutex_child_mode(argc, argv);
    }

    check(trtmc::installer::is_safe_payload_path("bin/trtmc.exe"), "normal path accepted");
    check(trtmc::installer::is_safe_payload_path("models/MiniMax-H3.bundle"),
          "model path accepted");
    for (const std::string& path : {"", "/rooted", "../escape", "bin/../escape", "bin\\trtmc.exe",
                                    "C:/absolute", "bin/con", "bin/file. ", "bin/file."}) {
        check(!trtmc::installer::is_safe_payload_path(path), "unsafe path rejected");
    }

    auto runtime_entries = valid_runtime_entries();
    trtmc::installer::validate_minimax_h3_runtime_payload(runtime_entries);
    runtime_entries.push_back(manifest_entry("bin/unapproved.dll"));
    check_throws([&] { trtmc::installer::validate_minimax_h3_runtime_payload(runtime_entries); },
                 "manifest entry outside exact runtime allowlist rejected");
    runtime_entries = valid_runtime_entries();
    runtime_entries.push_back(manifest_entry("bin/tensorrt_rtx_2_0.dll"));
    check_throws([&] { trtmc::installer::validate_minimax_h3_runtime_payload(runtime_entries); },
                 "second TensorRT-RTX runtime rejected");
    runtime_entries = valid_runtime_entries();
    runtime_entries[runtime_entries.size() - 2].relative_path =
        fs::u8path("models/not-the-h3-bundle.bundle");
    check_throws([&] { trtmc::installer::validate_minimax_h3_runtime_payload(runtime_entries); },
                 "wrong model bundle path rejected");

    const auto root = fs::temp_directory_path() /
                      (L"trtmc-h3-installer-test-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                       std::to_wstring(GetTickCount64()));
    const auto payload = root / L"layout" / L"payload";
    const auto manifest = root / L"layout" / L"payload.manifest";
    const auto install = root / L"installed";
    try {
        const std::string marker_name = ".minimax-h3-install-id";
        const std::string marker_contents = "trtmc-minimax-h3-native-install-v1\n";
        write_bytes(payload / L"bin" / L"trtmc.exe", "native-cli");
        write_bytes(payload / fs::u8path(marker_name), marker_contents);
        write_bytes(manifest, manifest_row(payload / L"bin" / L"trtmc.exe", "bin/trtmc.exe") +
                                  manifest_row(payload / fs::u8path(marker_name), marker_name));

        const auto entries = trtmc::installer::read_payload_manifest(manifest);
        check(entries.size() == 2, "manifest entry count");
        trtmc::installer::verify_payload(payload, entries);

        const auto mutex_install = root / L"mutex-install";
        const auto mutex_name_before =
            trtmc::installer::install_transaction_mutex_name(mutex_install);
        check(mutex_name_before.rfind(L"Global\\", 0) == 0,
              "installer mutex is visible across Windows sessions");
        {
            auto transaction_lock =
                trtmc::installer::acquire_install_transaction_lock(mutex_install, 1000);
            check_mutex_user_only_dacl(mutex_name_before);
            check(!transaction_lock.recovered_abandoned_owner(),
                  "fresh installer mutex is not abandoned");
            fs::create_directory(mutex_install);
            const auto mutex_name_after =
                trtmc::installer::install_transaction_mutex_name(mutex_install);
            check(mutex_name_before == mutex_name_after,
                  "installer mutex identity survives target creation");
            check(run_mutex_child(L"--mutex-probe", mutex_install, 150) == kMutexChildTimedOut,
                  "second process cannot enter the same installer transaction");
            check(run_mutex_child(L"--mutex-probe", root / L"different-install", 1000) == 0,
                  "different install roots do not share a transaction mutex");
        }
        check(run_mutex_child(L"--mutex-probe", mutex_install, 1000) == 0,
              "second process enters after installer transaction release");

        {
            auto alias_lock = trtmc::installer::acquire_install_transaction_lock(
                root / L"win32-alias-install.", 1000);
            (void)alias_lock;
            check(run_mutex_child(L"--mutex-probe", root / L"win32-alias-install", 150) ==
                      kMutexChildTimedOut,
                  "trailing-period Win32 aliases share an installer transaction mutex");
        }

        const auto abandoned_install = root / L"abandoned-install";
        HANDLE observer = nullptr;
        PROCESS_INFORMATION abandoning_child{};
        {
            auto transaction_lock =
                trtmc::installer::acquire_install_transaction_lock(abandoned_install, 1000);
            const auto mutex_name =
                trtmc::installer::install_transaction_mutex_name(abandoned_install);
            observer = OpenMutexW(SYNCHRONIZE, FALSE, mutex_name.c_str());
            if (observer == nullptr)
                throw std::runtime_error("Unable to retain abandoned installer test mutex");
            abandoning_child = launch_mutex_child(L"--mutex-abandon", abandoned_install, 5000);
        }
        check(wait_mutex_child(abandoning_child) == 0,
              "installer mutex abandonment child acquired the lock");
        auto recovered_lock =
            trtmc::installer::acquire_install_transaction_lock(abandoned_install, 1000);
        check(recovered_lock.recovered_abandoned_owner(),
              "abandoned installer mutex transfers ownership for recovery");
        CloseHandle(observer);

        const auto oversized_manifest = root / L"oversized.manifest";
        write_file_with_size(oversized_manifest, (64ULL << 20) + 1);
        const auto oversized_error = thrown_message(
            [&] { (void)trtmc::installer::read_payload_manifest(oversized_manifest); },
            "oversized manifest rejected before allocation");
        check(oversized_error.find("exceeds the installer size limit") != std::string::npos,
              "oversized manifest reports bounded-read limit");
        fs::remove(oversized_manifest);

        const auto anchored_executable = root / L"anchored-setup.exe";
        HMODULE anchor = load_manifest_anchor(anchored_executable, read_bytes(manifest));
        const auto authenticated_entries =
            trtmc::installer::read_authenticated_payload_manifest(manifest, anchor);
        check(authenticated_entries.size() == 2, "embedded manifest anchor accepted");

        // Re-hashing a modified payload into the mutable external manifest
        // must not authorize it: the trusted Setup resource remains the root.
        write_bytes(payload / L"bin" / L"trtmc.exe", "attacker-cli");
        write_bytes(manifest, manifest_row(payload / L"bin" / L"trtmc.exe", "bin/trtmc.exe") +
                                  manifest_row(payload / fs::u8path(marker_name), marker_name));
        check_throws(
            [&] { (void)trtmc::installer::read_authenticated_payload_manifest(manifest, anchor); },
            "tampered payload and matching external manifest rejected by embedded anchor");
        FreeLibrary(anchor);
        write_bytes(payload / L"bin" / L"trtmc.exe", "native-cli");
        write_bytes(manifest, manifest_row(payload / L"bin" / L"trtmc.exe", "bin/trtmc.exe") +
                                  manifest_row(payload / fs::u8path(marker_name), marker_name));

        const auto unexpected_sidecar = payload / L"models" / L"MiniMax-H3.effective_config.json";
        write_bytes(unexpected_sidecar, "{}");
        check_throws([&] { trtmc::installer::verify_payload(payload, entries); },
                     "payload file absent from manifest rejected");
        fs::remove(unexpected_sidecar);
        trtmc::installer::install_payload_transactional(payload, install, entries, marker_name,
                                                        marker_contents);
        check(fs::is_regular_file(install / L"bin" / L"trtmc.exe"), "transaction installs payload");
        check(trtmc::installer::installation_marker_matches(install, marker_name, marker_contents),
              "transaction preserves marker");

        write_bytes(payload / L"bin" / L"trtmc.exe", "mutated-source");
        check(fs::file_size(install / L"bin" / L"trtmc.exe") == 10,
              "installed payload is not hard-linked to mutable source");

        write_bytes(payload / L"bin" / L"trtmc.exe", "native-cli-v2");
        write_bytes(manifest, manifest_row(payload / L"bin" / L"trtmc.exe", "bin/trtmc.exe") +
                                  manifest_row(payload / fs::u8path(marker_name), marker_name));
        const auto updated_entries = trtmc::installer::read_payload_manifest(manifest);

        const trtmc::installer::InstallFaultInjector partial_commit =
            [](trtmc::installer::InstallTransactionOperation operation, const fs::path&,
               const fs::path&) -> std::uint32_t {
            return operation == trtmc::installer::InstallTransactionOperation::CommitStaging
                       ? ERROR_WRITE_FAULT
                       : ERROR_SUCCESS;
        };
        check_throws(
            [&] {
                trtmc::installer::install_payload_transactional(payload, install, updated_entries,
                                                                marker_name, marker_contents,
                                                                &partial_commit);
            },
            "partial commit failure reported");
        check(fs::file_size(install / L"bin" / L"trtmc.exe") == 10,
              "partial commit restores prior install");
        check(!fs::exists(root / L"installed.backup"), "partial commit leaves no backup");
        check(!fs::exists(root / L"installed.staging"), "partial commit leaves no staging");

        const trtmc::installer::InstallFaultInjector locked_rollback =
            [](trtmc::installer::InstallTransactionOperation operation, const fs::path&,
               const fs::path&) -> std::uint32_t {
            if (operation == trtmc::installer::InstallTransactionOperation::VerifyFinalPayload) {
                return ERROR_CRC;
            }
            if (operation ==
                trtmc::installer::InstallTransactionOperation::QuarantineFailedInstall) {
                return ERROR_SHARING_VIOLATION;
            }
            return ERROR_SUCCESS;
        };
        const auto locked_error = thrown_message(
            [&] {
                trtmc::installer::install_payload_transactional(payload, install, updated_entries,
                                                                marker_name, marker_contents,
                                                                &locked_rollback);
            },
            "locked rollback failure reported");
        check(locked_error.find((root / L"installed.backup").string()) != std::string::npos,
              "locked failure reports backup path");
        check(locked_error.find((root / L"installed.recovery-corrupt").string()) !=
                  std::string::npos,
              "locked failure reports recovery path");
        check(fs::file_size(root / L"installed.backup" / L"bin" / L"trtmc.exe") == 10,
              "locked failure retains prior backup");
        check(fs::file_size(install / L"bin" / L"trtmc.exe") == 13,
              "locked failure preserves current tree instead of deleting it");

        // A retry sees the retained backup first, quarantines the partial new
        // tree, restores the old install, and can then perform a clean update.
        trtmc::installer::install_payload_transactional(payload, install, updated_entries,
                                                        marker_name, marker_contents);
        check(fs::file_size(install / L"bin" / L"trtmc.exe") == 13,
              "transaction replaces an owned install");
        check(!fs::exists(root / L"installed.backup"), "retry consumes recovery backup");
        check(!fs::exists(root / L"installed.recovery-corrupt"),
              "retry cleans quarantined replacement");

        const auto duplicate_manifest = root / L"duplicate.manifest";
        const auto row = manifest_row(payload / L"bin" / L"trtmc.exe", "bin/trtmc.exe");
        write_bytes(duplicate_manifest, row + row);
        check_throws([&] { (void)trtmc::installer::read_payload_manifest(duplicate_manifest); },
                     "duplicate manifest path rejected");

        const auto unrelated = root / L"unrelated";
        write_bytes(unrelated / L"keep.txt", "owner-data");
        check_throws(
            [&] {
                trtmc::installer::install_payload_transactional(payload, unrelated, updated_entries,
                                                                marker_name, marker_contents);
            },
            "unrelated destination rejected");
        check(fs::is_regular_file(unrelated / L"keep.txt"), "unrelated destination remains intact");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);

    if (failures != 0) {
        std::cerr << failures << " Windows H3 installer test(s) failed\n";
        return 1;
    }
    return 0;
}
