/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/platform/windows_process_lockdown.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

struct EnvironmentValue {
    bool present{false};
    std::wstring value;
};

EnvironmentValue read_environment(const wchar_t* name) {
    std::vector<wchar_t> buffer(32768);
    SetLastError(ERROR_SUCCESS);
    const DWORD size =
        GetEnvironmentVariableW(name, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (size == 0) {
        const DWORD error = GetLastError();
        if (error == ERROR_ENVVAR_NOT_FOUND)
            return {};
        if (error == ERROR_SUCCESS)
            return {true, {}};
        throw std::runtime_error("Unable to read test environment");
    }
    if (size >= buffer.size())
        throw std::runtime_error("Test environment value is too large");
    return {true, std::wstring(buffer.data(), size)};
}

class ScopedEnvironment final {
  public:
    ScopedEnvironment(const wchar_t* name, const wchar_t* value)
        : name_(name), previous_(read_environment(name)) {
        if (!SetEnvironmentVariableW(name, value))
            throw std::runtime_error("Unable to set test environment");
    }

    ~ScopedEnvironment() {
        SetEnvironmentVariableW(name_.c_str(),
                                previous_.present ? previous_.value.c_str() : nullptr);
    }

  private:
    std::wstring name_;
    EnvironmentValue previous_;
};

std::wstring executable_path() {
    std::vector<wchar_t> buffer(32768);
    const DWORD size =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (size == 0 || size >= buffer.size())
        throw std::runtime_error("Unable to locate test executable");
    return std::wstring(buffer.data(), size);
}

void test_environment_override_is_rejected() {
    ScopedEnvironment override(L"TRTMC_BACKEND_DIR", L"C:\\untrusted");
    bool threw = false;
    try {
        trtmc::internal::reject_locked_runtime_override_environment();
    } catch (const std::runtime_error& error) {
        threw = true;
        check(std::string(error.what()).find("TRTMC_BACKEND_DIR") != std::string::npos,
              "environment error identifies the rejected override");
    }
    check(threw, "locked runtime rejects an override environment variable");
}

void test_job_blocks_child_processes() {
    trtmc::internal::enforce_locked_h3_process_policy();
    check(trtmc::internal::single_process_job_is_active(),
          "locked process has an active-process limit of one");
    trtmc::internal::enforce_locked_h3_process_policy();
    check(trtmc::internal::single_process_job_is_active(), "process lockdown is idempotent");

    const std::wstring executable = executable_path();
    std::wstring command_line = L"\"" + executable + L"\" --lockdown-child";
    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const BOOL created =
        CreateProcessW(executable.c_str(), mutable_command.data(), nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    if (created) {
        TerminateProcess(process.hProcess, 99);
        WaitForSingleObject(process.hProcess, 5000);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
    }
    check(created == FALSE, "one-process Job Object rejects child creation");
}

void test_concurrent_job_initialization() {
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int index = 0; index < 8; ++index) {
        threads.emplace_back([&] {
            try {
                trtmc::internal::enforce_locked_h3_process_policy();
            } catch (...) {
                ++errors;
            }
        });
    }
    for (auto& thread : threads)
        thread.join();
    check(errors.load() == 0, "concurrent process lockdown initialization succeeds");
    check(trtmc::internal::single_process_job_is_active(),
          "concurrent initialization retains the one-process limit");
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--lockdown-child")
        return 42;
    test_environment_override_is_rejected();
    test_concurrent_job_initialization();
    test_job_blocks_child_processes();
    return failures == 0 ? 0 : 1;
}
