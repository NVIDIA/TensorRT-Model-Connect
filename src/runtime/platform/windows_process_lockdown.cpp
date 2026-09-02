/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/platform/windows_process_lockdown.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <array>
#include <mutex>
#include <stdexcept>
#include <string>
#include <windows.h>

namespace trtmc::internal {
namespace {

HANDLE& process_job() noexcept {
    static HANDLE handle = nullptr;
    return handle;
}

std::once_flag& process_job_once() noexcept {
    static std::once_flag once;
    return once;
}

[[noreturn]] void throw_windows_error(const char* operation, DWORD error) {
    throw std::runtime_error(std::string(operation) + " failed with Windows error " +
                             std::to_string(error));
}

bool environment_variable_is_present(const wchar_t* name) {
    SetLastError(ERROR_SUCCESS);
    const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
    if (size != 0)
        return true;
    const DWORD error = GetLastError();
    if (error == ERROR_ENVVAR_NOT_FOUND)
        return false;
    if (error == ERROR_SUCCESS)
        return true; // A present, empty environment variable is still an override attempt.
    throw_windows_error("GetEnvironmentVariableW", error);
}

} // namespace

void reject_locked_runtime_override_environment() {
    struct BlockedVariable {
        const wchar_t* wide_name;
        const char* name;
    };
    constexpr std::array<BlockedVariable, 5> blocked{{
        {L"TRTMC_BACKEND_DIR", "TRTMC_BACKEND_DIR"},
        {L"TRTMC_MODEL_PLUGIN_DIR", "TRTMC_MODEL_PLUGIN_DIR"},
        {L"TRTMC_MODEL_PLUGIN_STRICT", "TRTMC_MODEL_PLUGIN_STRICT"},
        {L"TRTMC_KERNEL_BINDINGS", "TRTMC_KERNEL_BINDINGS"},
        {L"TRTMC_KERNEL_BINDINGS_PATH", "TRTMC_KERNEL_BINDINGS_PATH"},
    }};
    for (const auto& variable : blocked) {
        if (environment_variable_is_present(variable.wide_name)) {
            throw std::runtime_error(std::string("locked MiniMax-H3 runtime rejects environment ") +
                                     variable.name);
        }
    }
}

void enforce_single_process_job() {
    std::call_once(process_job_once(), [] {
        HANDLE job = CreateJobObjectW(nullptr, nullptr);
        if (job == nullptr)
            throw_windows_error("CreateJobObjectW", GetLastError());

        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
        limits.BasicLimitInformation.ActiveProcessLimit = 1;
        if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits,
                                     sizeof(limits))) {
            const DWORD error = GetLastError();
            CloseHandle(job);
            throw_windows_error("SetInformationJobObject", error);
        }
        if (!AssignProcessToJobObject(job, GetCurrentProcess())) {
            const DWORD error = GetLastError();
            CloseHandle(job);
            throw_windows_error("AssignProcessToJobObject", error);
        }
        process_job() = job;
    });

    if (!single_process_job_is_active()) {
        throw std::runtime_error("locked MiniMax-H3 process Job Object lost its limit");
    }
}

void enforce_locked_h3_process_policy() {
    reject_locked_runtime_override_environment();
    enforce_single_process_job();
}

bool single_process_job_is_active() noexcept {
    if (process_job() == nullptr)
        return false;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    if (!QueryInformationJobObject(process_job(), JobObjectExtendedLimitInformation, &limits,
                                   sizeof(limits), nullptr)) {
        return false;
    }
    return (limits.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_ACTIVE_PROCESS) != 0 &&
           limits.BasicLimitInformation.ActiveProcessLimit == 1;
}

} // namespace trtmc::internal
