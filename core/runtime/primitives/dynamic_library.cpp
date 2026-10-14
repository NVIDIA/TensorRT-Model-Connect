/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/dynamic_library.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#if defined(__GLIBC__)
#include <link.h>
#endif
#endif

namespace trtmc::platform {
namespace {

#if defined(_WIN32)
std::wstring widen(const std::string& value) {
    if (value.empty())
        return {};
    const int size =
        MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (size <= 0)
        throw std::runtime_error("Library name is not valid UTF-8: " + value);
    std::wstring result(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), result.data(),
                        size);
    return result;
}

std::string narrow(const std::wstring& value) {
    if (value.empty())
        return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                                         nullptr, 0, nullptr, nullptr);
    if (size <= 0)
        return {};
    std::string result(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), result.data(),
                        size, nullptr, nullptr);
    return result;
}

// The system text for a Windows error code, without the trailing period and line break.
std::string windows_system_message(DWORD code) {
    LPWSTR buffer = nullptr;
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPWSTR>(&buffer),
        0, nullptr);
    std::string message;
    if (length != 0 && buffer != nullptr)
        message = narrow(std::wstring(buffer, length));
    if (buffer != nullptr)
        LocalFree(buffer);
    message.erase(message.find_last_not_of("\r\n .") + 1);
    return message.empty() ? std::string("unknown error") : message;
}

// What usually causes the loader errors that a missing or mismatched DLL produces.
const char* windows_loader_hint(DWORD code) {
    switch (code) {
    case ERROR_MOD_NOT_FOUND:
        return "; the library or one of its dependent DLLs was not found on the DLL search "
               "path (application directory, the library's directory for a path, PATH)";
    case ERROR_PROC_NOT_FOUND:
        return "; a dependent DLL is missing an imported function (version mismatch)";
    case ERROR_BAD_EXE_FORMAT:
        return "; the file is not a 64-bit Windows DLL";
    default:
        return "";
    }
}

std::string windows_error_message(DWORD code) {
    return windows_system_message(code) + " (Windows error " + std::to_string(code) + ")" +
           windows_loader_hint(code);
}

std::filesystem::path module_file_name(HMODULE module) {
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length =
            GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            throw std::runtime_error("GetModuleFileNameW failed: " +
                                     windows_error_message(GetLastError()));
        }
        if (length < buffer.size()) {
            buffer.resize(length);
            return std::filesystem::path(buffer);
        }
        buffer.resize(buffer.size() * 2);
    }
}

bool has_directory(const std::string& value) {
    return value.find_first_of("\\/") != std::string::npos;
}
#endif

} // namespace

std::string shared_library_filename(std::string_view stem) {
#if defined(_WIN32)
    return std::string(stem) + ".dll";
#else
    return "lib" + std::string(stem) + ".so";
#endif
}

DynamicLibrary::DynamicLibrary(const std::string& name_or_path, std::string purpose)
    : name_(name_or_path), purpose_(std::move(purpose)) {
    if (name_.empty())
        throw std::runtime_error(purpose_ + ": empty shared library name");
#if defined(_WIN32)
    const std::wstring wide = widen(name_);
    // A path loads exactly that file and searches its directory for the
    // library's own dependencies; a bare name uses the standard DLL search
    // order (application directory, system directories, PATH).
    const DWORD flags = has_directory(name_) ? LOAD_WITH_ALTERED_SEARCH_PATH : 0;
    // Report a missing DLL as an exception instead of a modal dialog.
    const UINT previous_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);
    HMODULE module = LoadLibraryExW(wide.c_str(), nullptr, flags);
    const DWORD error = module == nullptr ? GetLastError() : ERROR_SUCCESS;
    SetErrorMode(previous_mode);
    if (module == nullptr) {
        throw std::runtime_error(purpose_ + ": unable to load '" + name_ +
                                 "': " + windows_error_message(error));
    }
    handle_ = module;
#else
    dlerror();
    handle_ = dlopen(name_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error(purpose_ + ": unable to load '" + name_ +
                                 "': " + (error != nullptr ? error : "unknown dlopen error"));
    }
#endif
}

DynamicLibrary::~DynamicLibrary() {
    if (handle_ == nullptr)
        return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
}

void* DynamicLibrary::find_symbol(const char* name) const noexcept {
    if (handle_ == nullptr || name == nullptr)
        return nullptr;
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    dlerror();
    void* symbol = dlsym(handle_, name);
    if (dlerror() != nullptr)
        return nullptr;
    return symbol;
#endif
}

void* DynamicLibrary::require_symbol(const char* name) const {
    void* symbol = find_symbol(name);
    if (symbol == nullptr) {
        throw std::runtime_error(purpose_ + ": library '" + loaded_path() +
                                 "' is missing required symbol '" +
                                 (name != nullptr ? name : "<null>") + "'");
    }
    return symbol;
}

std::string DynamicLibrary::loaded_path() const {
#if defined(_WIN32)
    try {
        return narrow(module_file_name(static_cast<HMODULE>(handle_)).wstring());
    } catch (const std::exception&) {
        return name_;
    }
#elif defined(__GLIBC__)
    struct link_map* map = nullptr;
    if (dlinfo(handle_, RTLD_DI_LINKMAP, &map) == 0 && map != nullptr && map->l_name != nullptr &&
        map->l_name[0] != '\0')
        return map->l_name;
    return name_;
#else
    return name_;
#endif
}

std::filesystem::path module_path_containing(const void* address) {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(address), &module) ||
        module == nullptr) {
        throw std::runtime_error("Unable to locate the module containing an address: " +
                                 windows_error_message(GetLastError()));
    }
    return std::filesystem::absolute(module_file_name(module)).lexically_normal();
#else
    Dl_info info{};
    if (dladdr(address, &info) == 0 || info.dli_fname == nullptr || info.dli_fname[0] == '\0')
        throw std::runtime_error("Unable to locate the shared library containing an address");
    return std::filesystem::absolute(info.dli_fname).lexically_normal();
#endif
}

std::filesystem::path current_executable_path() {
#if defined(_WIN32)
    return module_file_name(nullptr);
#else
    return std::filesystem::read_symlink("/proc/self/exe");
#endif
}

const char* default_nccl_library() {
#if defined(_WIN32)
    return "nccl.dll";
#else
    return "libnccl.so.2";
#endif
}

std::string nccl_library() {
    const char* configured = std::getenv(kNcclLibraryEnv);
    if (configured != nullptr && *configured != '\0')
        return configured;
    return default_nccl_library();
}

} // namespace trtmc::platform
