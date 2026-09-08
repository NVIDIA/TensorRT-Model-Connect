/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/platform/dynamic_library.h"

#include <system_error>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace trtmc::internal {
namespace {

namespace fs = std::filesystem;

void assign_error(std::string* output, std::string message) noexcept {
    if (output == nullptr)
        return;
    try {
        *output = std::move(message);
    } catch (...) {
        try {
            output->clear();
        } catch (...) {
        }
    }
}

#if defined(_WIN32)

std::string utf8_from_wide(const wchar_t* value, int length) {
    if (value == nullptr || length <= 0)
        return {};
    const int required =
        WideCharToMultiByte(CP_UTF8, 0, value, length, nullptr, 0, nullptr, nullptr);
    if (required <= 0)
        return {};
    std::string result(static_cast<std::size_t>(required), '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, value, length, result.data(), required, nullptr, nullptr) <=
        0) {
        return {};
    }
    return result;
}

std::string windows_error_message(DWORD code) {
    wchar_t* buffer = nullptr;
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
    std::string message = utf8_from_wide(buffer, static_cast<int>(length));
    if (buffer != nullptr)
        LocalFree(buffer);
    while (!message.empty() &&
           (message.back() == '\r' || message.back() == '\n' || message.back() == ' ')) {
        message.pop_back();
    }
    return message.empty() ? "Windows error " + std::to_string(code) : message;
}

#else

std::string loader_error() {
    const char* error = dlerror();
    return error == nullptr ? std::string("unknown dynamic-loader error") : std::string(error);
}

#endif

} // namespace

DynamicLibraryHandle open_dynamic_library(const fs::path& path, DynamicLibraryVisibility visibility,
                                          std::string* error) {
    if (error != nullptr)
        error->clear();
#if defined(_WIN32)
    (void)visibility;
    try {
        fs::path load_path = path;
        DWORD flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        if (path.is_absolute() || path.has_parent_path()) {
            std::error_code ec;
            const fs::path absolute = fs::absolute(path, ec);
            if (!ec)
                load_path = absolute;
            flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
        }
        HMODULE handle = LoadLibraryExW(load_path.c_str(), nullptr, flags);
        if (handle == nullptr) {
            assign_error(error, windows_error_message(GetLastError()));
            return nullptr;
        }
        return reinterpret_cast<DynamicLibraryHandle>(handle);
    } catch (const std::exception& exception) {
        assign_error(error, exception.what());
        return nullptr;
    }
#else
    dlerror();
    const int flags =
        RTLD_NOW | (visibility == DynamicLibraryVisibility::global ? RTLD_GLOBAL : RTLD_LOCAL);
    DynamicLibraryHandle handle = dlopen(path.c_str(), flags);
    if (handle == nullptr)
        assign_error(error, loader_error());
    return handle;
#endif
}

void* dynamic_library_symbol(DynamicLibraryHandle handle, const char* name, std::string* error) {
    if (error != nullptr)
        error->clear();
    if (handle == nullptr || name == nullptr) {
        assign_error(error, "invalid dynamic-library handle or symbol name");
        return nullptr;
    }
#if defined(_WIN32)
    FARPROC symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle), name);
    if (symbol == nullptr) {
        assign_error(error, windows_error_message(GetLastError()));
        return nullptr;
    }
    return reinterpret_cast<void*>(symbol);
#else
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* message = dlerror();
    if (message != nullptr) {
        assign_error(error, message);
        return nullptr;
    }
    return symbol;
#endif
}

bool close_dynamic_library(DynamicLibraryHandle handle, std::string* error) noexcept {
    if (error != nullptr)
        error->clear();
    if (handle == nullptr)
        return true;
#if defined(_WIN32)
    if (FreeLibrary(reinterpret_cast<HMODULE>(handle)) != 0)
        return true;
    try {
        assign_error(error, windows_error_message(GetLastError()));
    } catch (...) {
        assign_error(error, "unknown Windows dynamic-loader error");
    }
    return false;
#else
    if (dlclose(handle) == 0)
        return true;
    try {
        assign_error(error, loader_error());
    } catch (...) {
        assign_error(error, "unknown dynamic-loader error");
    }
    return false;
#endif
}

std::string dynamic_library_filename(std::string_view stem) {
#if defined(_WIN32)
    return std::string(stem) + ".dll";
#elif defined(__APPLE__)
    return "lib" + std::string(stem) + ".dylib";
#else
    return "lib" + std::string(stem) + ".so";
#endif
}

} // namespace trtmc::internal
