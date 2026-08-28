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
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
// psapi.h consumes Win32 base types from windows.h.
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <dlfcn.h>
#include <unistd.h>
#if defined(__linux__)
#include <link.h>
#endif
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

std::vector<HMODULE> process_modules() {
    std::vector<HMODULE> modules(128);
    while (true) {
        DWORD bytes_required = 0;
        if (!EnumProcessModules(GetCurrentProcess(), modules.data(),
                                static_cast<DWORD>(modules.size() * sizeof(HMODULE)),
                                &bytes_required)) {
            return {};
        }
        const std::size_t count = bytes_required / sizeof(HMODULE);
        if (count <= modules.size()) {
            modules.resize(count);
            return modules;
        }
        modules.resize(count + 32);
    }
}

fs::path module_path(HMODULE module) {
    std::vector<wchar_t> buffer(512);
    while (buffer.size() < 32768) {
        const DWORD size =
            GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (size == 0)
            return {};
        if (size < buffer.size() - 1)
            return fs::path(std::wstring(buffer.data(), size));
        buffer.resize(buffer.size() * 2);
    }
    return {};
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

void* dynamic_library_symbol_in_process(const char* name, std::string* error) {
    if (error != nullptr)
        error->clear();
    if (name == nullptr) {
        assign_error(error, "invalid dynamic-library symbol name");
        return nullptr;
    }
#if defined(_WIN32)
    for (HMODULE module : process_modules()) {
        FARPROC symbol = GetProcAddress(module, name);
        if (symbol != nullptr)
            return reinterpret_cast<void*>(symbol);
    }
    assign_error(error, std::string("symbol not found in loaded modules: ") + name);
    return nullptr;
#else
    dlerror();
    void* symbol = dlsym(RTLD_DEFAULT, name);
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

fs::path current_executable_path() noexcept {
    try {
#if defined(_WIN32)
        std::vector<wchar_t> buffer(512);
        while (buffer.size() < 32768) {
            const DWORD size =
                GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
            if (size == 0)
                return {};
            if (size < buffer.size() - 1)
                return fs::path(std::wstring(buffer.data(), size));
            buffer.resize(buffer.size() * 2);
        }
#else
        std::vector<char> buffer(512);
        while (buffer.size() < 1024 * 1024) {
            const ssize_t size = readlink("/proc/self/exe", buffer.data(), buffer.size());
            if (size < 0)
                return {};
            if (static_cast<std::size_t>(size) < buffer.size())
                return fs::path(std::string(buffer.data(), static_cast<std::size_t>(size)));
            buffer.resize(buffer.size() * 2);
        }
#endif
    } catch (...) {
    }
    return {};
}

std::vector<fs::path> loaded_dynamic_library_paths() {
    std::vector<fs::path> paths;
#if defined(_WIN32)
    for (HMODULE module : process_modules()) {
        fs::path path = module_path(module);
        if (!path.empty())
            paths.push_back(std::move(path));
    }
#elif defined(__linux__)
    struct CallbackData {
        std::vector<fs::path>* paths;
    } data{&paths};
    dl_iterate_phdr(
        [](dl_phdr_info* info, std::size_t, void* opaque) {
            auto* callback = static_cast<CallbackData*>(opaque);
            if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0')
                callback->paths->emplace_back(info->dlpi_name);
            return 0;
        },
        &data);
#endif
    return paths;
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

const char* dynamic_library_search_path_environment() noexcept {
#if defined(_WIN32)
    return "PATH";
#elif defined(__APPLE__)
    return "DYLD_LIBRARY_PATH";
#else
    return "LD_LIBRARY_PATH";
#endif
}

char path_list_separator() noexcept {
#if defined(_WIN32)
    return ';';
#else
    return ':';
#endif
}

} // namespace trtmc::internal
