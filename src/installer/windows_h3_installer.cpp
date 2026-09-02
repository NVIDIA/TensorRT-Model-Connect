/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "installer/windows_h3_installer.h"

#ifndef _WIN32
#error "The native MiniMax-H3 installer support is Windows-only"
#endif

#define WIN32_LEAN_AND_MEAN
// clang-format off
#include <windows.h>
#include <aclapi.h>
#include <bcrypt.h>
// clang-format on

#include <algorithm>
#include <cctype>
#include <charconv>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <utility>

namespace trtmc::installer {
namespace {

constexpr std::size_t kSha256Bytes = 32;
constexpr std::size_t kSha256HexChars = kSha256Bytes * 2;
constexpr std::size_t kCopyBufferBytes = 8U << 20;
constexpr std::size_t kMaximumManifestBytes = 64U << 20;
constexpr DWORD kInstallMutexAccess = SYNCHRONIZE | MUTEX_MODIFY_STATE | READ_CONTROL;

std::runtime_error windows_error(const std::string& operation, DWORD code = GetLastError()) {
    return std::runtime_error(operation + " failed with Windows error " + std::to_string(code));
}

struct AlgorithmCloser {
    void operator()(BCRYPT_ALG_HANDLE handle) const noexcept {
        if (handle != nullptr)
            BCryptCloseAlgorithmProvider(handle, 0);
    }
};

struct HashCloser {
    void operator()(BCRYPT_HASH_HANDLE handle) const noexcept {
        if (handle != nullptr)
            BCryptDestroyHash(handle);
    }
};

using AlgorithmHandle = std::unique_ptr<std::remove_pointer_t<BCRYPT_ALG_HANDLE>, AlgorithmCloser>;
using HashHandle = std::unique_ptr<std::remove_pointer_t<BCRYPT_HASH_HANDLE>, HashCloser>;

struct NativeHandleCloser {
    void operator()(void* handle) const noexcept {
        if (handle != nullptr)
            CloseHandle(static_cast<HANDLE>(handle));
    }
};

struct LocalMemoryCloser {
    void operator()(void* memory) const noexcept {
        if (memory != nullptr)
            LocalFree(static_cast<HLOCAL>(memory));
    }
};

using NativeHandle = std::unique_ptr<void, NativeHandleCloser>;
using LocalMemory = std::unique_ptr<void, LocalMemoryCloser>;

bool nt_success(NTSTATUS status) {
    return status >= 0;
}

std::array<std::uint8_t, kSha256Bytes> sha256_bytes(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() > (std::numeric_limits<ULONG>::max)())
        throw std::runtime_error("Installer mutex identity is unexpectedly large");
    BCRYPT_ALG_HANDLE raw_algorithm = nullptr;
    if (!nt_success(
            BCryptOpenAlgorithmProvider(&raw_algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0))) {
        throw std::runtime_error("Unable to initialize Windows SHA-256 provider");
    }
    AlgorithmHandle algorithm(raw_algorithm);
    DWORD object_bytes = 0;
    DWORD returned = 0;
    if (!nt_success(BCryptGetProperty(raw_algorithm, BCRYPT_OBJECT_LENGTH,
                                      reinterpret_cast<PUCHAR>(&object_bytes), sizeof(object_bytes),
                                      &returned, 0)) ||
        object_bytes == 0) {
        throw std::runtime_error("Unable to query Windows SHA-256 object size");
    }
    std::vector<std::uint8_t> hash_object(object_bytes);
    BCRYPT_HASH_HANDLE raw_hash = nullptr;
    if (!nt_success(BCryptCreateHash(raw_algorithm, &raw_hash, hash_object.data(), object_bytes,
                                     nullptr, 0, 0))) {
        throw std::runtime_error("Unable to create Windows SHA-256 hash");
    }
    HashHandle hash(raw_hash);
    if (!bytes.empty() && !nt_success(BCryptHashData(raw_hash, const_cast<PUCHAR>(bytes.data()),
                                                     static_cast<ULONG>(bytes.size()), 0))) {
        throw std::runtime_error("Windows SHA-256 update failed");
    }
    std::array<std::uint8_t, kSha256Bytes> result{};
    if (!nt_success(
            BCryptFinishHash(raw_hash, result.data(), static_cast<ULONG>(result.size()), 0))) {
        throw std::runtime_error("Windows SHA-256 finalization failed");
    }
    return result;
}

std::vector<std::uint8_t> current_user_sid() {
    HANDLE raw_token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw_token))
        throw windows_error("OpenProcessToken(installer transaction mutex)");
    NativeHandle token(raw_token);
    DWORD required = 0;
    if (GetTokenInformation(raw_token, TokenUser, nullptr, 0, &required) != FALSE ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER || required == 0) {
        throw windows_error("GetTokenInformation(TokenUser size)");
    }
    std::vector<std::uint8_t> token_user(required);
    if (!GetTokenInformation(raw_token, TokenUser, token_user.data(), required, &required))
        throw windows_error("GetTokenInformation(TokenUser)");
    const auto* user = reinterpret_cast<const TOKEN_USER*>(token_user.data());
    if (!IsValidSid(user->User.Sid))
        throw std::runtime_error("Windows returned an invalid current-user SID");
    const DWORD sid_bytes = GetLengthSid(user->User.Sid);
    std::vector<std::uint8_t> result(sid_bytes);
    if (!CopySid(sid_bytes, result.data(), user->User.Sid))
        throw windows_error("CopySid(installer transaction mutex)");
    return result;
}

std::wstring canonical_install_root_identity(const std::filesystem::path& install_root) {
    if (install_root.empty() || install_root.filename().empty())
        throw std::runtime_error("Installation directory is invalid");
    std::error_code error;
    const auto absolute = std::filesystem::absolute(install_root, error);
    if (error) {
        throw std::runtime_error("Unable to make installation directory absolute: " +
                                 install_root.string() + "; error=" + error.message());
    }
    // Resolve the deepest existing ancestor through a handle, then append the
    // normalized missing suffix. Therefore first install and later upgrades
    // derive the same key even though the target itself changes existence.
    auto ancestor = absolute.lexically_normal();
    std::vector<std::filesystem::path> missing_components;
    for (;;) {
        const DWORD attributes = GetFileAttributesW(ancestor.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES)
            break;
        const DWORD code = GetLastError();
        if (code != ERROR_FILE_NOT_FOUND && code != ERROR_PATH_NOT_FOUND) {
            throw windows_error(
                "GetFileAttributesW(installer mutex ancestor " + ancestor.string() + ")", code);
        }
        const auto component = ancestor.filename();
        const auto parent = ancestor.parent_path();
        if (component.empty() || parent.empty() || parent == ancestor) {
            throw std::runtime_error("Unable to find an existing ancestor for installation "
                                     "directory: " +
                                     install_root.string());
        }
        // Ordinary Win32 paths ignore trailing spaces and periods. Fold them
        // here so aliases such as "MiniMax-H3" and "MiniMax-H3." cannot obtain
        // different locks and then mutate the same on-disk transaction paths.
        auto normalized_component = component.wstring();
        while (!normalized_component.empty() &&
               (normalized_component.back() == L' ' || normalized_component.back() == L'.')) {
            normalized_component.pop_back();
        }
        if (normalized_component.empty()) {
            throw std::runtime_error("Installation directory has an ambiguous Windows path "
                                     "component: " +
                                     install_root.string());
        }
        missing_components.emplace_back(std::move(normalized_component));
        ancestor = parent;
    }

    HANDLE raw_ancestor =
        CreateFileW(ancestor.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (raw_ancestor == INVALID_HANDLE_VALUE) {
        throw windows_error("CreateFileW(installer mutex ancestor " + ancestor.string() + ")");
    }
    NativeHandle ancestor_handle(raw_ancestor);
    // The NT volume identity makes drive-letter, SUBST, mount-point, and
    // junction aliases converge on the same kernel-object name.
    const DWORD required =
        GetFinalPathNameByHandleW(raw_ancestor, nullptr, 0, FILE_NAME_NORMALIZED | VOLUME_NAME_NT);
    if (required == 0)
        throw windows_error("GetFinalPathNameByHandleW(installer mutex ancestor size)");
    std::vector<wchar_t> final_path(static_cast<std::size_t>(required) + 1, L'\0');
    const DWORD copied = GetFinalPathNameByHandleW(raw_ancestor, final_path.data(),
                                                   static_cast<DWORD>(final_path.size()),
                                                   FILE_NAME_NORMALIZED | VOLUME_NAME_NT);
    if (copied == 0 || static_cast<std::size_t>(copied) >= final_path.size())
        throw windows_error("GetFinalPathNameByHandleW(installer mutex ancestor)");
    std::filesystem::path canonical(std::wstring(final_path.data(), copied));
    for (auto component = missing_components.rbegin(); component != missing_components.rend();
         ++component) {
        canonical /= *component;
    }

    std::wstring value = canonical.lexically_normal().wstring();
    std::replace(value.begin(), value.end(), L'/', L'\\');
    if (value.rfind(L"\\\\?\\UNC\\", 0) == 0) {
        value = L"\\\\" + value.substr(8);
    } else if (value.rfind(L"\\\\?\\", 0) == 0) {
        value.erase(0, 4);
    }
    const int folded_required =
        LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_UPPERCASE, value.data(),
                      static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr, 0);
    if (folded_required <= 0)
        throw windows_error("LCMapStringEx(installer transaction mutex identity)");
    std::wstring folded(static_cast<std::size_t>(folded_required), L'\0');
    if (LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_UPPERCASE, value.data(),
                      static_cast<int>(value.size()), folded.data(), folded_required, nullptr,
                      nullptr, 0) != folded_required) {
        throw windows_error("LCMapStringEx(installer transaction mutex identity)");
    }
    return folded;
}

void append_identity_part(std::vector<std::uint8_t>& output, const void* data, std::size_t size) {
    if (size > (std::numeric_limits<std::uint32_t>::max)())
        throw std::runtime_error("Installer mutex identity component is unexpectedly large");
    const auto length = static_cast<std::uint32_t>(size);
    for (unsigned int shift = 0; shift < 32; shift += 8)
        output.push_back(static_cast<std::uint8_t>((length >> shift) & 0xffU));
    const auto* begin = static_cast<const std::uint8_t*>(data);
    output.insert(output.end(), begin, begin + size);
}

struct InstallMutexIdentity {
    std::vector<std::uint8_t> user_sid;
    std::wstring canonical_root;
    std::wstring name;
};

InstallMutexIdentity make_install_mutex_identity(const std::filesystem::path& install_root) {
    InstallMutexIdentity result;
    result.user_sid = current_user_sid();
    result.canonical_root = canonical_install_root_identity(install_root);
    std::vector<std::uint8_t> identity_bytes;
    append_identity_part(identity_bytes, result.user_sid.data(), result.user_sid.size());
    append_identity_part(identity_bytes, result.canonical_root.data(),
                         result.canonical_root.size() * sizeof(wchar_t));
    const auto digest = sha256_hex(sha256_bytes(identity_bytes));
    result.name =
        L"Global\\ModelConnectMiniMaxH3.Setup." + std::wstring(digest.begin(), digest.end());
    return result;
}

void require_user_only_mutex_security(HANDLE mutex, const std::vector<std::uint8_t>& user_sid) {
    PSID owner = nullptr;
    PACL dacl = nullptr;
    PSECURITY_DESCRIPTOR raw_descriptor = nullptr;
    const DWORD status = GetSecurityInfo(mutex, SE_KERNEL_OBJECT,
                                         OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                                         &owner, nullptr, &dacl, nullptr, &raw_descriptor);
    if (status != ERROR_SUCCESS) {
        throw windows_error("GetSecurityInfo(installer transaction mutex)", status);
    }
    LocalMemory descriptor(raw_descriptor);
    if (owner == nullptr || EqualSid(owner, const_cast<std::uint8_t*>(user_sid.data())) == FALSE) {
        throw std::runtime_error(
            "Installer transaction mutex is not owned by the current user SID");
    }
    if (dacl == nullptr || dacl->AceCount != 1)
        throw std::runtime_error("Installer transaction mutex does not have a user-only DACL");
    void* raw_ace = nullptr;
    if (!GetAce(dacl, 0, &raw_ace))
        throw windows_error("GetAce(installer transaction mutex)");
    const auto* ace = static_cast<const ACCESS_ALLOWED_ACE*>(raw_ace);
    if (ace->Header.AceType != ACCESS_ALLOWED_ACE_TYPE || ace->Mask != kInstallMutexAccess ||
        EqualSid(const_cast<DWORD*>(&ace->SidStart), const_cast<std::uint8_t*>(user_sid.data())) ==
            FALSE) {
        throw std::runtime_error("Installer transaction mutex does not have the required "
                                 "current-user-only access rule");
    }
}

bool ascii_hex(char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
}

std::uint8_t hex_nibble(char value) {
    if (value >= '0' && value <= '9')
        return static_cast<std::uint8_t>(value - '0');
    return static_cast<std::uint8_t>(10 + value - 'a');
}

std::array<std::uint8_t, kSha256Bytes> parse_sha256(const std::string& text) {
    if (text.size() != kSha256HexChars || !std::all_of(text.begin(), text.end(), ascii_hex)) {
        throw std::runtime_error("Payload manifest has a non-canonical SHA-256 digest");
    }
    std::array<std::uint8_t, kSha256Bytes> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<std::uint8_t>((hex_nibble(text[index * 2]) << 4U) |
                                                  hex_nibble(text[index * 2 + 1]));
    }
    return result;
}

std::uintmax_t parse_size(const std::string& text) {
    if (text.empty())
        throw std::runtime_error("Payload manifest has an empty size");
    std::uintmax_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("Payload manifest has an invalid decimal size");
    return value;
}

std::string lowercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool reserved_windows_component(const std::string& component) {
    const auto dot = component.find('.');
    const auto stem = lowercase_ascii(component.substr(0, dot));
    if (stem == "con" || stem == "prn" || stem == "aux" || stem == "nul")
        return true;
    if (stem.size() == 4 && (stem.rfind("com", 0) == 0 || stem.rfind("lpt", 0) == 0) &&
        stem[3] >= '1' && stem[3] <= '9') {
        return true;
    }
    return false;
}

bool is_versioned_rtx_runtime_path(const std::string& path) {
    constexpr char prefix[] = "bin/tensorrt_rtx_";
    constexpr char suffix[] = ".dll";
    if (path.rfind(prefix, 0) != 0 || path.size() <= sizeof(prefix) - 1 + sizeof(suffix) - 1 ||
        path.compare(path.size() - (sizeof(suffix) - 1), sizeof(suffix) - 1, suffix) != 0) {
        return false;
    }
    const std::size_t begin = sizeof(prefix) - 1;
    const std::size_t end = path.size() - (sizeof(suffix) - 1);
    bool previous_was_digit = false;
    for (std::size_t index = begin; index < end; ++index) {
        const char character = path[index];
        if (character >= '0' && character <= '9') {
            previous_was_digit = true;
            continue;
        }
        if (character != '_' || !previous_was_digit || index + 1 == end)
            return false;
        previous_was_digit = false;
    }
    return previous_was_digit;
}

std::string read_small_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return {};
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.eof() && input.fail())
        throw std::runtime_error("Unable to read installation marker");
    return contents.str();
}

std::string read_manifest_bytes(const std::filesystem::path& path) {
    std::error_code status_error;
    const auto status = std::filesystem::symlink_status(path, status_error);
    if (status_error || !std::filesystem::is_regular_file(status)) {
        throw std::runtime_error("payload.manifest is not a regular file: " + path.string() +
                                 (status_error ? "; error=" + status_error.message() : ""));
    }
    const auto size = std::filesystem::file_size(path, status_error);
    if (status_error)
        throw std::runtime_error("Unable to determine payload.manifest size: " + path.string() +
                                 "; error=" + status_error.message());
    if (size > kMaximumManifestBytes)
        throw std::runtime_error("payload.manifest exceeds the installer size limit: " +
                                 path.string());

    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Unable to open payload.manifest: " + path.string());
    std::string result(static_cast<std::size_t>(size), '\0');
    if (!result.empty()) {
        input.read(result.data(), static_cast<std::streamsize>(result.size()));
        if (input.gcount() != static_cast<std::streamsize>(result.size())) {
            throw std::runtime_error("payload.manifest changed during its bounded read: " +
                                     path.string());
        }
    }
    char unexpected = 0;
    if (input.get(unexpected)) {
        throw std::runtime_error("payload.manifest grew during its bounded read: " + path.string());
    }
    if (!input.eof())
        throw std::runtime_error("Unable to read payload.manifest: " + path.string());
    return result;
}

std::string read_embedded_manifest_bytes(void* module_handle) {
    HMODULE module =
        module_handle != nullptr ? static_cast<HMODULE>(module_handle) : GetModuleHandleW(nullptr);
    if (module == nullptr)
        throw windows_error("GetModuleHandleW(payload manifest anchor)");
    const HRSRC resource =
        FindResourceW(module, MAKEINTRESOURCEW(kPayloadManifestResourceId), MAKEINTRESOURCEW(10));
    if (resource == nullptr)
        throw windows_error("FindResourceW(payload manifest anchor)");
    const DWORD size = SizeofResource(module, resource);
    if (size == 0 || size > kMaximumManifestBytes)
        throw std::runtime_error("Embedded payload manifest anchor has an invalid size");
    const HGLOBAL loaded = LoadResource(module, resource);
    if (loaded == nullptr)
        throw windows_error("LoadResource(payload manifest anchor)");
    const void* bytes = LockResource(loaded);
    if (bytes == nullptr)
        throw windows_error("LockResource(payload manifest anchor)");
    return std::string(static_cast<const char*>(bytes), static_cast<std::size_t>(size));
}

std::vector<PayloadEntry> parse_payload_manifest_bytes(const std::string& bytes) {
    std::istringstream input(bytes);
    std::vector<PayloadEntry> result;
    std::set<std::string> paths;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        const auto first_tab = line.find('\t');
        const auto second_tab =
            first_tab == std::string::npos ? std::string::npos : line.find('\t', first_tab + 1);
        if (first_tab != kSha256HexChars || second_tab == std::string::npos ||
            line.find('\t', second_tab + 1) != std::string::npos) {
            throw std::runtime_error("Malformed payload manifest row " +
                                     std::to_string(line_number));
        }
        const auto relative_utf8 = line.substr(second_tab + 1);
        if (!is_safe_payload_path(relative_utf8))
            throw std::runtime_error("Unsafe payload path on manifest row " +
                                     std::to_string(line_number));
        if (!paths.insert(lowercase_ascii(relative_utf8)).second)
            throw std::runtime_error("Duplicate payload path on manifest row " +
                                     std::to_string(line_number));
        result.push_back(
            PayloadEntry{parse_sha256(line.substr(0, first_tab)),
                         parse_size(line.substr(first_tab + 1, second_tab - first_tab - 1)),
                         std::filesystem::u8path(relative_utf8)});
    }
    if (!input.eof() && input.fail())
        throw std::runtime_error("Unable to parse payload.manifest");
    if (result.empty())
        throw std::runtime_error("Payload manifest is empty");
    return result;
}

void materialize_file(const std::filesystem::path& source,
                      const std::filesystem::path& destination) {
    std::filesystem::create_directories(destination.parent_path());
    // Never hard-link an installed artifact to the mutable download layout.
    // A hard link would let a later edit of the package silently change the
    // installed bundle after its manifest had been verified.
    if (!CopyFileW(source.c_str(), destination.c_str(), TRUE))
        throw windows_error("CopyFileW");
}

} // namespace

InstallTransactionLock::InstallTransactionLock(InstallTransactionLock&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)), abandoned_(other.abandoned_) {
    other.abandoned_ = false;
}

InstallTransactionLock& InstallTransactionLock::operator=(InstallTransactionLock&& other) noexcept {
    if (this != &other) {
        release();
        handle_ = std::exchange(other.handle_, nullptr);
        abandoned_ = other.abandoned_;
        other.abandoned_ = false;
    }
    return *this;
}

InstallTransactionLock::~InstallTransactionLock() {
    release();
}

void InstallTransactionLock::release() noexcept {
    if (handle_ == nullptr)
        return;
    const HANDLE handle = static_cast<HANDLE>(handle_);
    // A mutex must be released by the thread that acquired it. Setup keeps the
    // guard on its main thread for the complete install/uninstall transaction.
    (void)ReleaseMutex(handle);
    CloseHandle(handle);
    handle_ = nullptr;
    abandoned_ = false;
}

std::wstring install_transaction_mutex_name(const std::filesystem::path& install_root) {
    return make_install_mutex_identity(install_root).name;
}

InstallTransactionLock acquire_install_transaction_lock(const std::filesystem::path& install_root,
                                                        std::uint32_t timeout_ms) {
    auto identity = make_install_mutex_identity(install_root);

    EXPLICIT_ACCESSW permission{};
    permission.grfAccessPermissions = kInstallMutexAccess;
    permission.grfAccessMode = SET_ACCESS;
    permission.grfInheritance = NO_INHERITANCE;
    permission.Trustee.pMultipleTrustee = nullptr;
    permission.Trustee.MultipleTrusteeOperation = NO_MULTIPLE_TRUSTEE;
    permission.Trustee.TrusteeForm = TRUSTEE_IS_SID;
    permission.Trustee.TrusteeType = TRUSTEE_IS_USER;
    permission.Trustee.ptstrName =
        reinterpret_cast<LPWSTR>(static_cast<void*>(identity.user_sid.data()));

    PACL raw_acl = nullptr;
    const DWORD acl_status = SetEntriesInAclW(1, &permission, nullptr, &raw_acl);
    if (acl_status != ERROR_SUCCESS)
        throw windows_error("SetEntriesInAclW(installer transaction mutex)", acl_status);
    LocalMemory acl(raw_acl);

    SECURITY_DESCRIPTOR descriptor{};
    if (!InitializeSecurityDescriptor(&descriptor, SECURITY_DESCRIPTOR_REVISION))
        throw windows_error("InitializeSecurityDescriptor(installer transaction mutex)");
    if (!SetSecurityDescriptorOwner(&descriptor, identity.user_sid.data(), FALSE))
        throw windows_error("SetSecurityDescriptorOwner(installer transaction mutex)");
    if (!SetSecurityDescriptorDacl(&descriptor, TRUE, raw_acl, FALSE))
        throw windows_error("SetSecurityDescriptorDacl(installer transaction mutex)");
    SECURITY_ATTRIBUTES attributes{};
    attributes.nLength = sizeof(attributes);
    attributes.lpSecurityDescriptor = &descriptor;
    attributes.bInheritHandle = FALSE;

    HANDLE raw_mutex = CreateMutexExW(&attributes, identity.name.c_str(), 0, kInstallMutexAccess);
    if (raw_mutex == nullptr) {
        throw windows_error("CreateMutexExW(installer transaction mutex for " +
                            install_root.string() + ")");
    }
    NativeHandle mutex(raw_mutex);
    require_user_only_mutex_security(raw_mutex, identity.user_sid);
    const DWORD wait_status = WaitForSingleObject(raw_mutex, timeout_ms);
    if (wait_status == WAIT_TIMEOUT) {
        throw std::runtime_error(
            "Timed out after " + std::to_string(timeout_ms) +
            " ms waiting for another MiniMax-H3 install/uninstall transaction at '" +
            install_root.string() + "'");
    }
    if (wait_status == WAIT_FAILED) {
        throw windows_error("WaitForSingleObject(installer transaction mutex for " +
                            install_root.string() + ")");
    }
    if (wait_status != WAIT_OBJECT_0 && wait_status != WAIT_ABANDONED) {
        throw std::runtime_error("Unexpected installer transaction mutex wait result " +
                                 std::to_string(wait_status) + " for '" + install_root.string() +
                                 "'");
    }
    const bool abandoned = wait_status == WAIT_ABANDONED;
    return InstallTransactionLock(mutex.release(), abandoned);
}

bool is_safe_payload_path(const std::string& utf8_path) {
    if (utf8_path.empty() || utf8_path.front() == '/' || utf8_path.back() == '/' ||
        utf8_path.find('\\') != std::string::npos || utf8_path.find(':') != std::string::npos ||
        utf8_path.find('\0') != std::string::npos) {
        return false;
    }
    std::size_t begin = 0;
    while (begin < utf8_path.size()) {
        const auto end = utf8_path.find('/', begin);
        const auto component =
            utf8_path.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (component.empty() || component == "." || component == ".." || component.back() == '.' ||
            component.back() == ' ' || reserved_windows_component(component)) {
            return false;
        }
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    const auto path = std::filesystem::u8path(utf8_path);
    return !path.empty() && !path.is_absolute() && !path.has_root_name() &&
           !path.has_root_directory() && path.lexically_normal() == path;
}

std::vector<PayloadEntry> read_payload_manifest(const std::filesystem::path& manifest_path) {
    return parse_payload_manifest_bytes(read_manifest_bytes(manifest_path));
}

std::vector<PayloadEntry>
read_authenticated_payload_manifest(const std::filesystem::path& manifest_path,
                                    void* module_handle) {
    const auto external = read_manifest_bytes(manifest_path);
    const auto embedded = read_embedded_manifest_bytes(module_handle);
    if (external != embedded) {
        throw std::runtime_error(
            "External payload.manifest does not match the manifest anchored in Setup: " +
            manifest_path.string());
    }
    return parse_payload_manifest_bytes(external);
}

std::array<std::uint8_t, kSha256Bytes> sha256_file(const std::filesystem::path& path) {
    BCRYPT_ALG_HANDLE raw_algorithm = nullptr;
    if (!nt_success(BCryptOpenAlgorithmProvider(&raw_algorithm, BCRYPT_SHA256_ALGORITHM, nullptr,
                                                BCRYPT_HASH_REUSABLE_FLAG))) {
        throw std::runtime_error("Unable to initialize Windows SHA-256 provider");
    }
    AlgorithmHandle algorithm(raw_algorithm);
    DWORD object_bytes = 0;
    DWORD returned = 0;
    if (!nt_success(BCryptGetProperty(raw_algorithm, BCRYPT_OBJECT_LENGTH,
                                      reinterpret_cast<PUCHAR>(&object_bytes), sizeof(object_bytes),
                                      &returned, 0)) ||
        object_bytes == 0) {
        throw std::runtime_error("Unable to query Windows SHA-256 object size");
    }
    std::vector<std::uint8_t> hash_object(object_bytes);
    BCRYPT_HASH_HANDLE raw_hash = nullptr;
    if (!nt_success(BCryptCreateHash(raw_algorithm, &raw_hash, hash_object.data(), object_bytes,
                                     nullptr, 0, BCRYPT_HASH_REUSABLE_FLAG))) {
        throw std::runtime_error("Unable to create Windows SHA-256 hash");
    }
    HashHandle hash(raw_hash);
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Unable to open payload file: " + path.string());
    std::vector<char> buffer(kCopyBufferBytes);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0 &&
            !nt_success(BCryptHashData(raw_hash, reinterpret_cast<PUCHAR>(buffer.data()),
                                       static_cast<ULONG>(count), 0))) {
            throw std::runtime_error("Windows SHA-256 update failed");
        }
    }
    if (!input.eof())
        throw std::runtime_error("Unable to read payload file: " + path.string());
    std::array<std::uint8_t, kSha256Bytes> result{};
    if (!nt_success(
            BCryptFinishHash(raw_hash, result.data(), static_cast<ULONG>(result.size()), 0))) {
        throw std::runtime_error("Windows SHA-256 finalization failed");
    }
    return result;
}

std::string sha256_hex(const std::array<std::uint8_t, kSha256Bytes>& digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(kSha256HexChars, '0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
        result[index * 2] = digits[digest[index] >> 4U];
        result[index * 2 + 1] = digits[digest[index] & 0x0FU];
    }
    return result;
}

void verify_payload(const std::filesystem::path& payload_root,
                    const std::vector<PayloadEntry>& entries) {
    if (!std::filesystem::is_directory(payload_root))
        throw std::runtime_error("Payload directory is missing");

    std::set<std::string> expected_paths;
    for (const auto& entry : entries) {
        const auto relative = entry.relative_path.generic_u8string();
        if (!expected_paths.insert(lowercase_ascii(relative)).second)
            throw std::runtime_error("Payload entries contain a duplicate path: " + relative);
    }

    std::set<std::string> actual_paths;
    for (const auto& item : std::filesystem::recursive_directory_iterator(payload_root)) {
        const auto status = item.symlink_status();
        if (std::filesystem::is_directory(status))
            continue;
        const auto relative_path = item.path().lexically_relative(payload_root);
        const auto relative = relative_path.generic_u8string();
        if (!std::filesystem::is_regular_file(status)) {
            throw std::runtime_error("Payload contains a non-regular entry: " + relative);
        }
        if (!is_safe_payload_path(relative))
            throw std::runtime_error("Payload contains an unsafe file path: " + relative);
        if (!actual_paths.insert(lowercase_ascii(relative)).second)
            throw std::runtime_error("Payload contains a duplicate file path: " + relative);
    }
    for (const auto& actual : actual_paths) {
        if (expected_paths.find(actual) == expected_paths.end())
            throw std::runtime_error("Payload contains a file absent from the manifest: " + actual);
    }
    for (const auto& expected : expected_paths) {
        if (actual_paths.find(expected) == actual_paths.end())
            throw std::runtime_error("Payload manifest names a missing file: " + expected);
    }

    for (const auto& entry : entries) {
        const auto source = payload_root / entry.relative_path;
        const auto status = std::filesystem::symlink_status(source);
        if (!std::filesystem::is_regular_file(status))
            throw std::runtime_error("Payload file is missing: " + entry.relative_path.string());
        std::error_code error;
        const auto actual_size = std::filesystem::file_size(source, error);
        if (error || actual_size != entry.size)
            throw std::runtime_error("Payload size mismatch: " + entry.relative_path.string());
        if (sha256_file(source) != entry.sha256)
            throw std::runtime_error("Payload SHA-256 mismatch: " + entry.relative_path.string());
    }
}

void validate_minimax_h3_runtime_payload(const std::vector<PayloadEntry>& entries) {
    const std::set<std::string> required{
        ".minimax-h3-install-id",
        "uninstallminimaxh3.exe",
        "bin/trtmc.exe",
        "bin/trtmc_core.dll",
        "bin/trtmc_backend_trt_rtx.dll",
        "bin/trtmc/models/minimax_h3/trtmc_model_minimax_h3.dll",
        "models/minimax-h3.bundle",
    };
    const std::set<std::string> optional{
        "licenses/license",
        "licenses/notice",
    };

    std::set<std::string> paths;
    std::size_t rtx_runtime_count = 0;
    for (const auto& entry : entries) {
        const std::string path = lowercase_ascii(entry.relative_path.generic_u8string());
        if (!paths.insert(path).second)
            throw std::runtime_error("MiniMax-H3 payload manifest contains a duplicate path: " +
                                     path);
        if (required.count(path) != 0 || optional.count(path) != 0)
            continue;
        if (is_versioned_rtx_runtime_path(path)) {
            ++rtx_runtime_count;
            continue;
        }
        throw std::runtime_error("MiniMax-H3 payload manifest contains an unexpected file: " +
                                 path);
    }
    for (const auto& path : required) {
        if (paths.count(path) == 0)
            throw std::runtime_error("MiniMax-H3 payload is missing required file: " + path);
    }
    if (rtx_runtime_count != 1) {
        throw std::runtime_error(
            "MiniMax-H3 payload must contain exactly one versioned TensorRT-RTX runtime DLL");
    }
}

bool installation_marker_matches(const std::filesystem::path& install_root,
                                 const std::string& marker_name,
                                 const std::string& marker_contents) {
    if (!is_safe_payload_path(marker_name))
        throw std::runtime_error("Installer marker name is unsafe");
    return read_small_text_file(install_root / std::filesystem::u8path(marker_name)) ==
           marker_contents;
}

namespace {

const char* transaction_operation_name(InstallTransactionOperation operation) {
    switch (operation) {
    case InstallTransactionOperation::BackupExisting:
        return "backup existing installation";
    case InstallTransactionOperation::CommitStaging:
        return "commit staged installation";
    case InstallTransactionOperation::VerifyFinalPayload:
        return "verify committed installation";
    case InstallTransactionOperation::QuarantineFailedInstall:
        return "quarantine failed installation";
    case InstallTransactionOperation::RestoreBackup:
        return "restore prior installation";
    case InstallTransactionOperation::RetireBackup:
        return "retire verified rollback backup";
    case InstallTransactionOperation::RemoveStaging:
        return "remove staging directory";
    case InstallTransactionOperation::RemoveRecovery:
        return "remove recovery directory";
    case InstallTransactionOperation::RemoveRetiredBackup:
        return "remove retired rollback backup";
    }
    return "perform installer transaction operation";
}

std::string quoted_path(const std::filesystem::path& path) {
    return "'" + path.string() + "'";
}

std::string current_exception_message(std::exception_ptr error) {
    try {
        if (error != nullptr)
            std::rethrow_exception(error);
    } catch (const std::exception& caught) {
        return caught.what();
    } catch (...) {
        return "unknown installer exception";
    }
    return "unknown installer exception";
}

void maybe_inject_fault(const InstallFaultInjector* injector, InstallTransactionOperation operation,
                        const std::filesystem::path& source,
                        const std::filesystem::path& destination) {
    if (injector == nullptr)
        return;
    const std::uint32_t code = (*injector)(operation, source, destination);
    if (code == 0)
        return;
    throw std::runtime_error(std::string("Unable to ") + transaction_operation_name(operation) +
                             "; simulated Windows error " + std::to_string(code) + "; source=" +
                             quoted_path(source) + "; destination=" + quoted_path(destination));
}

bool checked_path_exists(const std::filesystem::path& path) {
    const DWORD attributes = GetFileAttributesW(path.c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES)
        return true;
    const DWORD code = GetLastError();
    if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND)
        return false;
    throw windows_error("GetFileAttributesW(" + path.string() + ")", code);
}

void require_owned_transaction_directory(const std::filesystem::path& path,
                                         const std::filesystem::path& parent,
                                         const std::string& marker_name,
                                         const std::string& marker_contents) {
    if (path.empty() || path.parent_path() != parent || path == parent) {
        throw std::runtime_error("Refusing unsafe installer transaction path: " +
                                 quoted_path(path));
    }
    const DWORD attributes = GetFileAttributesW(path.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES)
        throw windows_error("GetFileAttributesW(" + path.string() + ")");
    if ((attributes & FILE_ATTRIBUTE_DIRECTORY) == 0 ||
        (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
        throw std::runtime_error("Installer transaction path is not a direct directory: " +
                                 quoted_path(path));
    }
    if (!installation_marker_matches(path, marker_name, marker_contents)) {
        throw std::runtime_error("Refusing to modify an unowned installer transaction directory: " +
                                 quoted_path(path));
    }
}

void rename_transaction_path(const std::filesystem::path& source,
                             const std::filesystem::path& destination,
                             InstallTransactionOperation operation,
                             const InstallFaultInjector* injector) {
    maybe_inject_fault(injector, operation, source, destination);
    if (!MoveFileExW(source.c_str(), destination.c_str(), 0)) {
        throw windows_error(std::string("Unable to ") + transaction_operation_name(operation) +
                            "; source=" + quoted_path(source) +
                            "; destination=" + quoted_path(destination));
    }
}

void remove_owned_transaction_directory(const std::filesystem::path& path,
                                        const std::filesystem::path& parent,
                                        const std::string& marker_name,
                                        const std::string& marker_contents,
                                        InstallTransactionOperation operation,
                                        const InstallFaultInjector* injector) {
    if (!checked_path_exists(path))
        return;
    require_owned_transaction_directory(path, parent, marker_name, marker_contents);
    maybe_inject_fault(injector, operation, path, {});
    std::error_code error;
    std::filesystem::remove_all(path, error);
    if (error || checked_path_exists(path)) {
        throw std::runtime_error(
            std::string("Unable to ") + transaction_operation_name(operation) + " at " +
            quoted_path(path) +
            (error ? "; error=" + error.message() : "; directory still exists"));
    }
}

void recover_interrupted_transaction(
    const std::filesystem::path& install_root, const std::filesystem::path& staging,
    const std::filesystem::path& backup, const std::filesystem::path& recovery,
    const std::filesystem::path& retired, const std::filesystem::path& parent,
    const std::string& marker_name, const std::string& marker_contents,
    const InstallFaultInjector* injector) {
    if (checked_path_exists(backup)) {
        require_owned_transaction_directory(backup, parent, marker_name, marker_contents);
        if (checked_path_exists(recovery)) {
            remove_owned_transaction_directory(recovery, parent, marker_name, marker_contents,
                                               InstallTransactionOperation::RemoveRecovery,
                                               injector);
        }
        if (checked_path_exists(install_root)) {
            try {
                rename_transaction_path(install_root, recovery,
                                        InstallTransactionOperation::QuarantineFailedInstall,
                                        injector);
            } catch (...) {
                throw std::runtime_error(
                    "Interrupted install recovery could not preserve the current tree at " +
                    quoted_path(recovery) + "; prior installation backup remains at " +
                    quoted_path(backup) + "; current tree remains at " + quoted_path(install_root) +
                    "; cause: " + current_exception_message(std::current_exception()));
            }
        }
        try {
            rename_transaction_path(backup, install_root,
                                    InstallTransactionOperation::RestoreBackup, injector);
        } catch (...) {
            throw std::runtime_error(
                "Interrupted install recovery could not restore the prior installation; backup "
                "remains at " +
                quoted_path(backup) + "; recovery tree is at " + quoted_path(recovery) +
                "; cause: " + current_exception_message(std::current_exception()));
        }
        try {
            remove_owned_transaction_directory(staging, parent, marker_name, marker_contents,
                                               InstallTransactionOperation::RemoveStaging,
                                               injector);
            remove_owned_transaction_directory(recovery, parent, marker_name, marker_contents,
                                               InstallTransactionOperation::RemoveRecovery,
                                               injector);
            remove_owned_transaction_directory(retired, parent, marker_name, marker_contents,
                                               InstallTransactionOperation::RemoveRetiredBackup,
                                               injector);
        } catch (...) {
            throw std::runtime_error("The prior installation was restored at " +
                                     quoted_path(install_root) +
                                     ", but recovery cleanup is incomplete; cause: " +
                                     current_exception_message(std::current_exception()));
        }
        return;
    }

    remove_owned_transaction_directory(staging, parent, marker_name, marker_contents,
                                       InstallTransactionOperation::RemoveStaging, injector);
    remove_owned_transaction_directory(recovery, parent, marker_name, marker_contents,
                                       InstallTransactionOperation::RemoveRecovery, injector);
    remove_owned_transaction_directory(retired, parent, marker_name, marker_contents,
                                       InstallTransactionOperation::RemoveRetiredBackup, injector);
}

} // namespace

void install_payload_transactional(const std::filesystem::path& payload_root,
                                   const std::filesystem::path& install_root,
                                   const std::vector<PayloadEntry>& entries,
                                   const std::string& marker_name,
                                   const std::string& marker_contents,
                                   const InstallFaultInjector* fault_injector) {
    if (install_root.empty() || install_root.filename().empty())
        throw std::runtime_error("Installation directory is invalid");

    const auto parent = install_root.parent_path();
    if (parent.empty())
        throw std::runtime_error("Installation directory must have a parent");
    std::filesystem::create_directories(parent);
    const auto base_name = install_root.filename().wstring();
    const auto staging = parent / (base_name + L".staging");
    const auto backup = parent / (base_name + L".backup");
    const auto recovery = parent / (base_name + L".recovery-corrupt");
    const auto retired = parent / (base_name + L".recovery-retired");

    recover_interrupted_transaction(install_root, staging, backup, recovery, retired, parent,
                                    marker_name, marker_contents, fault_injector);
    if (checked_path_exists(install_root)) {
        require_owned_transaction_directory(install_root, parent, marker_name, marker_contents);
    }
    // Recovery is independent of the new package. A damaged or unavailable
    // payload must not prevent restoration of a retained prior backup.
    verify_payload(payload_root, entries);

    try {
        std::error_code create_error;
        if (!std::filesystem::create_directory(staging, create_error) || create_error) {
            throw std::runtime_error("Unable to create installer staging directory at " +
                                     quoted_path(staging) +
                                     (create_error ? "; error=" + create_error.message() : ""));
        }

        const auto canonical_marker = lowercase_ascii(marker_name);
        const auto marker_entry =
            std::find_if(entries.begin(), entries.end(), [&](const PayloadEntry& entry) {
                return lowercase_ascii(entry.relative_path.generic_u8string()) == canonical_marker;
            });
        if (marker_entry == entries.end())
            throw std::runtime_error("Payload manifest does not contain the install marker");
        materialize_file(payload_root / marker_entry->relative_path,
                         staging / marker_entry->relative_path);
        if (!installation_marker_matches(staging, marker_name, marker_contents))
            throw std::runtime_error("Payload installation marker is missing or invalid");

        for (const auto& entry : entries) {
            if (&entry == &*marker_entry)
                continue;
            const auto source = payload_root / entry.relative_path;
            const auto destination = staging / entry.relative_path;
            materialize_file(source, destination);
        }
        // The source layout can change after the preflight verification. Hash
        // the bytes that will actually be committed, and audit the full staging
        // tree so a same-size replacement or unexpected file cannot cross the
        // verification/copy boundary.
        verify_payload(staging, entries);
        const bool replacing = checked_path_exists(install_root);
        if (replacing)
            rename_transaction_path(install_root, backup,
                                    InstallTransactionOperation::BackupExisting, fault_injector);
        try {
            rename_transaction_path(staging, install_root,
                                    InstallTransactionOperation::CommitStaging, fault_injector);
            // Attest the final path before discarding the rollback copy.
            maybe_inject_fault(fault_injector, InstallTransactionOperation::VerifyFinalPayload,
                               install_root, {});
            verify_payload(install_root, entries);
        } catch (...) {
            const auto commit_error = std::current_exception();
            try {
                if (checked_path_exists(install_root)) {
                    rename_transaction_path(install_root, recovery,
                                            InstallTransactionOperation::QuarantineFailedInstall,
                                            fault_injector);
                }
                if (replacing && checked_path_exists(backup)) {
                    rename_transaction_path(backup, install_root,
                                            InstallTransactionOperation::RestoreBackup,
                                            fault_injector);
                }
                remove_owned_transaction_directory(recovery, parent, marker_name, marker_contents,
                                                   InstallTransactionOperation::RemoveRecovery,
                                                   fault_injector);
            } catch (...) {
                throw std::runtime_error(
                    "Installation commit failed: " + current_exception_message(commit_error) +
                    "; rollback is incomplete. Prior installation backup path=" +
                    quoted_path(backup) + "; current install path=" + quoted_path(install_root) +
                    "; recovery path=" + quoted_path(recovery) +
                    "; rollback cause: " + current_exception_message(std::current_exception()));
            }
            std::rethrow_exception(commit_error);
        }
        if (replacing && checked_path_exists(backup)) {
            try {
                // The rename is the commit point: once the verified backup is
                // retired, a later cleanup failure cannot make recovery roll
                // back the newly verified installation.
                rename_transaction_path(backup, retired, InstallTransactionOperation::RetireBackup,
                                        fault_injector);
            } catch (...) {
                const auto retire_error = std::current_exception();
                try {
                    rename_transaction_path(install_root, recovery,
                                            InstallTransactionOperation::QuarantineFailedInstall,
                                            fault_injector);
                    rename_transaction_path(backup, install_root,
                                            InstallTransactionOperation::RestoreBackup,
                                            fault_injector);
                    remove_owned_transaction_directory(
                        recovery, parent, marker_name, marker_contents,
                        InstallTransactionOperation::RemoveRecovery, fault_injector);
                } catch (...) {
                    throw std::runtime_error(
                        "Verified install could not retire its rollback backup: " +
                        current_exception_message(retire_error) +
                        "; rollback is incomplete. Backup path=" + quoted_path(backup) +
                        "; install path=" + quoted_path(install_root) +
                        "; recovery path=" + quoted_path(recovery) +
                        "; rollback cause: " + current_exception_message(std::current_exception()));
                }
                std::rethrow_exception(retire_error);
            }
            remove_owned_transaction_directory(retired, parent, marker_name, marker_contents,
                                               InstallTransactionOperation::RemoveRetiredBackup,
                                               fault_injector);
        }
    } catch (...) {
        const auto primary_error = std::current_exception();
        try {
            remove_owned_transaction_directory(staging, parent, marker_name, marker_contents,
                                               InstallTransactionOperation::RemoveStaging,
                                               fault_injector);
        } catch (...) {
            throw std::runtime_error(
                "Installer failed: " + current_exception_message(primary_error) +
                "; staging cleanup at " + quoted_path(staging) +
                " also failed: " + current_exception_message(std::current_exception()));
        }
        std::rethrow_exception(primary_error);
    }
}

} // namespace trtmc::installer
