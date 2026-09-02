/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace trtmc::installer {

struct PayloadEntry {
    std::array<std::uint8_t, 32> sha256{};
    std::uintmax_t size{0};
    std::filesystem::path relative_path;
};

// The package step stamps the exact payload.manifest bytes into this RCDATA
// resource in the outer setup executable. Release signing happens afterwards.
inline constexpr std::uint16_t kPayloadManifestResourceId = 241;

// A second Setup invocation waits this long for an install/uninstall transaction
// on the same per-user destination. The mutex itself is cross-session so two
// logons for the same Windows user cannot mutate the fixed recovery paths at
// the same time.
inline constexpr std::uint32_t kInstallTransactionLockTimeoutMs = 30U * 60U * 1000U;

class InstallTransactionLock {
  public:
    InstallTransactionLock(const InstallTransactionLock&) = delete;
    InstallTransactionLock& operator=(const InstallTransactionLock&) = delete;
    InstallTransactionLock(InstallTransactionLock&& other) noexcept;
    InstallTransactionLock& operator=(InstallTransactionLock&& other) noexcept;
    ~InstallTransactionLock();

    // WAIT_ABANDONED transfers ownership to this caller. The fixed transaction
    // paths can then be recovered under the lock before new mutations begin.
    bool recovered_abandoned_owner() const noexcept { return abandoned_; }

  private:
    friend InstallTransactionLock acquire_install_transaction_lock(const std::filesystem::path&,
                                                                   std::uint32_t);

    explicit InstallTransactionLock(void* handle, bool abandoned) noexcept
        : handle_(handle), abandoned_(abandoned) {}

    void release() noexcept;

    void* handle_{nullptr};
    bool abandoned_{false};
};

// Return the collision-resistant kernel-object name used for this current
// user's canonical install root. Exposed so the native regression test can
// keep the object alive while proving abandoned-owner recovery.
std::wstring install_transaction_mutex_name(const std::filesystem::path& install_root);

InstallTransactionLock
acquire_install_transaction_lock(const std::filesystem::path& install_root,
                                 std::uint32_t timeout_ms = kInstallTransactionLockTimeoutMs);

enum class InstallTransactionOperation {
    BackupExisting,
    CommitStaging,
    VerifyFinalPayload,
    QuarantineFailedInstall,
    RestoreBackup,
    RetireBackup,
    RemoveStaging,
    RemoveRecovery,
    RemoveRetiredBackup,
};

// Deterministic filesystem fault injection for the native installer unit test.
// Returning zero performs the operation; any other value is reported as a
// simulated Windows error. Production callers leave this null.
using InstallFaultInjector = std::function<std::uint32_t(
    InstallTransactionOperation, const std::filesystem::path&, const std::filesystem::path&)>;

// Manifest rows use: lowercase-sha256<TAB>decimal-size<TAB>UTF-8-relative-path.
// Paths must use forward slashes and satisfy Windows canonical-name rules.
std::vector<PayloadEntry> read_payload_manifest(const std::filesystem::path& manifest_path);

// Read the manifest once, require its raw bytes to equal the RCDATA anchor in
// module_handle (or the current executable when null), then parse those same
// bytes. This closes the replace-manifest-and-payload authenticity gap.
std::vector<PayloadEntry>
read_authenticated_payload_manifest(const std::filesystem::path& manifest_path,
                                    void* module_handle = nullptr);

bool is_safe_payload_path(const std::string& utf8_path);

std::array<std::uint8_t, 32> sha256_file(const std::filesystem::path& path);
std::string sha256_hex(const std::array<std::uint8_t, 32>& digest);

void verify_payload(const std::filesystem::path& payload_root,
                    const std::vector<PayloadEntry>& entries);

// Reject manifests that contain anything outside the fixed native H3 runtime,
// one versioned TensorRT-RTX DLL, and the optional repository legal notices.
void validate_minimax_h3_runtime_payload(const std::vector<PayloadEntry>& entries);

// Verify the source, copy it into a sibling staging directory, verify the exact
// staged bytes, atomically replace the destination, and verify the final path
// before discarding the rollback copy. Existing destinations must carry the
// exact install marker, which prevents an installer invocation from replacing
// an unrelated directory. Installed files are never hard-linked to the mutable
// package layout. Interrupted replacements are recovered from fixed sibling
// backup/staging/recovery paths; an old backup always wins over a partially
// committed replacement.
void install_payload_transactional(const std::filesystem::path& payload_root,
                                   const std::filesystem::path& install_root,
                                   const std::vector<PayloadEntry>& entries,
                                   const std::string& marker_name,
                                   const std::string& marker_contents,
                                   const InstallFaultInjector* fault_injector = nullptr);

bool installation_marker_matches(const std::filesystem::path& install_root,
                                 const std::string& marker_name,
                                 const std::string& marker_contents);

} // namespace trtmc::installer
