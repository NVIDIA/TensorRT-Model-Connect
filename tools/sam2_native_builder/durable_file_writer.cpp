/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "durable_file_writer.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace trtmc::sam2::durable_file {
namespace {

#if defined(__linux__)

std::string systemError(std::string_view action, int error) {
    return std::string(action) + ": " + std::strerror(error);
}

class ScopedDescriptor final {
  public:
    explicit ScopedDescriptor(int value = -1) noexcept : value_(value) {}
    ~ScopedDescriptor() {
        if (value_ >= 0)
            (void)::close(value_);
    }

    ScopedDescriptor(const ScopedDescriptor&) = delete;
    ScopedDescriptor& operator=(const ScopedDescriptor&) = delete;

    int get() const noexcept { return value_; }
    void reset(int value) noexcept {
        if (value_ >= 0)
            (void)::close(value_);
        value_ = value;
    }

  private:
    int value_{-1};
};

class NamedTemporaryFile final {
  public:
    NamedTemporaryFile(int directory, std::string_view artifact)
        : directory_(directory), artifact_(artifact) {
        static std::atomic<std::uint64_t> sequence{0U};
        int flags = O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC;
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#endif
        for (std::size_t attempt = 0U; attempt < 128U; ++attempt) {
            name_ = ".trtmc-sam2-" + std::to_string(static_cast<long long>(::getpid())) + "-" +
                    std::to_string(sequence.fetch_add(1U, std::memory_order_relaxed)) + ".tmp";
            const int descriptor = ::openat(directory_, name_.c_str(), flags, 0600);
            if (descriptor >= 0) {
                descriptor_.reset(descriptor);
                return;
            }
            if (errno != EEXIST) {
                throw WriteError(systemError(
                    "Failed to create exclusive " + artifact_ + " temporary file", errno));
            }
        }
        throw WriteError("Failed to allocate a unique " + artifact_ + " temporary name");
    }

    ~NamedTemporaryFile() { removeNameIfStillOurs(); }

    int descriptor() const noexcept { return descriptor_.get(); }
    void unlinkAuthenticatedName() {
        if (descriptor_.get() < 0 || name_.empty())
            return;
        struct stat opened{};
        if (::fstat(descriptor_.get(), &opened) != 0) {
            throw WriteError(
                systemError("Failed to inspect the " + artifact_ + " temporary descriptor", errno));
        }
        struct stat named{};
        if (::fstatat(directory_, name_.c_str(), &named, AT_SYMLINK_NOFOLLOW) != 0) {
            throw WriteError(
                systemError("Failed to inspect the " + artifact_ + " temporary name", errno));
        }
        if (!S_ISREG(named.st_mode) || opened.st_dev != named.st_dev ||
            opened.st_ino != named.st_ino) {
            throw WriteError(artifact_ + " temporary name no longer matched its descriptor");
        }
        if (::unlinkat(directory_, name_.c_str(), 0) != 0) {
            throw WriteError(
                systemError("Failed to unlink the " + artifact_ + " temporary name", errno));
        }
        name_.clear();
    }

  private:
    void removeNameIfStillOurs() noexcept {
        if (descriptor_.get() < 0 || name_.empty())
            return;
        struct stat opened{};
        struct stat named{};
        if (::fstat(descriptor_.get(), &opened) != 0 ||
            ::fstatat(directory_, name_.c_str(), &named, AT_SYMLINK_NOFOLLOW) != 0 ||
            !S_ISREG(named.st_mode) || opened.st_dev != named.st_dev ||
            opened.st_ino != named.st_ino) {
            return;
        }
        (void)::unlinkat(directory_, name_.c_str(), 0);
    }

    int directory_{-1};
    std::string artifact_;
    std::string name_;
    ScopedDescriptor descriptor_;
};

void requireDestinationAbsent(int directory, const std::string& filename,
                              const std::string& artifact) {
    struct stat status{};
    if (::fstatat(directory, filename.c_str(), &status, AT_SYMLINK_NOFOLLOW) == 0)
        throw WriteError(artifact + " exclusive destination already exists");
    if (errno != ENOENT) {
        throw WriteError(systemError("Failed to inspect the " + artifact + " destination", errno));
    }
}

void syncDescriptor(int descriptor, std::string_view action) {
    int result = -1;
    do {
        result = ::fsync(descriptor);
    } while (result != 0 && errno == EINTR);
    if (result != 0)
        throw WriteError(systemError(action, errno));
}

int syncDirectory(int directory) noexcept {
    int result = -1;
    do {
        result = ::fsync(directory);
    } while (result != 0 && errno == EINTR);
    return result;
}

PublicationIdentity completedIdentity(int descriptor, const std::string& artifact) {
    struct stat status{};
    if (::fstat(descriptor, &status) != 0) {
        throw WriteError(
            systemError("Failed to inspect the completed " + artifact + " descriptor", errno));
    }
    if (!S_ISREG(status.st_mode) || status.st_size < 0)
        throw WriteError("Completed " + artifact + " descriptor was not a regular file");
    return {static_cast<std::uint64_t>(status.st_size), static_cast<std::uint64_t>(status.st_dev),
            static_cast<std::uint64_t>(status.st_ino)};
}

#ifdef AT_EMPTY_PATH
bool descriptorLinkFallbackAllowed(int error) noexcept {
    return error == EINVAL || error == ENOENT || error == EPERM || error == EOPNOTSUPP;
}
#endif

void linkDescriptorWithoutOverwrite(int descriptor, int directory, const std::string& filename,
                                    const std::string& artifact) {
    int result = -1;
    int error = 0;
#ifdef AT_EMPTY_PATH
    result = ::linkat(descriptor, "", directory, filename.c_str(), AT_EMPTY_PATH);
    if (result != 0)
        error = errno;
#endif
    if (result != 0) {
#ifdef AT_EMPTY_PATH
        if (!descriptorLinkFallbackAllowed(error)) {
            throw WriteError(systemError(
                "Failed to publish the " + artifact + " descriptor without overwrite", error));
        }
#endif
        const std::string descriptor_path = "/proc/self/fd/" + std::to_string(descriptor);
        result = ::linkat(AT_FDCWD, descriptor_path.c_str(), directory, filename.c_str(),
                          AT_SYMLINK_FOLLOW);
        if (result != 0) {
            throw WriteError(systemError(
                "Failed to publish the " + artifact + " descriptor without overwrite", errno));
        }
    }
}

void authenticatePublishedName(int directory, const std::string& filename,
                               const PublicationIdentity& identity, const std::string& artifact) {
    struct stat published{};
    if (::fstatat(directory, filename.c_str(), &published, AT_SYMLINK_NOFOLLOW) != 0 ||
        !S_ISREG(published.st_mode) ||
        identity.device != static_cast<std::uint64_t>(published.st_dev) ||
        identity.inode != static_cast<std::uint64_t>(published.st_ino) ||
        identity.size_bytes != static_cast<std::uint64_t>(published.st_size)) {
        throw WriteError("Published " + artifact + " did not match its completed file descriptor");
    }
}

std::string rollbackPublishedDescriptor(int descriptor, int directory, const std::string& filename,
                                        const std::string& artifact) {
    struct stat opened{};
    if (::fstat(descriptor, &opened) != 0) {
        return systemError("rollback could not inspect the completed " + artifact + " descriptor",
                           errno);
    }

    struct stat published{};
    if (::fstatat(directory, filename.c_str(), &published, AT_SYMLINK_NOFOLLOW) != 0) {
        if (errno == ENOENT)
            return "rollback found the " + artifact + " destination already absent";
        return systemError("rollback could not inspect the " + artifact + " destination", errno);
    }
    if (!S_ISREG(published.st_mode) || opened.st_dev != published.st_dev ||
        opened.st_ino != published.st_ino) {
        return "rollback skipped the " + artifact +
               " destination because it no longer matched the published inode";
    }
    if (::unlinkat(directory, filename.c_str(), 0) != 0) {
        return systemError("rollback could not remove the exact published " + artifact, errno);
    }
    return "rollback removed the exact published " + artifact + " inode";
}

[[noreturn]] void throwAfterRollback(int descriptor, int directory, const std::string& filename,
                                     const std::string& artifact, std::string failure) {
    failure += "; " + rollbackPublishedDescriptor(descriptor, directory, filename, artifact);
    const int sync_result = syncDirectory(directory);
    const int sync_error = sync_result == 0 ? 0 : errno;
    if (sync_result == 0) {
        failure += "; post-rollback destination directory fsync succeeded";
    } else {
        failure +=
            "; " + systemError("post-rollback destination directory fsync failed", sync_error);
    }
    throw WriteError(std::move(failure));
}

void cleanupTemporaryBeforePublication(NamedTemporaryFile& temporary, int directory) {
    temporary.unlinkAuthenticatedName();
    syncDescriptor(directory, "Failed to fsync the destination directory after temporary cleanup");
}

#endif

} // namespace

PublicationIdentity writeExclusiveDurably(const std::filesystem::path& destination,
                                          std::string_view artifact, const DescriptorAction& writer,
                                          const DescriptorAction& validate_after_sync) {
    if (destination.empty() || !destination.has_filename() || artifact.empty() || !writer)
        throw WriteError("Durable file destination, artifact, and writer must be nonempty");

#if defined(__linux__)
    const auto parent =
        destination.parent_path().empty() ? std::filesystem::path(".") : destination.parent_path();
    const std::string filename = destination.filename().string();
    if (filename.empty() || filename == "." || filename == ".." ||
        filename.find('\0') != std::string::npos) {
        throw WriteError(std::string(artifact) + " destination has an unsafe filename");
    }

    int directory_flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC;
#ifdef O_NOFOLLOW
    directory_flags |= O_NOFOLLOW;
#endif
    const ScopedDescriptor directory(::open(parent.c_str(), directory_flags));
    if (directory.get() < 0) {
        throw WriteError(systemError(
            "Failed to open the " + std::string(artifact) + " destination directory", errno));
    }
    const std::string artifact_string(artifact);
    requireDestinationAbsent(directory.get(), filename, artifact_string);

    NamedTemporaryFile temporary(directory.get(), artifact);
    const int descriptor = temporary.descriptor();
    PublicationIdentity identity;
    try {
        writer(descriptor);
        syncDescriptor(descriptor, "Failed to fsync the completed " + artifact_string);
        if (validate_after_sync)
            validate_after_sync(descriptor);
        identity = completedIdentity(descriptor, artifact_string);
    } catch (...) {
        const auto original = std::current_exception();
        try {
            cleanupTemporaryBeforePublication(temporary, directory.get());
        } catch (const std::exception& cleanup_error) {
            std::string message =
                "Durable " + artifact_string + " write failed and temporary cleanup also failed: ";
            try {
                std::rethrow_exception(original);
            } catch (const std::exception& original_error) {
                message += original_error.what();
            } catch (...) {
                message += "non-standard exception";
            }
            message += "; ";
            message += cleanup_error.what();
            throw WriteError(std::move(message));
        }
        std::rethrow_exception(original);
    }

    bool destination_linked = false;
    try {
        linkDescriptorWithoutOverwrite(descriptor, directory.get(), filename, artifact_string);
        destination_linked = true;
        authenticatePublishedName(directory.get(), filename, identity, artifact_string);
    } catch (const std::exception& error) {
        std::string failure = error.what();
        try {
            cleanupTemporaryBeforePublication(temporary, directory.get());
        } catch (const std::exception& cleanup_error) {
            failure += "; temporary cleanup failed: ";
            failure += cleanup_error.what();
        }
        if (destination_linked) {
            throwAfterRollback(descriptor, directory.get(), filename, artifact_string,
                               std::move(failure));
        }
        throw WriteError(std::move(failure));
    }

    try {
        temporary.unlinkAuthenticatedName();
    } catch (const std::exception& error) {
        std::string failure = error.what();
        try {
            temporary.unlinkAuthenticatedName();
            failure += "; retry removed the exact " + artifact_string + " temporary name";
        } catch (const std::exception& retry_error) {
            failure += "; temporary-name cleanup retry failed: ";
            failure += retry_error.what();
        }
        throwAfterRollback(descriptor, directory.get(), filename, artifact_string,
                           std::move(failure));
    }

    if (syncDirectory(directory.get()) != 0) {
        const int sync_error = errno;
        throwAfterRollback(descriptor, directory.get(), filename, artifact_string,
                           systemError("Failed to fsync the " + artifact_string +
                                           " destination directory after publication",
                                       sync_error));
    }
    return identity;
#else
    (void)validate_after_sync;
    throw WriteError("Durable exclusive file publication requires Linux");
#endif
}

} // namespace trtmc::sam2::durable_file
