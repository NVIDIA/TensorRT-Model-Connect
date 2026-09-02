/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Shared test helpers for model family integration tests.
// Provides temp-dir creation, file writing, and safetensors construction.

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#ifndef _WIN32
#include <ftw.h>
#endif
#include <stdexcept>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <utility>
#include <vector>

namespace trtmc_test {

// Windows uses its native std::filesystem implementation. POSIX avoids
// std::filesystem::remove_all because the aarch64 CI image has been observed
// to trampoline into libtorch's symbol-interposed implementation and segfault
// during teardown; nftw() only calls plain libc unlink/rmdir there.
inline int remove_all_safe(const std::filesystem::path& path) {
    if (path.empty())
        return 0;
#ifdef _WIN32
    std::error_code error;
    std::filesystem::remove_all(path, error);
    return error ? -1 : 0;
#else
    auto cb = [](const char* fpath, const struct stat* /*sb*/, int typeflag,
                 struct FTW* /*ftwbuf*/) -> int {
        if (typeflag == FTW_DP)
            return rmdir(fpath) == 0 ? 0 : -1;
        return std::remove(fpath) == 0 ? 0 : -1;
    };
    return nftw(path.string().c_str(), cb, 16, FTW_DEPTH | FTW_PHYS);
#endif
}

#ifdef _WIN32
inline std::filesystem::path create_unique_temp_dir(const std::filesystem::path& parent,
                                                    const std::string& prefix) {
    static std::atomic<std::uint64_t> sequence{0};
    for (int attempt = 0; attempt < 256; ++attempt) {
        const auto clock_value = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto candidate =
            parent / (prefix + std::to_string(clock_value) + "_" +
                      std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error))
            return candidate;
        if (error && error != std::errc::file_exists) {
            throw std::runtime_error("Failed to create temporary test directory: " +
                                     error.message());
        }
    }
    throw std::runtime_error("Unable to allocate a unique temporary test directory");
}
#endif

// RAII guard that saves an environment variable on construction and restores it
// on destruction. Prevents env var state leaks between tests if a test fails
// early or throws.
class EnvVarGuard {
  public:
    // If value is non-null, setenv to that value; if null, unsetenv.
    explicit EnvVarGuard(const std::string& name, const char* value = nullptr) : name_(name) {
        const char* old = std::getenv(name.c_str());
        had_value_ = (old != nullptr);
        if (had_value_)
            old_value_ = old;
#ifdef _WIN32
        const int status = _putenv_s(name.c_str(), value ? value : "");
        if (status != 0)
            throw std::runtime_error("Unable to update test environment variable");
#else
        if (value) {
            setenv(name.c_str(), value, 1);
        } else {
            unsetenv(name.c_str());
        }
#endif
    }
    ~EnvVarGuard() {
#ifdef _WIN32
        (void)_putenv_s(name_.c_str(), had_value_ ? old_value_.c_str() : "");
#else
        if (had_value_) {
            setenv(name_.c_str(), old_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
#endif
    }
    EnvVarGuard(const EnvVarGuard&) = delete;
    EnvVarGuard& operator=(const EnvVarGuard&) = delete;

  private:
    std::string name_;
    std::string old_value_;
    bool had_value_;
};

// RAII guard that creates a temporary directory on construction and removes it
// recursively on destruction. Prevents temp directory leaks if a test fails
// early or throws.
class TempDirGuard {
  public:
    TempDirGuard() {
#ifdef _WIN32
        path_ =
            create_unique_temp_dir(std::filesystem::temp_directory_path(), "trtmc_test_").string();
#else
        char tmpl[] = "/tmp/trtmc_test_XXXXXX";
        char* result = mkdtemp(tmpl);
        if (!result)
            throw std::runtime_error("mkdtemp failed");
        path_ = result;
#endif
    }
    ~TempDirGuard() {
        if (!path_.empty()) {
            remove_all_safe(path_);
        }
    }
    const std::string& path() const { return path_; }
    TempDirGuard(const TempDirGuard&) = delete;
    TempDirGuard& operator=(const TempDirGuard&) = delete;

  private:
    std::string path_;
};

struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

inline std::filesystem::path make_temp_dir_or_throw(const char* pattern) {
#ifdef _WIN32
    std::filesystem::path requested(pattern);
    auto parent = requested.parent_path();
    std::error_code error;
    if (parent.empty() || !std::filesystem::is_directory(parent, error))
        parent = std::filesystem::temp_directory_path();
    std::string prefix = requested.filename().string();
    const auto suffix = prefix.find_last_not_of('X');
    prefix.erase(suffix == std::string::npos ? 0 : suffix + 1);
    if (prefix.empty())
        prefix = "trtmc_test_";
    return create_unique_temp_dir(parent, prefix);
#else
    char buffer[256];
    std::strncpy(buffer, pattern, sizeof(buffer));
    buffer[sizeof(buffer) - 1] = '\0';
    char* created = mkdtemp(buffer);
    if (created == nullptr) {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::filesystem::path(created);
#endif
}

inline void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }
    out << content;
}

inline void write_u64_le(std::ofstream& out, uint64_t value) {
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xFFU);
    }
    out.write(reinterpret_cast<const char*>(bytes), 8);
}

inline void write_safetensors_f32(const std::filesystem::path& path,
                                  const std::vector<TensorSpec>& specs) {
    std::string header = "{";
    uint64_t offset = 0;
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const auto& spec = specs[i];
        const uint64_t bytes = static_cast<uint64_t>(spec.data.size() * sizeof(float));
        if (i != 0) {
            header += ",";
        }
        header += "\"" + spec.name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t d = 0; d < spec.shape.size(); ++d) {
            if (d != 0) {
                header += ",";
            }
            header += std::to_string(spec.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
                  std::to_string(offset + bytes) + "]}";
        offset += bytes;
    }
    header += "}";

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to write safetensors file: " + path.string());
    }

    write_u64_le(out, static_cast<uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& spec : specs) {
        out.write(reinterpret_cast<const char*>(spec.data.data()),
                  static_cast<std::streamsize>(spec.data.size() * sizeof(float)));
    }
}

// Writes a model.safetensors.index.json file mapping tensor names to shard filenames.
inline void
write_safetensors_index(const std::filesystem::path& path,
                        const std::vector<std::pair<std::string, std::string>>& weight_map) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to write safetensors index file: " + path.string());
    }

    out << "{\n";
    out << "  \"metadata\": {},\n";
    out << "  \"weight_map\": {\n";
    for (std::size_t i = 0; i < weight_map.size(); ++i) {
        out << "    \"" << weight_map[i].first << "\": \"" << weight_map[i].second << "\"";
        out << (i + 1 == weight_map.size() ? "\n" : ",\n");
    }
    out << "  }\n";
    out << "}\n";
}

// Creates a standard decoder model checkpoint (embedding, per-layer attention+MLP, final_norm,
// lm_head). Suitable for standard decoder variants. Use include_qk_norm=true for QK-normalized
// variants.
inline void write_standard_decoder_checkpoint(const std::filesystem::path& dir, int32_t vocab,
                                              int32_t hidden, int32_t q_hidden, int32_t kv_hidden,
                                              int32_t mlp, int32_t layers, bool include_qk_norm) {
    std::vector<float> embedding(static_cast<std::size_t>(vocab) * static_cast<std::size_t>(hidden),
                                 0.0F);
    for (int32_t i = 0; i < hidden; ++i) {
        embedding[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) +
                  static_cast<std::size_t>(i)] = 1.0F;
    }

    std::vector<float> q_proj(static_cast<std::size_t>(q_hidden) * static_cast<std::size_t>(hidden),
                              0.0F);
    for (int32_t i = 0; i < hidden; ++i) {
        q_proj[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) +
               static_cast<std::size_t>(i)] = 1.0F;
    }

    std::vector<float> o_proj(static_cast<std::size_t>(hidden) * static_cast<std::size_t>(q_hidden),
                              0.0F);
    for (int32_t i = 0; i < hidden; ++i) {
        o_proj[static_cast<std::size_t>(i) * static_cast<std::size_t>(q_hidden) +
               static_cast<std::size_t>(i)] = 1.0F;
    }

    std::vector<float> k_proj(
        static_cast<std::size_t>(kv_hidden) * static_cast<std::size_t>(hidden), 0.0F);
    std::vector<float> v_proj(
        static_cast<std::size_t>(kv_hidden) * static_cast<std::size_t>(hidden), 0.0F);
    for (int32_t i = 0; i < kv_hidden; ++i) {
        k_proj[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) +
               static_cast<std::size_t>(i)] = 1.0F;
        v_proj[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) +
               static_cast<std::size_t>(i)] = 1.0F;
    }

    std::vector<float> norm(static_cast<std::size_t>(hidden), 1.0F);
    const int32_t head_dim = q_hidden / 2; // assumes num_attention_heads=2
    std::vector<float> qk_norm(static_cast<std::size_t>(head_dim), 1.0F);
    std::vector<float> up_proj(static_cast<std::size_t>(mlp) * static_cast<std::size_t>(hidden),
                               0.0F);
    std::vector<float> gate_proj(static_cast<std::size_t>(mlp) * static_cast<std::size_t>(hidden),
                                 0.0F);
    std::vector<float> down_proj(static_cast<std::size_t>(hidden) * static_cast<std::size_t>(mlp),
                                 0.0F);
    std::vector<float> lm_head(static_cast<std::size_t>(vocab) * static_cast<std::size_t>(hidden),
                               -1.0F);
    for (int32_t i = 0; i < vocab && i < hidden; ++i) {
        lm_head[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) +
                static_cast<std::size_t>(i)] = 1.0F;
    }

    std::vector<TensorSpec> tensors;
    tensors.push_back({"model.embed_tokens.weight", {vocab, hidden}, embedding});
    for (int32_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        tensors.push_back({prefix + "input_layernorm.weight", {hidden}, norm});
        if (include_qk_norm) {
            tensors.push_back({prefix + "self_attn.q_norm.weight", {head_dim}, qk_norm});
            tensors.push_back({prefix + "self_attn.k_norm.weight", {head_dim}, qk_norm});
        }
        tensors.push_back({prefix + "self_attn.q_proj.weight", {q_hidden, hidden}, q_proj});
        tensors.push_back({prefix + "self_attn.k_proj.weight", {kv_hidden, hidden}, k_proj});
        tensors.push_back({prefix + "self_attn.v_proj.weight", {kv_hidden, hidden}, v_proj});
        tensors.push_back({prefix + "self_attn.o_proj.weight", {hidden, q_hidden}, o_proj});
        tensors.push_back({prefix + "post_attention_layernorm.weight", {hidden}, norm});
        tensors.push_back({prefix + "mlp.gate_proj.weight", {mlp, hidden}, gate_proj});
        tensors.push_back({prefix + "mlp.up_proj.weight", {mlp, hidden}, up_proj});
        tensors.push_back({prefix + "mlp.down_proj.weight", {hidden, mlp}, down_proj});
    }
    tensors.push_back({"model.norm.weight", {hidden}, norm});
    tensors.push_back({"lm_head.weight", {vocab, hidden}, lm_head});
    write_safetensors_f32(dir / "model.safetensors", tensors);
}

} // namespace trtmc_test
