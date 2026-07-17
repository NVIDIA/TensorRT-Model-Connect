/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace trtmc::internal {

class Sha256 {
  public:
    Sha256() = default;

    void update(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        total_size_ += size;
        while (size != 0) {
            const std::size_t count = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, bytes, count);
            block_size_ += count;
            bytes += count;
            size -= count;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    void update(std::string_view value) { update(value.data(), value.size()); }

    std::array<std::uint8_t, 32> digest() const {
        Sha256 copy = *this;
        return copy.finalize();
    }

    std::string hex_digest() const {
        const auto bytes = digest();
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const std::uint8_t byte : bytes)
            output << std::setw(2) << static_cast<unsigned int>(byte);
        return output.str();
    }

  private:
    static constexpr std::array<std::uint32_t, 64> kRoundConstants = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
        0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
        0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
        0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
        0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
        0xc67178f2U};

    static std::uint32_t rotate_right(std::uint32_t value, unsigned int count) {
        return (value >> count) | (value << (32U - count));
    }

    static std::uint32_t choose(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (x & y) ^ (~x & z);
    }

    static std::uint32_t majority(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    static std::uint32_t big_sigma_zero(std::uint32_t value) {
        return rotate_right(value, 2) ^ rotate_right(value, 13) ^ rotate_right(value, 22);
    }

    static std::uint32_t big_sigma_one(std::uint32_t value) {
        return rotate_right(value, 6) ^ rotate_right(value, 11) ^ rotate_right(value, 25);
    }

    static std::uint32_t small_sigma_zero(std::uint32_t value) {
        return rotate_right(value, 7) ^ rotate_right(value, 18) ^ (value >> 3);
    }

    static std::uint32_t small_sigma_one(std::uint32_t value) {
        return rotate_right(value, 17) ^ rotate_right(value, 19) ^ (value >> 10);
    }

    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            const std::size_t offset = i * 4;
            words[i] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                       (static_cast<std::uint32_t>(block[offset + 1]) << 16U) |
                       (static_cast<std::uint32_t>(block[offset + 2]) << 8U) |
                       static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            words[i] = small_sigma_one(words[i - 2]) + words[i - 7] +
                       small_sigma_zero(words[i - 15]) + words[i - 16];
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t i = 0; i < words.size(); ++i) {
            const std::uint32_t temporary_one =
                h + big_sigma_one(e) + choose(e, f, g) + kRoundConstants[i] + words[i];
            const std::uint32_t temporary_two = big_sigma_zero(a) + majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + temporary_one;
            d = c;
            c = b;
            b = a;
            a = temporary_one + temporary_two;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint8_t, 32> finalize() {
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56,
                  0);
        const std::uint64_t bit_length = total_size_ * 8U;
        for (std::size_t i = 0; i < 8; ++i)
            block_[63 - i] = static_cast<std::uint8_t>(bit_length >> (8U * i));
        transform(block_.data());

        std::array<std::uint8_t, 32> output{};
        for (std::size_t i = 0; i < state_.size(); ++i) {
            output[i * 4] = static_cast<std::uint8_t>(state_[i] >> 24U);
            output[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16U);
            output[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8U);
            output[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
        }
        return output;
    }

    std::array<std::uint32_t, 8> state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                           0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_{0};
    std::uint64_t total_size_{0};
};

} // namespace trtmc::internal
