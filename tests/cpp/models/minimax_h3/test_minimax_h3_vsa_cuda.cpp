/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/vsa_attention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int32_t kHeads = 1;
constexpr int32_t kPrefixTiles = 2;
constexpr int32_t kVideoTiles = 3;
constexpr int32_t kTotalTiles = kPrefixTiles + kVideoTiles;
constexpr int32_t kTopVideoTiles = 1;
constexpr int32_t kTile = trtmc::minimax_h3::vsa::kTileTokens;
constexpr int32_t kDim = trtmc::minimax_h3::vsa::kHeadDim;
constexpr float kScale = 0.08838834764831844055F;
constexpr float kLog2E = 1.4426950408889634074F;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__nv_bfloat16 to_bf16(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return data_; }
    const T* get() const { return data_; }
    std::size_t count() const { return count_; }

  private:
    T* data_{nullptr};
    std::size_t count_{0};
};

template <typename T>
void upload(DeviceBuffer<T>& destination, const std::vector<T>& source, cudaStream_t stream) {
    require(destination.count() == source.size(), "upload size mismatch");
    check_cuda(cudaMemcpyAsync(destination.get(), source.data(), source.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync H2D");
}

template <typename T>
std::vector<T> download(const DeviceBuffer<T>& source, cudaStream_t stream) {
    std::vector<T> result(source.count());
    check_cuda(cudaMemcpyAsync(result.data(), source.get(), result.size() * sizeof(T),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync D2H");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return result;
}

std::size_t tiled_offset(int32_t head, int32_t tile, int32_t row, int32_t dimension) {
    return (((static_cast<std::size_t>(head) * kTotalTiles + tile) * kTile + row) * kDim +
            dimension);
}

void test_tile_pool_and_untile(cudaStream_t stream) {
    const std::vector<int32_t> valid_sizes = {5, 3, 7, 4, 6};
    const int32_t logical_rows = 25;
    std::vector<int32_t> map(static_cast<std::size_t>(kTotalTiles) * kTile, -1);
    int32_t packed_row = 0;
    for (int32_t tile = 0; tile < kTotalTiles; ++tile) {
        for (int32_t row = 0; row < valid_sizes[static_cast<std::size_t>(tile)]; ++row)
            map[static_cast<std::size_t>(tile) * kTile + row] = packed_row++;
    }
    std::vector<__nv_bfloat16> packed(static_cast<std::size_t>(logical_rows) * kDim);
    for (int32_t row = 0; row < logical_rows; ++row) {
        for (int32_t dimension = 0; dimension < kDim; ++dimension) {
            packed[static_cast<std::size_t>(row) * kDim + dimension] =
                to_bf16(std::sin(static_cast<float>(row * 131 + dimension) * 0.017F) * 0.5F);
        }
    }

    DeviceBuffer<__nv_bfloat16> device_packed(packed.size());
    DeviceBuffer<__nv_bfloat16> device_tiled(static_cast<std::size_t>(kTotalTiles) * kTile * kDim);
    DeviceBuffer<__nv_bfloat16> device_roundtrip(packed.size());
    DeviceBuffer<int32_t> device_map(map.size());
    DeviceBuffer<int32_t> device_valid(valid_sizes.size());
    DeviceBuffer<float> device_pool(static_cast<std::size_t>(kTotalTiles) * kDim);
    upload(device_packed, packed, stream);
    upload(device_map, map, stream);
    upload(device_valid, valid_sizes, stream);

    trtmc::minimax_h3::vsa::tile_bhsd_async(device_packed.get(), device_map.get(),
                                            device_tiled.get(), kHeads, logical_rows, kTotalTiles,
                                            stream);
    trtmc::minimax_h3::vsa::mean_pool_tiles_async(device_tiled.get(), device_valid.get(),
                                                  device_pool.get(), kHeads, kTotalTiles, stream);
    trtmc::minimax_h3::vsa::untile_bhsd_async(device_tiled.get(), device_map.get(),
                                              device_roundtrip.get(), kHeads, logical_rows,
                                              kTotalTiles, stream);

    const auto tiled = download(device_tiled, stream);
    const auto roundtrip = download(device_roundtrip, stream);
    const auto pool = download(device_pool, stream);
    for (std::size_t index = 0; index < packed.size(); ++index)
        require(to_float(roundtrip[index]) == to_float(packed[index]),
                "tile/untile did not preserve a BF16 value");
    for (int32_t tile = 0; tile < kTotalTiles; ++tile) {
        for (int32_t dimension = 0; dimension < kDim; ++dimension) {
            float expected = 0.0F;
            for (int32_t row = 0; row < valid_sizes[static_cast<std::size_t>(tile)]; ++row) {
                const int32_t source = map[static_cast<std::size_t>(tile) * kTile + row];
                expected += to_float(packed[static_cast<std::size_t>(source) * kDim + dimension]);
            }
            expected /= static_cast<float>(valid_sizes[static_cast<std::size_t>(tile)]);
            require(std::abs(pool[static_cast<std::size_t>(tile) * kDim + dimension] - expected) <
                        2.0e-6F,
                    "mean-pool differs from the FP32 reference");
        }
        for (int32_t row = valid_sizes[static_cast<std::size_t>(tile)]; row < kTile; ++row) {
            require(to_float(tiled[tiled_offset(0, tile, row, 0)]) == 0.0F,
                    "tile padding must be exact BF16 zero");
        }
    }
}

struct AttentionFixture {
    std::vector<int32_t> valid_sizes{5, 3, 7, 4, 6};
    std::vector<__nv_bfloat16> query;
    std::vector<__nv_bfloat16> key;
    std::vector<__nv_bfloat16> value;
    std::vector<__nv_bfloat16> gate;

    AttentionFixture() {
        const std::size_t count = static_cast<std::size_t>(kHeads) * kTotalTiles * kTile * kDim;
        query.resize(count, to_bf16(0.0F));
        key.resize(count, to_bf16(0.0F));
        value.resize(count, to_bf16(0.0F));
        gate.resize(count, to_bf16(0.0F));
        for (int32_t tile = 0; tile < kTotalTiles; ++tile) {
            for (int32_t row = 0; row < valid_sizes[static_cast<std::size_t>(tile)]; ++row) {
                for (int32_t dimension = 0; dimension < kDim; ++dimension) {
                    const auto index = tiled_offset(0, tile, row, dimension);
                    const float x =
                        static_cast<float>(tile * 1009 + row * 137 + dimension * 17 + 1);
                    query[index] = to_bf16(0.18F * std::sin(x * 0.013F) + tile * 0.006F);
                    key[index] = to_bf16(0.17F * std::cos(x * 0.019F) - tile * 0.004F);
                    value[index] = to_bf16(0.35F * std::sin(x * 0.007F + 0.2F));
                    gate[index] = to_bf16(0.12F * std::cos(x * 0.011F) + 0.015F * row);
                }
            }
        }
    }
};

std::vector<float> pool_reference(const std::vector<__nv_bfloat16>& input,
                                  const std::vector<int32_t>& valid_sizes) {
    std::vector<float> result(static_cast<std::size_t>(kHeads) * kTotalTiles * kDim, 0.0F);
    for (int32_t head = 0; head < kHeads; ++head) {
        for (int32_t tile = 0; tile < kTotalTiles; ++tile) {
            for (int32_t dimension = 0; dimension < kDim; ++dimension) {
                float sum = 0.0F;
                for (int32_t row = 0; row < valid_sizes[static_cast<std::size_t>(tile)]; ++row)
                    sum += to_float(input[tiled_offset(head, tile, row, dimension)]);
                result[(static_cast<std::size_t>(head) * kTotalTiles + tile) * kDim + dimension] =
                    sum / static_cast<float>(valid_sizes[static_cast<std::size_t>(tile)]);
            }
        }
    }
    return result;
}

std::vector<float> qk_reference(const std::vector<float>& pooled_q,
                                const std::vector<float>& pooled_k) {
    std::vector<float> result(static_cast<std::size_t>(kHeads) * kTotalTiles * kTotalTiles, 0.0F);
    for (int32_t head = 0; head < kHeads; ++head) {
        for (int32_t query = 0; query < kTotalTiles; ++query) {
            for (int32_t key = 0; key < kTotalTiles; ++key) {
                float score = 0.0F;
                for (int32_t dimension = 0; dimension < kDim; ++dimension) {
                    score = std::fma(
                        pooled_q[(static_cast<std::size_t>(head) * kTotalTiles + query) * kDim +
                                 dimension],
                        pooled_k[(static_cast<std::size_t>(head) * kTotalTiles + key) * kDim +
                                 dimension],
                        score);
                }
                result[(static_cast<std::size_t>(head) * kTotalTiles + query) * kTotalTiles + key] =
                    score * kScale;
            }
        }
    }
    return result;
}

std::vector<__nv_bfloat16> sparse_reference(const AttentionFixture& fixture,
                                            const std::vector<int32_t>& selected) {
    std::vector<__nv_bfloat16> result(fixture.query.size(), to_bf16(0.0F));
    for (int32_t head = 0; head < kHeads; ++head) {
        for (int32_t query_tile = 0; query_tile < kTotalTiles; ++query_tile) {
            const int32_t* selection =
                selected.data() +
                (static_cast<std::size_t>(head) * kTotalTiles + query_tile) * kTopVideoTiles;
            const auto key_tiles = trtmc::minimax_h3::vsa::attended_key_tiles_reference(
                selection, query_tile, kPrefixTiles, kVideoTiles, kTopVideoTiles);
            for (int32_t query_row = 0;
                 query_row < fixture.valid_sizes[static_cast<std::size_t>(query_tile)];
                 ++query_row) {
                float maximum = -std::numeric_limits<float>::max();
                for (const int32_t key_tile : key_tiles) {
                    for (int32_t key_row = 0;
                         key_row < fixture.valid_sizes[static_cast<std::size_t>(key_tile)];
                         ++key_row) {
                        float dot = 0.0F;
                        for (int32_t dimension = 0; dimension < kDim; ++dimension)
                            dot = std::fma(
                                to_float(fixture.query[tiled_offset(head, query_tile, query_row,
                                                                    dimension)]),
                                to_float(
                                    fixture.key[tiled_offset(head, key_tile, key_row, dimension)]),
                                dot);
                        maximum = std::max(maximum, dot * kScale);
                    }
                }
                std::vector<float> numerator(kDim, 0.0F);
                float denominator = 0.0F;
                for (const int32_t key_tile : key_tiles) {
                    for (int32_t key_row = 0;
                         key_row < fixture.valid_sizes[static_cast<std::size_t>(key_tile)];
                         ++key_row) {
                        float dot = 0.0F;
                        for (int32_t dimension = 0; dimension < kDim; ++dimension)
                            dot = std::fma(
                                to_float(fixture.query[tiled_offset(head, query_tile, query_row,
                                                                    dimension)]),
                                to_float(
                                    fixture.key[tiled_offset(head, key_tile, key_row, dimension)]),
                                dot);
                        const float probability = std::exp2((dot * kScale - maximum) * kLog2E);
                        denominator += probability;
                        const float bf16_probability = to_float(to_bf16(probability));
                        for (int32_t dimension = 0; dimension < kDim; ++dimension) {
                            numerator[static_cast<std::size_t>(dimension)] =
                                std::fma(bf16_probability,
                                         to_float(fixture.value[tiled_offset(head, key_tile,
                                                                             key_row, dimension)]),
                                         numerator[static_cast<std::size_t>(dimension)]);
                        }
                    }
                }
                for (int32_t dimension = 0; dimension < kDim; ++dimension) {
                    result[tiled_offset(head, query_tile, query_row, dimension)] =
                        to_bf16(numerator[static_cast<std::size_t>(dimension)] / denominator);
                }
            }
        }
    }
    return result;
}

std::vector<float> gate_attention_reference(const std::vector<float>& scores,
                                            const std::vector<float>& pooled_v) {
    std::vector<float> result(static_cast<std::size_t>(kHeads) * kTotalTiles * kDim, 0.0F);
    for (int32_t head = 0; head < kHeads; ++head) {
        for (int32_t query = 0; query < kTotalTiles; ++query) {
            const float* score_row =
                scores.data() +
                (static_cast<std::size_t>(head) * kTotalTiles + query) * kTotalTiles;
            const float maximum = *std::max_element(score_row, score_row + kTotalTiles);
            float denominator = 0.0F;
            for (int32_t key = 0; key < kTotalTiles; ++key)
                denominator += std::exp2((score_row[key] - maximum) * kLog2E);
            for (int32_t dimension = 0; dimension < kDim; ++dimension) {
                float value = 0.0F;
                for (int32_t key = 0; key < kTotalTiles; ++key) {
                    const float probability =
                        std::exp2((score_row[key] - maximum) * kLog2E) / denominator;
                    value = std::fma(
                        probability,
                        pooled_v[(static_cast<std::size_t>(head) * kTotalTiles + key) * kDim +
                                 dimension],
                        value);
                }
                result[(static_cast<std::size_t>(head) * kTotalTiles + query) * kDim + dimension] =
                    value;
            }
        }
    }
    return result;
}

void test_complete_sparse_and_gate_path(cudaStream_t stream) {
    const AttentionFixture fixture;
    const std::size_t tiled_count = fixture.query.size();
    const std::size_t pooled_count = static_cast<std::size_t>(kHeads) * kTotalTiles * kDim;
    const std::size_t score_count = static_cast<std::size_t>(kHeads) * kTotalTiles * kTotalTiles;
    const std::size_t selected_count =
        static_cast<std::size_t>(kHeads) * kTotalTiles * kTopVideoTiles;

    DeviceBuffer<__nv_bfloat16> device_query(tiled_count);
    DeviceBuffer<__nv_bfloat16> device_key(tiled_count);
    DeviceBuffer<__nv_bfloat16> device_value(tiled_count);
    DeviceBuffer<__nv_bfloat16> device_gate(tiled_count);
    DeviceBuffer<__nv_bfloat16> device_sparse(tiled_count);
    DeviceBuffer<__nv_bfloat16> device_output(tiled_count);
    DeviceBuffer<int32_t> device_valid(fixture.valid_sizes.size());
    DeviceBuffer<int32_t> device_selected(selected_count);
    DeviceBuffer<float> device_pooled_q(pooled_count);
    DeviceBuffer<float> device_pooled_k(pooled_count);
    DeviceBuffer<float> device_pooled_v(pooled_count);
    DeviceBuffer<float> device_scores(score_count);
    DeviceBuffer<float> device_compressed(pooled_count);
    upload(device_query, fixture.query, stream);
    upload(device_key, fixture.key, stream);
    upload(device_value, fixture.value, stream);
    upload(device_gate, fixture.gate, stream);
    upload(device_valid, fixture.valid_sizes, stream);

    trtmc::minimax_h3::vsa::mean_pool_tiles_async(
        device_query.get(), device_valid.get(), device_pooled_q.get(), kHeads, kTotalTiles, stream);
    trtmc::minimax_h3::vsa::mean_pool_tiles_async(
        device_key.get(), device_valid.get(), device_pooled_k.get(), kHeads, kTotalTiles, stream);
    trtmc::minimax_h3::vsa::mean_pool_tiles_async(
        device_value.get(), device_valid.get(), device_pooled_v.get(), kHeads, kTotalTiles, stream);
    trtmc::minimax_h3::vsa::pooled_qk_scores_async(device_pooled_q.get(), device_pooled_k.get(),
                                                   device_scores.get(), kHeads, kTotalTiles,
                                                   stream);
    trtmc::minimax_h3::vsa::select_video_topk_async(device_scores.get(), device_selected.get(),
                                                    kHeads, kTotalTiles, kPrefixTiles, kVideoTiles,
                                                    kTopVideoTiles, stream);
    trtmc::minimax_h3::vsa::block_sparse_attention_64_async(
        device_query.get(), device_key.get(), device_value.get(), device_valid.get(),
        device_selected.get(), device_sparse.get(), kHeads, kTotalTiles, kPrefixTiles, kVideoTiles,
        kTopVideoTiles, stream);
    trtmc::minimax_h3::vsa::pooled_gate_attention_async(device_scores.get(), device_pooled_v.get(),
                                                        device_compressed.get(), kHeads,
                                                        kTotalTiles, stream);
    trtmc::minimax_h3::vsa::merge_gate_async(device_sparse.get(), device_gate.get(),
                                             device_compressed.get(), device_output.get(), kHeads,
                                             kTotalTiles, stream);

    const auto pooled_q = download(device_pooled_q, stream);
    const auto pooled_k = download(device_pooled_k, stream);
    const auto pooled_v = download(device_pooled_v, stream);
    const auto scores = download(device_scores, stream);
    const auto selected = download(device_selected, stream);
    const auto sparse = download(device_sparse, stream);
    const auto compressed = download(device_compressed, stream);
    const auto output = download(device_output, stream);

    const auto expected_pooled_q = pool_reference(fixture.query, fixture.valid_sizes);
    const auto expected_pooled_k = pool_reference(fixture.key, fixture.valid_sizes);
    const auto expected_pooled_v = pool_reference(fixture.value, fixture.valid_sizes);
    for (std::size_t index = 0; index < pooled_count; ++index) {
        require(std::abs(pooled_q[index] - expected_pooled_q[index]) < 2.0e-6F,
                "pooled Q differs from reference");
        require(std::abs(pooled_k[index] - expected_pooled_k[index]) < 2.0e-6F,
                "pooled K differs from reference");
        require(std::abs(pooled_v[index] - expected_pooled_v[index]) < 2.0e-6F,
                "pooled V differs from reference");
    }
    const auto expected_scores = qk_reference(expected_pooled_q, expected_pooled_k);
    for (std::size_t index = 0; index < score_count; ++index)
        require(std::abs(scores[index] - expected_scores[index]) < 2.0e-5F,
                "pooled QK score differs from reference");

    const auto expected_selected = trtmc::minimax_h3::vsa::select_video_topk_reference(
        scores.data(), kHeads, kTotalTiles, kPrefixTiles, kVideoTiles, kTopVideoTiles);
    require(selected == expected_selected, "CUDA selector differs from CPU top-k");

    const auto expected_sparse = sparse_reference(fixture, selected);
    for (std::size_t index = 0; index < tiled_count; ++index) {
        const float difference =
            std::abs(to_float(sparse[index]) - to_float(expected_sparse[index]));
        require(difference <= 0.012F,
                "BF16 tensor-core sparse attention differs from CPU reference");
    }

    const auto expected_compressed = gate_attention_reference(scores, expected_pooled_v);
    for (std::size_t index = 0; index < pooled_count; ++index)
        require(std::abs(compressed[index] - expected_compressed[index]) < 3.0e-5F,
                "pooled gate attention differs from reference");

    for (int32_t tile = 0; tile < kTotalTiles; ++tile) {
        for (int32_t row = 0; row < kTile; ++row) {
            for (int32_t dimension = 0; dimension < kDim; ++dimension) {
                const auto index = tiled_offset(0, tile, row, dimension);
                float expected = 0.0F;
                if (row < fixture.valid_sizes[static_cast<std::size_t>(tile)]) {
                    expected =
                        to_float(expected_sparse[index]) +
                        to_float(fixture.gate[index]) *
                            expected_compressed[static_cast<std::size_t>(tile) * kDim + dimension];
                    expected = to_float(to_bf16(expected));
                }
                require(std::abs(to_float(output[index]) - expected) <= 0.014F,
                        "gate merge differs from sparse + gate * compressed");
                if (row >= fixture.valid_sizes[static_cast<std::size_t>(tile)])
                    require(to_float(output[index]) == 0.0F,
                            "padded output row must stay exact zero");
            }
        }
    }
}

void test_worst_profile_selector_capacity(cudaStream_t stream) {
    constexpr int32_t prefix_tiles = 27;
    constexpr int32_t video_tiles = 2080;
    constexpr int32_t total_tiles = prefix_tiles + video_tiles;
    constexpr int32_t top_video_tiles = 208;
    const std::size_t score_count = static_cast<std::size_t>(total_tiles) * total_tiles;
    const std::size_t selected_count = static_cast<std::size_t>(total_tiles) * top_video_tiles;
    DeviceBuffer<float> scores(score_count);
    DeviceBuffer<int32_t> selected(selected_count);
    check_cuda(cudaMemsetAsync(scores.get(), 0, score_count * sizeof(float), stream),
               "cudaMemsetAsync worst-profile scores");
    trtmc::minimax_h3::vsa::select_video_topk_async(scores.get(), selected.get(), 1, total_tiles,
                                                    prefix_tiles, video_tiles, top_video_tiles,
                                                    stream);
    const auto chosen = download(selected, stream);
    for (int32_t query = 0; query < total_tiles; ++query) {
        const auto begin = static_cast<std::size_t>(query) * top_video_tiles;
        for (int32_t rank = 0; rank < top_video_tiles; ++rank) {
            require(chosen[begin + static_cast<std::size_t>(rank)] == prefix_tiles + rank,
                    "worst-profile selector capacity or stable tie order mismatch");
        }
    }
}

} // namespace

int main() {
    cudaStream_t stream = nullptr;
    try {
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
        test_tile_pool_and_untile(stream);
        test_complete_sparse_and_gate_path(stream);
        test_worst_profile_selector_capacity(stream);
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    } catch (const std::exception& error) {
        if (stream != nullptr)
            cudaStreamDestroy(stream);
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
