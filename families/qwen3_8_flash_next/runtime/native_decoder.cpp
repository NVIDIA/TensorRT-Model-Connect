/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "native_decoder.h"

#include "decoder_layer.h"

#include <algorithm>
#include <cstring>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

namespace trtmc::qwen3_8_flash_next {
namespace {
using Cache = std::unordered_map<std::string, SavedTensor>;
using Module = std::unique_ptr<ITrtModule>;

nlohmann::json metadata(const NativeDecoder::ReadSection& read, const char* key) {
    const auto bytes = read(key);
    return nlohmann::json::parse(bytes.begin(), bytes.end());
}
NgramLookup lookup(const NativeDecoder::ReadSection& read, int vocab, int eos) {
    const auto v = metadata(read, "ngram_layout.json");
    const auto signed_multipliers = v.at("layer_multipliers").get<std::array<std::int64_t, 3>>();
    NgramLayout l{};
    std::copy(signed_multipliers.begin(), signed_multipliers.end(), l.multipliers.begin());
    l.sizes = v.at("ngram_heads_vocab_sizes").get<decltype(l.sizes)>();
    l.offsets = v.at("ngram_heads_offsets").get<decltype(l.offsets)>();
    l.eos = eos;
    l.vocabulary_size = vocab;
    return NgramLookup(l, v.at("table_rows").get<std::int64_t>());
}
SavedTensor zeros(DType type, std::vector<std::int64_t> shape) {
    SavedTensor t{{}, std::move(shape), type};
    t.bytes.resize(t.view().nbytes());
    return t;
}
void require_shape(const Tensor& t, DType type, std::vector<std::int64_t> shape) {
    if (!t.data || t.dtype != type || t.shape != shape)
        throw std::runtime_error("Unexpected native Flash-Next stage output");
}
} // namespace
struct NativeDecoder::Impl {
    ReadSection read;
    ReadRows read_rows;
    int capacity, chunk, vocab, eos, used{0};
    float theta;
    std::vector<std::string> kinds;
    NgramLookup ngram;
    std::array<std::int32_t, 2> history;
    std::vector<float> rotary;
    Module embed, head, expert, combine;
    std::vector<Module> layers;
    std::vector<Cache> caches;

    Impl(IBackend& backend, const ReadSection& sections, ReadRows rows,
         const nlohmann::json& layout)
        : read(sections), read_rows(std::move(rows)), capacity(layout.at("capacity")),
          chunk(layout.at("max_chunk")), vocab(layout.at("vocab_size")),
          eos(layout.at("ngram_eos")), theta(layout.at("rope_theta")),
          kinds(layout.at("layers").get<std::vector<std::string>>()),
          ngram(lookup(read, vocab, eos)) {
        if (layout.at("format") != 1 || capacity <= 0 || capacity > 262144 || capacity % 4 ||
            chunk <= 0 || chunk > capacity || vocab != 248320 || kinds.size() != 48 ||
            layout.at("ple_layer_ids") != std::vector<int>{2} || theta != 10000000.F || !read_rows)
            throw std::invalid_argument("Invalid prepared Flash-Next decoder layout");
        auto load = [&](const std::string& name) {
            const auto plan = read(name + ".plan");
            ModuleCreateOptions options{};
            if (embed)
                options.stream = embed->stream();
            auto m = backend.create_module(plan.data(), plan.size(), options);
            if (!m || !m->ok())
                throw std::runtime_error("Cannot load Flash-Next " + name + " plan");
            return m;
        };
        embed = load("embed");
        head = load("head");
        expert = load("expert");
        combine = load("combine");
        rotary.resize(static_cast<std::size_t>(capacity) * 64);
        for (std::size_t i = 0; i < kinds.size(); ++i) {
            Cache c;
            if (kinds[i] == "linear_attention") {
                c.emplace("history", zeros(DType::kFloat16, {3, 10240}));
                c.emplace("state", zeros(DType::kFloat32, {48, 128, 128}));
            } else if (kinds[i] == "full_attention") {
                for (const char* name : {"key", "value"})
                    c.emplace(name, zeros(DType::kFloat16, {capacity, 2, 256}));
                c.emplace("index_key", zeros(DType::kFloat16, {capacity, 128}));
            } else
                throw std::invalid_argument("Unrecognized Flash-Next attention type");
            if (i == 1)
                c.emplace("lexical_history", zeros(DType::kFloat16, {9, 10240}));
            caches.push_back(std::move(c));
            layers.push_back(load("layer." + std::to_string(i)));
        }
        history = ngram.initial_history();
    }
};
NativeDecoder::NativeDecoder(IBackend& backend, const ReadSection& read, ReadRows rows)
    : impl_(std::make_unique<Impl>(backend, read, std::move(rows),
                                   metadata(read, "decoder_layout.json"))) {}
NativeDecoder::~NativeDecoder() = default;
int NativeDecoder::capacity() const {
    return impl_->capacity;
}
int NativeDecoder::max_chunk() const {
    return impl_->chunk;
}
void NativeDecoder::reset() {
    auto& d = *impl_;
    d.used = 0;
    d.history = d.ngram.initial_history();
    std::fill(d.rotary.begin(), d.rotary.end(), 0.F);
    for (auto& layer : d.caches)
        for (auto& field : layer)
            std::fill(field.second.bytes.begin(), field.second.bytes.end(), 0);
}
std::vector<std::uint16_t> NativeDecoder::forward(const std::vector<std::int32_t>& ids,
                                                  const std::vector<Position>& positions,
                                                  const std::vector<std::uint8_t>& media_mask,
                                                  const std::vector<std::uint16_t>& media,
                                                  bool want_logits) {
    auto& d = *impl_;
    if (ids.empty() ||
        ids.size() > static_cast<std::size_t>(std::min(d.chunk, d.capacity - d.used)) ||
        ids.size() != positions.size() || (!media_mask.empty() && ids.size() != media_mask.size()))
        throw std::invalid_argument(
            "Flash-Next chunk exceeds its profile or has inconsistent inputs");
    const auto n = static_cast<std::int64_t>(ids.size());
    std::vector<std::int32_t> mask(n), slot(n);
    std::vector<std::uint16_t> vectors(n * 2560);
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!media_mask.empty() && media_mask[i]) {
            if (media_mask[i] != 1 || media.size() - cursor < 2560)
                throw std::invalid_argument("Invalid Flash-Next media row");
            mask[i] = 1;
            std::copy_n(media.begin() + cursor, 2560, vectors.begin() + i * 2560);
            cursor += 2560;
        }
    }
    if (cursor != media.size())
        throw std::invalid_argument("Unconsumed Flash-Next media rows");
    const auto lexical = d.ngram.lookup(ids, d.history);
    auto words = d.read_rows(lexical.rows);
    if (words.size() != ids.size() * 2560)
        throw std::runtime_error("Invalid Flash-Next lexical row data");
    std::iota(slot.begin(), slot.end(), d.used);
    // Write only uncommitted rotary slots. A retry overwrites them, and no
    // preceding causal query can attend to a block containing a future slot.
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto row = rotary_row(positions[i], d.theta);
        std::copy(row.begin(), row.end(),
                  d.rotary.begin() + static_cast<std::size_t>(slot[i]) * 64);
    }
    const auto embedded =
        d.embed->forward({{"ids", {const_cast<std::int32_t*>(ids.data()), {n}, DType::kInt32}},
                          {"media_mask", {mask.data(), {n, 1}, DType::kInt32}},
                          {"media_vectors", {vectors.data(), {n, 2560}, DType::kFloat16}}});
    auto hyper = embedded.at("hyper");
    require_shape(hyper, DType::kFloat16, {n, 10240});
    SavedTensor working;
    std::vector<Cache> next;
    next.reserve(d.layers.size());
    for (std::size_t i = 0; i < d.layers.size(); ++i) {
        TensorMap inputs{{"hyper", hyper}};
        if (i == 1)
            inputs.emplace("lexical_words", Tensor{words.data(), {n, 640}, DType::kInt32});
        if (d.kinds[i] == "full_attention") {
            inputs.emplace("position", Tensor{slot.data(), {n}, DType::kInt32});
            inputs.emplace("rotary", Tensor{d.rotary.data(), {d.capacity, 64}, DType::kFloat32});
        }
        for (auto& field : d.caches[i])
            inputs.emplace(field.first, field.second.view());
        auto y =
            run_decoder_layer(*d.layers[i], *d.expert, *d.combine, inputs, [&](std::int32_t id) {
                return d.read("expert." + std::to_string(i) + "." + std::to_string(id) + ".bin");
            });
        next.push_back(std::move(y.state));
        working = std::move(y.hyper);
        hyper = working.view();
    }
    std::vector<std::uint16_t> result;
    if (want_logits) {
        auto* final_row = working.bytes.data() + (n - 1) * 10240 * 2;
        const auto output = d.head->forward({{"hyper", {final_row, {1, 10240}, DType::kFloat16}}});
        const auto& logits = output.at("logits");
        require_shape(logits, DType::kFloat16, {1, d.vocab});
        result.resize(d.vocab);
        std::memcpy(result.data(), logits.data, logits.nbytes());
    }
    d.caches.swap(next);
    d.history = lexical.history;
    d.used += static_cast<int>(n);
    return result;
}
} // namespace trtmc::qwen3_8_flash_next
