/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-only contract of the MiniCPM-V 4.7 runtime against values recorded from the provider
// (transformers 5.22.2, tokenizers 0.23.1): image slicing, vision position ids, canvas M-RoPE,
// the chat format, request preparation, the generation budget and sampling, BPE merge order, the
// tokenizer's NFC and its token ids.

#include "families/minicpmv4_7/runtime/bpe_merge.h"
#include "families/minicpmv4_7/runtime/image_preprocess.h"
#include "families/minicpmv4_7/runtime/prompt.h"
#include "families/minicpmv4_7/runtime/request.h"
#include "families/minicpmv4_7/runtime/sampler.h"
#include "families/minicpmv4_7/runtime/tokenizer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace trtmc::minicpmv4_7;
namespace internal = trtmc::internal;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << "\n";
        ++failures;
    }
}

// The message of the exception `body` throws; empty when it returns.
template <typename Body>
std::string error_of(Body body) {
    try {
        body();
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

int64_t digest(const std::vector<int32_t>& values) {
    int64_t sum = 0;
    for (std::size_t i = 0; i < values.size(); ++i)
        sum = (sum + static_cast<int64_t>(i + 1) * (values[i] + 7)) % 1000000007;
    return sum;
}

void slicing() {
    struct Case {
        int32_t height, width, rows, cols, thumb_h, thumb_w, slice_h, slice_w;
    };
    // MiniCPMV4_6ImageProcessor target_sizes (patches) for black images of these sizes.
    const std::vector<Case> cases{
        {382, 640, 1, 2, 24, 40, 36, 28},   {200, 300, 0, 0, 28, 40, 0, 0},
        {1600, 1600, 3, 3, 32, 32, 32, 32}, {10, 2000, 0, 0, 4, 444, 0, 0},
        {2000, 10, 0, 0, 452, 4, 0, 0},     {448, 448, 0, 0, 32, 32, 0, 0},
        {449, 449, 2, 1, 32, 32, 24, 44},   {1000, 3000, 2, 4, 20, 56, 28, 40},
        {3000, 1000, 4, 2, 56, 20, 40, 28}, {700, 701, 2, 2, 32, 32, 32, 32},
        {2500, 1300, 4, 2, 44, 24, 32, 32}, {90, 3500, 1, 3, 4, 196, 8, 116},
    };
    for (const auto& c : cases) {
        const std::string name = std::to_string(c.height) + "x" + std::to_string(c.width);
        const auto grid = sliced_grid(c.height, c.width, 9, 448);
        expect(grid[0] == c.rows && grid[1] == c.cols, "slice grid " + name);
        const auto thumb = best_resize(c.height, c.width, 448, 14, 4, grid[0] == 0);
        expect(thumb[0] / 14 == c.thumb_h && thumb[1] / 14 == c.thumb_w, "thumbnail " + name);
        if (grid[0] > 0) {
            const auto refine = refine_size(c.height, c.width, grid, 448, 14, 4);
            expect(refine[0] / grid[0] / 14 == c.slice_h && refine[1] / grid[1] / 14 == c.slice_w,
                   "slice size " + name);
        }
    }
}

void positions() {
    struct Case {
        int32_t h, w;
        int64_t digest;
    };
    // get_vision_nearest_position_ids(target_sizes, 70)
    const std::vector<Case> cases{{4, 4, 349796},      {24, 40, 464372553}, {36, 28, 631848645},
                                  {28, 28, 984666172}, {32, 32, 676626553}, {140, 4, 512571220},
                                  {68, 52, 207599368}};
    for (const auto& c : cases)
        expect(digest(nearest_position_ids(c.h, c.w, 70)) == c.digest,
               "position ids " + std::to_string(c.h) + "x" + std::to_string(c.w));
    expect(block_order(4, 4) ==
               std::vector<int32_t>{0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15},
           "2x2 block order");
    expect(canvas_coordinates(9, 6, 0) == std::vector<int32_t>{0, 2, 3, 5, 6, 8}, "linspace 9/6");
    expect(canvas_coordinates(6, 3, 10) == std::vector<int32_t>{10, 12, 15},
           "linspace tie to even");
    expect(canvas_coordinates(7, 1, 0) == std::vector<int32_t>{0}, "linspace single");
}

std::vector<int32_t> image_tokens(const TokenIds& t, std::array<int32_t, 2> thumb,
                                  const std::vector<std::array<int32_t, 2>>& slices, int32_t cols) {
    std::vector<int32_t> ids{t.image_start};
    ids.insert(ids.end(), static_cast<std::size_t>(thumb[0] * thumb[1] / 16), t.image_pad);
    ids.push_back(t.image_end);
    for (std::size_t k = 0; k < slices.size(); ++k) {
        if (k && k % static_cast<std::size_t>(cols) == 0)
            ids.push_back(t.newline);
        ids.push_back(t.slice_start);
        ids.insert(ids.end(), static_cast<std::size_t>(slices[k][0] * slices[k][1] / 16),
                   t.image_pad);
        ids.push_back(t.slice_end);
    }
    return ids;
}

void canvas() {
    TokenIds t;
    t.image_pad = 248056;
    t.image_start = 248078;
    t.image_end = 248079;
    t.slice_start = 248088;
    t.slice_end = 248089;
    t.image_id_start = 248090;
    t.image_id_end = 248091;
    t.newline = 198;
    auto join = [](std::vector<std::vector<int32_t>> pieces) {
        std::vector<int32_t> out;
        for (const auto& piece : pieces)
            out.insert(out.end(), piece.begin(), piece.end());
        return out;
    };
    auto repeat = [&](int32_t count) {
        return std::vector<int32_t>(static_cast<std::size_t>(count), 1000);
    };
    struct Case {
        std::vector<int32_t> ids;
        std::vector<std::array<int32_t, 2>> grids;
        int64_t digest;
        int32_t decode;
    };
    const std::vector<std::array<int32_t, 2>> four(4, {28, 28});
    std::vector<std::array<int32_t, 2>> pair{{28, 44}};
    pair.insert(pair.end(), four.begin(), four.end());
    auto both = pair;
    both.insert(both.end(), pair.begin(), pair.end());
    // MiniCPMV4_7Model.get_rope_index on the same sequences.
    const std::vector<Case> cases{
        {join({repeat(5), image_tokens(t, {24, 40}, {{36, 28}, {36, 28}}, 2), repeat(3)}),
         {{24, 40}, {36, 28}, {36, 28}},
         2941071,
         23},
        {join({repeat(2), image_tokens(t, {32, 32}, {}, 1), repeat(4)}), {{32, 32}}, 297264, 15},
        {join({repeat(1),
               image_tokens(t, {28, 44}, four, 2),
               {198, 1000},
               image_tokens(t, {28, 44}, four, 2)}),
         both, 36113584, 32},
    };
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto result = canvas_positions(cases[i].ids, t, cases[i].grids, 4);
        std::vector<int32_t> flat;
        for (const auto& position : result.positions)
            flat.insert(flat.end(), position.begin(), position.end());
        expect(digest(flat) == cases[i].digest && result.decode_position == cases[i].decode,
               "canvas M-RoPE case " + std::to_string(i));
    }
    // Rows and columns of the square thumbnail sum alike, so its digest would not see H and W
    // swapped: tokens (row 0, column 1) and (row 1, column 0) after the text and <image>.
    const auto square = canvas_positions(cases[1].ids, t, cases[1].grids, 4);
    expect(square.positions[4] == std::array<int32_t, 3>{2, 2, 3} &&
               square.positions[11] == std::array<int32_t, 3>{2, 3, 2},
           "canvas H and W of the square thumbnail");
}

// Recipe extracted from the checkpoint template at build time (enable_thinking=false).
const nlohmann::json& chat_recipe() {
    static const auto recipe = nlohmann::json::parse(R"({
      "default_prefix": "", "user": {"prefix": "<|im_start|>user\n", "suffix": "<|im_end|>\n"},
      "system": {"prefix": "<|im_start|>system\n", "suffix": "<|im_end|>\n"},
      "empty_system_prefix": "<|im_start|>system\n<|im_end|>\n",
      "generation": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
      "assistant_recent": {"prefix": "<|im_start|>assistant\n<think>\n", "include_reasoning": true,
        "reasoning_prefix": "", "reasoning_suffix": "\n</think>\n\n", "empty_reasoning": "\n</think>\n\n",
        "suffix": "<|im_end|>\n"},
      "assistant_history": {"prefix": "<|im_start|>assistant\n<think>\n\n</think>\n\n", "include_reasoning": false,
        "reasoning_prefix": "", "reasoning_suffix": "", "empty_reasoning": "", "suffix": "<|im_end|>\n"}})");
    return recipe;
}

void conversation() {
    const auto& recipe = chat_recipe();
    const std::string image = kImagePlaceholder;
    expect(format_conversation(recipe, {{Role::User, join_parts({image, "What is shown?"})}}) ==
               "<|im_start|>user\n<|image_pad|>\nWhat is shown?<|im_end|>\n<|im_start|>assistant\n"
               "<think>\n\n</think>\n\n",
           "single image turn");
    expect(format_conversation(recipe, {{Role::System, " Be brief. "},
                                        {Role::User, join_parts({image, "Describe."})},
                                        {Role::Assistant, "<think>\nlooking\n</think>\n\nA car."},
                                        {Role::User, join_parts({"Color?", image})}}) ==
               "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n<|image_pad|>\nDescribe."
               "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nA car.<|im_end|>\n"
               "<|im_start|>user\nColor?\n<|image_pad|><|im_end|>\n<|im_start|>assistant\n"
               "<think>\n\n</think>\n\n",
           "multi-turn conversation");

    ImageCrops crops;
    crops.rows = 1;
    crops.cols = 2;
    crops.crops.resize(3);
    crops.crops[0].grid_h = 8;
    crops.crops[0].grid_w = 4;
    crops.crops[1].grid_h = crops.crops[2].grid_h = 4;
    crops.crops[1].grid_w = crops.crops[2].grid_w = 4;
    expect(expand_images("A" + image + "B", {crops}, 4) ==
               "A<image_id>0</image_id><image>" + image + image + "</image><slice>" + image +
                   "</slice><slice>" + image + "</slice>B",
           "image expansion");
}

void generation() {
    expect(generation_budget(std::nullopt, 4096, 219) == 128, "default budget, large capacity");
    expect(generation_budget(std::nullopt, 64, 30) == 35, "default budget limited by the capacity");
    expect(generation_budget(8, 64, 30) == 8, "requested budget");
    const auto rejected = [](std::optional<int64_t> requested, int64_t capacity, int64_t prompt) {
        try {
            (void)generation_budget(requested, capacity, prompt);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    expect(rejected(36, 64, 30), "requested budget beyond the capacity");
    expect(rejected(std::nullopt, 64, 65), "prompt filling the capacity");
    expect(rejected(std::nullopt, 64, 0), "empty prompt");

    // Top-2 of three equal maxima keeps the lower ids 1 and 2, never id 4 or a smaller logit.
    Sampler top_k(SamplingConfig{1.0, 2, 1.0, 7});
    const std::vector<float> logits{1.0F, 3.0F, 3.0F, 2.0F, 3.0F, 0.0F};
    std::array<int, 6> counts{};
    for (int draw = 0; draw < 300; ++draw)
        ++counts[static_cast<std::size_t>(top_k.sample(logits))];
    expect(counts[1] > 0 && counts[2] > 0 && counts[1] + counts[2] == 300, "top-k ties by id");
    // Nucleus over the full order: the dominant candidate alone covers top_p = 0.5.
    Sampler nucleus(SamplingConfig{1.0, 0, 0.5, 7});
    bool dominant = true;
    for (int draw = 0; draw < 50; ++draw)
        dominant = dominant && nucleus.sample({0.0F, 5.0F, 1.0F}) == 1;
    expect(dominant, "nucleus keeps the dominant candidate");
}

// Fixed prompt ids; counts how often a prompt was tokenized.
class CountingTokenizer final : public ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override {
        ++calls;
        return std::vector<int32_t>(4, 1000);
    }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return -1; }
    mutable int calls = 0;
};

void request_preparation() {
    const nlohmann::json prompts{{"false", chat_recipe()}, {"true", chat_recipe()}};
    const ProcessorConfig processor{};
    CountingTokenizer tokenizer;
    const RequestContext context{prompts, processor, tokenizer, 64};
    const std::vector<uint8_t> pixels(28 * 28 * 3, 128);
    internal::ImageView image;
    image.data = pixels.data();
    image.byte_size = pixels.size();
    image.height = image.width = 28;
    image.format = internal::ImageFormat::UInt8;
    const auto user_turn = [](const std::vector<internal::ImagesTextPart>& parts) {
        return std::vector<internal::ImagesTextMessage>{
            {internal::MessageRole::User, {parts.data(), parts.size()}}};
    };
    const std::vector<internal::ImagesTextPart> parts{image, internal::TextPartView{"Describe."}};
    const auto messages = user_turn(parts);
    const internal::ImagesTextToTextRequest request{{messages.data(), messages.size()}};
    const auto prepared = prepare_request(request, {}, context);
    expect(prepared.ids.size() == 4 && prepared.images.size() == 1 &&
               prepared.max_new_tokens == 61 && tokenizer.calls == 1,
           "a valid request is prepared");

    // top_p = 0 passes the shared type check; the sampler rejects it before any image work, so a
    // request that also carries an invalid image still fails on its options.
    internal::ImageView four_channels = image;
    four_channels.channels = 4;
    const std::vector<internal::ImagesTextPart> bad_parts{four_channels,
                                                          internal::TextPartView{"Describe."}};
    const auto bad_messages = user_turn(bad_parts);
    const internal::ImagesTextToTextRequest bad_image{{bad_messages.data(), bad_messages.size()}};
    const internal::ConfigEntry zero_top_p[] = {{"top_p", 0.0}};
    tokenizer.calls = 0;
    expect(error_of([&] {
               (void)prepare_request(bad_image, zero_top_p, context);
           }).find("sampling") != std::string::npos &&
               tokenizer.calls == 0,
           "sampling options fail before images and the prompt");
    expect(error_of([&] { (void)prepare_request(bad_image, {}, context); }).find("RGB") !=
               std::string::npos,
           "an invalid image fails with valid options");

    // The sides and the byte size are checked before the copy is allocated: a (2^31 - 1)^2 RGB
    // copy would take about 13.8 EB.
    const auto rejected = [](const internal::ImageView& view) {
        try {
            (void)rgb_bytes(view);
        } catch (const std::invalid_argument&) {
            return true;
        } catch (...) {
        }
        return false;
    };
    internal::ImageView huge = image;
    huge.height = huge.width = 2147483647U;
    huge.byte_size = 3;
    expect(rejected(huge), "image byte size checked before the copy is allocated");
    huge.height = huge.width = 2147483648U;
    expect(rejected(huge), "image sides beyond int32 rejected");
    const std::vector<float> unit{0.0F, 1.0F, 0.5F, 0.2F, 0.4F, 0.6F};
    internal::ImageView floats;
    floats.data = unit.data();
    floats.byte_size = unit.size() * sizeof(float);
    floats.height = 2;
    floats.width = 1;
    floats.format = internal::ImageFormat::Float32;
    expect(rgb_bytes(floats) == std::vector<uint8_t>{0, 255, 128, 51, 102, 153},
           "Float32 pixels scale to bytes");
}

void sampling() {
    // Non-finite logits fail on both paths instead of reaching the orderings and the weights.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const auto rejected = [](SamplingConfig config, const std::vector<float>& logits) {
        Sampler sampler(config);
        try {
            (void)sampler.sample(logits);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    expect(rejected({1.0, 1, 1.0, 7}, {1.0F, nan, 0.0F}) &&
               rejected({1.0, 0, 1.0, 7}, {1.0F, nan, 0.0F}) &&
               rejected({1.0, 0, 1.0, 7}, {inf, 0.0F}) && rejected({0.0, 0, 1.0, 7}, {-inf, 0.0F}),
           "non-finite logits are rejected");
}

// BPE merges in the order of the tokenizers BPE model: lowest (rank, position) first over the
// current symbols, so a pair formed by a merge competes with the pairs queued before it.
void bpe_merges() {
    using Rules = std::map<std::pair<int32_t, int32_t>, BpeMergeRule>;
    std::size_t lookups = 0;
    const auto merge = [&](std::vector<int32_t> ids, const Rules& rules) {
        lookups = 0;
        return merge_bpe(std::move(ids), [&](int32_t left, int32_t right) -> const BpeMergeRule* {
            ++lookups;
            const auto found = rules.find({left, right});
            return found == rules.end() ? nullptr : &found->second;
        });
    };
    constexpr int32_t a = 0, b = 1, c = 2, ab = 3, aba = 4, aa = 5, aaaa = 6, bc = 7;
    const Rules pairs{{{a, a}, {0, aa}}, {{aa, aa}, {1, aaaa}}};
    expect(merge({a, a, a}, pairs) == std::vector<int32_t>{aa, a}, "repeated pair, left first");
    expect(merge({a, a, a, a, a}, pairs) == std::vector<int32_t>{aaaa, a}, "merged pairs merge");
    expect(merge({a, b, c}, Rules{{{b, c}, {0, bc}}, {{a, b}, {1, ab}}}) ==
               std::vector<int32_t>{a, bc},
           "lowest rank first; a queued pair that changed is skipped");
    expect(merge({a, b, a, b}, Rules{{{a, b}, {1, ab}}, {{ab, a}, {0, aba}}}) ==
               std::vector<int32_t>{aba, b},
           "a pair formed by a merge is ranked against the queued pairs");
    expect(merge({}, pairs).empty() && merge({c}, pairs) == std::vector<int32_t>{c},
           "empty and single-symbol words");

    // A word whose merges extend one token left to right: one merge per pass for a full rescan
    // (about n * n / 2 pair lookups), at most 3 lookups per symbol here.
    constexpr int32_t n = 4096;
    Rules chain;
    std::vector<int32_t> word(static_cast<std::size_t>(n));
    for (int32_t index = 0; index < n; ++index) {
        word[static_cast<std::size_t>(index)] = index;
        if (index > 0)
            chain[{index == 1 ? 0 : n + index - 1, index}] = BpeMergeRule{index, n + index};
    }
    expect(merge(word, chain) == std::vector<int32_t>{2 * n - 1}, "chained merges");
    expect(lookups <= 3 * static_cast<std::size_t>(n), "pair lookups grow linearly with the word");
}

// NFC as the tokenizers normalizer applies it: Unicode 9.0 data, so a mark assigned later
// (U+09FE, U+1DF6; Unicode 10.0) is not reordered before an older mark of a lower class.
void normalization() {
    expect(tokenizer_nfc("e\u0301") == "\u00e9", "composition");
    expect(tokenizer_nfc("\u1e0a\u0323") == "\u1e0c\u0307", "reordering, then composition");
    expect(tokenizer_nfc("\u09fe\u06e3") == "\u09fe\u06e3", "a Unicode 10.0 mark stays in place");
    expect(tokenizer_nfc("x\u1df6\u0316") == "x\u1df6\u0316", "no reordering across it");
}

// tokenizer.json ids are JSON integers checked before narrowing, so 2^32 + 5, -(2^32 - 5) or 9.5
// never stand for the ids 5 or 9.
void token_ids() {
    const auto id = [](const char* text, int32_t end) {
        return tokenizer_json_id(nlohmann::json::parse(text), end);
    };
    expect(id("0", 10) == 0 && id("5", 10) == 5 && id("9", 10) == 9, "token ids in range");
    expect(!id("9.5", 10) && !id("5.0", 10), "fractional token ids");
    expect(!id("4294967301", 10) && !id("-4294967291", 10), "token ids that narrow into range");
    expect(!id("10", 10) && !id("-1", 10) && !id("\"5\"", 10) && !id("null", 10),
           "out-of-range and non-numeric token ids");
}

} // namespace

int main() {
    slicing();
    positions();
    canvas();
    conversation();
    request_preparation();
    generation();
    sampling();
    bpe_merges();
    normalization();
    token_ids();
    if (failures == 0)
        std::cout << "minicpmv4_7 host contract: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
