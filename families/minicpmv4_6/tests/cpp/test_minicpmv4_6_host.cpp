/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-only checks of the MiniCPM-V 4.6 runtime: image slicing, resampling and
// grid indices, placeholder text, conversation formatting, pre-tokenization and
// BPE merges, metadata parsing, sampling, and the engine I/O contracts.
// Golden values were produced with the Transformers 5.22.2 processor, the
// PyTorch operations the reference model runs, and Hugging Face tokenizers 0.23.3.

#include "families/minicpmv4_6/runtime/engine_io.h"
#include "families/minicpmv4_6/runtime/image.h"
#include "families/minicpmv4_6/runtime/metadata.h"
#include "families/minicpmv4_6/runtime/prompt.h"
#include "families/minicpmv4_6/runtime/sampler.h"
#include "families/minicpmv4_6/runtime/tokenizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace trtmc::minicpmv4_6;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Callable>
bool rejects(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

template <typename Callable>
bool fails(Callable&& callable) {
    try {
        callable();
    } catch (const std::runtime_error&) {
        return true;
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// Deterministic RGB8 test pattern (the generator of the resize goldens).
std::vector<std::uint8_t> pattern(std::int32_t height, std::int32_t width) {
    std::vector<std::uint8_t> values;
    for (std::int32_t y = 0; y < height; ++y)
        for (std::int32_t x = 0; x < width; ++x)
            for (std::int32_t c = 0; c < 3; ++c)
                values.push_back(
                    static_cast<std::uint8_t>((y * 37 + x * 101 + c * 59 + (y * x) % 23) % 256));
    return values;
}

struct PlanCase {
    std::int32_t height, width, source_height, source_width, rows, cols, refine_height,
        refine_width;
};

void slice_plans() {
    // {image height, image width} -> source size, grid, refined size (pixels).
    const std::vector<PlanCase> cases{
        {382, 640, 336, 560, 1, 2, 504, 784},    {468, 931, 336, 616, 1, 2, 448, 896},
        {448, 448, 448, 448, 0, 0, 0, 0},        {10, 10, 448, 448, 0, 0, 0, 0},
        {1000, 1000, 448, 448, 2, 2, 896, 896},  {300, 4000, 112, 1624, 1, 7, 336, 4312},
        {4000, 300, 1624, 112, 7, 1, 4312, 336}, {2000, 3000, 392, 560, 2, 4, 1008, 1568},
        {449, 449, 448, 448, 2, 1, 672, 616},    {100, 3000, 56, 2408, 1, 3, 168, 4200},
    };
    const ImageSettings settings;
    for (const auto& c : cases) {
        const auto plan = plan_slices(settings, c.height, c.width);
        const auto label = "plan " + std::to_string(c.height) + "x" + std::to_string(c.width);
        check(plan.source_height == c.source_height && plan.source_width == c.source_width,
              label + " source");
        check(plan.rows == c.rows && plan.cols == c.cols, label + " grid");
        check(plan.refine_height == c.refine_height && plan.refine_width == c.refine_width,
              label + " refine");
    }
    check(rejects([&] { plan_slices(settings, 0, 10); }), "plan rejects an empty image");
}

void grid_indices() {
    check(block_order(2, 4) == std::vector<std::int32_t>{0, 1, 4, 5, 2, 3, 6, 7},
          "block order 2x4");
    check(block_order(4, 2) == std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 6, 7},
          "block order 4x2");
    check(rejects([] { block_order(3, 4); }), "block order rejects odd grids");
    const auto ids = bucket_position_ids(4, 8, 70);
    check(ids.size() == 32, "position ids size");
    check(ids.front() == 0, "position ids origin");
    bool inside = true;
    for (const auto id : ids)
        inside = inside && id >= 0 && id < 70 * 70;
    check(inside, "position ids address the 70 x 70 table");

    // get_vision_nearest_position_ids evaluated on the GPU, as the reference model does; its
    // CPU evaluation buckets these sizes differently.
    const std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>> buckets{
        {25, {0,  2,  5,  8,  11, 13, 16, 19, 22, 25, 27, 30, 33,
              36, 39, 41, 44, 47, 50, 53, 55, 58, 61, 64, 67}},
        {49, {0,  1,  2,  4,  5,  7,  8,  9,  11, 12, 14, 15, 17, 18, 19, 21, 22,
              24, 25, 27, 28, 30, 31, 32, 34, 35, 37, 38, 39, 41, 42, 44, 45, 47,
              48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 62, 64, 65, 67, 68}},
        {50, {0,  1,  2,  4,  5,  6,  8,  9,  11, 12, 13, 15, 16, 18, 19, 20, 22,
              23, 25, 26, 27, 29, 30, 32, 33, 35, 36, 37, 39, 40, 41, 43, 44, 46,
              47, 49, 50, 51, 53, 54, 55, 57, 58, 60, 61, 63, 64, 65, 67, 68}},
        {55, {0,  1,  2,  3,  5,  6,  7,  8,  10, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22,
              24, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 47,
              48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 61, 62, 63, 64, 66, 67, 68}},
        {82, {0,  0,  1,  2,  3,  4,  5,  5,  6,  7,  8,  9,  10, 11, 11, 12, 13, 14, 15, 16, 17,
              17, 18, 19, 20, 21, 22, 23, 23, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 33, 34, 34,
              35, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 52,
              53, 54, 55, 56, 57, 58, 58, 59, 60, 61, 62, 63, 64, 64, 65, 66, 67, 68, 69}},
    };
    for (const auto& [width, expected] : buckets)
        check(bucket_position_ids(1, width, 70) == expected,
              "position buckets of width " + std::to_string(width));
}

void resampling() {
    // A uniform image stays uniform through both antialiased passes.
    const std::vector<std::uint8_t> gray(37 * 53 * 3, 117);
    const auto smaller = resize_rgb8(gray.data(), 37, 53, 14, 28);
    bool uniform = smaller.size() == static_cast<std::size_t>(14 * 28 * 3);
    for (const auto value : smaller)
        uniform = uniform && value == 117;
    check(uniform, "uniform image resamples to itself");
    const auto same = resize_rgb8(gray.data(), 37, 53, 37, 53);
    check(same == gray, "same-size resize is the identity");

    // torchvision resize(uint8, BICUBIC, antialias=True), as the reference processor calls it.
    struct ResizeCase {
        std::int32_t height, width, out_height, out_width;
        std::vector<std::uint8_t> expected;
    };
    const std::vector<ResizeCase> cases{
        {5, 7, 3, 4, {64,  119, 125, 161, 77,  120, 136, 168, 86,  135, 130, 181,
                      119, 152, 121, 112, 136, 131, 140, 125, 147, 131, 160, 142,
                      179, 123, 108, 146, 107, 141, 94,  121, 118, 192, 110, 106}},
        {5, 7, 9, 11, {0,   49,  107, 34,  106, 175, 108, 152, 231, 190, 33,  97,  151, 20,  88,
                       44,  103, 162, 96,  186, 55,  184, 173, 16,  255, 54,  98,  170, 99,  171,
                       78,  156, 216, 7,   61,  125, 45,  118, 144, 121, 164, 154, 210, 45,  86,
                       170, 32,  108, 57,  116, 175, 116, 205, 68,  174, 194, 29,  190, 68,  111,
                       141, 113, 191, 98,  170, 237, 26,  85,  160, 69,  141, 91,  145, 188, 16,
                       237, 69,  68,  197, 57,  145, 82,  141, 200, 153, 233, 94,  159, 222, 55,
                       54,  95,  138, 91,  140, 219, 137, 198, 255, 46,  104, 184, 96,  169, 94,
                       159, 224, 1,   156, 95,  80,  128, 83,  175, 105, 164, 237, 194, 166, 123,
                       185, 143, 81,  42,  111, 165, 97,  179, 173, 164, 237, 159, 67,  126, 203,
                       125, 184, 123, 172, 231, 39,  54,  113, 107, 43,  112, 190, 127, 186, 245,
                       211, 81,  140, 200, 43,  109, 82,  126, 194, 129, 202, 111, 187, 246, 31,
                       88,  157, 242, 146, 136, 154, 192, 109, 60,  64,  96,  138, 61,  152, 158,
                       149, 226, 121, 126, 99,  106, 100, 57,  147, 97,  153, 236, 161, 142, 136,
                       213, 116, 41,  108, 186, 223, 167, 100, 153, 215, 12,  83,  98,  90,  170,
                       98,  169, 135, 172, 211, 23,  59,  118, 86,  21,  98,  169, 113, 183, 223,
                       190, 93,  149, 233, 6,   79,  132, 210, 70,  191, 126, 80,  239, 39,  114,
                       123, 127, 199, 123, 132, 160, 197, 71,  51,  87,  83,  126, 49,  136, 153,
                       140, 211, 88,  208, 112, 90,  245, 21,  102, 144, 221, 0,   203, 142, 39,
                       251, 60,  130, 135, 149, 214, 135, 109, 176, 210, 0,   72,  106, 62,  149,
                       70,  156, 143, 154, 225, 12,  216, 126, 56,  251, 36,  114}},
        {6, 4, 6, 9, {0,   49,  108, 10,  72,  131, 55,  131, 190, 107, 157, 216, 168, 83,  142,
                      203, 8,   67,  148, 34,  93,  66,  93,  152, 32,  116, 175, 27,  86,  170,
                      47,  109, 136, 92,  169, 54,  145, 195, 0,   206, 121, 36,  242, 47,  97,
                      188, 73,  150, 106, 133, 195, 72,  156, 215, 64,  123, 206, 87,  146, 173,
                      147, 206, 92,  174, 233, 38,  101, 160, 75,  27,  86,  136, 54,  113, 189,
                      114, 173, 234, 137, 196, 255, 101, 184, 243, 125, 152, 211, 185, 70,  129,
                      213, 17,  77,  139, 54,  129, 65,  116, 181, 93,  170, 129, 154, 215, 47,
                      177, 236, 15,  138, 221, 0,   162, 189, 21,  222, 108, 67,  251, 56,  121,
                      178, 109, 184, 104, 161, 221, 133, 109, 168, 193, 28,  87,  217, 0,   55,
                      199, 255, 37,  167, 226, 61,  86,  145, 122, 35,  94,  151, 72,  147, 78,
                      135, 200, 5,   190, 149, 34,  236, 68,  95,  255, 36,  119}},
        {13, 3, 4, 3, {46,  110, 167, 158, 172, 73,  143, 53, 123, 158, 173, 86,
                       148, 82,  132, 118, 172, 115, 74,  89, 146, 133, 182, 116,
                       107, 95,  97,  137, 201, 112, 106, 92, 106, 94,  158, 224}},
    };
    for (const auto& c : cases)
        check(resize_rgb8(pattern(c.height, c.width).data(), c.height, c.width, c.out_height,
                          c.out_width) == c.expected,
              "resize " + std::to_string(c.height) + "x" + std::to_string(c.width) + " -> " +
                  std::to_string(c.out_height) + "x" + std::to_string(c.out_width));
}

void placeholders() {
    PreparedImage image;
    image.rows = 1;
    image.cols = 2;
    VisionSlice source, slice;
    source.height = 4;
    source.width = 8;               // 2 tokens
    slice.height = slice.width = 4; // 1 token
    image.slices = {source, slice, slice};
    check(image_placeholder(image, 3) ==
              "<image_id>3</image_id><image><|image_pad|><|image_pad|></image>"
              "<slice><|image_pad|></slice><slice><|image_pad|></slice>",
          "placeholder of a 1 x 2 grid");
    image.rows = 2;
    image.cols = 1;
    check(image_placeholder(image, 0) ==
              "<image_id>0</image_id><image><|image_pad|><|image_pad|></image>"
              "<slice><|image_pad|></slice>\n<slice><|image_pad|></slice>",
          "placeholder rows are newline separated");
}

void conversations() {
    const auto recipe = nlohmann::json::parse(R"({
        "default_prefix": "", "empty_system_prefix": "<s>\n</s>\n", "generation": "<a>\n",
        "parse_assistant_thinking": true,
        "user": {"prefix": "<u>\n", "suffix": "</u>\n"},
        "system": {"prefix": "<s>\n", "suffix": "</s>\n"},
        "assistant_recent": {"prefix": "<a>\n<t>\n", "include_reasoning": true,
                             "reasoning_suffix": "\n</t>\n\n", "suffix": "</a>\n"},
        "assistant_history": {"prefix": "<a>\n", "include_reasoning": false,
                              "reasoning_suffix": "", "suffix": "</a>\n"}})");
    validate_recipe(recipe);
    check(format_conversation(recipe, {{Role::User, " \n Hi \xe3\x80\x80"}}) ==
              "<u>\nHi</u>\n<a>\n",
          "user message is stripped like Python");
    check(format_conversation(recipe, {{Role::System, " Be brief. "},
                                       {Role::User, "Q1"},
                                       {Role::Assistant, "<think>r</think>\n\nA1"},
                                       {Role::User, "Q2"}}) ==
              "<s>\nBe brief.</s>\n<u>\nQ1</u>\n<a>\nA1</a>\n<u>\nQ2</u>\n<a>\n",
          "history assistant drops its reasoning");
    check(rejects([&] { format_conversation(recipe, {{Role::Assistant, "A"}}); }),
          "a conversation needs a user query");
    check(strip("\xc2\xa0 x \t") == "x", "strip removes Unicode white space");
}

void pre_tokenization() {
    check(pre_tokenize("Hello world") == std::vector<std::string>{"Hello", " world"},
          "words keep their leading space");
    check(pre_tokenize("2026") == std::vector<std::string>{"2", "0", "2", "6"},
          "digits split one by one");
    check(pre_tokenize("it's") == std::vector<std::string>{"it", "'s"}, "contractions split");
    // Pieces from tokenizers 0.23.3 on the same texts. Its NFC uses Unicode 9.0 tables, where
    // U+1DF6 (Unicode 10.0) is a starter: it stays before U+0316 and blocks e + U+0301, which
    // current NFC reorders and composes.
    check(pre_tokenize("x\xE1\xB7\xB6\xCC\x96") ==
              std::vector<std::string>{"x\xE1\xB7\xB6\xCC\x96"},
          "a mark unknown to Unicode 9.0 is not reordered");
    check(pre_tokenize("e\xE1\xB7\xB6\xCC\x81") ==
              std::vector<std::string>{"e\xE1\xB7\xB6\xCC\x81"},
          "a mark unknown to Unicode 9.0 blocks composition");
    check(pre_tokenize("e\xCC\x81") == std::vector<std::string>{"\xC3\xA9"}, "NFC composes");
    // Its pattern classes are Unicode 16.0's: GARAY DIGIT ONE (U+10D41) is a number and CYRILLIC
    // CAPITAL LETTER TJE (U+1C89) a letter, while U+10940 (assigned in Unicode 17.0) is neither.
    check(pre_tokenize("a\xF0\x90\xB5\x81"
                       "b") == std::vector<std::string>{"a", "\xF0\x90\xB5\x81", "b"},
          "a Unicode 16.0 digit splits alone");
    check(pre_tokenize("a\xE1\xB2\x89"
                       "b") == std::vector<std::string>{"a\xE1\xB2\x89"
                                                        "b"},
          "a Unicode 16.0 letter joins the word");
    // Unicode 15.0 (ICU 72 and newer): KAWI LETTER A (U+11F04) and KAWI DIGIT ONE (U+11F51).
    check(pre_tokenize("a\xF0\x91\xBC\x84"
                       "b") == std::vector<std::string>{"a\xF0\x91\xBC\x84"
                                                        "b"},
          "a Unicode 15.0 letter joins the word");
    check(pre_tokenize("a\xF0\x91\xBD\x91"
                       "b") == std::vector<std::string>{"a", "\xF0\x91\xBD\x91", "b"},
          "a Unicode 15.0 digit splits alone");
    check(pre_tokenize("a\xF0\x90\xA5\x80"
                       "b") == std::vector<std::string>{"a", "\xF0\x90\xA5\x80"
                                                             "b"},
          "a code point unassigned in Unicode 16.0 is not a letter");
    // \s is White_Space in both regex engines, VT and NEL included.
    check(pre_tokenize("x\x0B\x0By") == std::vector<std::string>{"x", "\x0B", "\x0By"},
          "vertical tabs are white space");
    check(pre_tokenize("x\xC2\x85\xC2\x85y") ==
              std::vector<std::string>{"x", "\xC2\x85", "\xC2\x85y"},
          "next-line characters are white space");
    check(utf8_lossy(std::string("a\xff", 2)) == "a\xef\xbf\xbd", "invalid UTF-8 is replaced");
}

// A module that only answers introspection queries, for the engine I/O contracts. Like the TensorRT
// module it reports every tensor at a current, concrete shape: a dynamic dimension (-1 in `shape`,
// the engine's declared shape) at its profile maximum, or 1 without a profile.
class FakeModule final : public trtmc::ITrtModule {
  public:
    struct Entry {
        trtmc::DType dtype;
        std::vector<int64_t> shape;
        bool input;
        std::vector<int64_t> minimum, maximum;
        std::vector<int64_t> current() const {
            auto result = shape;
            for (std::size_t i = 0; i < result.size(); ++i)
                if (result[i] < 0)
                    result[i] = maximum.size() == shape.size() ? maximum[i] : 1;
            return result;
        }
    };
    std::map<std::string, Entry> tensors;

    void add(const std::string& name, trtmc::DType dtype, std::vector<int64_t> shape, bool input,
             std::vector<int64_t> minimum = {}, std::vector<int64_t> maximum = {}) {
        tensors[name] = {dtype, std::move(shape), input, std::move(minimum), std::move(maximum)};
    }

    trtmc::TensorMap forward(const trtmc::TensorMap&) override { throw std::logic_error("fake"); }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override {
        throw std::logic_error("fake");
    }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return info(true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return info(false); }
    bool has_input(const std::string& name) const override {
        const auto found = tensors.find(name);
        return found != tensors.end() && found->second.input;
    }
    bool has_output(const std::string& name) const override {
        const auto found = tensors.find(name);
        return found != tensors.end() && !found->second.input;
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return tensors.at(name).dtype;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return tensors.at(name).current();
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector selector) const override {
        const auto& entry = tensors.at(name);
        return selector == trtmc::ProfileShapeSelector::kMin ? entry.minimum : entry.maximum;
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string& name) const override {
        return static_cast<int32_t>(tensors.at(name).shape.size());
    }
    bool input_is_dynamic(const std::string& name) const override {
        const auto& entry = tensors.at(name);
        return entry.input &&
               std::any_of(entry.shape.begin(), entry.shape.end(), [](int64_t d) { return d < 0; });
    }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::vector<trtmc::TensorInfo> info(bool input) const {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& [name, entry] : tensors)
            if (entry.input == input)
                result.push_back({name, entry.current(), entry.dtype, input});
        return result;
    }
};

DecoderGeometry small_geometry() {
    DecoderGeometry geometry;
    geometry.attention_layers = 1;
    geometry.deltanet_layers = 2;
    geometry.kv_width = 8;
    geometry.conv_width = 6;
    geometry.conv_kernel = 4;
    geometry.deltanet_heads = 2;
    geometry.deltanet_head_dim = 3;
    geometry.capacity = 16;
    geometry.cache_dtype = trtmc::DType::kFloat16;
    return geometry;
}

// The tensors the builder declares for the decoder (rows = 1) or the prefill (rows = -1).
FakeModule decoder_engine(const DecoderGeometry& g, bool prefill, int64_t vocab, int64_t hidden) {
    using trtmc::DType;
    FakeModule module;
    const int64_t rows = prefill ? -1 : 1;
    // The prefill profile admits chunks of 1 to 8 rows, as the builder declares it.
    const int64_t most = prefill ? 8 : 1;
    module.add("token_id", DType::kInt32, {rows}, true, {1}, {most});
    module.add("position_id", DType::kInt32, {rows}, true, {1}, {most});
    module.add("attention_mask", DType::kFloat32, {rows, prefill ? -1 : g.capacity + 1}, true,
               {1, g.capacity + 1}, {most, g.capacity + most});
    if (prefill) {
        module.add("visual_embedding", g.cache_dtype, {-1, hidden}, true, {1, hidden},
                   {most, hidden});
        module.add("use_visual_embedding", DType::kInt32, {-1, 1}, true, {1, 1}, {most, 1});
    }
    module.add("logits", DType::kFloat32, {1, vocab}, false);
    for (int i = 0; i < g.deltanet_layers; ++i) {
        const auto n = std::to_string(i);
        for (const char* stem : {"conv_state_", "present_conv_"})
            module.add(stem + n, DType::kFloat32, {g.conv_width, g.conv_kernel}, stem[0] == 'c');
        for (const char* stem : {"ssm_state_", "present_ssm_"})
            module.add(stem + n, DType::kFloat32,
                       {g.deltanet_heads, g.deltanet_head_dim, g.deltanet_head_dim},
                       stem[0] == 's');
    }
    for (int i = 0; i < g.attention_layers; ++i) {
        const auto n = std::to_string(i);
        module.add("cache_k_" + n, g.cache_dtype, {g.capacity, g.kv_width}, true);
        module.add("cache_v_" + n, g.cache_dtype, {g.capacity, g.kv_width}, true);
        // Outputs are sized for the profile's largest chunk, as the TensorRT module allocates them.
        module.add("present_k_" + n, g.cache_dtype, {rows, g.kv_width}, false, {},
                   {most, g.kv_width});
        module.add("present_v_" + n, g.cache_dtype, {rows, g.kv_width}, false, {},
                   {most, g.kv_width});
    }
    return module;
}

void engine_contracts() {
    using trtmc::DType;
    const auto geometry = small_geometry();
    for (const bool prefill : {false, true}) {
        const std::string kind = prefill ? "prefill" : "decoder";
        auto engine = decoder_engine(geometry, prefill, 32, 4);
        validate_decoder_engine(engine, geometry, prefill, 32, 4);
        auto wrong_ssm = engine;
        wrong_ssm.tensors["ssm_state_1"].shape = {2, 3, 2};
        check(fails([&] { validate_decoder_engine(wrong_ssm, geometry, prefill, 32, 4); }),
              kind + ": a recurrent state of another shape is rejected");
        auto wrong_present = engine;
        wrong_present.tensors["present_conv_0"].dtype = DType::kFloat16;
        check(fails([&] { validate_decoder_engine(wrong_present, geometry, prefill, 32, 4); }),
              kind + ": a present state of another dtype is rejected");
        auto wrong_logits = engine;
        wrong_logits.tensors["logits"].shape = {1, 16};
        check(fails([&] { validate_decoder_engine(wrong_logits, geometry, prefill, 32, 4); }),
              kind + ": logits narrower than the vocabulary are rejected");
        auto missing = engine;
        missing.tensors.erase("present_v_0");
        check(fails([&] { validate_decoder_engine(missing, geometry, prefill, 32, 4); }),
              kind + ": a missing present output is rejected");
        auto extra = engine;
        extra.add("unexpected", DType::kInt32, {1}, true);
        check(fails([&] { validate_decoder_engine(extra, geometry, prefill, 32, 4); }),
              kind + ": an input the runtime does not bind is rejected");
    }
    auto prefill = decoder_engine(geometry, true, 32, 4);
    prefill.tensors["visual_embedding"].shape = {-1, 8};
    check(fails([&] { validate_decoder_engine(prefill, geometry, true, 32, 4); }),
          "a vision row of another width is rejected");
    // Row inputs must be dynamic in the prefill engine and static in the decoder.
    auto fixed_rows = decoder_engine(geometry, true, 32, 4);
    fixed_rows.tensors["token_id"].shape = {8};
    check(fails([&] { validate_decoder_engine(fixed_rows, geometry, true, 32, 4); }),
          "a prefill engine with a fixed row count is rejected");
    auto dynamic_rows = decoder_engine(geometry, false, 32, 4);
    dynamic_rows.tensors["position_id"].shape = {-1};
    dynamic_rows.tensors["position_id"].maximum = {1};
    check(fails([&] { validate_decoder_engine(dynamic_rows, geometry, false, 32, 4); }),
          "a decoder engine with a dynamic row count is rejected");

    // Every chunk input's profile must admit chunks of 1 to 8 rows.
    const auto chunked = decoder_engine(geometry, true, 32, 4);
    check(prefill_rows(chunked, geometry, 4) == 8, "the prefill chunk has 8 rows");
    const std::vector<std::pair<std::string, std::vector<int64_t>>> narrow_maxima{
        {"token_id", {17}},
        {"position_id", {4}},
        {"attention_mask", {8, 23}},
        {"visual_embedding", {4, 4}},
        {"use_visual_embedding", {4, 1}}};
    for (const auto& [name, maximum] : narrow_maxima) {
        auto engine = chunked;
        engine.tensors[name].maximum = maximum;
        check(fails([&] { prefill_rows(engine, geometry, 4); }),
              "a prefill profile whose " + name + " maximum is too small is rejected");
    }
    // A present K/V output smaller than the largest chunk would be read past its end.
    for (const char* name : {"present_k_0", "present_v_0"}) {
        auto engine = chunked;
        engine.tensors[name].shape = {1, geometry.kv_width};
        check(fails([&] { prefill_rows(engine, geometry, 4); }),
              std::string("a prefill engine whose ") + name + " holds one row is rejected");
    }
    auto late_start = chunked;
    late_start.tensors["attention_mask"].minimum = {2, geometry.capacity + 2};
    check(fails([&] { prefill_rows(late_start, geometry, 4); }),
          "a prefill profile that does not admit one-row chunks is rejected");

    FakeModule vision;
    vision.add("patches", DType::kFloat32, {-1, 12}, true, {16, 12}, {64, 12});
    vision.add("position_ids", DType::kInt32, {-1}, true, {16}, {64});
    vision.add("window_index", DType::kInt32, {-1}, true, {16}, {64});
    vision.add("merge_index", DType::kInt32, {-1}, true, {4}, {16});
    vision.add("visual_features", DType::kFloat16, {-1, 4}, false);
    validate_vision_engine(vision, DType::kFloat16, 12, 4, 64);
    auto bad_index = vision;
    bad_index.tensors["window_index"].dtype = DType::kFloat32;
    check(fails([&] { validate_vision_engine(bad_index, DType::kFloat16, 12, 4, 64); }),
          "a grid index of another dtype is rejected");
    auto bad_profile = vision;
    bad_profile.tensors["merge_index"].maximum = {64};
    check(fails([&] { validate_vision_engine(bad_profile, DType::kFloat16, 12, 4, 64); }),
          "a merge-index profile that does not match the patch capacity is rejected");
    auto bad_features = vision;
    bad_features.tensors["visual_features"].shape = {-1, 8};
    check(fails([&] { validate_vision_engine(bad_features, DType::kFloat16, 12, 4, 64); }),
          "features of another width are rejected");
    // The smallest slice is one 4 x 4 block: 16 patches and 4 merged rows.
    const std::vector<std::pair<std::string, std::vector<int64_t>>> late_minima{
        {"patches", {32, 12}},
        {"position_ids", {32}},
        {"window_index", {32}},
        {"merge_index", {8}}};
    for (const auto& [name, minimum] : late_minima) {
        auto engine = vision;
        engine.tensors[name].minimum = minimum;
        check(fails([&] { validate_vision_engine(engine, DType::kFloat16, 12, 4, 64); }),
              "a vision profile whose " + name + " excludes the smallest slice is rejected");
    }
}

void metadata_parsing() {
    const auto object = nlohmann::json::parse(R"({
        "int": 7, "fraction": 7.5, "flag": true, "huge": 8589934592, "negative": -3, "nothing": null,
        "float3": [0.5, 1, 0.25], "two": [0.5, 0.5], "four": [1, 2, 3, 4], "bool3": [0.5, true, 0.5],
        "big3": [1e39, 0.5, 0.5], "ids": [3, 5], "repeated": [3, 3], "outside": [3, 99],
        "fraction_ids": [3.0], "empty": []})");
    check(json_int32(object, "int", 1) == 7, "an integer field is read");
    for (const char* name : {"fraction", "flag", "huge", "negative", "nothing", "missing"})
        check(fails([&] { json_int32(object, name, 0); }),
              std::string("integer field '") + name + "' is rejected");
    check(json_float3(object, "float3") == std::array<float, 3>{0.5F, 1.0F, 0.25F},
          "three numbers are read");
    for (const char* name : {"two", "four", "bool3", "big3", "int"})
        check(fails([&] { json_float3(object, name); }),
              std::string("float triple '") + name + "' is rejected");
    check(json_token_ids(object, "ids", 10) == std::vector<std::int32_t>{3, 5}, "token ids read");
    for (const char* name : {"repeated", "outside", "fraction_ids", "empty", "int"})
        check(fails([&] { json_token_ids(object, name, 10); }),
              std::string("token id list '") + name + "' is rejected");
    check(checked_product(4, 256, "kv") == 1024, "a small product is kept");
    check(fails([&] { checked_product(65536, 65536, "kv"); }), "an int32 overflow is rejected");

    const auto parse = [](const std::string& text) {
        return parse_json_unique(text.data(), text.size(), "test.json");
    };
    check(parse(R"({"a": {"b": 1}, "c": {"b": 2}})")["c"]["b"] == 2,
          "the same key in sibling objects is accepted");
    check(fails([&] { parse(R"({"a": 1, "a": 2})"); }), "a repeated top-level key is rejected");
    check(fails([&] { parse(R"({"x": [{"b": 1, "b": 2}]})"); }),
          "a repeated nested key is rejected");
    check_vocabulary(248094, 248094);
    check_vocabulary(248094, 248320);
    check(fails([&] { check_vocabulary(248321, 248320); }),
          "token ids beyond the embedding rows are rejected");
}

void sampling() {
    SamplingOptions options;
    Sampler greedy(options);
    check(greedy.sample({0.5F, 2.0F, 1.0F}) == 1, "greedy picks the argmax");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    check(fails([&] { greedy.sample({0.5F, nan, 1.0F}); }), "NaN logits are rejected");
    check(fails([&] { greedy.sample({0.5F, inf, 1.0F}); }), "infinite logits are rejected");
    options.top_k = 0;
    options.seed = 7;
    Sampler drawing(options);
    check(fails([&] { drawing.sample({nan, 0.0F}); }), "NaN logits are rejected when sampling");
    options.seed = -2;
    check(rejects([&] { Sampler invalid(options); }), "a seed below -1 is rejected");

    // Request values are checked before FP32 narrows them.
    const auto valid = sampling_options(0.7, 40, 0.9, 7);
    check(valid.temperature == 0.7F && valid.top_k == 40 && valid.top_p == 0.9F && valid.seed == 7,
          "valid sampling values are kept");
    check(sampling_options(0.0, 1, 1.0, -1).temperature == 0.0F, "a zero temperature is greedy");
    const double nan_double = std::numeric_limits<double>::quiet_NaN();
    for (const double temperature : {-1e-50, 1e-50, 1e39, nan_double})
        check(rejects([&] { sampling_options(temperature, 1, 1.0, -1); }),
              "temperature " + std::to_string(temperature) + " is rejected");
    for (const double top_p : {1.00000001, 0.0, 1e-50, -0.5, nan_double})
        check(rejects([&] { sampling_options(1.0, 1, top_p, -1); }),
              "top_p " + std::to_string(top_p) + " is rejected");
    for (const std::int64_t top_k : {std::int64_t{-1}, std::int64_t{1} << 31})
        check(rejects([&] { sampling_options(1.0, top_k, 1.0, -1); }),
              "an invalid top_k is rejected");
    check(rejects([&] { sampling_options(1.0, 1, 1.0, -2); }), "a seed below -1 is rejected");
}

void planned_image_rows() {
    // The rows counted from the slice plan before any allocation equal the prepared slices' rows.
    ImageSettings settings;
    for (const auto& [height, width] : std::vector<std::pair<std::int32_t, std::int32_t>>{
             {382, 640}, {468, 931}, {448, 448}, {10, 10}, {1000, 1000}, {300, 1200}}) {
        const auto pixels = pattern(height, width);
        trtmc::internal::ImageView image{};
        image.data = pixels.data();
        image.byte_size = pixels.size();
        image.height = static_cast<std::uint32_t>(height);
        image.width = static_cast<std::uint32_t>(width);
        image.channels = 3;
        image.format = trtmc::internal::ImageFormat::UInt8;
        std::int64_t prepared = 0;
        for (const auto& slice : prepare_image(settings, image).slices)
            prepared += slice.tokens();
        check(planned_rows(settings, plan_image(settings, image)) == prepared,
              "planned rows of " + std::to_string(height) + "x" + std::to_string(width) +
                  " equal the prepared slices");
    }
}

std::string message_of(const std::function<void()>& callable) {
    try {
        callable();
    } catch (const std::exception& error) {
        return error.what();
    }
    return "";
}

void planning_before_pixels() {
    // A tiny engine capacity: one 4 x 4 block of patches per slice.
    ImageSettings settings;
    settings.scale_resolution = 56;
    settings.max_patches = 16;
    settings.validate();
    const std::int32_t height = 10, width = 200;
    const auto plan = plan_slices(settings, height, width);
    check(static_cast<std::int64_t>(plan.source_height / 14) * (plan.source_width / 14) > 16,
          "the elongated test image plans a slice beyond the capacity");
    // The float pixels hold a NaN: the slice plan must reject the image before they are read.
    std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3, 0.5F);
    pixels[7] = std::numeric_limits<float>::quiet_NaN();
    trtmc::internal::ImageView image{};
    image.data = pixels.data();
    image.byte_size = pixels.size() * sizeof(float);
    image.height = height;
    image.width = width;
    image.channels = 3;
    image.format = trtmc::internal::ImageFormat::Float32;
    check(message_of([&] { prepare_image(settings, image); }).find("patch capacity") !=
              std::string::npos,
          "the slice plan is checked before the pixels are converted");
}

void slice_bounds() {
    ImageSettings settings;
    settings.max_slice_nums = kMaxSliceNums;
    settings.validate();
    for (const std::int32_t count : {kMaxSliceNums + 1, std::numeric_limits<std::int32_t>::max()}) {
        auto large = settings;
        large.max_slice_nums = count;
        check(rejects([&] { large.validate(); }),
              "max_slice_nums " + std::to_string(count) + " is rejected");
    }
    // The largest accepted slice count on the largest square image.
    const std::int32_t side = std::numeric_limits<std::int32_t>::max();
    const auto plan = plan_slices(settings, side, side);
    check(plan.rows == 64 && plan.cols == 64, "the largest image plans a 64 x 64 grid");
    // Dimensions that the storage does not hold are rejected before the slice plan.
    const std::vector<std::uint8_t> pixel{1, 2, 3};
    trtmc::internal::ImageView image{};
    image.data = pixel.data();
    image.byte_size = pixel.size();
    image.height = static_cast<std::uint32_t>(side);
    image.width = static_cast<std::uint32_t>(side);
    image.channels = 3;
    image.format = trtmc::internal::ImageFormat::UInt8;
    check(message_of([&] { plan_image(settings, image); }).find("storage") != std::string::npos,
          "the slice plan checks the image storage first");
}

void normalization_bounds() {
    ImageSettings settings;
    settings.validate();
    auto tiny = settings;
    tiny.stddev[1] = 1e-40F;
    check(rejects([&] { tiny.validate(); }), "a vanishing std is rejected");
    auto large = settings;
    large.mean[0] = 1e38F;
    check(rejects([&] { large.validate(); }), "a mean that overflows FP32 at 255x is rejected");
    auto narrow = settings;
    narrow.stddev[2] = 1e-6F;
    check(rejects([&] { narrow.validate(); }), "normalized pixels beyond FP16 are rejected");
    auto patch = settings;
    patch.patch_size = std::numeric_limits<std::int32_t>::max();
    check(rejects([&] { patch.validate(); }), "a patch size whose 4x overflows int32 is rejected");
    auto resolution = settings;
    resolution.scale_resolution = 1000000;
    check(rejects([&] { resolution.validate(); }),
          "a nominal slice beyond the vision engine's patches is rejected");
}

// GPT-2 byte-level alphabet: one vocabulary token per byte, in byte order.
std::vector<std::string> byte_alphabet() {
    std::vector<int> direct;
    for (int b = 33; b <= 126; ++b)
        direct.push_back(b);
    for (int b = 161; b <= 172; ++b)
        direct.push_back(b);
    for (int b = 174; b <= 255; ++b)
        direct.push_back(b);
    std::vector<std::string> tokens(256);
    int displaced = 0;
    for (int b = 0; b < 256; ++b) {
        const bool kept = std::find(direct.begin(), direct.end(), b) != direct.end();
        const auto codepoint = static_cast<char32_t>(kept ? b : 256 + displaced++);
        std::string utf8;
        if (codepoint < 0x80) {
            utf8.push_back(static_cast<char>(codepoint));
        } else {
            utf8.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            utf8.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
        tokens[static_cast<std::size_t>(b)] = utf8;
    }
    return tokens;
}

// A minimal tokenizer.json of the pinned schema: bytes, "xy" (256), "xyx" (257), one added token.
nlohmann::json small_tokenizer() {
    nlohmann::json vocab = nlohmann::json::object();
    const auto bytes = byte_alphabet();
    for (std::size_t i = 0; i < bytes.size(); ++i)
        vocab[bytes[i]] = i;
    vocab["xy"] = 256;
    vocab["xyx"] = 257;
    nlohmann::json byte_level = nlohmann::json::object();
    byte_level["type"] = "ByteLevel";
    byte_level["add_prefix_space"] = false;
    byte_level["trim_offsets"] = false;
    byte_level["use_regex"] = false;
    nlohmann::json split = nlohmann::json::object();
    split["type"] = "Split";
    split["pattern"]["Regex"] =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    split["behavior"] = "Isolated";
    split["invert"] = false;
    nlohmann::json added = nlohmann::json::object();
    added["id"] = 258;
    added["content"] = "<|end|>";
    for (const char* flag : {"single_word", "lstrip", "rstrip", "normalized"})
        added[flag] = false;
    added["special"] = true;
    nlohmann::json model = nlohmann::json::object();
    model["type"] = "BPE";
    model["dropout"] = nullptr;
    model["unk_token"] = nullptr;
    model["continuing_subword_prefix"] = "";
    model["end_of_word_suffix"] = "";
    model["fuse_unk"] = false;
    model["byte_fallback"] = false;
    model["ignore_merges"] = false;
    model["vocab"] = vocab;
    model["merges"] = nlohmann::json::array();
    model["merges"].push_back(nlohmann::json::array({"xy", "x"}));
    model["merges"].push_back(nlohmann::json::array({"x", "y"}));
    nlohmann::json root = nlohmann::json::object();
    root["version"] = "1.0";
    root["truncation"] = nullptr;
    root["padding"] = nullptr;
    root["added_tokens"] = nlohmann::json::array({added});
    root["normalizer"]["type"] = "NFC";
    root["pre_tokenizer"]["type"] = "Sequence";
    root["pre_tokenizer"]["pretokenizers"] = nlohmann::json::array({split, byte_level});
    root["post_processor"] = byte_level;
    root["decoder"] = byte_level;
    root["model"] = model;
    return root;
}

std::unique_ptr<Tokenizer> load(const nlohmann::json& json) {
    const auto text = json.dump();
    return create_tokenizer(text.data(), text.size());
}

// The ids of a text that fits a generous limit.
std::vector<std::int32_t> ids(const Tokenizer& tokenizer, std::string_view text) {
    const auto result = tokenizer.encode(text, 1000);
    check(result.has_value(), "the text fits 1000 ids");
    return result.value_or(std::vector<std::int32_t>{});
}

void bpe_merges() {
    // Hugging Face tokenizers merge one occurrence at a time and re-rank after each
    // merge: in "xyxy" the first "xy" joins the next "x" (rank 0) before the second
    // "xy" (rank 1) can form. Expected ids from tokenizers 0.23.3 on the same file.
    const auto tokenizer = load(small_tokenizer());
    check(ids(*tokenizer, "xyxy") == std::vector<std::int32_t>{257, 121},
          "xyxy merges as tokenizers does");
    check(ids(*tokenizer, "xyxyxy") == std::vector<std::int32_t>{257, 121, 256},
          "xyxyxy merges as tokenizers does");
    check(ids(*tokenizer, "x xyxy") == std::vector<std::int32_t>{120, 32, 257, 121},
          "a spaced word merges as tokenizers does");
    check(ids(*tokenizer, "yxyx<|end|>") == std::vector<std::int32_t>{121, 257, 258},
          "added tokens split the text");

    // A non-special added token with a character outside the byte alphabet decodes as its own
    // UTF-8 text (tokenizers 0.23.3 on the same file: "你é", not "你\uFFFD").
    auto with_text_token = small_tokenizer();
    nlohmann::json text_token = with_text_token["added_tokens"][0];
    text_token["id"] = 259;
    text_token["content"] = "\xE4\xBD\xA0\xC3\xA9";
    text_token["special"] = false;
    with_text_token["added_tokens"].push_back(text_token);
    const auto text_tokenizer = load(with_text_token);
    check(text_tokenizer->decode({259}) == "\xE4\xBD\xA0\xC3\xA9",
          "an added text token decodes whole");
    check(text_tokenizer->decode({257, 259, 121}) == "xyx\xE4\xBD\xA0\xC3\xA9y",
          "byte-level tokens around an added text token decode as tokenizers does");
    check(text_tokenizer->decode({195, 259}) == "\xEF\xBF\xBD\xE4\xBD\xA0\xC3\xA9",
          "a dangling lead byte before an added text token is replaced");
    check(ids(*text_tokenizer, "a\xE4\xBD\xA0\xC3\xA9"
                               "b") == std::vector<std::int32_t>{97, 259, 98},
          "an added text token is matched whole");

    auto unknown_result = small_tokenizer();
    unknown_result["model"]["merges"].push_back(nlohmann::json::array({"y", "y"}));
    check(fails([&] { load(unknown_result); }), "a merge whose result is not a token is rejected");
    auto added_operand = small_tokenizer();
    added_operand["model"]["vocab"]["xy<|end|>"] = 258;
    added_operand["added_tokens"][0]["id"] = 259;
    added_operand["model"]["merges"].push_back(nlohmann::json::array({"xy", "<|end|>"}));
    check(fails([&] { load(added_operand); }),
          "a merge of a token outside the base vocabulary is rejected");
    auto repeated = small_tokenizer().dump();
    // The same valid value twice: nlohmann alone would keep one and accept the file.
    repeated.replace(repeated.find("\"dropout\":null"), 14, "\"dropout\":null,\"dropout\":null");
    check(fails([&] { create_tokenizer(repeated.data(), repeated.size()); }),
          "a tokenizer.json that repeats a key is rejected");
    auto wrapped_id = small_tokenizer();
    wrapped_id["model"]["vocab"]["xyx"] = 4294967296LL + 257;
    check(fails([&] { load(wrapped_id); }), "a vocabulary id that wraps to a valid id is rejected");
    auto fractional_added = small_tokenizer();
    fractional_added["added_tokens"][0]["id"] = 258.0;
    check(fails([&] { load(fractional_added); }), "a fractional added-token id is rejected");
}

void bounded_encoding() {
    const auto tokenizer = load(small_tokenizer());
    // The longest entry is the added token "<|end|>": 7 code points, 28 per id.
    check(tokenizer->max_code_points(10) == 280, "ten ids stand for at most 280 code points");
    check(tokenizer->encode("xyxy", 2) == std::vector<std::int32_t>{257, 121},
          "a text that needs exactly the limit fits");
    check(!tokenizer->encode("xyxy", 1), "a text that needs one id more than the limit does not");
    check(!tokenizer->encode(std::string(281, 'x'), 10),
          "a text longer than the limit can stand for is refused");
    check(!tokenizer->encode(std::string(std::size_t{1} << 26, 'x'), 4096),
          "a 64 MiB text is refused against a 4096-id limit");
    // Code points as ICU decodes UTF-8: each maximal ill-formed subpart is one U+FFFD.
    check(code_points("a\x80\x80"
                      "b") == 4,
          "stray continuation bytes count one each");
    check(code_points("\xE4\xBD") == 1, "a truncated sequence counts once");
    check(code_points("\xE4\xBD\xA0\xC3\xA9") == 2, "well-formed text counts its code points");
}

} // namespace

int main() {
    try {
        slice_plans();
        grid_indices();
        resampling();
        placeholders();
        conversations();
        pre_tokenization();
        engine_contracts();
        metadata_parsing();
        sampling();
        normalization_bounds();
        slice_bounds();
        planning_before_pixels();
        planned_image_rows();
        bpe_merges();
        bounded_encoding();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "test_minicpmv4_6_host passed\n";
    return 0;
}
