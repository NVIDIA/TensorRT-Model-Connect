/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-side GLM-OCR runtime checks that need no GPU and no checkpoint (run without
// arguments; with a command it is the host-parity probe of host_probe.cpp). Expected values come
// from the reference implementations (Transformers 5.22.2 smart_resize and get_rope_index,
// torchvision 0.24 uint8 antialiased bicubic resize on the CPU, the tokenizers Split
// pre-tokenizer with the checkpoint's pattern, the checkpoint's chat template).

#include "families/glm_ocr/runtime/image.h"
#include "families/glm_ocr/runtime/model.h"
#include "families/glm_ocr/runtime/prompt.h"
#include "families/glm_ocr/runtime/tokenizer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

bool throws(const std::function<void()>& body) {
    try {
        body();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

bool throws_invalid_argument(const std::function<void()>& body) {
    try {
        body();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (const std::exception&) {
        return false;
    }
    return false;
}

// A refused request must stay a ConfigError: the server answers it with a client error, while
// other exception types mark the worker as failed.
bool throws_config_error(const std::function<void()>& body) {
    try {
        body();
    } catch (const trtmc::internal::ConfigError&) {
        return true;
    } catch (const std::exception&) {
        return false;
    }
    return false;
}

trtmc::glm_ocr::ImageConfig image_config() {
    trtmc::glm_ocr::ImageConfig config;
    config.patch_size = 14;
    config.temporal_patch_size = 2;
    config.merge_size = 2;
    config.min_pixels = 12544;
    config.max_pixels = 9633792;
    config.mean = {0.48145466F, 0.4578275F, 0.40821073F};
    config.std = {0.26862954F, 0.26130258F, 0.27577711F};
    config.head_dim = 64;
    config.rope_theta = 10000.0F;
    config.max_patches = 24576;
    return config;
}

void test_smart_resize() {
    struct Case {
        int height, width, out_height, out_width;
    };
    const Case cases[] = {
        {56, 840, 56, 840},       {448, 924, 448, 924},     {468, 931, 476, 924},
        {14, 14, 84, 84},         {27, 300, 28, 308},       {70, 70, 84, 84},
        {42, 42, 84, 84},         {1000, 3, -1, -1},        {5000, 5000, 2184, 2184},
        {3000, 2000, 2688, 1792}, {2193, 2193, 2184, 2184}, {150, 150, 140, 140},
        {98, 182, 112, 168},      {1, 250, -1, -1},         {300, 1, -1, -1},
        {12, 2500, -1, -1},       {140, 210, 140, 224},     {500, 30, 504, 28},
    };
    const auto config = image_config();
    for (const auto& c : cases) {
        const auto label =
            "smart_resize " + std::to_string(c.height) + "x" + std::to_string(c.width);
        if (c.out_height < 0) {
            check(throws([&] { trtmc::glm_ocr::smart_resize(c.height, c.width, config); }), label);
            continue;
        }
        const auto size = trtmc::glm_ocr::smart_resize(c.height, c.width, config);
        check(size.height == c.out_height && size.width == c.out_width, label);
    }
}

void test_resize() {
    std::vector<std::uint8_t> source(4 * 5 * 3);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 5; ++x)
            for (int c = 0; c < 3; ++c)
                source[(y * 5 + x) * 3 + c] =
                    static_cast<std::uint8_t>((y * 29 + x * 13 + c * 71) % 256);
    struct Case {
        int height, width;
        std::vector<int> pixels;
    };
    const Case cases[] = {
        {3, 4, {8,  79,  150, 24, 95,  166, 40,  111, 182, 56,  127, 198,
                46, 117, 188, 62, 133, 203, 78,  149, 225, 94,  165, 245,
                83, 154, 225, 99, 170, 248, 115, 186, 139, 131, 202, 71}},
        {7, 9, {0,   68,  139, 1,   72,  143, 9,   80,  151, 17,  88,  159, 24,  95,  166, 31,
                102, 173, 39,  110, 181, 47,  118, 189, 51,  122, 193, 7,   78,  149, 11,  82,
                153, 19,  90,  161, 27,  98,  169, 34,  105, 176, 41,  112, 183, 49,  120, 191,
                57,  128, 199, 61,  132, 203, 26,  97,  168, 30,  101, 172, 38,  109, 180, 46,
                117, 188, 53,  124, 195, 60,  131, 202, 68,  139, 210, 76,  147, 218, 80,  151,
                222, 42,  114, 185, 47,  118, 189, 55,  126, 197, 63,  134, 204, 70,  141, 212,
                77,  148, 228, 85,  156, 243, 93,  164, 251, 97,  168, 255, 59,  130, 201, 63,
                134, 205, 71,  142, 213, 79,  150, 222, 86,  157, 228, 93,  164, 229, 101, 172,
                231, 109, 180, 239, 113, 184, 244, 78,  149, 220, 82,  153, 224, 90,  161, 232,
                98,  169, 245, 105, 176, 247, 112, 183, 152, 120, 191, 75,  128, 199, 85,  132,
                203, 96,  88,  159, 230, 92,  163, 234, 100, 171, 242, 108, 179, 255, 115, 186,
                255, 122, 193, 107, 130, 201, 0,   138, 209, 0,   142, 213, 8}},
        {4, 2, {11, 82,  153, 41, 112, 183, 40, 111, 182, 70,  141, 212,
                69, 140, 211, 99, 170, 241, 98, 169, 241, 128, 199, 75}},
        {2, 5, {17, 88,  159, 30, 101, 172, 43, 114, 185, 56,  127, 208, 69,  140, 221,
                70, 141, 212, 83, 154, 225, 96, 167, 238, 109, 180, 134, 122, 193, 147}},
    };
    for (const auto& c : cases) {
        const auto out = trtmc::glm_ocr::resize_bicubic(source, 4, 5, c.height, c.width);
        bool same = out.size() == c.pixels.size();
        for (std::size_t i = 0; same && i < out.size(); ++i)
            same = out[i] == c.pixels[i];
        check(same, "resize to " + std::to_string(c.height) + "x" + std::to_string(c.width));
    }
    check(trtmc::glm_ocr::resize_bicubic(source, 4, 5, 4, 5) == source,
          "resize keeps an unchanged size");
}

void test_preprocess() {
    // A 56x840 image keeps its size; each patch row repeats the frame for both temporal slices.
    std::vector<float> pixels(56 * 840 * 3);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<float>((i * 7) % 256) / 255.0F;
    trtmc::internal::ImageView view{pixels.data(),
                                    pixels.size() * sizeof(float),
                                    56,
                                    840,
                                    3,
                                    trtmc::internal::ImageFormat::Float32};
    const auto config = image_config();
    const auto input = trtmc::glm_ocr::preprocess_image(view, config);
    check(input.grid_h == 4 && input.grid_w == 60, "preprocess grid");
    check(input.patches.size() == static_cast<std::size_t>(240) * 1176, "preprocess patch count");
    // Patch 1 is merge-window (0, 0) position (0, 1): rows 0..13, columns 14..27.
    const float* patch = input.patches.data() + 1176;
    const std::size_t source = (0 * 840 + 14) * 3 + 1; // pixel (0, 14), channel 1
    const float value = static_cast<float>((source * 7) % 256);
    const float expected = (value - config.mean[1] * 255.0F) / (config.std[1] * 255.0F);
    check(patch[196 * 2] == expected && patch[196 * 3] == expected,
          "preprocess channel/frame layout");
    // Patch 2 starts the second row of the first merge window: position (1, 0).
    const float* rotary = input.rotary_cos.data() + 2 * 32;
    check(rotary[0] == std::cos(1.0F) && rotary[16] == 1.0F, "preprocess rotary row/column split");
    check(throws([&] {
              auto small = config;
              small.max_patches = 200;
              trtmc::glm_ocr::preprocess_image(view, small);
          }),
          "preprocess enforces the patch budget");
    trtmc::internal::ImageView gray = view;
    gray.channels = 1;
    check(throws([&] { trtmc::glm_ocr::to_rgb8(gray); }), "non-RGB images are refused");
    // Dimensions far beyond the buffer: refused before the pixel buffer is allocated (an
    // allocation of this size would throw std::length_error instead).
    trtmc::internal::ImageView huge = view;
    huge.height = huge.width = static_cast<std::uint32_t>(std::numeric_limits<int>::max());
    check(throws_invalid_argument([&] { trtmc::glm_ocr::to_rgb8(huge); }),
          "an image view larger than its buffer is refused before allocation");
}

void test_pre_tokenize() {
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const Case cases[] = {
        {"Text Recognition:", {"Text", " Recognition", ":"}},
        {"  word", {" ", " word"}},
        {"x 1 2345", {"x", " ", "1", " ", "234", "5"}},
        {"Hello world!!\n\n  next", {"Hello", " world", "!!\n\n", " ", " next"}},
        {"it's DON'T 'll", {"it", "'s", " DON", "'T", " '", "ll"}},
        {"\t\ttab\n", {"\t", "\ttab", "\n"}},
        {"end   ", {"end", "   "}},
        {"a\r\nb", {"a", "\r\n", "b"}},
        {"数学公式 x²+y²=1", {"数学公式", " x", "²", "+y", "²", "=", "1"}},
        {"Ünïcödé ŝtring, ok?", {"Ünïcödé", " ŝtring", ",", " ok", "?"}},
        {" .;\n", {" .;\n"}},
        {"$12.50 (approx.)", {"$", "12", ".", "50", " (", "approx", ".)"}},
        {"line1\n  line2\n\n", {"line", "1", "\n", " ", " line", "2", "\n\n"}},
        {"ſ's 'S", {"ſ", "'s", " '", "S"}},
        {"x'ſun", {"x", "'ſ", "un"}},
        {"a　　b", {"a", "　", "　b"}},
    };
    for (const auto& c : cases) {
        const auto pieces = trtmc::glm_ocr::pre_tokenize(c.text);
        bool same = pieces.size() == c.pieces.size();
        for (std::size_t i = 0; same && i < pieces.size(); ++i)
            same = pieces[i] == c.pieces[i];
        check(same, "pre_tokenize " + c.text);
    }
    check(throws([] { trtmc::glm_ocr::pre_tokenize("bad \xFF byte"); }),
          "invalid UTF-8 is refused");
}

// A miniature tokenizer.json with the checkpoint's model, pre-tokenizer and decoder settings.
const char* kTinyTokenizer = R"json({
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 11, "content": "<|x|>", "single_word": false, "lstrip": false, "rstrip": false,
     "normalized": false, "special": true},
    {"id": 12, "content": "<y>", "single_word": false, "lstrip": false, "rstrip": false,
     "normalized": false, "special": false}],
  "normalizer": null,
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
     "behavior": "Isolated", "invert": false},
    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
  "post_processor": null,
  "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
  "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
    "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": true,
    "vocab": {"a": 0, "b": 1, "c": 2, "Ġ": 3, "ab": 4, "abc": 5, "Ġa": 6, "!": 7,
              "Ċ": 8, "Ã": 9, "©": 10},
    "merges": [["a", "b"], ["ab", "c"]]}
})json";

// kTinyTokenizer with ignore_merges off and two competing merges: (b, c) ranks before (a, b),
// so "abc" must become [a, bc] although it is a vocabulary entry and (a, b) comes first in the
// text. Negative ids are refused.
std::string merge_rank_tokenizer() {
    std::string json = kTinyTokenizer;
    const auto replace = [&json](const std::string& from, const std::string& to) {
        const auto at = json.find(from);
        if (at == std::string::npos)
            throw std::logic_error("fixture text not found: " + from);
        json.replace(at, from.size(), to);
    };
    replace("\"ignore_merges\": true", "\"ignore_merges\": false");
    replace("\"Ã\": 9, \"©\": 10}", "\"Ã\": 9, \"©\": 10, \"bc\": 13}");
    replace("[[\"a\", \"b\"], [\"ab\", \"c\"]]",
            "[[\"b\", \"c\"], [\"a\", \"b\"], [\"ab\", \"c\"]]");
    return json;
}

void test_tokenizer() {
    const trtmc::glm_ocr::Tokenizer tokenizer(kTinyTokenizer);
    using Ids = std::vector<std::int32_t>;
    check(tokenizer.encode("abc") == Ids{5}, "ignore_merges takes a whole vocabulary word");
    const trtmc::glm_ocr::Tokenizer ranked(merge_rank_tokenizer());
    check(ranked.encode("abc") == Ids{0, 13}, "BPE applies the lowest-ranked merge first");
    check(ranked.encode("ab") == Ids{4}, "BPE merges a pair that has a rank");
    std::string negative = kTinyTokenizer;
    negative.replace(negative.find("\"b\": 1"), 6, "\"b\": -1");
    check(throws([&] { trtmc::glm_ocr::Tokenizer bad(negative); }),
          "a negative token id is refused");
    check(tokenizer.encode("ab ab") == Ids{4, 3, 4}, "BPE stops when no merge applies");
    check(tokenizer.encode(" a") == Ids{6}, "ignore_merges takes a whole vocabulary piece");
    check(tokenizer.encode("a<|x|>b<y>") == Ids{0, 11, 1, 12}, "added tokens split the text");
    const Ids ids{5, 11, 12, 8};
    check(tokenizer.decode({ids.data(), ids.size()}, true) == "abc<y>\n", "special tokens skipped");
    check(tokenizer.decode({ids.data(), ids.size()}, false) == "abc<|x|><y>\n",
          "added tokens kept");
    const Ids accent{9, 10}, broken{9, 0};
    check(tokenizer.decode({accent.data(), accent.size()}, true) == "\xC3\xA9",
          "bytes join to UTF-8");
    check(tokenizer.decode({broken.data(), broken.size()}, true) == "\xEF\xBF\xBD"
                                                                    "a",
          "an invalid byte becomes U+FFFD");
    const Ids unknown{5, 13, 4096, -1, 12};
    check(tokenizer.decode({unknown.data(), unknown.size()}, true) == "abc<y>",
          "ids without a token decode to nothing");
    check(tokenizer.added_token_id("<y>") == 12, "added token lookup");
}

trtmc::internal::ImagesTextToTextRequest
request_of(std::vector<std::vector<trtmc::internal::ImagesTextPart>>& parts,
           std::vector<trtmc::internal::ImagesTextMessage>& messages,
           const std::vector<trtmc::internal::MessageRole>& roles) {
    messages.clear();
    for (std::size_t i = 0; i < parts.size(); ++i)
        messages.push_back({roles[i], {parts[i].data(), parts[i].size()}});
    return {{messages.data(), messages.size()}, {}};
}

void test_prompt() {
    using trtmc::internal::ImageView;
    using trtmc::internal::MessageRole;
    using trtmc::internal::TextPartView;
    const float pixel[3] = {0, 0, 0};
    const ImageView image{pixel, sizeof(pixel), 1, 1, 3, trtmc::internal::ImageFormat::Float32};
    std::vector<trtmc::internal::ImagesTextMessage> messages;

    std::vector<std::vector<trtmc::internal::ImagesTextPart>> single{
        {image, TextPartView{"Text Recognition:"}}};
    auto prompt = trtmc::glm_ocr::render_prompt(request_of(single, messages, {MessageRole::User}));
    check(prompt.text == "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>Text "
                         "Recognition:<|assistant|>\n" &&
              prompt.images.size() == 1,
          "single image prompt");

    std::vector<std::vector<trtmc::internal::ImagesTextPart>> turns{
        {TextPartView{"You are OCR."}},
        {image, TextPartView{"A"}},
        {TextPartView{"  spaced  \n"}},
        {TextPartView{"a"}, TextPartView{"b"}, image}};
    prompt = trtmc::glm_ocr::render_prompt(request_of(
        turns, messages,
        {MessageRole::System, MessageRole::User, MessageRole::Assistant, MessageRole::User}));
    check(prompt.text ==
                  "[gMASK]<sop><|system|>\nYou are OCR.<|user|>\n<|begin_of_image|><|image|>"
                  "<|end_of_image|>A<|assistant|>\n<think></think>\nspaced<|user|>\nab<|begin_"
                  "of_image|><|image|><|end_of_image|><|assistant|>\n" &&
              prompt.images.size() == 2,
          "system, assistant history and two images");

    std::vector<std::vector<trtmc::internal::ImagesTextPart>> ends_with_assistant{
        {image, TextPartView{"A"}}, {TextPartView{"B"}}};
    check(throws([&] {
              trtmc::glm_ocr::render_prompt(request_of(
                  ends_with_assistant, messages, {MessageRole::User, MessageRole::Assistant}));
          }),
          "a conversation must end with a user message");
    std::vector<std::vector<trtmc::internal::ImagesTextPart>> image_in_assistant{
        {TextPartView{"A"}}, {image}, {TextPartView{"B"}}};
    check(throws([&] {
              trtmc::glm_ocr::render_prompt(
                  request_of(image_in_assistant, messages,
                             {MessageRole::User, MessageRole::Assistant, MessageRole::User}));
          }),
          "assistant messages carry text only");
}

void test_controls() {
    using trtmc::glm_ocr::check_greedy_controls;
    check(!throws([] { check_greedy_controls(1.0, 1, 1.0); }), "top_k 1 is greedy");
    check(!throws([] { check_greedy_controls(0.7, 1, 0.8); }), "top_k 1 at any temperature");
    check(!throws([] { check_greedy_controls(0.0, 50, 0.9); }), "temperature 0 is greedy");
    check(throws_config_error([] { check_greedy_controls(0.7, 5, 1.0); }), "sampling is rejected");
    check(throws_config_error([] { check_greedy_controls(1.0, 0, 1.0); }),
          "top_k 0 samples everything");
    check(throws_config_error([] { check_greedy_controls(-1.0, 1, 1.0); }), "negative temperature");
    check(throws_config_error([] { check_greedy_controls(std::nan(""), 1, 1.0); }),
          "NaN temperature");
    check(throws_config_error([] { check_greedy_controls(1.0, -1, 1.0); }), "negative top_k");
    check(throws_config_error([] { check_greedy_controls(1.0, 1, 0.0); }), "top_p 0");
}

void test_layout() {
    const std::int32_t image = 59280;
    const std::vector<std::int32_t> ids{11, 12, 13, image, 21, 22, image, 31, 32, 33, 34};
    const auto layout = trtmc::glm_ocr::layout_prompt(ids, image, {{4, 6}, {6, 2}}, 2);
    const std::vector<std::array<std::int32_t, 3>> expected{
        {0, 0, 0}, {1, 1, 1},  {2, 2, 2},    {3, 3, 3},    {3, 3, 4},    {3, 3, 5},
        {3, 4, 3}, {3, 4, 4},  {3, 4, 5},    {6, 6, 6},    {7, 7, 7},    {8, 8, 8},
        {8, 9, 8}, {8, 10, 8}, {11, 11, 11}, {12, 12, 12}, {13, 13, 13}, {14, 14, 14}};
    check(layout.ids.size() == 18 && layout.positions == expected, "mrope positions");
    check(layout.next_position == 15, "first generated position");
    check(layout.image_offsets == std::vector<std::size_t>{3, 11}, "image rows");
    check(throws([&] { trtmc::glm_ocr::layout_prompt(ids, image, {{4, 6}}, 2); }),
          "placeholder count must match the images");

    std::vector<float> cos, sin;
    trtmc::glm_ocr::text_rotary({{5, 6, 35}}, 128, 10000.0F, {16, 24, 24}, cos, sin);
    const auto angle = [](int position, int pair) {
        return position / std::pow(10000.0, 2.0 * pair / 128.0);
    };
    check(std::abs(cos[0] - std::cos(angle(5, 0))) < 1e-6 &&
              std::abs(sin[20] - std::sin(angle(6, 20))) < 1e-6 &&
              std::abs(cos[50] - std::cos(angle(35, 50))) < 1e-6,
          "text rotary reads each section's position");
}

} // namespace

namespace trtmc::glm_ocr::testing {
int run_host_probe(int argc, char** argv); // host_probe.cpp
} // namespace trtmc::glm_ocr::testing

int main(int argc, char** argv) {
    if (argc > 1)
        return trtmc::glm_ocr::testing::run_host_probe(argc, argv);
    test_smart_resize();
    test_resize();
    test_preprocess();
    test_pre_tokenize();
    test_tokenizer();
    test_prompt();
    test_layout();
    test_controls();
    if (failures == 0)
        std::printf("glm_ocr runtime tests passed\n");
    return failures == 0 ? 0 : 1;
}
