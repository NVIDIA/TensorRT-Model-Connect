/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU contract of the TrOCR runtime: preprocessing bit for bit against
// Pillow and numpy, the byte-level decoder against Python, the request
// contract and the runtime parameter. No engine and no checkpoint.

#include "families/trocr/runtime/detokenizer.h"
#include "families/trocr/runtime/image_preprocess.h"
#include "families/trocr/runtime/pipeline.h"
#include "families/trocr/tests/cpp/goldens.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

template <class Exception>
void check_throws(const std::function<void()>& call, const std::string& name) {
    try {
        call();
    } catch (const Exception&) {
        return;
    }
    check(false, name + " is rejected");
}

std::vector<std::uint8_t> pattern(int height, int width) {
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(height) * width * 3);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c)
                rgb[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
                    static_cast<std::uint8_t>((y * 31 + x * 17 + c * 7 + (x * y) % 13) % 256);
    return rgb;
}

std::uint32_t checksum(const std::vector<std::uint8_t>& values) {
    std::uint64_t total = 0;
    for (auto value : values)
        total = (total * 1000003 + value) % (std::uint64_t{1} << 32);
    return static_cast<std::uint32_t>(total);
}

trtmc::internal::ImageView view(const std::vector<std::uint8_t>& rgb, int height, int width,
                                std::uint32_t channels = 3) {
    trtmc::internal::ImageView image;
    image.data = rgb.data();
    image.byte_size = rgb.size();
    image.height = static_cast<std::uint32_t>(height);
    image.width = static_cast<std::uint32_t>(width);
    image.channels = channels;
    image.format = trtmc::internal::ImageFormat::UInt8;
    return image;
}

void test_resize_matches_pillow() {
    for (const auto& golden : trocr_goldens::resizes()) {
        const auto source = pattern(golden.height, golden.width);
        const auto resized = trtmc::trocr::resize_bilinear(
            source.data(), golden.height, golden.width, golden.output_height, golden.output_width);
        const auto name = "Pillow bilinear " + std::to_string(golden.height) + "x" +
                          std::to_string(golden.width) + " -> " +
                          std::to_string(golden.output_height) + "x" +
                          std::to_string(golden.output_width);
        check(checksum(resized) == golden.checksum, name);
        if (!golden.values.empty())
            check(resized == golden.values, name + " values");
    }
}

void test_normalization_matches_numpy() {
    // A 1x256 strip holds every 8-bit level; resizing it to its own size is
    // the identity, so the output is the rescale/normalize chain alone.
    std::vector<std::uint8_t> strip(256 * 3);
    for (int level = 0; level < 256; ++level)
        for (int c = 0; c < 3; ++c)
            strip[static_cast<std::size_t>(level) * 3 + c] = static_cast<std::uint8_t>(level);
    trtmc::trocr::PreprocessConfig config;
    config.height = 1;
    config.width = 256;
    const auto values = trtmc::trocr::preprocess(view(strip, 1, 256), config);
    bool exact = values.size() == 3 * 256;
    for (int c = 0; exact && c < 3; ++c) {
        for (int level = 0; level < 256; ++level) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &values[static_cast<std::size_t>(c) * 256 + level], sizeof(bits));
            exact = exact && bits == trocr_goldens::normalized_bits[level];
        }
    }
    check(exact, "rescale and normalize are bit-identical to numpy for every level");
}

void test_channels_and_formats() {
    // Pillow's convert("RGB"): grayscale is replicated, alpha is dropped.
    const std::vector<std::uint8_t> gray{10, 200};
    const auto rgb = trtmc::trocr::to_rgb8(view(gray, 1, 2, 1));
    check(rgb == std::vector<std::uint8_t>({10, 10, 10, 200, 200, 200}), "grayscale is replicated");
    const std::vector<std::uint8_t> rgba{1, 2, 3, 0, 4, 5, 6, 255};
    check(trtmc::trocr::to_rgb8(view(rgba, 1, 2, 4)) ==
              std::vector<std::uint8_t>({1, 2, 3, 4, 5, 6}),
          "alpha is dropped");

    // The CLI decodes 8-bit files and divides by 255; rounding recovers the level.
    std::vector<float> floats(256 * 3);
    for (std::size_t index = 0; index < floats.size(); ++index)
        floats[index] = static_cast<float>(index / 3) / 255.0F;
    trtmc::internal::ImageView image;
    image.data = floats.data();
    image.byte_size = floats.size() * sizeof(float);
    image.height = 1;
    image.width = 256;
    image.format = trtmc::internal::ImageFormat::Float32;
    const auto levels = trtmc::trocr::to_rgb8(image);
    bool recovered = true;
    for (std::size_t index = 0; index < levels.size(); ++index)
        recovered = recovered && levels[index] == index / 3;
    check(recovered, "float32 [0, 1] pixels round back to their 8-bit levels");

    floats[5] = 1.5F;
    check_throws<std::invalid_argument>([&] { trtmc::trocr::to_rgb8(image); },
                                        "a float pixel outside [0, 1]");
    floats[5] = std::numeric_limits<float>::quiet_NaN();
    check_throws<std::invalid_argument>([&] { trtmc::trocr::to_rgb8(image); }, "a NaN pixel");
    const std::vector<std::uint8_t> two(4);
    check_throws<std::invalid_argument>([&] { trtmc::trocr::to_rgb8(view(two, 1, 2, 2)); },
                                        "a two-channel image");
    check_throws<std::invalid_argument>([&] { trtmc::trocr::to_rgb8(view(two, 1, 1, 3)); },
                                        "an image whose bytes do not match its size");
}

void test_utf8_replacement_matches_python() {
    for (const auto& item : trocr_goldens::utf8_cases) {
        const auto decoded =
            trtmc::trocr::utf8_replace_invalid(std::string_view(item.bytes, item.size));
        check(decoded == item.expected, "UTF-8 replacement of " + std::to_string(item.size) +
                                            "-byte case \"" + decoded + "\"");
    }
}

void test_byte_level_decoding() {
    // GPT-2's byte table: U+0120 stands for the space byte, and U+00E2 U+0124
    // U+00AC for E2 82 AC, the three bytes of the euro sign.
    std::vector<std::string> vocabulary{"<s>",
                                        "<pad>",
                                        "</s>",
                                        "<unk>",
                                        "TOTAL",
                                        "\xc4\xa0RM",
                                        "\xc4\xa0"
                                        "25",
                                        ".",
                                        "\xc3\xa2\xc4\xa4\xc2\xac",
                                        "\xc3\xa2\xc4\xa4",
                                        "<mask>"};
    const trtmc::trocr::ByteLevelDetokenizer detokenizer(vocabulary, {0, 1, 2, 3, 10});
    const std::vector<std::int32_t> ids{4, 5, 6, 7, 8, 2};
    check(detokenizer.decode({ids.data(), ids.size()}) == "TOTAL RM 25.\xe2\x82\xac",
          "byte-level tokens decode to text and the end token is skipped");
    const std::vector<std::int32_t> special{0, 4, 3, 10, 1, 5, 2};
    check(detokenizer.decode({special.data(), special.size()}) == "TOTAL RM",
          "every special token is skipped wherever it occurs");
    const std::vector<std::int32_t> truncated{4, 9};
    check(detokenizer.decode({truncated.data(), truncated.size()}) == "TOTAL\xef\xbf\xbd",
          "a token ending inside a character becomes one replacement character");
    const std::vector<std::int32_t> outside{4, 11};
    check_throws<std::out_of_range>([&] { detokenizer.decode({outside.data(), outside.size()}); },
                                    "an id outside the vocabulary");
    check_throws<std::invalid_argument>(
        [&] { trtmc::trocr::ByteLevelDetokenizer(vocabulary, {11}); },
        "a special id outside the vocabulary");
}

trtmc::internal::ImagesTextToTextRequest
request(const std::vector<trtmc::internal::ImagesTextMessage>& messages) {
    return {{messages.data(), messages.size()}};
}

void test_request_contract() {
    using namespace trtmc::internal;
    const std::vector<std::uint8_t> rgb(3, 0);
    const ImageView image = view(rgb, 1, 1);
    const std::vector<ImagesTextPart> one{image, TextPartView{""}};
    const std::vector<ImagesTextMessage> user{{MessageRole::User, {one.data(), one.size()}}};
    check(&trtmc::trocr::single_image(request(user)) == &std::get<ImageView>(one[0]),
          "one image with an empty text part is the request");

    const std::vector<ImagesTextPart> prompt{image, TextPartView{"Read this"}};
    const std::vector<ImagesTextMessage> instructed{
        {MessageRole::User, {prompt.data(), prompt.size()}}};
    check_throws<std::invalid_argument>([&] { trtmc::trocr::single_image(request(instructed)); },
                                        "a text instruction");
    const std::vector<ImagesTextPart> two{image, image, TextPartView{""}};
    const std::vector<ImagesTextMessage> pair{{MessageRole::User, {two.data(), two.size()}}};
    check_throws<std::invalid_argument>([&] { trtmc::trocr::single_image(request(pair)); },
                                        "two images");
    const std::vector<ImagesTextPart> text{TextPartView{""}};
    const std::vector<ImagesTextMessage> blind{{MessageRole::User, {text.data(), text.size()}}};
    check_throws<std::invalid_argument>([&] { trtmc::trocr::single_image(request(blind)); },
                                        "a request without an image");
    const std::vector<ImagesTextMessage> system{{MessageRole::System, {text.data(), text.size()}},
                                                {MessageRole::User, {one.data(), one.size()}}};
    check_throws<std::invalid_argument>([&] { trtmc::trocr::single_image(request(system)); },
                                        "a system message");
    const std::vector<ImagesTextPart> reasoning{image, ReasoningPartView{""}};
    const std::vector<ImagesTextMessage> thought{
        {MessageRole::User, {reasoning.data(), reasoning.size()}}};
    check_throws<std::invalid_argument>([&] { trtmc::trocr::single_image(request(thought)); },
                                        "a reasoning part");
}

void test_max_new_tokens() {
    using trtmc::internal::ConfigEntry;
    const auto fields = trtmc::trocr::config_fields();
    check(fields.size() == 1 && fields[0].name == "max_new_tokens" && !fields[0].default_value,
          "max_new_tokens is the only parameter and its default follows the capacity");
    check(trtmc::trocr::resolve_max_new_tokens({}, 512) == 512, "the default is the capacity");
    const ConfigEntry five[] = {{"max_new_tokens", std::int64_t{5}}};
    check(trtmc::trocr::resolve_max_new_tokens(five, 512) == 5, "an explicit limit is kept");
    for (const std::int64_t value : {std::int64_t{0}, std::int64_t{513}}) {
        const ConfigEntry entry[] = {{"max_new_tokens", value}};
        check_throws<trtmc::internal::ConfigError>(
            [&] { trtmc::trocr::resolve_max_new_tokens(entry, 512); },
            "max_new_tokens=" + std::to_string(value));
    }
}

void test_greedy_token() {
    const std::vector<float> tie{0.5F, 2.0F, -1.0F, 2.0F};
    check(trtmc::trocr::greedy_token(tie.data(), 4) == 1, "a tie picks the first index");
    const std::vector<float> nan{0.5F, std::numeric_limits<float>::quiet_NaN()};
    check_throws<std::runtime_error>([&] { trtmc::trocr::greedy_token(nan.data(), 2); },
                                     "a non-finite logit");
}

} // namespace

int main() {
    test_resize_matches_pillow();
    test_normalization_matches_numpy();
    test_channels_and_formats();
    test_utf8_replacement_matches_python();
    test_byte_level_decoding();
    test_request_contract();
    test_max_new_tokens();
    test_greedy_token();
    if (g_failures != 0) {
        std::cerr << g_failures << " TrOCR runtime contract check(s) failed\n";
        return 1;
    }
    std::cout << "TrOCR runtime contract: all checks passed\n";
    return 0;
}
