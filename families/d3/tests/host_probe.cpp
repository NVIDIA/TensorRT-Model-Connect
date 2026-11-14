/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/d3/runtime/batch.h"
#include "families/d3/runtime/media.h"
#include "families/d3/runtime/record.h"
#include "families/d3/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3)
            throw std::invalid_argument("usage: host_probe RECORD [TOKENIZER]");
        if (std::string(argv[1]) == "--tokenize") {
            if (argc != 3)
                throw std::invalid_argument("--tokenize requires TOKENIZER");
            std::ifstream f(argv[2]);
            if (!f)
                throw std::invalid_argument("cannot read tokenizer");
            const std::string bytes{std::istreambuf_iterator<char>(f), {}};
            const std::string text{std::istreambuf_iterator<char>(std::cin), {}};
            auto tokenizer = trtmc::d3::CreateBpeTokenizer(bytes.data(), bytes.size(), false);
            std::cout << trtmc::d3::Record(tokenizer->encode(text)).dump() << '\n';
            return 0;
        }
        if (std::string(argv[1]) == "--batch") {
            using namespace trtmc::d3;
            std::vector<BatchRow> rows{{0, {}, {11, 12}, std::vector<float>(64, 1)},
                                       {1, {}, {21, 22, 23}, std::vector<float>(96, 2)}};
            const auto packed = pack_rows(rows, 0, 2);
            std::vector<bool> visible;
            for (const float value : packed.mask)
                visible.push_back(value == 0);
            VisualInputs a, b;
            a.count = b.count = 2;
            a.indices = {0, 1, 2, 3, 4, 5, 6, 7};
            b.indices = {10, 11, 12, 13, 14, 15, 16, 17};
            a.factors.assign(8, 1);
            b.factors.assign(8, 2);
            a.patches = {31, 32};
            b.patches = {41, 42};
            a.angles = {1, 2};
            b.angles = {3, 4};
            const auto visual = pack_visual({a, b}, 0, 2);
            std::cout << Record{{"ids", packed.ids},
                                {"angles", packed.angles},
                                {"lengths", packed.lengths},
                                {"visible", visible},
                                {"visual_indices", visual.indices},
                                {"visual_factors", visual.factors},
                                {"patches", visual.patches},
                                {"visual_angles", visual.angles}}
                             .dump()
                      << '\n';
            return 0;
        }
        if (std::string(argv[1]) == "--media") {
            trtmc::ImageResult image;
            image.height = image.width = 256;
            image.pixels.resize(256 * 256 * 3);
            for (std::size_t i = 0; i < image.pixels.size(); ++i)
                image.pixels[i] = static_cast<float>(i % 256);
            const auto visual = trtmc::d3::prepare_image(image, 256, 256, 2304, 64);
            std::cout << trtmc::d3::Record{{"patches", visual.patches},
                                           {"indices", visual.indices},
                                           {"weights", visual.factors},
                                           {"angles", visual.angles}}
                             .dump()
                      << '\n';
            return 0;
        }
        std::ifstream file(argv[1]);
        const auto record = trtmc::d3::Record::parse(file);
        std::vector<std::string> codes;
        for (char c = 'A'; c <= 'Z'; ++c)
            codes.emplace_back(1, c);
        for (char a = 'A'; a <= 'Z'; ++a)
            for (char b = 'A'; b <= 'Z'; ++b)
                codes.push_back(std::string(1, a) + b);
        std::unique_ptr<trtmc::d3::ITokenizer> tokenizer;
        if (argc == 3) {
            std::ifstream f(argv[2]);
            const std::string bytes{std::istreambuf_iterator<char>(f), {}};
            tokenizer = trtmc::d3::CreateBpeTokenizer(bytes.data(), bytes.size(), false);
        }
        trtmc::d3::Record result = trtmc::d3::Record::array();
        for (const auto& q : trtmc::d3::questions(record)) {
            const auto text =
                trtmc::d3::prompt(record.value("state", trtmc::d3::Record()), q, codes);
            std::vector<float> p(q.keys.size(), 1.0F / q.keys.size());
            trtmc::d3::Record row = {
                {"prompt", text}, {"answer", trtmc::d3::answer(q, p)}, {"keys", q.keys}};
            if (tokenizer)
                row["ids"] = tokenizer->encode(text);
            result.push_back(row);
        }
        std::cout << result.dump() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
