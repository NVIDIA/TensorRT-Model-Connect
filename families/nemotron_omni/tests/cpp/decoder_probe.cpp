/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native_decoder.h"
#include "trtmc/runtime/trt_backend.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::runtime_error("Usage: probe bundle input.json output.json");
        trtmc::BundleReader reader(argv[1]);
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        if (!backend)
            throw std::runtime_error("Cannot create native MC backend");
        trtmc::FamilyContext context{reader, *backend, 0};
        trtmc::nemotron_omni::NativeDecoder decoder(context);
        std::ifstream input(argv[2]);
        nlohmann::json data;
        input >> data;
        const auto embeddings = data.get<std::vector<std::vector<float>>>();
        nlohmann::json result = nlohmann::json::array();
        for (int round = 0; round < 2; ++round) {
            decoder.reset();
            nlohmann::json output = nlohmann::json::array();
            bool rejected = false;
            try {
                decoder.step({});
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected || decoder.position() != 0)
                throw std::runtime_error("Invalid embedding changed decoder state");
            for (const auto& embedding : embeddings)
                output.push_back(decoder.step(embedding));
            if (decoder.position() != static_cast<int>(embeddings.size()))
                throw std::runtime_error("Decoder did not advance one step per token");
            while (decoder.position() < decoder.capacity())
                decoder.step(embeddings.back());
            rejected = false;
            try {
                decoder.step(embeddings.back());
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("Decoder silently exceeded capacity");
            result.push_back(output);
        }
        std::ofstream output(argv[3]);
        output << result.dump();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
