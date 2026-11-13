/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/laya/runtime/routing.h"

#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("usage: laya_routing_probe TABLES");
        std::ifstream file(argv[1]);
        trtmc::laya::Routing router(trtmc::laya::Json::parse(file));
        std::string line;
        while (std::getline(std::cin, line))
            std::cout << router.route(trtmc::laya::Json::parse(line)).dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
