#pragma once

#include <cstdint>
#include <string>

namespace trtmc {

inline std::string granite_expand_layer_name(const std::string& pattern, int32_t layer) {
    std::string result = pattern;
    auto replace_all = [&](const std::string& token, const std::string& value) {
        std::size_t pos = 0;
        while ((pos = result.find(token, pos)) != std::string::npos) {
            result.replace(pos, token.size(), value);
            pos += value.size();
        }
    };

    replace_all("{2i+2}", std::to_string(2 * layer + 2));
    replace_all("{2i+1}", std::to_string(2 * layer + 1));
    replace_all("{2i}", std::to_string(2 * layer));
    replace_all("{i}", std::to_string(layer));
    return result;
}

inline std::string granite_layer_tensor_name(const char* stem, int32_t layer) {
    return std::string(stem) + "_" + std::to_string(layer);
}

} // namespace trtmc
