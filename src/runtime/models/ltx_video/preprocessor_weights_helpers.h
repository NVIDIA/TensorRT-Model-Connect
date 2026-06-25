#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace trtmc {
namespace ltx_video_preprocessor_weights {

inline bool extract_preprocessor_index(const std::vector<char>& data, std::string& index_json,
                                       const char*& blob, std::size_t& blob_size) {
    if (data.size() < 4) {
        std::cerr << "[ltx-video] preprocessor_weights section too small\n";
        return false;
    }

    uint32_t index_len = 0;
    std::memcpy(&index_len, data.data(), 4);
    if (4 + index_len > data.size()) {
        std::cerr << "[ltx-video] preprocessor_weights index length overflow\n";
        return false;
    }

    index_json.assign(data.data() + 4, data.data() + 4 + index_len);
    blob = data.data() + 4 + index_len;
    blob_size = data.size() - 4 - index_len;
    return true;
}

inline bool parse_shape_csv(const std::string& csv, std::vector<int32_t>& shape) {
    shape.clear();
    std::istringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        const auto begin = token.find_first_not_of(" \t");
        if (begin == std::string::npos) {
            continue;
        }
        const auto end = token.find_last_not_of(" \t");
        try {
            shape.push_back(std::stoi(token.substr(begin, end - begin + 1)));
        } catch (const std::exception&) {
            return false;
        }
    }
    return true;
}

inline bool parse_offset_after(const std::string& index_json, std::size_t offset_pos,
                               std::size_t& offset) {
    const auto colon = index_json.find(':', offset_pos + 8);
    if (colon == std::string::npos) {
        return false;
    }
    try {
        offset = static_cast<std::size_t>(std::stoul(index_json.substr(colon + 1)));
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

inline bool find_preprocessor_entry(const std::string& index_json, const std::string& key,
                                    std::size_t& offset, std::vector<int32_t>& shape) {
    const std::string search = "\"" + key + "\"";
    const auto pos = index_json.find(search);
    if (pos == std::string::npos) {
        return false;
    }

    const auto off_pos = index_json.find("\"offset\"", pos);
    if (off_pos == std::string::npos || !parse_offset_after(index_json, off_pos, offset)) {
        return false;
    }

    const auto shape_pos = index_json.find("\"shape\"", pos);
    if (shape_pos == std::string::npos) {
        return false;
    }
    const auto bracket = index_json.find('[', shape_pos);
    const auto end_bracket = index_json.find(']', bracket);
    if (bracket == std::string::npos || end_bracket == std::string::npos) {
        return false;
    }

    return parse_shape_csv(index_json.substr(bracket + 1, end_bracket - bracket - 1), shape);
}

inline bool load_preprocessor_floats(const std::string& index_json, const char* blob,
                                     std::size_t blob_size, const std::string& key,
                                     std::vector<float>& dst) {
    std::size_t offset = 0;
    std::vector<int32_t> shape;
    if (!find_preprocessor_entry(index_json, key, offset, shape)) {
        return false;
    }

    std::size_t count = 1;
    for (const auto s : shape) {
        count *= static_cast<std::size_t>(s);
    }
    const std::size_t nbytes = count * sizeof(float);
    if (offset + nbytes > blob_size) {
        std::cerr << "[ltx-video] weight " << key << " overflows blob\n";
        return false;
    }

    dst.resize(count);
    std::memcpy(dst.data(), blob + offset, nbytes);
    return true;
}

inline bool load_with_fallback(const std::string& index_json, const char* blob,
                               std::size_t blob_size, const std::string& primary,
                               const std::string& fallback, std::vector<float>& dst) {
    if (load_preprocessor_floats(index_json, blob, blob_size, primary, dst)) {
        return true;
    }
    if (fallback.empty()) {
        return false;
    }
    return load_preprocessor_floats(index_json, blob, blob_size, fallback, dst);
}

} // namespace ltx_video_preprocessor_weights
} // namespace trtmc
