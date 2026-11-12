/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/minicpmv4_7/runtime/prompt.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc::minicpmv4_7 {
namespace {

std::string field(const nlohmann::json& object, const char* name) {
    return object.at(name).get<std::string>();
}

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string repeat(const char* token, int32_t count) {
    std::string out;
    for (int32_t i = 0; i < count; ++i)
        out += token;
    return out;
}

// Span of one image-placeholder run (a crop) in the token sequence: [begin, end).
struct Span2 {
    std::size_t begin, end;
};

struct Frame {
    std::size_t thumb_begin, thumb_end, end;
    std::vector<Span2> slices;
};

struct Group {
    std::size_t begin, end;
    std::vector<Frame> frames;
};

// MiniCPMV4_7Model._group_visual_frames for an image-only sequence (Transformers 5.22.2,
// Apache-2.0; see NOTICE).
std::vector<Group> group_frames(const std::vector<int32_t>& ids, const TokenIds& t) {
    const std::array<int32_t, 5> markers{t.slice_start, t.slice_end, t.image_start, t.image_end,
                                         t.newline};
    auto is_marker = [&](int32_t id) {
        return std::find(markers.begin(), markers.end(), id) != markers.end();
    };
    struct Pending {
        std::size_t begin;
        std::vector<Frame> frames;
    };
    std::vector<Pending> groups;
    Frame* open = nullptr;
    std::size_t previous_end = 0;
    for (std::size_t i = 0; i < ids.size();) {
        if (ids[i] != t.image_pad) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < ids.size() && ids[j] == t.image_pad)
            ++j;
        const Span2 crop{i, j};
        if (crop.begin == 0)
            throw std::invalid_argument("MiniCPM-V 4.7 image tokens must follow a marker");
        if (open != nullptr && ids[crop.begin - 1] == t.slice_start) {
            open->slices.push_back(crop);
        } else {
            bool adjacent = !groups.empty();
            for (std::size_t k = previous_end; adjacent && k + 1 < crop.begin; ++k)
                adjacent = is_marker(ids[k]);
            if (!adjacent)
                groups.push_back({crop.begin - 1, {}});
            groups.back().frames.push_back({crop.begin, crop.end, 0, {}});
            open = &groups.back().frames.back();
        }
        previous_end = crop.end;
        i = j;
    }
    std::vector<Group> result;
    for (auto& pending : groups) {
        for (auto& frame : pending.frames) {
            std::size_t end = frame.slices.empty() ? frame.thumb_end : frame.slices.back().end;
            while (end < ids.size() && is_marker(ids[end]))
                ++end;
            frame.end = end;
        }
        const std::size_t group_end = pending.frames.back().end;
        result.push_back({pending.begin, group_end, std::move(pending.frames)});
    }
    return result;
}

} // namespace

std::string strip(const std::string& text) {
    static const std::vector<std::string> spaces{" ",
                                                 "\t",
                                                 "\n",
                                                 "\r",
                                                 "\v",
                                                 "\f",
                                                 "\x1c",
                                                 "\x1d",
                                                 "\x1e",
                                                 "\x1f",
                                                 "\xc2\x85",
                                                 "\xc2\xa0",
                                                 "\xe1\x9a\x80",
                                                 "\xe2\x80\x80",
                                                 "\xe2\x80\x81",
                                                 "\xe2\x80\x82",
                                                 "\xe2\x80\x83",
                                                 "\xe2\x80\x84",
                                                 "\xe2\x80\x85",
                                                 "\xe2\x80\x86",
                                                 "\xe2\x80\x87",
                                                 "\xe2\x80\x88",
                                                 "\xe2\x80\x89",
                                                 "\xe2\x80\x8a",
                                                 "\xe2\x80\xa8",
                                                 "\xe2\x80\xa9",
                                                 "\xe2\x80\xaf",
                                                 "\xe2\x81\x9f",
                                                 "\xe3\x80\x80"};
    std::size_t begin = 0, end = text.size();
    for (bool found = true; found && begin < end;) {
        found = false;
        for (const auto& space : spaces)
            if (space.size() <= end - begin && text.compare(begin, space.size(), space) == 0) {
                begin += space.size();
                found = true;
                break;
            }
    }
    for (bool found = true; found && begin < end;) {
        found = false;
        for (const auto& space : spaces)
            if (space.size() <= end - begin &&
                text.compare(end - space.size(), space.size(), space) == 0) {
                end -= space.size();
                found = true;
                break;
            }
    }
    return text.substr(begin, end - begin);
}

std::string join_parts(const std::vector<std::string>& parts) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i)
        out += (i ? "\n" : "") + parts[i];
    return out;
}

std::string format_conversation(const nlohmann::json& recipe,
                                const std::vector<ChatMessage>& messages) {
    if (messages.empty())
        throw std::invalid_argument("MiniCPM-V 4.7 requires at least one message");
    std::vector<std::string> contents;
    int64_t last_user = -1;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        contents.push_back(strip(messages[i].content));
        const auto& text = contents.back();
        const bool tool_response =
            starts_with(text, "<tool_response>") && ends_with(text, "</tool_response>");
        if (messages[i].role == Role::User && !tool_response)
            last_user = static_cast<int64_t>(i);
    }
    if (last_user < 0)
        throw std::invalid_argument("MiniCPM-V 4.7 requires a user query");
    std::string output;
    if (messages[0].role != Role::System)
        output = field(recipe, "default_prefix");
    else if (contents[0].empty())
        output = field(recipe, "empty_system_prefix");
    else
        output = field(recipe.at("system"), "prefix") + contents[0] +
                 field(recipe.at("system"), "suffix");
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        if (message.role == Role::System) {
            if (i != 0)
                throw std::invalid_argument("MiniCPM-V 4.7 system message must be first");
            continue;
        }
        if (message.role == Role::User) {
            output += field(recipe.at("user"), "prefix") + contents[i] +
                      field(recipe.at("user"), "suffix");
            continue;
        }
        std::string body = contents[i], reasoning;
        const auto close = body.find("</think>");
        if (close != std::string::npos) {
            reasoning = body.substr(0, close);
            while (!reasoning.empty() && reasoning.back() == '\n')
                reasoning.pop_back();
            const auto open = reasoning.rfind("<think>");
            if (open != std::string::npos)
                reasoning.erase(0, open + 7);
            reasoning.erase(0, reasoning.find_first_not_of('\n'));
            reasoning = strip(reasoning);
            body.erase(0, body.rfind("</think>") + 8);
            const auto first = body.find_first_not_of('\n');
            body.erase(0, first == std::string::npos ? body.size() : first);
        }
        const auto& form = recipe.at(static_cast<int64_t>(i) > last_user ? "assistant_recent"
                                                                         : "assistant_history");
        output += field(form, "prefix");
        if (form.at("include_reasoning").get<bool>() && !reasoning.empty())
            output += field(form, "reasoning_prefix") + reasoning + field(form, "reasoning_suffix");
        else
            output += field(form, "empty_reasoning");
        output += body + field(form, "suffix");
    }
    return output + field(recipe, "generation");
}

// The placeholder replacement of MiniCPMV4_7Processor (Transformers 5.22.2, Apache-2.0).
std::string expand_images(const std::string& prompt, const std::vector<ImageCrops>& images,
                          int32_t merge_factor) {
    const std::string placeholder = kImagePlaceholder;
    std::string output;
    std::size_t cursor = 0, index = 0;
    while (true) {
        const auto found = prompt.find(placeholder, cursor);
        if (found == std::string::npos)
            break;
        if (index == images.size())
            throw std::invalid_argument(
                "MiniCPM-V 4.7 prompt has more image placeholders than images");
        output.append(prompt, cursor, found - cursor);
        const auto& image = images[index];
        output += "<image_id>" + std::to_string(index) + "</image_id><image>" +
                  repeat(kImagePlaceholder, image.crops.front().tokens(merge_factor)) + "</image>";
        if (image.rows > 0) {
            if (image.crops.size() != static_cast<std::size_t>(image.rows) * image.cols + 1)
                throw std::logic_error("MiniCPM-V 4.7 slice count disagrees with its grid");
            for (int32_t r = 0; r < image.rows; ++r) {
                if (r)
                    output += "\n";
                for (int32_t c = 0; c < image.cols; ++c) {
                    const auto& crop =
                        image.crops[1 + static_cast<std::size_t>(r) * image.cols + c];
                    output += "<slice>" + repeat(kImagePlaceholder, crop.tokens(merge_factor)) +
                              "</slice>";
                }
            }
        }
        cursor = found + placeholder.size();
        ++index;
    }
    if (index != images.size())
        throw std::invalid_argument(
            "MiniCPM-V 4.7 prompt has fewer image placeholders than images");
    output.append(prompt, cursor, std::string::npos);
    return output;
}

std::vector<int32_t> canvas_coordinates(int32_t canvas, int32_t count, int32_t offset) {
    // torch.linspace(0, canvas - 1, count) on the GPU: FP32 step, the first half counted from the
    // start and the second half from the end, each as one fused multiply-add.
    std::vector<int32_t> out(static_cast<std::size_t>(count));
    const float start = 0.0F, end = static_cast<float>(canvas - 1);
    if (count == 1) {
        out[0] = offset;
        return out;
    }
    const float step = (end - start) / static_cast<float>(count - 1);
    const int32_t halfway = count / 2;
    for (int32_t i = 0; i < count; ++i) {
        const float value = i < halfway ? std::fma(step, static_cast<float>(i), start)
                                        : std::fma(-step, static_cast<float>(count - i - 1), end);
        out[static_cast<std::size_t>(i)] = static_cast<int32_t>(std::nearbyint(value)) + offset;
    }
    return out;
}

// Ports MiniCPMV4_7Model.get_rope_index and get_vision_position_ids (Transformers 5.22.2,
// Apache-2.0; see NOTICE) for one unpadded image-only sequence.
CanvasPositions canvas_positions(const std::vector<int32_t>& ids, const TokenIds& tokens,
                                 const std::vector<std::array<int32_t, 2>>& crop_grids,
                                 int32_t merge_factor) {
    const std::size_t n = ids.size();
    CanvasPositions result;
    result.positions.resize(n);
    auto set_all = [&](std::size_t index, int32_t value) {
        result.positions[index] = {value, value, value};
    };
    for (std::size_t i = 0; i < n; ++i)
        set_all(i, static_cast<int32_t>(i));
    std::size_t next_grid = 0;
    auto take_grid = [&]() {
        if (next_grid == crop_grids.size())
            throw std::invalid_argument("MiniCPM-V 4.7 prompt has more image crops than grids");
        return crop_grids[next_grid++];
    };
    // Write the grid of one crop: rows of `count_h` x `count_w` LLM tokens.
    auto write_grid = [&](std::size_t begin, std::size_t end, int32_t origin, int32_t count_h,
                          int32_t count_w, int32_t canvas_h, int32_t canvas_w, int32_t h_offset,
                          int32_t w_offset) {
        if (end - begin != static_cast<std::size_t>(count_h) * count_w)
            throw std::invalid_argument("MiniCPM-V 4.7 image tokens disagree with the crop grid");
        const auto rows = canvas_coordinates(canvas_h, count_h, h_offset);
        const auto cols = canvas_coordinates(canvas_w, count_w, w_offset);
        for (int32_t y = 0; y < count_h; ++y)
            for (int32_t x = 0; x < count_w; ++x)
                result.positions[begin + static_cast<std::size_t>(y) * count_w + x] = {
                    origin, origin + rows[static_cast<std::size_t>(y)],
                    origin + cols[static_cast<std::size_t>(x)]};
    };

    int32_t current_pos = 0;
    std::size_t current = 0;
    for (const auto& group : group_frames(ids, tokens)) {
        for (std::size_t i = current; i < group.begin; ++i)
            set_all(i, current_pos++);
        std::size_t cursor = group.begin;
        for (const auto& frame : group.frames) {
            const auto thumb = take_grid();
            const std::size_t frame_start = frame.thumb_begin - 1;
            if (frame_start > cursor) {
                const auto gap = static_cast<int32_t>(frame_start - cursor);
                for (std::size_t i = cursor; i < frame_start; ++i)
                    set_all(i, current_pos + static_cast<int32_t>(i - cursor));
                current_pos += gap + 1;
            }
            const std::size_t frame_end = std::min(frame.end, group.end);
            const int32_t origin = current_pos;
            const int32_t halo = std::max(origin - 1, 0);
            int32_t slice_h = 0, slice_w = 0, rows = 0, cols = 0;
            int32_t canvas_h = thumb[0] / merge_factor, canvas_w = thumb[1] / merge_factor;
            std::array<int32_t, 2> first{};
            const auto& slices = frame.slices;
            if (!slices.empty()) {
                cols = static_cast<int32_t>(slices.size());
                for (std::size_t k = 0; k + 1 < slices.size(); ++k)
                    if (slices[k + 1].begin - slices[k].end > 2) {
                        cols = static_cast<int32_t>(k + 1);
                        break;
                    }
                rows = static_cast<int32_t>(slices.size()) / cols;
                if (static_cast<std::size_t>(rows) * cols != slices.size()) {
                    rows = 1;
                    cols = static_cast<int32_t>(slices.size());
                }
                first = take_grid();
                slice_h = first[0] / merge_factor;
                slice_w = first[1] / merge_factor;
                canvas_h = rows * slice_h;
                canvas_w = cols * slice_w;
            }
            for (std::size_t i = frame_start; i < frame_end; ++i)
                set_all(i, origin);
            result.positions[frame_start][1] = halo;
            result.positions[frame_start][2] = halo;
            if (frame.thumb_end < frame_end) {
                result.positions[frame.thumb_end][1] = origin + canvas_h;
                result.positions[frame.thumb_end][2] = origin + canvas_w;
            }
            write_grid(frame.thumb_begin, frame.thumb_end, origin, thumb[0] / merge_factor,
                       thumb[1] / merge_factor, canvas_h, canvas_w, 0, 0);
            for (std::size_t k = 0; k < slices.size(); ++k) {
                const int32_t h_offset = static_cast<int32_t>(k) / cols * slice_h;
                const int32_t w_offset = static_cast<int32_t>(k) % cols * slice_w;
                const auto grid = k == 0 ? first : take_grid();
                const int32_t sh = grid[0] / merge_factor, sw = grid[1] / merge_factor;
                const std::size_t marker = slices[k].begin - 1;
                if (marker >= frame_start) {
                    result.positions[marker][1] = origin + h_offset;
                    result.positions[marker][2] = origin + w_offset;
                }
                if (slices[k].end < frame_end) {
                    result.positions[slices[k].end][1] = origin + h_offset + sh - 1;
                    result.positions[slices[k].end][2] = origin + w_offset + sw - 1;
                }
                write_grid(slices[k].begin, slices[k].end, origin, sh, sw, sh, sw, h_offset,
                           w_offset);
            }
            for (std::size_t k = 0; k + 1 < slices.size(); ++k) {
                const std::size_t gap_begin = slices[k].end, gap_end = slices[k + 1].begin;
                if (gap_end - gap_begin <= 2)
                    continue;
                const int32_t boundary_h = (static_cast<int32_t>(k) / cols + 1) * slice_h - 1;
                for (std::size_t i = gap_begin + 1; i + 1 < gap_end; ++i) {
                    result.positions[i][1] = origin + boundary_h;
                    result.positions[i][2] = origin + cols * slice_w;
                }
            }
            current_pos = origin + std::max(canvas_h, canvas_w) + 1;
            cursor = frame_end;
        }
        for (std::size_t i = cursor; i < group.end; ++i)
            set_all(i, current_pos++);
        current = group.end;
    }
    for (std::size_t i = current; i < n; ++i)
        set_all(i, current_pos++);
    if (next_grid != crop_grids.size())
        throw std::invalid_argument("MiniCPM-V 4.7 prompt has fewer image crops than grids");
    int32_t maximum = -1;
    for (const auto& position : result.positions)
        maximum = std::max({maximum, position[0], position[1], position[2]});
    result.decode_position = maximum + 1;
    return result;
}

} // namespace trtmc::minicpmv4_7
