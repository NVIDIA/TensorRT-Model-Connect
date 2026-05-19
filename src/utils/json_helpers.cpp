#include "utils/json_helpers.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

enum class ArrayParseState { kReady, kEnd };

bool is_digit_char(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

bool is_space_char(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

bool is_space_or_comma(char c) {
    if (c == ',') {
        return true;
    }
    return is_space_char(c);
}

bool is_int_char(char c) {
    if (is_digit_char(c)) {
        return true;
    }
    return c == '-';
}

bool is_float_char(char c) {
    if (is_digit_char(c)) {
        return true;
    }
    return c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E';
}

std::size_t skip_whitespace(const std::string& text, std::size_t pos) {
    while (pos < text.size() && is_space_char(text[pos])) {
        ++pos;
    }
    return pos;
}

std::size_t skip_space_or_commas(const std::string& text, std::size_t pos) {
    while (pos < text.size() && is_space_or_comma(text[pos])) {
        ++pos;
    }
    return pos;
}

bool find_key_colon(const std::string& text, const std::string& key, std::size_t& colon) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }

    colon = text.find(':', key_pos);
    return colon != std::string::npos;
}

bool find_array_start(const std::string& text, std::size_t colon, std::size_t& open_bracket) {
    open_bracket = text.find('[', colon + 1);
    return open_bracket != std::string::npos;
}

std::size_t scan_while(const std::string& text, std::size_t pos, bool (*is_allowed)(char)) {
    std::size_t end = pos;
    while (end < text.size() && is_allowed(text[end])) {
        ++end;
    }
    return end;
}

ArrayParseState advance_array_pos(const std::string& text, std::size_t& pos) {
    pos = skip_space_or_commas(text, pos);
    if (pos >= text.size()) {
        return ArrayParseState::kEnd;
    }
    if (text[pos] == ']') {
        return ArrayParseState::kEnd;
    }
    return ArrayParseState::kReady;
}

bool append_json_escape(char escaped, std::string& out) {
    switch (escaped) {
    case '"':
        out.push_back('"');
        return true;
    case '\\':
        out.push_back('\\');
        return true;
    case '/':
        out.push_back('/');
        return true;
    case 'b':
        out.push_back('\b');
        return true;
    case 'f':
        out.push_back('\f');
        return true;
    case 'n':
        out.push_back('\n');
        return true;
    case 'r':
        out.push_back('\r');
        return true;
    case 't':
        out.push_back('\t');
        return true;
    default:
        return false;
    }
}

bool read_quoted_token(const std::string& text, std::size_t& pos, std::string& out) {
    if (pos >= text.size() || text[pos] != '"') {
        return false;
    }

    out.clear();
    ++pos;
    while (pos < text.size()) {
        const char c = text[pos++];
        if (c == '"') {
            return !out.empty();
        }
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (pos >= text.size() || !append_json_escape(text[pos++], out)) {
            return false;
        }
    }
    return false;
}

template <typename T, typename Parser>
std::vector<T> extract_numeric_array_impl(const std::string& text, const std::string& key,
                                          std::size_t max_count, bool (*is_allowed)(char),
                                          Parser parse) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return {};
    }

    std::size_t open_bracket = 0;
    if (!find_array_start(text, colon, open_bracket)) {
        return {};
    }

    std::vector<T> out;
    std::size_t pos = open_bracket + 1;
    while (pos < text.size() && out.size() < max_count) {
        if (advance_array_pos(text, pos) != ArrayParseState::kReady) {
            break;
        }

        const std::size_t end = scan_while(text, pos, is_allowed);
        if (end == pos) {
            break;
        }

        if (!parse(text.substr(pos, end - pos), out)) {
            break;
        }
        pos = end;
    }

    return out;
}

} // namespace

std::string extract_json_string(const std::string& text, const std::string& key,
                                const std::string& fallback) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return fallback;
    }

    const std::size_t first_quote = text.find('"', colon + 1);
    if (first_quote == std::string::npos) {
        return fallback;
    }

    std::size_t pos = first_quote;
    std::string parsed;
    if (!read_quoted_token(text, pos, parsed)) {
        return fallback;
    }
    return parsed;
}

std::vector<std::string> extract_json_string_array(const std::string& text,
                                                   const std::string& key) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return {};
    }

    std::size_t open_bracket = 0;
    if (!find_array_start(text, colon, open_bracket)) {
        return {};
    }

    std::vector<std::string> out;
    std::size_t pos = open_bracket + 1;
    while (pos < text.size()) {
        if (advance_array_pos(text, pos) != ArrayParseState::kReady) {
            break;
        }

        std::string parsed;
        if (!read_quoted_token(text, pos, parsed)) {
            break;
        }
        out.push_back(parsed);
    }

    return out;
}

int32_t extract_json_int(const std::string& text, const std::string& key, int32_t fallback) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return fallback;
    }

    const std::size_t pos = skip_whitespace(text, colon + 1);
    const std::size_t end = scan_while(text, pos, is_int_char);

    if (end == pos) {
        return fallback;
    }

    return static_cast<int32_t>(std::stoi(text.substr(pos, end - pos)));
}

int32_t extract_json_int_or_first_array(const std::string& text, const std::string& key,
                                        int32_t fallback) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return fallback;
    }

    std::size_t pos = skip_whitespace(text, colon + 1);

    if (pos < text.size() && text[pos] == '[') {
        pos = skip_whitespace(text, pos + 1);
    }

    const std::size_t end = scan_while(text, pos, is_int_char);

    if (end == pos) {
        return fallback;
    }

    return static_cast<int32_t>(std::stoi(text.substr(pos, end - pos)));
}

float extract_json_float(const std::string& text, const std::string& key, float fallback) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return fallback;
    }

    const std::size_t pos = skip_whitespace(text, colon + 1);
    const std::size_t end = scan_while(text, pos, is_float_char);

    if (end == pos) {
        return fallback;
    }

    try {
        return std::stof(text.substr(pos, end - pos));
    } catch (const std::exception&) {
        return fallback;
    }
}

std::vector<float> extract_json_float_array(const std::string& text, const std::string& key,
                                            std::size_t max_count) {
    auto parse_float = [](const std::string& token, std::vector<float>& out) {
        try {
            out.push_back(std::stof(token));
            return true;
        } catch (const std::exception&) {
            return false;
        }
    };
    return extract_numeric_array_impl<float>(text, key, max_count, is_float_char, parse_float);
}

std::vector<int32_t> extract_json_int_array(const std::string& text, const std::string& key,
                                            std::size_t max_count) {
    auto parse_int = [](const std::string& token, std::vector<int32_t>& out) {
        try {
            out.push_back(static_cast<int32_t>(std::stoi(token)));
            return true;
        } catch (const std::exception&) {
            return false;
        }
    };
    return extract_numeric_array_impl<int32_t>(text, key, max_count, is_int_char, parse_int);
}

namespace {

// Match either JSON literal `true`/`false` or numeric 0/1 at `pos`. Returns
// {parsed_value, end_index} on success; `end == pos` on failure.
struct BoolMatch {
    bool value;
    std::size_t end;
    bool ok;
};

BoolMatch read_bool_token(const std::string& text, std::size_t pos) {
    BoolMatch m{false, pos, false};
    if (pos >= text.size()) {
        return m;
    }
    if (text.compare(pos, 4, "true") == 0) {
        m.value = true;
        m.end = pos + 4;
        m.ok = true;
        return m;
    }
    if (text.compare(pos, 5, "false") == 0) {
        m.value = false;
        m.end = pos + 5;
        m.ok = true;
        return m;
    }
    if (is_digit_char(text[pos])) {
        const std::size_t end = scan_while(text, pos, is_digit_char);
        try {
            const int v = std::stoi(text.substr(pos, end - pos));
            m.value = (v != 0);
            m.end = end;
            m.ok = true;
        } catch (const std::exception&) {
            // fall through
        }
    }
    return m;
}

} // namespace

bool extract_json_bool(const std::string& text, const std::string& key, bool fallback) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return fallback;
    }
    const std::size_t pos = skip_whitespace(text, colon + 1);
    const BoolMatch m = read_bool_token(text, pos);
    if (!m.ok) {
        return fallback;
    }
    return m.value;
}

std::vector<bool> extract_json_bool_array(const std::string& text, const std::string& key,
                                          std::size_t max_count) {
    std::vector<bool> out;
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return out;
    }
    std::size_t open_bracket = 0;
    if (!find_array_start(text, colon, open_bracket)) {
        return out;
    }
    std::size_t pos = open_bracket + 1;
    while (pos < text.size() && out.size() < max_count) {
        if (advance_array_pos(text, pos) != ArrayParseState::kReady) {
            break;
        }
        const BoolMatch m = read_bool_token(text, pos);
        if (!m.ok) {
            break;
        }
        out.push_back(m.value);
        pos = m.end;
    }
    return out;
}

namespace {
// Updates in_string / escape based on c. Returns true if c is inside a string
// (or a quote that toggled string state) and should be ignored by the
// brace-depth tracker.
bool inside_string_state(char c, bool& in_string, bool& escape) {
    if (in_string) {
        if (escape) {
            escape = false;
        } else if (c == '\\') {
            escape = true;
        } else if (c == '"') {
            in_string = false;
        }
        return true;
    }
    if (c == '"') {
        in_string = true;
        return true;
    }
    return false;
}
} // namespace

std::string extract_json_object_text(const std::string& text, const std::string& key) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return "";
    }
    const std::size_t brace = text.find('{', colon + 1);
    if (brace == std::string::npos) {
        return "";
    }
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (std::size_t i = brace; i < text.size(); ++i) {
        const char c = text[i];
        if (inside_string_state(c, in_string, escape)) {
            continue;
        }
        if (c == '{') {
            ++depth;
        } else if (c == '}' && --depth == 0) {
            return text.substr(brace, i - brace + 1);
        }
    }
    return "";
}

} // namespace trtmc
