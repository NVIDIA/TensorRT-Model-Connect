#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

std::string extract_json_string(const std::string& text, const std::string& key,
                                const std::string& fallback);
std::vector<std::string> extract_json_string_array(const std::string& text, const std::string& key);
int32_t extract_json_int(const std::string& text, const std::string& key, int32_t fallback);
int32_t extract_json_int_or_first_array(const std::string& text, const std::string& key,
                                        int32_t fallback);
float extract_json_float(const std::string& text, const std::string& key, float fallback);
std::vector<float> extract_json_float_array(const std::string& text, const std::string& key,
                                            std::size_t max_count = 16);
std::vector<int32_t> extract_json_int_array(const std::string& text, const std::string& key,
                                            std::size_t max_count = 16);

// Parse a JSON literal boolean (true/false) for the given key. Also accepts
// integer 0/1. Returns `fallback` if the key is missing or unparseable.
bool extract_json_bool(const std::string& text, const std::string& key, bool fallback);

// Parse a JSON array of booleans (true/false), returning each element.
std::vector<bool> extract_json_bool_array(const std::string& text, const std::string& key,
                                          std::size_t max_count = 16);

// Extract a nested JSON object's text (including the enclosing braces) for the
// given key. Returns an empty string if the key is absent or the value is not
// an object. Useful for scoping further extractions to a sub-section without
// risking key collisions with siblings.
std::string extract_json_object_text(const std::string& text, const std::string& key);

} // namespace trtmc
