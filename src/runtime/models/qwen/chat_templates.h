#pragma once

#include <string>

namespace trtmc {

std::string qwen_detect_chat_template_format(const std::string& jinja_template);
std::string qwen_apply_chat_template(const std::string& format, const std::string& prompt,
                                     bool enable_thinking = true);

} // namespace trtmc
