#pragma once

#include <string>

namespace trtmc {

std::string nemotron_labs_diffusion_detect_chat_template_format(const std::string& jinja_template);
std::string nemotron_labs_diffusion_apply_chat_template(const std::string& format,
                                                        const std::string& prompt,
                                                        bool enable_thinking = true);

} // namespace trtmc
