#pragma once

// Generic chat template registry. Runtime model plugins own the concrete
// template markers and rendering functions they support.

#include <initializer_list>
#include <string>

namespace trtmc {

using ChatTemplateFormat = std::string;
using ChatTemplateApplyFn = std::string (*)(const std::string& prompt, bool enable_thinking);

/// Register a model-owned chat template format.
///
/// Registration is idempotent by id. The first matching marker wins during
/// detection, so plugins should register more-specific formats before
/// broader fallback formats.
void register_chat_template_format(const std::string& id,
                                   std::initializer_list<const char*> markers,
                                   ChatTemplateApplyFn apply);

/// Detect chat template format from the raw chat_template Jinja2 string
/// stored in tokenizer_config.json inside the bundle. Returns an empty id when
/// no registered template matches.
ChatTemplateFormat detect_chat_template_format(const std::string& jinja_template);

/// Apply chat template to a user prompt.
/// @param format  The detected template id.
/// @param prompt  The user's raw prompt text.
/// @param enable_thinking  If false, emit the family-specific no-thinking
///                         marker after the assistant prefix when supported.
/// @return The formatted prompt string ready for tokenization.
std::string apply_chat_template(const ChatTemplateFormat& format, const std::string& prompt,
                                bool enable_thinking = true);

} // namespace trtmc
