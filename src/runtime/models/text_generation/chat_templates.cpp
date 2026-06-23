#include "runtime/models/text_generation/chat_templates.h"

#include "runtime/core/chat_template.h"

#include <string>

namespace trtmc {
namespace {

std::string apply_chatml(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (!enable_thinking)
        r += "<think>\n\n</think>\n\n";
    return r;
}

std::string apply_mistral(const std::string& prompt, bool /*enable_thinking*/) {
    return "[INST] " + prompt + " [/INST]";
}

std::string apply_phi(const std::string& prompt, bool /*enable_thinking*/) {
    return "<|user|>\n" + prompt + "<|end|>\n<|assistant|>\n";
}

std::string apply_gemma(const std::string& prompt, bool /*enable_thinking*/) {
    return "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";
}

std::string apply_llama3(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|begin_of_text|>";
    if (!enable_thinking)
        r += "<|start_header_id|>system<|end_header_id|>\n\ndetailed thinking off<|eot_id|>";
    r += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt +
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return r;
}

std::string apply_nemotron(const std::string& prompt, bool /*enable_thinking*/) {
    return "<extra_id_0>System\n\n<extra_id_1>User\n" + prompt + "\n<extra_id_1>Assistant\n";
}

std::string apply_nemotron_h(const std::string& prompt, bool enable_thinking) {
    std::string r =
        "<SPECIAL_10>System\n\n<SPECIAL_11>User\n" + prompt + "\n<SPECIAL_11>Assistant\n";
    r += enable_thinking ? "<think>\n" : "<think></think>";
    return r;
}

std::string apply_nemotron_labs_diffusion(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" + prompt +
                    "<|im_end|>\n<|im_start|>assistant\n";
    r += enable_thinking ? "<think>\n" : "<think></think>";
    return r;
}

} // namespace

void register_text_generation_chat_templates() {
    register_chat_template_format(
        "nemotron_labs_diffusion",
        {"truncate_history_thinking", "enable_thinking if enable_thinking is defined else False"},
        apply_nemotron_labs_diffusion);
    register_chat_template_format("chatml", {"<|im_start|>"}, apply_chatml);
    register_chat_template_format("mistral", {"[INST]"}, apply_mistral);
    register_chat_template_format("phi", {"<|user|>", "<|assistant|>"}, apply_phi);
    register_chat_template_format("gemma", {"<start_of_turn>"}, apply_gemma);
    register_chat_template_format("llama3", {"<|start_header_id|>"}, apply_llama3);
    register_chat_template_format("nemotron", {"<extra_id_0>"}, apply_nemotron);
    register_chat_template_format("nemotron_h", {"<SPECIAL_10>"}, apply_nemotron_h);
}

} // namespace trtmc
