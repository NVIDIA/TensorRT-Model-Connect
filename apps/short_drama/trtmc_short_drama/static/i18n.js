/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The page speaks English or Chinese. t("English text", { name: value }) looks the text up in the
// current language and fills its {name} slots; text without a translation stays English. Text the
// server writes (stage details, plan checks, job phases) is matched by the patterns below.

const LANGUAGES = { zh: "中文", en: "English" };

let LANG = (() => {
  try {
    const saved = localStorage.getItem("short-drama-lang");
    if (saved in LANGUAGES) return saved;
  } catch (_) { /* private mode */ }
  return (navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en";
})();

function setLanguage(language) {
  LANG = language in LANGUAGES ? language : "en";
  try { localStorage.setItem("short-drama-lang", LANG); } catch (_) { /* private mode */ }
}

function t(text, values = {}) {
  let template = text;
  if (LANG === "zh") {
    template = ZH[text] ?? translatePattern(text) ?? text;
  }
  return template.replace(/\{(\w+)\}/g, (whole, key) => (key in values ? String(values[key]) : whole));
}

function translatePattern(text) {
  for (const [pattern, replacement] of ZH_PATTERNS) {
    if (pattern.test(text)) return text.replace(pattern, replacement);
  }
  return null;
}

// "3 shots" in English, "3 个镜头" in Chinese.
function plural(n, word) {
  if (LANG === "zh") return `${n} ${ZH_UNITS[word] || word}`;
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

const ZH_UNITS = {
  episode: "集", shot: "个镜头", picture: "张图片", "reference picture": "张参考图", "outdated picture": "张过期图片",
  "missing picture": "张缺失图片", problem: "个问题", thing: "项", note: "条意见", finding: "条结论",
  "outdated shot": "个过期镜头", still: "张图",
};

const ZH = {
  // shell and editor
  "Expand": "展开", "Open in a large editor": "在大窗口中编辑", "Text": "文本", "{n} characters": "{n} 字",
  "Apply": "应用", "Cancel": "取消", "Applying puts the text back in the page; save there as usual.": "应用后文字会回到页面里，再像平常一样保存。",
  "Running {step}": "正在运行 {step}", ", episode {n}": "，第 {n} 集", "about {n} s": "约 {n} 秒", "about {n} min": "约 {n} 分钟",
  "just now": "刚刚", "{n} min ago": "{n} 分钟前", "{n} h ago": "{n} 小时前", "Dismiss": "关闭",
  "GPU idle · {models} loaded": "GPU 空闲 · 已加载 {models}", "GPU idle": "GPU 空闲",
  "Loading {models} onto the GPU": "正在把 {models} 加载到 GPU", "{step} failed": "{step}失败", "+{n} queued": "+{n} 排队",
  "could not read the log: {error}": "无法读取日志：{error}", "The model server has not written anything yet.": "模型服务还没有输出日志。",
  "Activity": "活动", "Cancel all": "全部取消", "Close": "关闭", "{gpu} · tier {tier}": "{gpu} · {tier} 档",
  "loaded": "已加载", "loading": "加载中", "Loading {models} for": "正在加载 {models}，已用",
  ". The model server reads each bundle into GPU memory before the step starts.": "。模型服务会在步骤开始前把每个 bundle 读入显存。",
  "The model server failed: {error}": "模型服务出错：{error}", "Now": "当前", "Nothing is running.": "没有正在运行的任务。",
  "Recent": "最近", "No step has run since the app started.": "应用启动后还没有运行过步骤。", "Model server log": "模型服务日志",
  "Show the log": "显示日志", "Loading…": "加载中…",
  "Cancel every queued step and stop the running one at its next check?": "取消所有排队的步骤，并在下一个检查点停止正在运行的步骤？",
  "Leave without saving your edits?": "不保存修改就离开吗？",
  "Light": "浅色", "Dark": "深色", "What the GPU is doing; click for the activity panel": "GPU 正在做什么；点击打开活动面板",
  "Short Drama projects": "短剧项目", "Switch the page language": "切换页面语言",
  // projects
  "Next: write the story from the idea": "下一步：根据创意写故事", "Next: plan the drama": "下一步：规划短剧",
  "Next: reference pictures · {done} of {all} approved": "下一步：参考图 · 已批准 {done}/{all}",
  "Next: shots · {done} of {all} rendered": "下一步：镜头 · 已渲染 {done}/{all}",
  "Next: episodes · {done} of {all} assembled": "下一步：成片 · 已拼接 {done}/{all}",
  "Done · {episodes} assembled": "完成 · 已拼接 {episodes}", "From an idea": "来自创意", "From a novel": "来自小说",
  "{source} · {episodes} of up to {seconds} s · {language}": "{source} · {episodes}，每集最长 {seconds} 秒 · {language}",
  "queued": "排队中", "Changed {when}": "{when}修改", "Rename": "重命名", "Delete": "删除", "New title": "新标题", "Save": "保存",
  "Delete “{title}” with every picture and video in it? This cannot be undone.": "删除「{title}」及其中所有图片和视频？此操作无法撤销。",
  "Deleted “{title}”": "已删除「{title}」", "Find a project": "查找项目", "Projects": "项目",
  "Turn a novel or an idea into vertical episodes. Every model runs on this GPU through TensorRT Model Connect.":
    "把小说或一个创意变成竖屏短剧。所有模型都通过 TensorRT Model Connect 在这块 GPU 上运行。",
  "New project": "新建项目", "Start your first drama": "开始你的第一部短剧",
  "A language model writes the story and plans the episodes, an image model draws the cast and places, a video model films each shot with sound, and Whisper times the subtitles.":
    "语言模型写故事、规划分集，图像模型画角色和场景，视频模型拍出带声音的镜头，Whisper 给字幕定时。",
  // new project
  "Visual style": "画风", "Custom": "自定义", "Language": "语言", "Your idea": "你的创意",
  "A panda cub named Tuantuan finds a glowing seed in the bamboo forest…": "熊猫宝宝团团在竹林里捡到一颗会发光的种子……",
  "Paste the novel": "粘贴小说", "Chapter 1 …": "第一章 …", "Or open a text file": "或打开文本文件",
  "《阿Q正传》 by Lu Xun: nine chapters, about 22,000 characters.": "鲁迅《阿Q正传》：九章，约两万两千字。",
  "for example: watercolor picture-book illustration, soft daylight": "例如：水彩绘本插画，柔和的日光",
  "About {shots} in all (up to {each} per episode). ": "共约 {shots}（每集最多 {each} 个）。",
  "Rendering them takes {time} of GPU time on this machine; you render a sample shot first and decide after.":
    "在这台机器上渲染需要 {time} 的 GPU 时间；会先渲染一个样片，看过再决定。",
  "Rendering time is measured on your first shot, which you render as a sample before the rest.": "渲染时间会在第一个镜头（样片）上实测。",
  "Taken from the idea or the novel when empty": "留空时取自创意或小说",
  "Five choices, then the app takes you through the steps one at a time.": "做完五个选择，应用会一步一步带你完成。",
  "Where does the story come from?": "故事从哪里来？",
  "The language model writes the story, the plan, the lines and every review in this language, and the characters speak it.":
    "语言模型用这种语言写故事、规划、台词和审阅意见，角色也说这种语言。",
  "Every reference picture and shot is drawn in this style. You can change it later, before drawing.": "所有参考图和镜头都按这个画风生成。画图之前还可以改。",
  "Describe the style in any language; English works best for the models": "用任何语言描述画风；模型对英文最熟",
  "Length": "长度", "Seconds per episode, at most": "每集最长秒数", "Title": "标题", "Optional.": "可选。",
  "Create project": "创建项目",
  "idea": "创意", "An idea": "一个创意", "One or two sentences; the language model writes the story first.": "一两句话；语言模型先写成故事。",
  "novel": "小说", "My novel": "我的小说", "Paste the text or open a .txt file.": "粘贴文本或打开 .txt 文件。",
  "sample": "示例", "The sample novel": "示例小说", "《阿Q正传》 by Lu Xun, public domain.": "鲁迅《阿Q正传》，公有领域。",
  "Auto": "自动", "The planner picks a style that suits the story.": "由规划模型选一个适合故事的画风。",
  "Live action": "真人实拍", "3D cartoon": "3D 卡通", "Anime": "日式动画", "Ink painting": "水墨",
  // settings
  "Another OpenAI-compatible API": "其他 OpenAI 兼容 API", "empty: measure only; or a model that reads pictures, such as the chat model above": "留空：只做本机测量；也可以填能看图的模型，比如上面的聊天模型",
  "saved; type to replace": "已保存；输入新值替换", "paste your key": "粘贴你的 key", "Get a free key": "获取免费 key", "current models": "当前模型",
  "Any API that serves /chat/completions the OpenAI way.": "任何按 OpenAI 方式提供 /chat/completions 的 API。",
  "Saved": "已保存", "Using {api} · {model}": "正在使用 {api} · {model}", "key saved": "已保存 key", "no key": "没有 key",
  "Using {model} on this GPU": "正在使用本机 GPU 上的 {model}",
  "No language model yet: pick an API, paste your key and save.": "还没有语言模型：选择一个 API，粘贴 key 并保存。",
  "Settings": "设置", "The language model that writes and plans, and the models this GPU serves.": "负责写作和规划的语言模型，以及这块 GPU 上运行的模型。",
  "Language model": "语言模型",
  "It writes stories from ideas, plans the episodes, reviews and revises. The story is sent to the API you choose; start the app with --llm local to keep it on this machine. The key is stored with mode 600 in the data directory and never shown again.":
    "它根据创意写故事、规划分集、审阅和修改。故事会发送到你选择的 API；用 --llm local 启动可以只在本机处理。key 以 600 权限存在数据目录里，不会再显示。",
  "Base URL": "Base URL", "Model": "模型", "API key": "API key", "Vision model for video analysis (optional)": "用于视频分析的视觉模型（可选）",
  "Video analysis sends frames of a clip to a model that reads pictures. Without one, it checks the sound, the spoken lines and the continuity of the frames on this machine only.":
    "视频分析会把片段的帧图发给能看图的模型。没有配置时，只在本机检查声音、台词和画面衔接。",
  "Use {model} on this GPU": "使用本机 GPU 上的 {model}", "Models on this GPU": "这块 GPU 上的模型",
  "A step loads the models it needs; the page shows the load while it happens.": "每个步骤会加载需要的模型；加载过程会显示在页面上。",
  "Used for": "用途", "Bundle": "Bundle", "Last load failed: {error}": "上次加载失败：{error}", "Show the model server log": "显示模型服务日志",
  "Hardware tiers": "硬件档位", "The app picks the tier by GPU memory. Lower tiers keep the same steps with smaller models.":
    "应用按显存自动选档。低档位步骤相同，换用更小的模型。",
  "Tier": "档位", "GPU memory": "显存", "Shots": "镜头", "Speech": "台词", "Status": "状态", "this GPU": "本机",
  "{n} GB or more": "{n} GB 及以上", "ready": "可用", "planned": "规划中",
  "Free for development with an NVIDIA Developer Program account; requests per minute are limited.": "加入 NVIDIA 开发者计划即可免费用于开发；每分钟请求数有限制。",
  "Models ending in :free cost nothing: 20 requests a minute and 50 a day, 1,000 a day after buying $10 of credits once. Some need free-model training allowed in the privacy settings.":
    "以 :free 结尾的模型免费：每分钟 20 次、每天 50 次，一次性充值 10 美元后每天 1000 次。部分模型需要在隐私设置里允许免费模型训练。",
  // tiers from the server
  "MiniMax H3 reference-to-video with native audio": "MiniMax H3 参考图生视频，自带声音",
  "keyframes from the reference pictures, Wan2.2-TI2V-5B image-to-video and text-to-speech": "按参考图生成关键帧，Wan2.2-TI2V-5B 图生视频，语音合成配音",
  "keyframes and Wan2.2-TI2V-5B image-to-video with text-to-speech": "关键帧加 Wan2.2-TI2V-5B 图生视频，语音合成配音",
  "MiniMax H3 films each shot from up to 6 reference pictures": "MiniMax H3 用最多 6 张参考图拍出每个镜头",
  "native audio from the video model": "视频模型自带声音",
  "Qwen-Image-Edit draws each shot's first frame from the reference pictures; Wan2.2-TI2V-5B animates it": "Qwen-Image-Edit 按参考图画出每个镜头的首帧，Wan2.2-TI2V-5B 让它动起来",
  "FLUX.2 klein 4B draws each shot's first frame from the reference pictures; Wan2.2-TI2V-5B animates it": "FLUX.2 klein 4B 按参考图画出每个镜头的首帧，Wan2.2-TI2V-5B 让它动起来",
  "text-to-speech, one voice per character": "语音合成，每个角色一个音色",
  "image-to-video for Wan2.2-TI2V-5B in TRTMC (its family renders from text only)": "TRTMC 的 Wan2.2-TI2V-5B 还不支持图生视频（目前只能文生视频）",
  "Chinese text-to-speech in TRTMC (Qwen3-TTS or CosyVoice3) and a speech endpoint in trtmc-server": "TRTMC 的中文语音合成（Qwen3-TTS 或 CosyVoice3），以及 trtmc-server 的语音接口",
  "keyframes from reference pictures in the app (Qwen-Image-Edit-2511 is in TRTMC)": "应用里按参考图生成关键帧（TRTMC 已支持 Qwen-Image-Edit-2511）",
  "FLUX.2 klein 4B in TRTMC, with reference pictures as input": "TRTMC 支持 FLUX.2 klein 4B，并能输入参考图",
  // roles, scopes, statuses
  "llm": "语言模型", "image": "画图", "video": "视频", "asr": "语音识别", "tts": "语音合成", "vision": "看图",
  "project": "项目", "episode": "分集", "running": "运行中", "done": "完成", "failed": "失败", "cancelled": "已取消",
  "error": "错误", "warning": "提醒", "estimated": "估计", "the {role} model": "{role}模型",
  // steps
  "Writing the story": "写故事", "Planning the drama": "规划短剧", "Revising the plan": "修改规划", "Reviewing the plan": "审阅规划",
  "Drawing reference pictures": "画参考图", "Rendering shots": "渲染镜头", "Timing subtitles": "字幕定时", "Assembling the episode": "拼接成片",
  "Analyzing the video": "分析视频", "Applying the analysis to the plan": "把分析结果写回规划",
  "Story": "故事", "Plan": "规划", "Reference pictures": "参考图", "Episodes": "成片", "Number of episodes": "集数",
  "Character": "角色", "Place": "场景", "Prop": "道具",
  "Read the novel and write characters, places, episodes and shot lists": "读小说，写出角色、场景、分集和分镜",
  "Draw a reference still for every character, place and prop": "为每个角色、场景和道具画参考图",
  "Render each shot from its approved reference stills, with sound": "用已批准的参考图渲染每个镜头，带声音",
  "Check every line with speech recognition and time its subtitle": "用语音识别检查每句台词并给字幕定时",
  "Join the clips, burn in subtitles and export the episode": "拼接片段、压制字幕并导出成片",
  "Write a short story from the project's idea, to plan from": "根据项目创意写一个短故事，用来做规划",
  "Rewrite one shot, character, location or prop from a note": "按意见改写一个镜头、角色、场景或道具",
  "Ask the language model to review every episode's shots before rendering": "渲染前请语言模型审阅每集的分镜",
  "Measure the sound, the lines, the picture and the cuts of the rendered shots, and ask a vision model what it sees":
    "测量已渲染镜头的声音、台词、画面和衔接，并请视觉模型看画面",
  // project page
  "Started: {what}": "已开始：{what}", "rendering, subtitles and assembly": "渲染、字幕和拼接", "{time} of GPU time": "{time} GPU 时间",
  "the time is measured on the first one": "时间会在第一个上实测", "{what}: {time}. Continue?": "{what}：{time}。继续吗？",
  "from your idea": "来自你的创意", "Write the story first": "先写故事", "not planned yet": "还没有规划", "Plan the drama first": "先规划短剧",
  "Approve the pictures first": "先批准参考图", "Render the shots first": "先渲染镜头", "Steps": "步骤", "running…": "运行中…",
  "{source} · {episodes} of up to {seconds} s · written in {language} · {style}": "{source} · {episodes}，每集最长 {seconds} 秒 · {language} · {style}",
  "style: {style}": "画风：{style}", "style chosen when planning": "规划时选择画风", "Flow (advanced)": "流程（高级）",
  "A new title for this project": "这个项目的新标题",
  "Waiting for the GPU: “{title}” is busy with {step} first.": "等待 GPU：「{title}」正在{step}。", "Starting…": "正在启动…",
  "Loading {models} onto the GPU. A step first loads the models it needs; this takes a minute or more.": "正在把 {models} 加载到 GPU。每个步骤先加载需要的模型，通常要一分钟以上。",
  "Running now": "正在运行", "usually {time}": "通常{time}", "Activity and log": "活动和日志", "Failed": "失败",
  "no reason was given": "没有给出原因", "Try again": "重试", "All done": "全部完成", "Next · step {n} of 5": "下一步 · 第 {n} 步，共 5 步",
  "Choose a language model": "选择语言模型",
  "It writes the story and plans the drama. The NVIDIA API catalog and OpenRouter have free tiers: pick one in Settings and paste a key.":
    "它负责写故事和规划短剧。NVIDIA API catalog 和 OpenRouter 有免费额度：在设置里选一个并粘贴 key。",
  "Open Settings": "打开设置", "Write the story from your idea": "根据你的创意写故事",
  "The language model turns the idea into a short story, one chapter per episode. It takes about a minute, and you can edit the story afterwards.":
    "语言模型把创意写成短故事，每集一章。大约一分钟，写完可以修改。",
  "Write the story": "写故事", "Plan the drama": "规划短剧",
  "The language model reads the story ({n} characters) and writes the cast, places, props and a shot list for each episode. It takes a few minutes; you can edit every part afterwards.":
    "语言模型读完故事（{n} 字），写出角色、场景、道具和每集的分镜。需要几分钟，之后每部分都可以改。",
  "Fix {problems} in the plan": "修正规划里的 {problems}", "The checks list them all on the Plan step.": "规划步骤里列出了全部问题。",
  "Show the problems": "查看问题", "Draw {pictures}": "画 {pictures}",
  "The image model ({model}) draws every character, place and prop once in the plan's style; each shot is filmed from these pictures.":
    "图像模型（{model}）按规划的画风把每个角色、场景和道具各画一次；每个镜头都以这些图为参考拍摄。",
  "The first one also loads the image model onto the GPU.": "第一张还会先把图像模型加载到 GPU。",
  "Redraw {pictures}": "重画 {pictures}", "Their description or the style changed after they were drawn.": "画完之后描述或画风改了。",
  "Redraw {n}": "重画 {n} 张", "Look at them": "看一看", "Check and approve {pictures}": "检查并批准 {pictures}",
  "Shots use approved pictures only. Look at each one: approve it, or redraw it with a new seed or your own prompt.":
    "镜头只用已批准的图。逐张看：批准，或换种子、改提示词重画。",
  "Approve all {n}": "全部批准（{n} 张）", "Look at them first": "先看一看", "{time} of GPU time.": "需要 {time} GPU 时间。",
  "The first shot also loads the video model; then each shot takes several minutes, and this card shows the time as it runs.":
    "第一个镜头还要加载视频模型；之后每个镜头需要几分钟，这张卡片会显示实时用时。",
  "Render a sample shot": "渲染一个样片",
  "Render {shot} ({seconds} s{line}) first, to check the look and the voices before the other {others}.":
    "先渲染 {shot}（{seconds} 秒{line}），看看画面和声音，再渲染其余 {others}。",
  "Render the sample": "渲染样片", "Re-render {shots} of episode {n}": "重新渲染第 {n} 集的 {shots}",
  "Something they were rendered from changed.": "它们依据的内容有了变化。", "Re-render {n}": "重新渲染 {n} 个",
  "Re-render {n} outdated shots": "重新渲染 {n} 个过期镜头", "Render {shots} of episode {n}": "渲染第 {n} 集的 {shots}",
  "Or render them, time the subtitles and assemble the episode in one go.": "也可以一次完成渲染、字幕定时和拼接。",
  "Render {shots}": "渲染 {shots}", "Render {n} shots": "渲染 {n} 个镜头", "Render, subtitle and assemble": "渲染、字幕、拼接一次完成",
  "Subtitle and assemble episode {n}": "给第 {n} 集定字幕并拼接",
  "The speech model ({model}) times each line from the voice in its shot, then the clips are joined into one 9:16 video.":
    "语音模型（{model}）根据镜头里的声音给每句台词定时，然后把片段拼成一个 9:16 视频。",
  "Subtitle and assemble": "定字幕并拼接", "The drama is ready": "短剧完成了",
  "{episodes} assembled. Watch and download them, analyze the video and send the findings back into the plan, or re-render any shot and assemble again.":
    "已拼接 {episodes}。可以观看下载，也可以分析视频、把结论写回规划，或重渲染任意镜头后再拼接。",
  "Watch and download": "观看和下载", "Render {n} shots, then subtitle and assemble": "渲染 {n} 个镜头，然后定字幕并拼接",
  "{pictures} approved": "已批准 {pictures}", "Go to {step}": "前往{step}", "What should change about {name}?": "{name} 要怎么改？",
  "Revise with the model": "让模型修改", "Revise": "修改", "Remove": "删除", "Add": "添加", "You have unsaved edits.": "有未保存的修改。",
  "Discard": "放弃", "Save edits": "保存修改", "Plan saved": "规划已保存",
  "Save your edits first: the model revises the saved plan.": "请先保存修改：模型修改的是已保存的规划。",
  "The story appears here once it is written.": "故事写好后会显示在这里。", "The story": "故事", "Story saved": "故事已保存",
  "Save the story": "保存故事",
  "To use an edited story, plan again on the Plan step; the current plan is kept under Earlier versions.": "要使用修改后的故事，请在规划步骤重新规划；当前规划会保存在历史版本里。",
  "Edit freely; the plan is written from the saved story.": "随意修改；规划依据保存后的故事。",
  "{things} to check before rendering": "渲染前有 {things}需要检查", "{n} must be fixed": "{n} 项必须修正",
  "The plan checks found nothing to fix": "规划检查没有发现问题", "Review again with the model": "让模型再审一遍",
  "Ask the model to review": "请模型审阅", "Model review: {notes}": "模型审阅：{notes}", "no problems": "没有问题",
  "written for an earlier version of the shots": "针对的是之前版本的分镜", "Fix": "修改建议", "Revise this shot": "修改这个镜头",
  "Earlier versions of the plan ({n})": "规划的历史版本（{n}）", "Restore this version? The current plan is kept as a version too.": "恢复这个版本？当前规划也会保存为一个版本。",
  "Restore": "恢复", "Note for the model": "给模型的意见",
  "What should change? For example: let him touch his head before he speaks, and keep the line shorter": "要怎么改？例如：让他先摸摸头再开口，台词再短一些",
  "line from the novel": "原著台词", "adapted line": "改编台词", "{n} to check": "{n} 项待查", "Remove shot": "删除镜头",
  "Seconds": "秒数", "Characters": "角色", "Props": "道具", "Speaker": "说话人", "Summary": "概要", "Line": "台词",
  "Camera": "镜头运动", "Action": "动作", "After": "之后", "Then": "接着", "Sound": "声音",
  "The plan is written from the story; write the story first.": "规划依据故事；请先写故事。",
  "Plan again from the story? The current plan is kept under Earlier versions.": "根据故事重新规划？当前规划会保存在历史版本里。",
  "Plan again from the story": "根据故事重新规划", "Run {step}": "运行 {step}", "{step}: not installed, skipped": "{step}：未安装，已跳过",
  "Edit anything below, in any language, then save; the checks run again on every save.": "下面的内容都可以用任何语言修改，然后保存；每次保存都会重新检查。",
  "Title and setting": "标题和背景", "Setting": "背景", "The visual style is set on the Reference pictures step.": "画风在参考图步骤里设置。",
  "Each character's look and voice go into the picture and video prompts.": "每个角色的外貌和声音会写进画图和视频的提示词。",
  "Relationships": "人物关系", "Places": "场景", "Episode {n}": "第 {n} 集", "Add shot": "添加镜头",
  "id": "id", "name": "名字", "role": "身份", "look": "外貌", "voice": "声音", "from": "从", "to": "到", "relation": "关系",
  // pictures
  "Style saved. Pictures drawn in the old style are marked outdated.": "画风已保存。旧画风画的图已标为过期。",
  "Every picture and every shot prompt starts from this style. Pick one, or describe your own in any language; the presets are English, which the models follow best.":
    "每张图和每个镜头的提示词都从这个画风开始。选一个预设，或用任何语言自己描述；预设是英文的，模型最熟悉。",
  "Style": "画风", "Save style": "保存画风", "Prompt of {name}": "{name} 的提示词", "Drawing…": "正在画…", "Not drawn yet": "还没画",
  "outdated": "过期", "approved": "已批准", "to check": "待检查", "{kind} · in {shots}": "{kind} · 出现在 {shots}", "and {n} more": "等另外 {n} 个",
  "Drawn before its description or the style changed; redraw it.": "描述或画风改过，需要重画。", "Unapprove": "取消批准", "Approve": "批准",
  "Redraw": "重画", "Draw": "画", "Prompt": "提示词", "Keep this prompt for the next drawing": "下次画图使用这个提示词",
  "Pictures are drawn from the plan's cast, places and props; plan the drama first.": "参考图依据规划里的角色、场景和道具；请先规划短剧。",
  "Pictures": "参考图", "{done} of {all} approved": "已批准 {done}/{all}", "Draw {n} pictures": "画 {n} 张图",
  "Only what appears in a shot gets a picture. A shot renders once every picture it uses is approved.": "只有镜头里出现的才画参考图。一个镜头用到的图全部批准后才能渲染。",
  "Not in any shot, so not drawn: {names}.": "没有出现在任何镜头里，所以不画：{names}。",
  // shots
  "rendering": "渲染中", "Shots come from the plan; plan the drama first.": "镜头来自规划；请先规划短剧。",
  "{pictures} still to approve: a shot renders once every picture it uses is approved.": "还有 {pictures}待批准：镜头用到的图全部批准后才能渲染。",
  "Episode {n}: {title}": "第 {n} 集：{title}", "{done} of {all} rendered": "已渲染 {done}/{all}", "{time} of GPU time per shot here": "本机每个镜头 {time} GPU 时间",
  "Render a sample: {shot}": "渲染样片：{shot}", "Render the other {n}": "渲染其余 {n} 个", "Render all {n}": "全部渲染（{n} 个）",
  "Re-render {n} outdated": "重新渲染 {n} 个过期镜头",
  "Use the last frame of episode {n} as an extra reference for the first shot": "把第 {n} 集的最后一帧作为第一个镜头的额外参考",
  "default: on, same place": "默认：开，同一场景", "default: off, the place changes": "默认：关，场景变了",
  "Open each shot where the shot before it ends (same place): its last frame becomes an extra reference. Render in order; a new take of a shot marks the next one outdated.":
    "同一场景里，每个镜头从前一个镜头结束的地方开始：前一镜的最后一帧作为额外参考。请按顺序渲染；某个镜头重拍后，下一个镜头会被标为过期。",
  "Video prompt of {shot}": "{shot} 的视频提示词", "Rendering…": "正在渲染…", "Not rendered yet": "还没渲染", "take {n}": "第 {n} 条",
  "Outdated: {why}.": "已过期：{why}。", "no line": "没有台词",
  "heard “{heard}” · match {match}% · {start}–{end} s ({timing})": "识别到「{heard}」· 匹配 {match}% · {start}–{end} 秒（{timing}）",
  "Frames of {shot}": "{shot} 的帧", "Approve the pictures this shot uses first": "请先批准这个镜头用到的图",
  "Render another take": "再渲染一条", "Render": "渲染", "Video prompt of this take": "这一条的视频提示词",
  "Video prompt (preview)": "视频提示词（预览）", "Keep this prompt for the next take": "下一条使用这个提示词",
  "a picture it used was redrawn or replaced": "用到的图被重画或替换了",
  "the shot before it has a new take, so it no longer opens where that one ends": "前一个镜头换了新的一条，这个镜头已经接不上它的结尾",
  "its prompt changed since this take (the plan, a kept prompt or the app's template)": "这一条之后提示词变了（规划、保留的提示词或应用模板）",
  // analysis
  "Video analysis": "视频分析", "Analyze again": "重新分析", "Analyze the video": "分析视频",
  "Checks the sound, the spoken lines, the picture and the cuts of every rendered shot. ": "检查每个已渲染镜头的声音、台词、画面和镜头衔接。",
  "{model} also looks at the frames, the reference pictures and the previous shot's last frame.": "{model} 还会看帧图、参考图和上一镜的最后一帧。",
  "Set a vision model in Settings to also check content, on-screen text, continuity and whether people match their pictures.":
    "在设置里配置视觉模型后，还能检查内容、画面文字、连续性，以及人物是否和参考图一致。",
  "Analyzing…": "分析中…", "Suggestion for {target}": "{target} 的修改建议", "seen by {model}": "由 {model} 发现",
  "Suggestion (edit freely, in any language)": "修改建议（可用任何语言自由修改）", "Your own note": "你自己的意见",
  "Your own note for the model, in any language": "写给模型的意见，任何语言都可以", "new place": "换场景",
  "Analyzed {when} · {findings}": "{when}分析 · {findings}", "vision model: {model}": "视觉模型：{model}",
  "measured on this machine only": "仅本机测量", "picture change at the cuts: {cuts}": "切镜处画面变化：{cuts}",
  "Some shots have a newer take than the one analyzed; analyze again.": "有些镜头在分析后换了新的一条；请重新分析。",
  "The vision model failed on: {errors}": "视觉模型出错：{errors}", "No problems found.": "没有发现问题。", "Change": "要改的内容",
  "Apply the selected changes": "应用选中的修改",
  "Notes revise the plan with the language model; the shots and pictures they touch become outdated, and the next card offers to render them again.":
    "意见会交给语言模型修改规划；涉及的镜头和参考图会被标为过期，下一步卡片会提示重新渲染。",
  "Select at least one change, or write your own note.": "请至少选一项修改，或写下你自己的意见。",
  "Queued {n} changes. Revised shots and pictures become outdated; render them again from the next card.": "已排队 {n} 项修改。改过的镜头和参考图会标为过期，请从下一步卡片重新渲染。",
  "Revise the plan": "修改规划", "Redraw the picture": "重画参考图", "Ignore": "忽略",
  "Picture": "画面", "Content": "内容", "On-screen text": "画面文字", "Continuity": "连续性", "Consistency": "一致性", "Reference picture": "参考图",
  // episodes
  "Episodes are assembled from rendered shots; plan the drama first.": "成片由已渲染的镜头拼接；请先规划短剧。",
  "Ready to assemble": "可以拼接了", "{done} of {all} shots rendered": "已渲染 {done}/{all} 个镜头", "analysis: {findings}": "分析：{findings}",
  "Download": "下载", "Subtitle and assemble again": "重新定字幕并拼接", "Time subtitles only": "只做字幕定时", "Render its shots": "渲染它的镜头",
  // flow
  "Flow": "流程", "what changed": "改了什么",
  "A flow lists the steps, the model of each and its parameters, as numbered versions. Most projects never need to change it.":
    "流程列出各个步骤、每步的模型和参数，按编号保存版本。大多数项目不需要改。",
  "Use this version for the project": "这个项目使用此版本", "Flow name": "流程名", "Note": "说明", "Saved {flow}": "已保存 {flow}",
  "Save as a new version": "另存为新版本", "Installed steps": "已安装的步骤", "action · {step} model": "动作 · 使用 {step} 的模型",
  "Reload plugins": "重新加载插件",
  "A plugin is one Python file in the plugins directory that defines STEP. Add its name to a flow version to run it; delete the file and reload to remove it. Actions (story, revise, review, analyze) run with the model of a flow step.":
    "插件是 plugins 目录里定义了 STEP 的一个 Python 文件。把它的名字加进某个流程版本就会运行；删除文件并重新加载即可移除。动作（story、revise、review、analyze）使用某个流程步骤的模型。",
};

// Text the server writes, matched when no exact translation exists.
const ZH_PATTERNS = [
  [/^(\d+) of (\d+) approved$/, "已批准 $1/$2"], [/^(\d+) of (\d+) drawn$/, "已画 $1/$2"],
  [/^(\d+) of (\d+) rendered$/, "已渲染 $1/$2"], [/^(\d+) of (\d+) assembled$/, "已拼接 $1/$2"],
  [/^(\d+) approved$/, "已批准 $1 张"], [/^(\d+) rendered$/, "已渲染 $1 个"],
  [/^(\d+) episodes? · (\d+) shots? · (\d+) cast$/, "$1 集 · $2 个镜头 · $3 个角色"],
  [/^Draw (\d+) stills?$/, "画 $1 张图"], [/^Approve (\d+) stills?$/, "批准 $1 张图"],
  [/^Re-render (\d+) outdated$/, "重新渲染 $1 个过期镜头"], [/^Render (\d+) more$/, "再渲染 $1 个"],
  [/^Assemble episode (\d+)$/, "拼接第 $1 集"], [/^Fix (\d+) problems? in the plan$/, "修正规划里的 $1 个问题"],
  [/^episode (\d+)$/, "第 $1 集"], [/^has no shots$/, "没有镜头"],
  [/^runs (\S+) s, over the (\S+) s limit$/, "时长 $1 秒，超过上限 $2 秒"], [/^shorten or remove shots$/, "缩短或删除镜头"],
  [/^uses (.+), which the plan does not have$/, "用到了规划里没有的 $1"], [/^pick ids from Characters, Places and Props$/, "从角色、场景和道具里选 id"],
  [/^lasts (\S+) s; a shot is one clip of at most (\S+) s$/, "时长 $1 秒；一个镜头最多 $2 秒"], [/^set 4 to 9 seconds$/, "设为 4 到 9 秒"],
  [/^shows (\d+) things; the video model takes (\d+) reference pictures, the rest are dropped$/, "出现了 $1 样东西；视频模型最多用 $2 张参考图，其余会被丢掉"],
  [/^the line is spoken by (.+), who is not in the shot, so it becomes an off-screen voice$/, "台词由不在镜头里的 $1 说出，会变成画外音"],
  [/^add the speaker or change it$/, "把说话人加进镜头或换一个说话人"],
  [/^the line needs about (\S+) s with the action around it; the shot has (\S+) s$/, "加上前后动作，这句台词大约需要 $1 秒；镜头只有 $2 秒"],
  [/^shorten the line$/, "缩短台词"],
  [/^mentions (.+), who is not in the shot's characters, so the video model will invent them$/, "提到了不在镜头角色里的 $1，视频模型会凭空画出这个人"],
  [/^add (\S+) to the shot or reword it$/, "把 $1 加进镜头，或改写描述"],
  [/^(.+) is named (\d+) times in its chapters but is not in Characters$/, "$1 在对应章节里出现了 $2 次，但不在角色表里"],
  [/^add them if they appear on screen, then add them to their shots$/, "如果会出镜，先加进角色表，再加进他们的镜头"],
  [/^episode (\d+) shot (\d+) of (\d+) \((\w+)\)$/, "第 $1 集，第 $2/$3 个镜头（$4）"],
  [/^measuring (\w+) \((\d+) of (\d+)\)$/, "正在测量 $1（$2/$3）"], [/^asking (.+) about (\w+)$/, "正在请 $1 看 $2"],
  [/^reading chapter (\d+) of (\d+)$/, "正在读第 $1/$2 章"], [/^planning characters, places and episodes$/, "正在规划角色、场景和分集"],
  [/^writing the shots of episode (\d+)$/, "正在写第 $1 集的分镜"], [/^writing the story$/, "正在写故事"],
  [/^revising (\w+) (\w+)$/, "正在修改 $1 $2"], [/^reviewing episode (\d+)$/, "正在审阅第 $1 集"],
  [/^still (\d+) of (\d+): (.+)$/, "第 $1/$2 张参考图：$3"], [/^episode (\d+): timing line (\d+) of (\d+)$/, "第 $1 集：第 $2/$3 句台词定时"],
  [/^episode (\d+): joining (\d+) clips$/, "第 $1 集：拼接 $2 个片段"],
  [/^episode (\d+): cutting (\w+) again$/, "第 $1 集：按新规则重剪 $2"],
  [/^episode (\d+): putting the descriptions into English for the video model$/, "第 $1 集：把描述翻成英文给视频模型"],
];
