# E2E CI 单一事实来源文档

副标题: 面向管理层与工程团队的统一方案，覆盖现状、问题、目标、参考真值、每类模型合同、多阶段 diff 测试框架、实施计划与风险说明

---

## 文档定位

这是一份单独、完整、可向老板直接汇报的文档。

它整合并替代了当前 `docs/context/plans/` 目录下与 E2E / CI / 参考真值 / 多模态 diff 测试相关的工作文档，目的是把原来分散在多份 markdown 中的信息，收敛成一份可以从头读到尾的统一方案。

这份文档重点回答七个问题:

1. 我们当前的 E2E CI 到底哪里不够可靠。
2. 为什么不能把所有模型都当成“assistant 行为测试”。
3. 每一类模型的 ground truth reference 应该来自哪里。
4. 什么叫 user contract，为什么它比 intermediate logits 更重要。
5. 多 stage 模型应该怎么做 diff testing，哪些 stage 该比，哪些不该比。
6. acceptance / nightly / debug 三条测试通道应该如何分工。
7. 这套方案具体需要做哪些工程改造，顺序是什么，风险是什么。

---

## 1. 管理层摘要

### 1.1 一句话结论

我们应该把 CI 的信心来源，从“部分模型与 Python debug runner 的中间值对比”，升级成“基于官方参考实现定义的 user-contract 行为测试 + 多 stage diff 测试框架”。

### 1.2 为什么现在还不够

当前 CI 已经有统一 harness 和一定数量的 E2E case，这比完全没有框架强很多，但它仍然有四个核心问题:

- blocking E2E 并没有完全做到 `C++ binary only`
- 很多模型的 expected behavior 还没有被准确定义
- 多 stage 模型没有被纳入一套统一的 diff testing 框架
- determinism rerun 还没有真正实现

### 1.3 这次方案的核心改动

我们不再把所有模型一概而论，而是把它们拆成两种根本不同的类型:

- 强行为模型
  - 官方参考实现本身就定义了明确的 end-user 行为
  - 例如 chat / translation / OCR / ASR / TTS / VL QA / segmentation / diffusion
- 弱基础模型
  - 官方 checkpoint 本身并没有 assistant 或强产品语义
  - 例如 GPT-2、BERT、RoBERTa、base GLM、base Granite、base seq2seq backbones

对强行为模型:

- CI 应该验证 user contract
- ground truth 必须来自官方 HF / diffusers / NeMo / 官方代码

对弱基础模型:

- CI 不应该伪装成“行为正确”
- 只能诚实地验证 continuation parity、representation parity、或 probe-set ranking parity

### 1.4 管理层需要拍板的四件事

1. 是否接受双通道策略
   - `acceptance`: blocking，CLI-only，测 user contract
   - `nightly parity`: 非 blocking，允许深度 reference parity、judge model、crossover
2. 是否接受“不是每个模型都能被定义成 assistant 行为测试”
3. 是否接受“多 stage 模型不比每个 tensor，只比稳定、可解释、可隔离的 stage boundary”
4. 是否接受先做框架和高价值模型，再逐步补齐所有模型

### 1.5 推荐决策

建议全部接受。

这是唯一既工程上可落地、又能对外讲清楚、又能长期扩展到多模态的路径。

---

## 2. 当前 CI 的问题，到底在哪里

### 2.1 当前 harness 的优点

现有 E2E harness 已经有几个正确方向:

- 统一 orchestrator
- 统一 manifest
- task-based runner / comparator / reference backend
- 已经支持 text、encoder、embedding、rerank、VL、speech、diffusion、segmentation 等多种任务

这说明我们不是从零开始。

### 2.2 当前 harness 的四个根本缺陷

#### 缺陷 A: blocking E2E 还没有完全脱离 Python/debug 路径

有些路径，尤其是 text generation 和一部分 VL / diffusion 相关路径，仍然混着使用:

- TRT CLI 输出
- Python debug runner
- 内部 torch / custom reference

这会引入一个非常糟糕的问题:

> 即使 C++ binary 和 debug runner 不一致，CI 也未必能准确告诉我们“用户真正会看到什么”。

#### 缺陷 B: 不是所有模型的“正确行为”都被正确定义了

现在最大的逻辑漏洞，不是“没有比较”，而是“比较的对象并不总是对的”。

例如:

- `gpt2` 这种基础语言模型，本来就没有 assistant 行为
- `bert-base` 这种 encoder，本来就不应该被理解成“会回答用户问题”
- `bart-base` 这种 raw backbone，本身并不是强产品行为模型

如果我们错误地给这类模型套上 assistant contract，就会得到一个看起来高级、实际上不真实的 CI 指标。

#### 缺陷 C: 多阶段模型没有统一的 stage diff 框架

现在对多 stage 模型的处理更像是:

- 能跑几个 stage 就先跑几个
- 某些 family 单独写一些 comparator
- diffusion / omni / speech 各自有各自的习惯

这样的问题是:

- 很难统一复用
- 很难解释为什么某个 stage 要比较、另一个不比较
- 很难定义“这个 stage 应该比较什么 artifact”
- 很难统一 determinism / crossover / projection 的概念

#### 缺陷 D: determinism 是显式需求，但还没实现

当前框架里 determinism rerun 只是一个 TODO。

这意味着今天 CI 证明的是:

- “它跑了一次”

而不是:

- “它稳定地跑对了”

对生成模型来说，这差别非常大。

---

## 3. 我们到底要测什么

### 3.1 真正应该被 CI 保护的对象

CI 真正应该保护的是:

- 用户通过 `trtmc` CLI 看到的行为
- 这个行为是否与官方 reference 一致
- 如果这个模型是多 stage pipeline，内部关键 stage boundary 是否与 reference 一致
- 同一个 engine 是否 deterministic

### 3.2 什么不是主要目标

以下这些都不应该成为主目标:

- 为了比较而比较所有中间 tensor
- 为了高数值相似度而牺牲用户语义
- 用内部 debug runner 替代真实产品面
- 把所有模型都强行定义成“assistant quality”

### 3.3 user contract 的定义

所谓 user contract，就是“用户从这个模型能稳定期待的产品行为”。

例子:

- chat model
  - 用户输入消息，得到 assistant answer
- translation model
  - 用户输入源语言文本，得到目标语言文本
- ASR
  - 用户输入音频，得到 transcript
- OCR
  - 用户输入文档图像，得到提取出的 markdown 或 text
- TTS
  - 用户输入文本，得到语义正确、内容正确的音频
- segmentation
  - 用户输入图像，得到 mask
- rerank
  - 用户输入 query + passages，得到排序
- embedding
  - 用户输入 query / document，得到 retrieval order

### 3.4 为什么 user contract 比 intermediate logits 更重要

因为用户不关心某层 logits 是否和 HF 在第 17 个小数点上接近。

用户关心的是:

- 这是不是正确答案
- 输出有没有泄漏 prompt
- 排序是不是对的
- transcript 对不对
- 图像里到底是不是 prompt 说的内容
- 音频是不是在说正确的话

intermediate logits 有价值，但它的价值是:

- 帮助定位问题
- 作为夜间 parity / debug 的强辅助信号

而不是替代 user contract。

---

## 4. ground truth 应该来自哪里

### 4.1 总原则

ground truth reference 必须来自该 checkpoint 官方定义的参考实现。

优先级如下:

1. `transformers`
2. `diffusers`
3. `sentence-transformers`
4. `NeMo`
5. 模型卡明确给出的官方推理代码

### 4.2 为什么不能自己“发明行为”

因为很多 checkpoint 并不承诺那种行为。

一个典型例子:

- `openai-community/gpt2`

它是一个 causal language model，官方定义的是 next-token text continuation。
它不是一个 chat assistant。

如果我们把它拿来做 assistant behavior regression test，结果无论过还是不过，结论都不真实。

### 4.3 强行为 reference 与弱行为 reference

#### 强行为 reference

这类 checkpoint 在官方 surface 上就有明确的 end-user 行为:

- instruct/chat 模型
- translation 模型
- sentence embedding
- reranker
- VL instruction / OCR
- Whisper / Canary
- Bark / Magpie / PersonaPlex
- SegFormer / SAM
- FLUX / PixArt / Wan / Z-Image

#### 弱行为 reference

这类 checkpoint 没有强产品语义:

- GPT-2、GPT-Neo、Pythia、OPT、BLOOM 等 base Causal LM
- BERT / RoBERTa / ALBERT / DeBERTa 等 raw encoder
- raw BART
- 部分 toy / random checkpoints

对这类模型，ground truth 也必须来自官方 reference，但测试目标只能诚实地定义成:

- continuation parity
- representation parity
- probe-set retrieval parity

而不能包装成“assistant correctness”。

---

## 5. 参考家族定义: 每一类模型应该如何定义 exact reference

这一节是整套方案最关键的部分。

### 5.1 `CAUSAL_BASE_CONTINUATION`

定义:

- 官方 surface 是 raw prompt -> continuation
- 不使用 chat template

本地模型包括:

- `gpt2-125m`
- `gpt-neo-125m`
- `pythia-70m`
- `bloom-560m`
- `opt-125m`
- `granite-3.1-2b`
- `glm-4-9b`
- `deepseek-v2-lite`
- `falcon-rw-1b`
- `stablelm2-1.6b`
- `xglm-564m`
- `mamba-130m`
- `rwkv-169m`
- `olmo-1b`
- `olmo2-1b`
- `minitron-4b-depth`
- `minitron-4b-width`
- `nemotron-hindi-4b`
- `falcon3-1b`
- `deepseek-v2-tiny`
- `mixtral-stories-15m`

reference backend:

- `AutoTokenizer`
- `AutoModelForCausalLM`

expected behavior:

- continuation text
- optionally generated token ids

结论:

- 这些模型不是 assistant behavior coverage
- 它们是 raw continuation parity coverage

### 5.2 `CODE_BASE_COMPLETION`

定义:

- 官方 surface 是 code prefix -> code completion

本地模型:

- `codegen-350m`
- `starcoder2-3b`

reference backend:

- `AutoTokenizer`
- `AutoModelForCausalLM`

expected behavior:

- code continuation
- exact whitespace-sensitive prefix

### 5.3 `CHAT_INSTRUCT_TEMPLATE`

定义:

- 官方 surface 是 chat template + assistant response

本地模型:

- `gemma-2-2b`
- `mistral-7b`
- `phi3-mini`
- `phi-moe`
- `gpt-oss-20b`
- `nemotron-mini-4b`
- `tinyllama-1.1b`
- `internlm2-1.8b`

reference backend:

- `AutoTokenizer.apply_chat_template(...)`
- `AutoModelForCausalLM`

expected behavior:

- assistant response only
- no system/user leakage
- normalized assistant text exact match

### 5.4 `CHAT_QWEN3_POSTTRAINED`

定义:

- Qwen3 某些 checkpoint 虽然名字不含 `Instruct`，但 HF 页面定义的是 conversational/post-trained surface

本地模型:

- `qwen3-0.6b`
- `qwen3-0.6b-fp16`
- `qwen3-4b-instruct-2507`
- `qwen3-moe-30b-a3b`

reference backend:

- Qwen chat template
- 非 thinking 模式
- greedy decode

expected behavior:

- assistant response

结论:

- `qwen3-0.6b` 不应该按 raw continuation 来定义 reference
- 它应该按 chat surface 来定义

为什么 `Qwen` 这一组需要单独说明:

- 不是因为整个 `Qwen family` 天生特殊，而是因为它的 checkpoint 命名和官方 surface 之间不能只靠文件名推断。
- 对很多家族来说，名字里是否含有 `Instruct`，已经足够强地暗示它应该走 `chat template` 还是 `raw prompt`。
- 但 Qwen3 里存在一种更复杂的情况:
  - 名字里不一定含 `Instruct`
  - 但 HF 官方 quickstart 仍然让用户按 chat / conversational surface 来使用
  - 而且部分 checkpoint 同时存在 thinking / non-thinking 两种官方使用模式

这意味着:

- 我们不能用“是否带 `Instruct`”来决定 reference
- 也不能把所有 `qwen` 都统一归为 raw continuation 或统一归为 generic chat
- 必须按具体 checkpoint 的官方 HF 页面来定义

Qwen 文本 checkpoint 的精确定义:

#### `qwen3-0.6b` / `qwen3-0.6b-fp16`

官方 reference:

- `transformers.AutoTokenizer`
- `transformers.AutoModelForCausalLM`
- `tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)`

为什么这样定义:

- 这个 checkpoint 虽然名字不含 `Instruct`，但官方页面定义的是 post-trained conversational surface
- 如果按 raw continuation 去测，会把 checkpoint 的官方使用方式测错
- 如果不显式关掉 thinking，还会把 `<think>` 相关内容引入 CI，导致 reference 和真正产品 contract 不稳定

CI 含义:

- acceptance contract = `CHAT_RESPONSE`
- 不是 `CAUSAL_CONTINUATION`

#### `qwen3-moe-30b-a3b`

官方 reference:

- `AutoTokenizer`
- `AutoModelForCausalLM`
- Qwen chat template
- 非 thinking 模式

为什么这样定义:

- 它和 `qwen3-0.6b` 一样，官方 surface 是 conversational / post-trained，而不是 raw base continuation

#### `qwen3-4b-instruct-2507`

官方 reference:

- `AutoTokenizer`
- `AutoModelForCausalLM`
- `apply_chat_template(messages, add_generation_prompt=True)`

为什么这样定义:

- 这是显式 instruct checkpoint
- 官方 surface 直接就是 chat instruction following
- 它仍然属于 assistant response contract，但不需要再按“base vs post-trained”去猜

Qwen 多模态 checkpoint 需要单独处理:

#### `qwen25vl-3b` / `qwen3-vl-2b`

官方 reference:

- `transformers.AutoProcessor`
- 对应的 VL conditional generation model
- `processor.apply_chat_template(...)`

为什么这样定义:

- 它们不是 text-only LM
- 官方 surface 是 multimodal chat，不是 image caption 的自由发挥接口
- 因此最合理的 acceptance contract 是 `VL_CLOSED_QA`，不是 open-ended description

Qwen 组的最终工程规则:

- 不按 `family=qwen` 做统一粗暴规则
- 按 checkpoint 官方 HF 页面定义 exact surface
- text Qwen3:
  - 一律先判断是不是 conversational/post-trained
  - 如果是，就走 chat template
- VL Qwen:
  - 一律走 processor + multimodal chat template
- 若 checkpoint 支持 thinking / non-thinking:
  - acceptance 固定到 non-thinking
  - nightly 可以额外覆盖 thinking mode，但不建议作为 blocking CI

### 5.5 `MULTIMODAL_CHAT_QWEN35`

本地模型:

- `qwen35-9b`

reference backend:

- `AutoProcessor`
- 官方 multimodal chat format

expected behavior:

- 文本或多模态 assistant response

### 5.6 `TRANSLATION_CHAT_TEMPLATE`

本地模型:

- `riva-translate-4b`

reference backend:

- chat template
- `system` 传语言对
- `user` 传源文本

expected behavior:

- target text

结论:

- 这是 translation contract，不是 generic continuation

### 5.7 `SEQ2SEQ_TEXT2TEXT`

本地模型:

- `t5-small`

reference backend:

- `AutoTokenizer`
- `AutoModelForSeq2SeqLM`

expected behavior:

- task-prefixed text-to-text output

### 5.8 `SEQ2SEQ_TRANSLATION`

本地模型:

- `marian-en-ru`
- `nllb-200`

reference backend:

- Marian / NLLB 官方 translation API

expected behavior:

- translated text

### 5.9 `SEQ2SEQ_BASE_WEAK`

本地模型:

- `bart-base`

结论:

- raw BART 不是强产品行为模型
- 若保留，应定义成弱 seq2seq parity / smoke
- 如果想测真正 summarization / translation / QA 行为，应换成 task-tuned BART checkpoint

### 5.10 `ENCODER_BASE_FEATURES`

本地模型:

- `albert-base`
- `bert-base-uncased`
- `roberta-base`
- `roberta-large`
- `xlm-roberta-base`
- `camembert-base`
- `deberta-base`
- `distilbert-base-uncased`
- `modernbert-base`
- `convbert-base`
- `fnet-base`
- `xlnet-base`
- `electra-base-discriminator`

reference backend:

- `AutoTokenizer`
- `AutoModel`

expected behavior:

- hidden states / pooled representation
- probe-set ranking or separation behavior

结论:

- 这些模型没有 assistant behavior
- 最强 honest contract 是 representation extraction + probe set

### 5.11 `DPR_CONTEXT_EMBED`

本地模型:

- `dpr-ctx-encoder`

expected behavior:

- passage embedding
- retrieval parity

### 5.12 `SENTENCE_TRANSFORMER_EMBED`

本地模型:

- `all-minilm-l6-v2`
- `all-mpnet-base-v2`
- `paraphrase-multilingual-minilm-l12-v2`

reference backend:

- `SentenceTransformer.encode`

expected behavior:

- sentence embedding
- retrieval / clustering / similarity order

### 5.13 `BGE_RETRIEVAL_EMBED`

本地模型:

- `bge-small-en-v1.5`

expected behavior:

- instruction-aware query embedding
- retrieval ranking

### 5.14 `VL_EMBED_RETRIEVAL`

本地模型:

- `eagle-embed-vl-1b-v2`
- `nemotron-embed-vl-1b-v2`

expected behavior:

- multimodal embedding
- retrieval order

### 5.15 `VL_RERANK`

本地模型:

- `eagle-rerank-vl-1b-v2`

expected behavior:

- exact ranking order

### 5.16 `VL_INSTRUCT_QA`

本地模型:

- `qwen25vl-3b`
- `qwen3-vl-2b`
- `internvl3-8b`
- `phi4-multimodal`

expected behavior:

- image + question -> exact normalized answer

### 5.17 `OCR_MARKDOWN`

本地模型:

- `deepseek-ocr`

expected behavior:

- exact OCR markdown / text

### 5.18 `ASR_WHISPER`

本地模型:

- `whisper-tiny`
- `whisper-tiny-fp16`
- `whisper-large-v3-turbo`

expected behavior:

- transcript
- optional timestamps

### 5.19 `ASR_CANARY`

本地模型:

- `canary-1b-v2`

expected behavior:

- transcript 或 translation
- language metadata
- timestamps

### 5.20 `TTS_BARK`

本地模型:

- `bark-small`
- `bark-large`

expected behavior:

- TTS audio
- transcript recovery
- semantic/coarse token stability

### 5.21 `TTS_MAGPIE`

本地模型:

- `magpie-tts-357m`

expected behavior:

- TTS audio
- transcript recovery
- 如果可导出，最好加 codec / acoustic token

### 5.22 `S2S_PERSONAPLEX`

本地模型:

- `personaplex-7b`

expected behavior:

- speech-to-speech response
- 官方 token 或 projected transcript

### 5.23 `SEGMENTATION_SEGFORMER`

本地模型:

- `segformer-b0-ade`

expected behavior:

- semantic class map / mask

### 5.24 `PROMPTED_SEGMENTATION_SAM`

本地模型:

- `sam-vit-base`

expected behavior:

- prompt-conditioned mask

### 5.25 `DIFFUSERS_IMAGE_GEN`

本地模型:

- `flux-schnell`
- `flux-2-dev`
- `pixart-sigma-1024`
- `z-image-turbo`

expected behavior:

- prompt-conditioned image
- official diffusers pipeline as primary truth source

### 5.26 `DIFFUSERS_VIDEO_GEN`

本地模型:

- `wan21-t2v-1.3b`

expected behavior:

- prompt-conditioned video / frames
- official Wan diffusers pipeline as primary truth source

---

## 6. 我们的 CI 应该分成哪几条通道

### 6.1 `acceptance` 通道

用途:

- merge request blocking

要求:

- 只能使用 C++ CLI
- 不允许 debug runner 作为 required stage
- 测 user contract
- 做 determinism rerun

适合测:

- exact text
- exact transcript
- exact OCR
- exact ranking
- exact / thresholded masks
- image/audio/video 的健康度 + semantic acceptance

### 6.2 `nightly parity` 通道

用途:

- 深度信心
- stage diff
- reference parity
- expensive judge

允许:

- HF / diffusers / NeMo
- stage-level tensor / embedding / token comparisons
- judge model
- heavy artifacts

### 6.3 `crossover debug` 通道

用途:

- 定位问题

例子:

- HF encoder + TRT decoder
- TRT encoder + HF decoder
- TRT semantic tokens + HF vocoder

特点:

- 非 blocking
- 高信号
- 高成本

### 6.4 `golden` 与 `live reference` 的分工

这是整个 CI 设计里最容易摇摆、但也最需要明确的一件事。

问题不是“二选一”，而是“不同 CI lane 应该怎么组合使用”。

#### 方案 A: 只用 golden

优点:

- 快
- 稳
- 不依赖 nightly 当下的外部 reference 环境
- 适合 MR blocking

缺点:

- 如果最初生成 golden 的方式错了，错误会被固化
- 当官方 reference 行为、tokenizer、processor、chat template 变化时，golden 会慢慢与真实官方 surface 脱节
- 很难发现“我们的 golden 过期了”

#### 方案 B: 每次都 live 跑 official reference

优点:

- 永远对齐“当前官方实现”
- 不容易把错误的 snapshot 永久固化
- 对 stage parity 和 root-cause analysis 很强

缺点:

- 慢
- 不稳定
- 依赖外部包版本、模型下载、`trust_remote_code`、环境一致性
- 不适合作为所有 MR 的 blocking 路径

#### 推荐方案: 混合策略

我们不应该只选一个。

应该采用:

- `bootstrap reference`
  - 用官方 HF / diffusers / NeMo 跑一次或若干次
  - 生成经过 review 的 golden artifacts
- `acceptance lane`
  - CLI-only
  - 对 frozen golden 做比较
- `nightly parity lane`
  - 重新运行 official reference
  - 检查 CLI 与 live reference 是否仍一致
  - 同时检查 existing golden 是否需要刷新

也就是说:

- golden 是 presubmit 的执行载体
- live reference 是 ground truth 的持续校验机制

#### 最终建议

不要“只生成 golden”，也不要“所有 CI 都实时跑 reference”。

正确做法是:

- blocking CI 主要信任 reviewed golden
- nightly / parity job 持续运行 live official reference
- 任何 checkpoint、tokenizer、processor、template、prompt fixture 改动，都应该触发 golden refresh review

#### 哪些任务更适合 golden-first

- chat / translation / OCR / ASR
- closed QA
- reranking
- embedding probe-set ranking
- segmentation / SAM

原因:

- 输出结构化
- deterministic 程度高
- artifact 小

#### 哪些任务更需要 live reference 保底

- diffusion
- text-to-video
- TTS / speech-to-speech
- trust-remote-code 的复杂多模态模型

原因:

- 产物大
- 语义与视觉/音频质量容易漂移
- 需要持续确认 golden 没过期

#### stage 级别的建议

- end-to-end acceptance:
  - 优先 golden
- stable stage boundary:
  - nightly 跑 live reference
  - 必要时也可冻结 stage golden
- unstable stage:
  - 不建议长期冻结完整 tensor golden
  - 优先 crossover 或 projection

---

## 7. 多 stage 模型，应该怎么测

### 7.1 基本原则

不要把 multi-stage model 当成“一个黑盒跑完就算了”。

也不要反过来，把 every internal tensor 都拿出来比较。

正确做法是把它当成一个 `typed pipeline graph`。

### 7.2 stage graph 的五个核心概念

#### `PipelineSpec`

定义完整 pipeline:

- stages
- dependencies
- final outputs

#### `StageNode`

定义单个 stage:

- 这个 stage 输入什么
- 输出什么
- 用 TRT 跑还是 reference 跑
- 哪些 lane 要测

#### `ArtifactSpec`

定义 stage output 的类型:

- token ids
- normalized text
- embedding
- class map
- semantic tokens
- waveform
- image
- video frames

#### `OracleSpec`

定义 reference 怎么来:

- 官方 HF
- 官方 diffusers
- 官方 NeMo
- frozen official golden
- crossover hybrid

#### `ComparisonSpec`

定义如何比较:

- exact text
- token exact
- numeric tensor
- ranking
- mask overlap
- media similarity
- semantic judgment

### 7.3 什么叫“稳定的 stage boundary”

以下这类 boundary 应该优先测试:

- externally meaningful
- precision-stable
- implementation-independent enough
- failure 时能帮助定位问题

例子:

- generated token ids
- pooled embeddings
- image encoder features
- segmentation class map
- speech semantic tokens
- text encoder hidden states for diffusion

### 7.4 哪些 boundary 不该硬比

不应优先比较:

- allocator / kernel 相关临时 buffer
- fusion 后的内部 cache layout
- 每一步 scheduler 的所有内部状态
- 没有公共语义的隐藏张量

### 7.5 四层测试金字塔

#### Layer A: user-contract acceptance

例子:

- `trtmc run` 返回 exact normalized answer
- `trtmc transcribe` 返回 exact transcript
- `trtmc run --image` 返回 closed QA exact answer
- `trtmc generate-audio` 产物被 ASR 恢复为预期文本

#### Layer B: stage-boundary diff

例子:

- diffusion `text_encode`
- VL `vision_encode`
- TTS `semantic_decode`
- segmentation `class_map`

#### Layer C: crossover isolation

例子:

- HF text encoder + TRT denoiser + HF decoder
- TRT semantic tokens + HF vocoder

#### Layer D: determinism / stability

例子:

- rerun greedy decode
- rerun diffusion fixed seed
- rerun TTS fixed seed

---

## 8. 比较模式: 每种 artifact 应该怎么比

### 8.1 `exact_text`

适用:

- chat answer
- OCR markdown
- translation result
- VL closed QA answer

### 8.2 `prefix_text`

适用:

- code completion
- 长文本 continuation

### 8.3 `token_exact`

适用:

- greedy token ids
- semantic tokens
- codec tokens

### 8.4 `numeric_tensor`

适用:

- hidden states
- logits
- embeddings
- latent snapshots

常见指标:

- cosine
- relative L2
- top-k agreement

### 8.5 `structured_ranking`

适用:

- embedding retrieval
- reranking

指标:

- top-1 exact
- top-k order
- nDCG
- Kendall tau

### 8.6 `mask_overlap`

适用:

- segmentation
- SAM

指标:

- IoU
- Dice

### 8.7 `detection_match`

适用:

- object detection

### 8.8 `media_similarity`

适用:

- image
- video
- 部分 waveform sanity

指标:

- PSNR
- SSIM
- LPIPS

### 8.9 `semantic_judgment`

适用:

- 图像 / 音频 / 视频最终产物

例子:

- VLM closed QA
- SigLIP / CLIP 相似度
- audio -> ASR transcript recovery

重要说明:

- 它是 secondary oracle
- 不是 primary truth source

### 8.10 `invariant_check`

适用:

- 暂时无法做强 parity 的阶段

例子:

- 输出存在
- 尺寸正确
- 没有 NaN
- 音频非静音
- 图像非全黑

---

## 9. 各模态的标准测试形态

### 9.1 text / chat / seq2seq

应有 stages:

- `prompt_encode`
- `prefill`
- `decode`
- `final_text`

acceptance:

- final normalized text

nightly:

- prefill logits / top-k agreement
- token parity

### 9.2 encoder / embedding / rerank

应有 stages:

- `encode`
- `pool`
- `score`

acceptance:

- probe-set retrieval / ranking

nightly:

- vector parity
- similarity matrix parity

### 9.3 VL / OCR

应有 stages:

- `vision_encode`
- `fusion_decode`
- `answer_text`

acceptance:

- closed QA exact
- OCR exact markdown

nightly:

- image embedding cosine
- token/logit parity

### 9.4 ASR

应有 stages:

- `audio_encode`
- `decode`
- `transcript`

acceptance:

- transcript exact

nightly:

- timestamps
- language metadata

### 9.5 TTS

应有 stages:

- `text_encode`
- `semantic_decode`
- `acoustic_decode`
- `vocoder`
- `final_audio`

acceptance:

- ASR transcript recovery
- duration / RMS / silence checks

nightly:

- semantic / codec token parity

### 9.6 speech-to-speech

应有 stages:

- `audio_encode`
- `thinker_decode`
- `speaker_or_codec_decode`
- `final_audio`

acceptance:

- output speech projected transcript / meaning

nightly:

- token parity

### 9.7 segmentation / SAM

acceptance:

- mask / class map quality

nightly:

- detailed IoU
- score ordering

### 9.8 diffusion image / video

应有 stages:

- `text_encode`
- `denoise_core`
- `decode`
- `final_media`

acceptance:

- seed rerun stable
- output healthy
- semantic acceptance through judge

nightly:

- text encoder hidden parity
- selected latent snapshots
- final media similarity against frozen official reference

---

## 10. 图像、音频、视频为什么不能只看“像不像”

### 10.1 图像

图像测试必须分层:

#### 第一层: 健康度

- 文件存在
- 分辨率正确
- 像素分布合理
- 非全黑
- 非 NaN

#### 第二层: 与官方 reference 的接近程度

- 官方 diffusers 输出作为 primary truth
- PSNR / SSIM / LPIPS

#### 第三层: 语义正确性

- VLM closed QA
- SigLIP / CLIP 相似度

结论:

- judge model 很有价值
- 但它不应该替代官方 diffusers 参考

### 10.2 音频

音频也必须分层:

#### 第一层: 健康度

- WAV 存在
- duration 合理
- RMS 合理
- 非静音 / 非爆音

#### 第二层: 结构稳定性

- semantic tokens
- acoustic / codec tokens

#### 第三层: 内容语义

- 生成音频 -> ASR
- transcript recovery 是否正确

结论:

- ASR judge 非常关键
- 但如果模型本身能导出稳定 token，token parity 更强

### 10.3 视频

视频应分层:

- frame health
- temporal consistency
- key frame semantics
- optional video-level judge

---

## 11. shared fixtures 和 expected artifacts 应该如何定义

### 11.1 fixtures 的原则

fixture 必须满足:

- 小
- 固定
- 可解释
- 覆盖真实 user flow
- 适合 deterministic regression

### 11.2 推荐共享 fixture 类型

#### 文本 fixtures

- continuation prompt
- code completion prompt
- chat Q&A prompt
- translation source sentence

#### retrieval fixtures

- probe-set queries
- candidate corpus
- expected ranking

#### 视觉 fixtures

- `test_img.jpeg`
- `ocr_test_img.jpeg`
- 闭集问题和标准答案

#### 音频 fixtures

- `Recording.wav`
- canonical transcript
- optional language metadata

#### diffusion fixtures

- 简单 prompt
- 固定 seed
- 固定 resolution / steps

### 11.3 expected artifact 的形式

推荐统一存为:

- `goldens/text/...json`
- `goldens/vl/...json`
- `goldens/ocr/...md`
- `goldens/audio/...json`
- `goldens/image/...png` + `...json`
- `goldens/video/...frames/` + `...json`

### 11.4 expected artifact 不只是“文件”

它还必须有 schema。

例如 transcript golden:

- `transcript`
- `normalized_transcript`
- `language_code`
- `segments`

例如 VL answer golden:

- `question_id`
- `expected_answer`
- `normalized_answer`

例如 image golden:

- `seed`
- `steps`
- `reference_image_path`
- `semantic_questions`
- `expected_answers`
- `health_thresholds`

---

## 12. 当前模型清单应该如何向老板解释

这一节的目标不是把 84 个模型逐行念给老板听，而是让老板理解“模型很多，但它们实际上落在少数几种 reference family 上”。

### 12.1 Chat / Instruct / Conversational

包括:

- Gemma instruct
- Mistral instruct
- Phi instruct
- GPT-OSS
- Nemotron Instruct
- Qwen3 conversational / instruct
- TinyLlama Chat
- InternLM math-plus chat
- Qwen3.5 multimodal chat

管理层应该理解:

- 这些模型适合做真正 assistant 行为测试
- expected behavior 就是 assistant response

### 12.2 Base language models

包括:

- GPT-2
- GPT-Neo
- Pythia
- OPT
- BLOOM
- Granite base
- GLM base
- OLMo
- XGLM
- base Nemotron / Minitron / Falcon / DeepSeek base

管理层应该理解:

- 这些模型不是 assistant KPI
- 它们更多是 runtime / parity / correctness coverage

### 12.3 Encoder / embedding / rerank

包括:

- BERT / RoBERTa / ALBERT / DistilBERT / DeBERTa / ModernBERT / XLNet
- sentence-transformers
- BGE
- DPR
- Eagle / Nemotron embed/rerank

管理层应该理解:

- 这些模型不适合用 open-ended generation 评估
- 最合理的是 retrieval / ranking / embedding probe set

### 12.4 Speech / audio

包括:

- Whisper
- Canary
- Bark
- Magpie
- PersonaPlex

管理层应该理解:

- audio 测试不能只靠波形差异
- 必须加入 transcript recovery 和 token-level structure

### 12.5 Vision / OCR / segmentation

包括:

- Qwen-VL
- InternVL
- Phi-4 multimodal
- DeepSeek-OCR
- SegFormer
- SAM

管理层应该理解:

- open-ended caption 太弱
- closed QA / OCR exact / mask overlap 才是强信号

### 12.6 Diffusion / video

包括:

- FLUX
- PixArt
- Z-Image
- Wan

管理层应该理解:

- judge model 有用，但官方 diffusers reference 才是主真值
- acceptance 与 nightly 的分工必须明确

---

## 13. 具体工程改造应该怎么做

### 13.1 Phase 0: 定规则，不先定实现细节

先统一以下规则:

- blocking CI 只能用 CLI
- 所有模型先归类到 exact reference family
- 每个模型先定义 user contract
- 不是所有模型都必须是 assistant behavior

### 13.2 Phase 1: 把现有 harness 提升到能承载这套方案

需要改的核心文件:

- `tests/e2e_harness/contracts.py`
- `tests/e2e_harness/orchestrator.py`
- `tests/e2e_harness/manifest_loader.py`
- `tests/e2e_harness/artifact_sink.py`

需要做的事情:

- 给 stage 和 artifact 加类型
- 补 determinism rerun
- 明确 lane 概念
- 明确 oracle tier

### 13.3 Phase 2: 先把高价值家族改对

优先级最高:

- chat/instruct text
- seq2seq translation
- Whisper / Canary
- VL closed QA
- OCR
- rerank / embedding probe set
- Bark / Magpie
- diffusion text/image

### 13.4 Phase 3: 实现 stage diff 和 crossover

优先做:

- diffusion
- TTS
- speech-to-speech
- omni / multi-branch

### 13.5 Phase 4: 把所有模型补齐到统一框架

最终状态:

- 每个模型都有明确 family
- 每个 family 都有 reference policy
- 每个模型都有 acceptance lane contract
- 多 stage 模型都有 stage diff / crossover / determinism

---

## 14. 风险与取舍

### 14.1 风险: 文档比实现跑得快

这是当前最大的现实风险。

我们已经把设计方向定义得越来越清楚，但 goldens、fixtures、typed stage schema、determinism 实现还没有全部落地。

应对方式:

- 先以框架和高价值模型为主
- 不要求一次性补齐所有模型

### 14.2 风险: 过度追求 exact tensor parity

如果把所有多模态内部态都拿来硬比，结果会是:

- 框架复杂度极高
- 误报多
- 团队会失去信心

应对方式:

- 只比较 stable boundary

### 14.3 风险: 过度依赖 judge model

如果把 VLM / ASR judge 当唯一真值，也会有问题:

- judge 自己会错
- judge 可能掩盖 reference drift

应对方式:

- judge 作为 secondary oracle
- primary truth 仍来自官方 checkpoint reference

### 14.4 风险: base model coverage 看起来“不够高级”

这其实不是坏事，而是更诚实。

对 GPT-2 这种模型，如果我们说:

- “它的 CI 保护 continuation parity”

这比说:

- “它的 assistant behavior 通过了”

要真实得多。

---

## 15. 推荐的落地顺序

### 第一周

- 锁定 reference family policy
- 锁定 acceptance / nightly / debug lane policy
- 给核心 harness 补 determinism 骨架

### 第二周

- 改 text/chat/seq2seq
- 改 Whisper / Canary
- 改 rerank / embedding probe-set

### 第三周

- 改 VL closed QA / OCR
- 改 Bark / Magpie

### 第四周

- 改 diffusion 和多 stage framework
- 开始上 crossover

这会比“一次性把所有模型一次做完”更稳。

---

## 16. 管理层可以直接复述的版本

如果老板只需要三分钟版本，可以这样说:

> 我们现在的 E2E CI 已经有统一框架，但它还没有真正做到以用户看到的 CLI 行为为中心，也没有把多阶段多模态模型纳入统一的 diff testing 体系。  
> 下一步我们会把所有模型先按官方 HF / diffusers / NeMo 定义的能力进行分类。不是每个模型都适合做 assistant 行为测试，比如 GPT-2 这类 base model 只能做 continuation parity。  
> 对强行为模型，我们用官方 reference 定义 expected behavior，并在 blocking CI 中只验证 C++ binary 的 user contract；对复杂多阶段模型，再在 nightly 中增加 stage diff、crossover 和 judge model。这样既能提高 CI 信心，又不会引入虚假的质量指标。  

---

## 17. 本文档与 `docs/context/plans/` 其它文件的关系

### 17.1 本文档直接整合的 E2E/CI 相关文档

以下文档的核心内容已经被并入本文档:

- `e2e_ci_contracts_report.md`
- `e2e_ci_model_assessment_matrix.md`
- `e2e_model_test_directions_and_methods.md`
- `e2e_ci_boss_ready_full_spec.md`
- `e2e_ci_boss_ready_typed_spec.md`
- `e2e_hf_reference_by_family.md`
- `multimodal_diff_testing_framework.md`

这些文档可以保留作工作记录，但今后不应再作为对外汇报的主文档。

### 17.2 `docs/context/plans/` 中其它非 E2E 文档

`docs/context/plans/` 目录里还有一批与 runtime / plugin architecture / migration 相关的文档，例如:

- `MR2-todo.md`
- `MR2-worklog.md`
- `MR3-plan.md`
- `MR4-plan.md`
- `TASK-01_whisper_bark_migration.md`
- `TASK-02_magpie_speech_omni_migration.md`
- `TASK-03_diffusion_migration.md`
- `TASK-04_cleanup_delete_legacy.md`
- `TASK-05_plugin_architecture.md`

这些文档描述的是另一条工作流:

- runtime 重构
- plugin architecture
- pipeline migration
- legacy cleanup

它们和本文档不是矛盾关系，而是上下游关系:

- 本文档定义“应该怎么测试”
- 那些文档定义“runtime/pipeline 自身怎么重构”

如果以后要做第二份老板版文档，可以专门为 runtime architecture 单独做一份，而不应该跟 E2E testing 策略混写在一起。

---

## 18. 最终建议

建议把这份文档作为今后 E2E CI 讨论的唯一入口。

后续新的细节更新，原则上都应直接回收进这份文档，而不是继续在 `docs/context/plans/` 下新增并列的“老板版”“完整版”“typed 版”“framework 版”。

这样团队才能真正拥有单一事实来源。
