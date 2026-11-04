<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Short drama (`trtmc_short_drama`)

Turn a Chinese novel into vertical (9:16) short-drama episodes with sound and subtitles, on
your own GPU. The pictures, video and speech recognition run on TensorRT Model Connect through
`trtmc-server`; the language model is a chat API of your choice, or runs here too with `--llm local`:

| Step | What it does | Model (tier A) |
| --- | --- | --- |
| Plan | reads every chapter, writes the cast, relationships, places, props, episodes and a shot list per episode | a chat API chosen in Settings (free tiers: the NVIDIA API catalog, OpenRouter), or `Qwen/Qwen3.6-27B` with `--llm local` |
| Reference pictures | draws a still of each character, place and prop for you to approve or redraw | `Qwen/Qwen-Image` |
| Shots | films each shot from its approved stills, with dialogue and sound | `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` (MiniMax H3 reference-to-video) |
| Subtitles | checks every line by speech recognition and times its subtitle | `openai/whisper-large-v3-turbo` |
| Assemble | joins the shots, burns in subtitles, exports 1080x1920 MP4 | FFmpeg |

Everything the models write can be edited in the app before anything is rendered. The five
steps at the top of a project (story, plan, reference pictures, shots, episodes) show what is
done, and the card under them names the next thing to do.

On one GB300, planning the bundled 9-chapter novel into 4 episodes with the local 27B model
takes about 9 minutes (14 chat requests), a reference still about 10 seconds, and each 5 to 9
second shot about 10 minutes (one H3 call renders 226 frames, 9.4 s, with audio).

## Requirements

- An NVIDIA GPU and driver. Tier A needs about 140 GB of GPU memory (MiniMax H3 peaks near
  110 GB at 768x1344 and 226 frames); it was run on a GB300.
  Tier B (80 GB, for example an RTX PRO 6000) renders shots without voices on a TRTMC build that has
  what it lists; tier C (24 GB) waits for Z-Image-Turbo pictures within 24 GB. See [Tiers](#tiers).
- A TRTMC build with the server, bring-your-own kernels and the H3 sparse-attention kernel:

  ```bash
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTRTMC_BUILD_SERVER=ON \
    -DTRTMC_ENABLE_BYOK=ON -DTRTMC_BUILD_MINIMAX_H3_OMNIREF_VSA=ON \
    -DTRTMC_BUILD_MINIMAX_H3_OMNIREF_VSA_SM100A=ON   # Blackwell sm100/sm103 fast path
  ninja -C build trtmc-server trtmc_backend_trt trtmc_byok_tvm_ffi trtmc_minimax_h3_omniref_vsa \
    trtmc_model_minimax_h3 trtmc_model_qwen_image trtmc_model_whisper
  ```

  Add `trtmc_model_qwen3_8` for `--llm local`.
- Python 3.10+ with `pip install -r apps/short_drama/requirements.txt` and the server extras
  (`pip install -e '.[serve]'`).
- FFmpeg built with libass, and a CJK font such as Noto Sans CJK SC for the subtitles.
- A key for a chat API (see [Language model](#language-model)), or `--llm local`.
- Disk: about 250 GB while the H3 bundles build (130 GB each), a few GB for the image and speech
  bundles, plus the checkpoints; `--llm local` adds 110 GB for the 27B language model.

## Run

From the repository root:

```bash
PYTHONPATH=apps/short_drama python -m trtmc_short_drama up
```

`up` picks the GPU with the most memory and the tier that fits it, builds every missing bundle
into `short-drama-data/bundles/` with `python -m tensorrt_model_connect build`, and serves the
app at <http://127.0.0.1:8700>. A second run reuses the bundles. `--dry-run` prints the plan
and the build commands without running them.

| Option | Meaning |
| --- | --- |
| `--data DIR` | projects, bundles, settings and logs (default `./short-drama-data`) |
| `--tier a\|b\|c` | force a tier instead of choosing by GPU memory |
| `--allow-planned` | start a tier that TRTMC cannot run end to end yet; without it `up` names what is missing and stops |
| `--gpu N` | GPU index as `nvidia-smi` numbers it |
| `--llm api\|local` | `api` (default): the chat API chosen in Settings; `local`: build and serve the tier's language model on this GPU |
| `--bundle ROLE=PATH` | use an existing bundle for `llm` (with `--llm local`), `image`, `video`, `asr` or `tts` |
| `--runtime-root DIR` | directory with `trtmc-server` and the runtime libraries (default `build/`) |
| `--host`, `--port`, `--server-port` | where the app (8700) and `trtmc-server` (8701) listen; any host other than loopback prints a URL with a `?token=` that the browser needs once. The token is kept in `app-token.json` (mode 600) in the data directory, so the link survives a restart; delete the file for a new one |

The app starts one `trtmc-server` and loads only the bundles the current step needs, so the
models never have to fit in memory together. Speech recognition stays loaded next to the video
model because it is small.

### Language model

Planning, writing a story from an idea, the model review and revisions use a chat model. By
default that is an OpenAI-compatible chat API: open *Settings*, pick one, paste your key and
save. Two have free tiers and are preset:

| API | Models offered first | Free tier |
| --- | --- | --- |
| [NVIDIA API catalog](https://build.nvidia.com/models) | `deepseek-ai/deepseek-v4.1-flash`, `moonshotai/kimi-k2.6`, `z-ai/glm-5.3-flash` | free for development with an NVIDIA Developer Program account; requests per minute are limited |
| [OpenRouter](https://openrouter.ai/models?max_price=0) | `nvidia/nemotron-3-ultra-550b-a55b:free`, `google/gemma-4-31b-it:free` | models ending in `:free`: 20 requests a minute and 50 a day, 1,000 a day after buying $10 of credits once |

Free lineups change often; type any other model name the API serves. *Another
OpenAI-compatible API* takes any base URL. Planning one 9-chapter novel into 4 episodes takes
14 requests. When an API answers 429 (limit reached) or 5xx, the app waits and tries again three
times, using the API's `Retry-After` when it sends one. Some free OpenRouter models answer 404
until free-model training is allowed in the account's privacy settings.

The story and plan are sent to the API you choose, and free tiers may keep prompts. To keep
everything on your machine, start with `--llm local`: `up` then also builds `Qwen/Qwen3.6-27B`
(tier B: `Qwen/Qwen3.5-4B`) and plans on the GPU. A chat API saved in Settings still takes
precedence; *Use … on this GPU* switches back.

The Whisper family fixes the decoder language when the bundle is built and falls back to
English, which turns Chinese speech into English text. `up` therefore builds Whisper from a
local view of the checkpoint (`short-drama-data/checkpoints/`, weights linked, not copied) whose
generation config forces `<|zh|>`; the bundle is named `whisper-zh-sm<arch>.bundle`.

### The H3 bundle

MiniMax H3 reference-to-video is an explicit experimental build in the `minimax_h3` family
(see `families/minimax_h3/README.md`). `up` runs the family command
`python -m tensorrt_model_connect minimax_h3 build-omniref`: the original `MiniMaxAI/MiniMax-H3`
two-window bundle first (kept in `bundles/scratch/`), then the FastH3 OmniRef 8-step student
that reuses six of its eight plans. Plan for about 250 GB of
disk and 100 GB of host memory while it builds, or pass an existing bundle with
`--bundle video=PATH`.

## Use

The page opens on *Projects*: one card per project with its progress through the five steps,
what it does next and when it last changed; rename or delete a project from its card. The
header switches the page between Chinese and English. *New project* asks five things: where the
story comes from, the project's language, the visual style, the length, and an optional title.
It shows how many shots that makes and, once one has rendered on this machine, how much GPU time
they take. The language model writes the story, the plan, the lines and every review in the
project's language; any field can be edited in any language, and a line is spoken in its own
language. Every text box grows with its text and opens in a large editor (*Expand*).

A project page shows the five steps below, and under them one card with the next thing to do
and the button that does it. While a step runs, the card shows its current phase (for example
*Loading fasth3 and whisper onto the GPU*, *Still 2 of 5*, *Episode 1 shot 1 of 3*), how long
it has run and, from earlier runs on this machine, how long it usually takes. When a step
fails, the card shows why, with *Try again* and the model server's log. The status in the
header says what the GPU is doing; click it for the activity panel: the models loaded, the
running and queued steps of every project, recent steps with their errors, and the end of
`logs/trtmc-server.log`.

1. **Story.** Paste a novel, open a text file, use the bundled sample, 《阿Q正传》 by Lu Xun
   (public domain), or start from an idea: one or two sentences that the language model turns
   into a short story with one chapter per episode, which you can edit before planning.
2. **Plan.** One click, once a language model is chosen in Settings. Edit the cast (each character's `look` and `voice` feed the image and
   video prompts), places, props, episodes and every shot: place, characters, seconds, the
   line, and what happens before, during and after it.
   - The plan is checked on every change, before any GPU time is spent: people the chapters
     name often but nobody plays, someone acting in a shot without being in it, a speaker who
     is not in the shot, a line too long for its shot, ids the plan does not have, an episode
     over its limit. *Ask the model to review* adds the language model's notes on missing
     events, continuity between neighbouring shots and actions too big for one clip.
   - Every line is marked as quoted from the novel or adapted.
   - *Revise with the model* rewrites one shot, character, place or prop from a note and leaves
     the rest of the plan alone. A re-plan, a revision, a save or a restore first keeps the plan
     it replaces under *Earlier versions*.
3. **Reference pictures.** Pick the visual style (or keep the one chosen with the project),
   draw the stills, then approve each one or redraw it, optionally with your own prompt. Only
   characters, places and props that appear in a shot get a still, and each still lists the
   shots that use it. Shots only use approved stills; a shot's render buttons wait for them.
4. **Shots.** Render a sample shot first, check the look, then render the rest; a batch first
   says how much GPU time it takes, from the times measured on this machine. The strip at the
   top shows every shot of the episode. A shot in the same place as the shot before it gets
   that shot's last frame as its first reference, so positions, clothes and light carry over
   (switch it off per episode); the last frame of an episode does the same for the first shot
   of the next one. Every take keeps its raw clip, a contact sheet, the exact prompt and the
   stills it used: when a still is redrawn, the shot before it gets a new take or the shot
   changes, the take is marked outdated and *Re-render outdated* renders only those shots.
   The FastH3 OmniRef student often opens on reference pictures held still (a portrait, or the
   previous shot's last frame for up to 1.4 s) and then cuts into the shot, so a clip starts
   after those pictures, and never before 0.25 s: TRTMC's H3 opens every clip with a short loud
   burst that MiniMax's own samples do not have. Sound fades in and out at every cut.
5. **Episodes.** Time the subtitles and assemble; play or download the MP4. Assembling cuts
   takes made under an older version of these rules again from their raw clips (no GPU). The
   episode is normalized to one loudness as a whole; a shot much quieter than its neighbours
   stays quieter, and the analysis reports it.

**Video analysis.** *Analyze the video* on the Shots step measures every rendered shot on this
machine: a burst at the start, clipping, a nearly silent clip, the planned line against speech
recognition, speech where the shot has no line, black or frozen frames, and how far the picture
and the level jump at each cut. With a vision model set in Settings (any OpenAI-compatible model
that reads pictures), each shot's frames, up to two of its reference pictures and the previous
shot's last frame go to that model as one picture, and it reports content that does not match the
plan, text drawn on screen, continuity breaks, people or places that do not look like their
pictures, and reference pictures that do not fit the plan. `up --vision local` on tier A serves
Qwen3.5-9B built for pictures instead (experimental): trtmc-server keeps its thinking on for
pictures, so a shot takes 2 to 4 minutes and some run out of tokens, and with thinking off the 9B
model takes the reference pictures for frames of the clip. Each finding names the plan item it
would change and suggests a change; edit the suggestion in any language, pick an action (revise
the plan with the language model, render another take, redraw a picture) or write your own
note, then *Apply the selected changes*. Revised shots and pictures become outdated, and the
next card offers to render them again; analyze again to check the result.

### How shots are prompted

The language model only fills short fields per shot. The app writes the H3 prompt from them:
subject definitions that bind each `<Picture N>` to a character, place or prop; an action that
runs through the whole shot; the line in the first seconds inside `<d>[Chinese] …</d>` (or
`[English]`, by the line's own language); a light
camera move; and a second-window prompt that keeps the action going without new dialogue.
Text in double quotes is drawn on screen by H3, so the app removes quotes from descriptions.
H3 reads Chinese, Japanese or Korean prose in the description aloud, so when a plan is written in
one of them (or edited in one), the shots step first asks the language model for the English of
every such description and keeps it in the project's `translations.json`; the line itself stays
as written. A shot without a line says that no one speaks.

## Customize

**Flow.** *Flow (advanced)* on a project page shows the steps, their order, the model of each step and its parameters
as JSON. Save edits as a new version (`default@2`, …) and select the version a project uses;
earlier versions never change.

**Plugins.** A step is a Python file in `apps/short_drama/plugins/` (or `--plugins DIR`) that
defines `STEP = Step(name, run, role=..., scope="project"|"episode")`. Click *Reload plugins*,
then add `{"step": "<name>", "model": ..., "params": {...}}` to a flow version.
`plugins/dialogue_polish.py` is an example that tightens every line with the language model.
Delete the file and reload to remove the step; a flow that still names it skips it with a
message.

**Actions.** Writing a story, revising one item and the model review are actions, not stages:
they run on request with the model of the flow's `plan` step. Video analysis runs with the
model of the `subtitles` step. A plugin step becomes an action with `model_from="plan"`.

**Chat API key.** The key entered in Settings is written to `secrets.json` (mode 600) in the
data directory, only ever sent to the base URL it was entered with, and never logged or sent back
to the browser.

## Tiers

`up` picks the largest tier the GPU's memory holds; GPUs that share system memory (DGX Spark,
Jetson Thor) count their system memory. Settings shows the table below for the running GPU.

| Tier | GPU memory | For example | Shots | Speech | Status |
| --- | --- | --- | --- | --- | --- |
| A | 140 GB or more | GB300, B200, H200 | MiniMax H3 films each shot from up to 6 reference pictures | native audio from the video model | runs end to end |
| B | 80 GB or more | RTX PRO 6000, H100, DGX Spark | Qwen-Image-Edit draws each shot's first frame from the reference pictures; Wan2.2-TI2V-5B animates it | none yet: each line is a subtitle | runs end to end on a TRTMC build with image-to-video for Wan2.2-TI2V-5B and vertical Qwen-Image-Edit pictures; start it with `--allow-planned` until TRTMC main has both |
| C | 24 GB or more | RTX 4090, RTX 5090, L40S | Wan2.2-TI2V-5B animates each shot from its speaker's reference picture | none yet: each line is a subtitle | planned: shots, subtitles and assembly run in 24 GB on a TRTMC build with image-to-video for Wan2.2-TI2V-5B (peak 20.7 GiB); the reference pictures wait for Z-Image-Turbo within 24 GB, because TRTMC's fp16 build renders white pictures and fp32 needs about 31 GiB |

On tiers B and C a shot starts from one picture. A shot in the same place as the shot before it
starts on that shot's last frame. Any other shot starts from the reference picture of its speaker,
else of its first person, else of its place. Tier B's image-edit model draws the shot's first
frame from that picture and describes the other people of the shot in words. Tier C has no room
for an edit model, so the shot starts on the picture itself and shows one person at first.
Wan2.2-TI2V-5B then animates 121 frames, about 5 s, so a shot planned longer ends after 5 s. The
clips have no sound: each line becomes a subtitle that starts with the speaker's name, and the
analysis skips the sound and speech checks.

The models of tiers B and C are released under Apache-2.0 or MIT. MiniMax H3 and the FastH3
student of tier A are released under the MiniMax H3 Community License, which limits the
territories where they may be used and asks products built on them to say so; read it before
you use or ship tier A.

## Data layout

```text
short-drama-data/
  app-token.json           the page's token when the app listens beyond loopback (mode 600)
  settings.json            the chat API; its key is in secrets.json (mode 600)
  bundles/                 built bundles, named <model>-sm<arch>.bundle
  flows/<name>/v<N>.json   flow versions
  projects/<id>/
    novel.txt  plan.json  stills.json  stills/  review.json  history/ (earlier plans)
    episodes/<NN>/raw/ clips/ sheets/ frames/ shots.json analysis.json episode.srt episode.mp4 last_frame.png
  logs/trtmc-server.log
```

## Tests

```bash
CUDA_VISIBLE_DEVICES= python -m pytest apps/short_drama/tests
```

The tests use fake models; the tests that need FFmpeg are skipped when it is not installed.
