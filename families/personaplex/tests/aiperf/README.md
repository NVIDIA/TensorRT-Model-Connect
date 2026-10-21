# PersonaPlex speech question answering

The `personaplex-7b` AIPerf profile uses all 455 questions from VoiceBench
OpenBookQA, pinned to dataset revision
`b02edcef1330480be3a11bd6f7434ac32f05ad08`. Each question has an objective
A/B/C/D answer. Only the question audio enters the model; the question text and
gold answer are available to the scorer only. The suite resamples the 16 kHz
questions once to Mimi's 24 kHz rate (SciPy polyphase), so both sides read the
same samples and neither pipeline resamples.

Both sides use the official NATM1 voice and the same teacher role prompt,
greedy decoding, an FP16 temporal/depth model, and an FP32 Mimi codec. The
official voice asset's BF16 embeddings are cast to FP16 on both sides. Each
request resets its state and processes the full question followed by 125
silent input frames (10 seconds) for the reply. The prepared bundle's temporal
capacity covers the longest question, reply window and system prompt. Original
weights remain in the shared HF cache; the prepared model directory contains
symlinks, one config overlay and small derived prompt files.

Native applies the pinned official `step_system_prompts` implementation. TRTMC
replays the same input embeddings through its temporal engine, restores the
official delayed-token state, and then processes the user stream. Every stream
is forced during the prefix, so its embeddings and token cache can be prepared
without generating predictions. Prefix execution is included in each Task
call; preparation, model loading and output scoring are excluded.

During the question, both implementations force the agent's text PAD and official
silence audio tokens while processing the real user audio. Generation is released
for the reply window, with the EPAD text token (0) triggering the response as in
[Moshi's spoken-QA protocol](https://arxiv.org/html/2410.00037v2#S5.SS5).
This gives the model the complete question and all options
before answering, as required by this turn-based QA protocol. Original full-duplex
barge-in behavior remains available when `listen_before_answer` is disabled.

The same AIPerf inference outputs provide both results:

- **Acc:** transcribe the generated audio using pinned
  `openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9`,
  evaluating the final 10-second answer window while retaining complete raw audio,
  then match an explicit option letter or a unique complete option text.
  Explicit answer declarations take precedence over words in explanations;
  conflicting declarations and negated option mentions do not become answers.
  Empty, ambiguous and unparsable replies count as wrong. The ASR process
  receives output paths only, without gold labels. Native and TRTMC each get
  an absolute score and a paired comparison with a 2-point regression margin;
  there is no minimum Native score. ASR errors remain a measurement limitation.
- **Perf:** Native and TRTMC Task-call P50 over those same questions, after
  excluded warmup. Input/reply lengths and output frame counts must agree.
  Offline ASR and grading do not enter either P50. AIPerf client measurements
  remain informational.

Run the standard profile:

```bash
trtmc-aiperf-qual run --profile personaplex-7b \
  --environment environment.yaml --out results/personaplex-7b
```

For a pilot, resolve the profile and set
`absolute[0].suite_definition.selection.count` to 50 before `campaign.run_one`.
Keep the pilot and full results in separate directories. Preserve generated
audio for scoring and review with `retention.temporary_files: retain`; bundles
can be deleted after qualification while keeping `retention.hf_cache: retain`.

This evaluates spoken single-question answering. It does not establish
interruption handling, turn-taking quality or voice similarity. Previous
LibriSpeech waveform-parity results remain separate historical diagnostics;
their scores and passing criteria are not reinterpreted as QA accuracy.

The profile's own requirements file pins dataset preparation and ASR dependencies
in its reference environment. Offline postprocessing uses that interpreter by
default; `scoring_python` in the environment can select a separate compatible
scoring interpreter. Neither path changes the measured inference duration.
