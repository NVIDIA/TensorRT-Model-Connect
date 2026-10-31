# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example plugin step: let the language model tighten every line of dialogue.

Add {"step": "dialogue_polish", "model": "<chat model>", "params": {}} after "plan" in a flow
version to use it. Delete this file and reload plugins to remove it again.
"""

import json

from trtmc_short_drama.flow import Step

PROMPT = """Rewrite these lines of a vertical short drama so each one sounds natural when spoken,
keeps its meaning and the speaker's voice, and has at most {limit} Chinese characters.
Return a JSON object that maps each id to its new line. Lines: {lines}"""


def run(context):
    plan = context.project.plan
    lines = {shot["id"]: shot["line"] for episode in plan.get("episodes", [])
             for shot in episode.get("shots", []) if shot.get("line")}
    if not lines:
        return
    context.progress(f"polishing {len(lines)} lines")
    limit = int(context.params.get("max_characters", 16))
    chat = context.services.chat(context.model)
    polished = chat.complete_json("Reply with one JSON object and nothing else.",
                                  PROMPT.format(limit=limit, lines=json.dumps(lines, ensure_ascii=False)),
                                  max_tokens=4096, temperature=0.4)
    changed = 0
    for episode in plan["episodes"]:
        for shot in episode["shots"]:
            new = polished.get(shot["id"]) if isinstance(polished, dict) else None
            if isinstance(new, str) and new.strip() and len(new.strip()) <= limit + 4:
                changed += new.strip() != shot["line"]
                shot["line"] = new.strip()
    context.project.save_plan(plan)
    context.project.log(f"dialogue_polish changed {changed} lines")


STEP = Step("dialogue_polish", run, role="llm", scope="project",
            description="Tighten every line of dialogue with the language model")
