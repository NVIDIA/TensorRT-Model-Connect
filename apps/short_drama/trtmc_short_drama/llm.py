# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat completions against the local trtmc-server or an optional OpenAI-compatible API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

import httpx

TIMEOUT = httpx.Timeout(3600.0, connect=10.0)


@dataclass(frozen=True)
class ChatClient:
    base_url: str
    model: str
    api_key: str = ""
    local: bool = True

    def __repr__(self) -> str:  # keep keys out of logs and tracebacks
        return f"ChatClient({self.base_url!r}, {self.model!r}, local={self.local})"

    def complete(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        body: dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.local:
            body["enable_thinking"] = False
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = httpx.post(f"{self.base_url.rstrip('/')}/v1/chat/completions" if self.local
                              else f"{self.base_url.rstrip('/')}/chat/completions",
                              json=body, headers=headers, timeout=TIMEOUT)
        if not response.is_success:
            raise RuntimeError(f"chat model {self.model} returned {response.status_code}")
        return response.json()["choices"][0]["message"]["content"] or ""

    def complete_json(self, system: str, user: str, *, max_tokens: int, temperature: float,
                      attempts: int = 2) -> Any:
        """Ask for one JSON value; retry once with the parse error when the reply is not JSON."""

        reply = ""
        for attempt in range(attempts):
            prompt = user if not reply else (
                f"{user}\n\nYour previous reply was not valid JSON. Reply again with only the JSON value.")
            reply = self.complete(system, prompt, max_tokens=max_tokens,
                                  temperature=temperature + 0.3 * attempt)
            try:
                return parse_json(reply)
            except ValueError:
                continue
        raise ValueError(f"the chat model did not return JSON: {reply[:200]!r}")


def parse_json(text: str) -> Any:
    """The first JSON object or array in a reply, ignoring code fences and thinking blocks."""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise ValueError("no JSON value in the reply")
    text = text[min(starts):]
    try:
        return json.JSONDecoder().raw_decode(text)[0]
    except ValueError:
        return json.JSONDecoder().raw_decode(escape_inner_quotes(text))[0]


def escape_inner_quotes(text: str) -> str:
    """Escape double quotes that sit inside prose, such as 用“阿Q"称呼, which models often emit.

    A structural quote opens after one of `{[,:` or whitespace, or closes before one of `,:]}` or
    whitespace; any other unescaped quote is part of a string.
    """

    characters = list(text)
    for index, character in enumerate(characters):
        if character != '"' or (index and text[index - 1] == "\\"):
            continue
        before = text[index - 1] if index else " "
        after = text[index + 1] if index + 1 < len(text) else " "
        if not (before in "{[,:" or before.isspace() or after in ",:]}" or after.isspace()):
            characters[index] = '\\"'
    return "".join(characters)
