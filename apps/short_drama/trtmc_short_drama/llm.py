# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat completions against an OpenAI-compatible API or the local trtmc-server."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any

import httpx

TIMEOUT = httpx.Timeout(3600.0, connect=10.0)
# Free tiers answer 429 once the requests of a minute or a day are used; busy hosts answer 5xx.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
RETRY_WAITS = (5.0, 20.0, 60.0)  # before each retry, unless the reply's Retry-After says otherwise
MAX_WAIT = 120.0
HINTS = {401: "check the API key in Settings", 403: "check the API key in Settings",
         404: "check the model name in Settings",
         429: "the free requests of this minute or day are used up; wait, or pick another model"}


@dataclass(frozen=True)
class ChatApi:
    """An OpenAI-compatible API with a free tier; everyone brings their own key."""

    label: str
    base_url: str
    models: tuple[str, ...]  # the first is the default; any other model of the API works too
    key_url: str
    models_url: str
    note: str


# Free lineups change often, so Settings also links each provider's current list.
FREE_APIS = {
    "nvidia": ChatApi(
        "NVIDIA API catalog", "https://integrate.api.nvidia.com/v1",
        ("deepseek-ai/deepseek-v4.1-flash", "moonshotai/kimi-k2.6", "z-ai/glm-5.3-flash"),
        "https://build.nvidia.com/settings/api-keys", "https://build.nvidia.com/models",
        "Free for development with an NVIDIA Developer Program account; requests per minute are limited."),
    "openrouter": ChatApi(
        "OpenRouter free models", "https://openrouter.ai/api/v1",
        ("nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-31b-it:free"),
        "https://openrouter.ai/settings/keys", "https://openrouter.ai/models?max_price=0",
        "Models ending in :free cost nothing: 20 requests a minute and 50 a day, 1,000 a day after "
        "buying $10 of credits once. Some need free-model training allowed in the privacy settings."),
}


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
        url = (f"{self.base_url.rstrip('/')}/v1/chat/completions" if self.local
               else f"{self.base_url.rstrip('/')}/chat/completions")
        waits = [] if self.local else list(RETRY_WAITS)  # the local server fails the same way again
        while True:
            try:
                response = httpx.post(url, json=body, headers=headers, timeout=TIMEOUT)
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as error:
                if not waits:
                    raise RuntimeError(f"cannot reach the chat API at {self.base_url}: {error}") from None
                time.sleep(waits.pop(0))
                continue
            text = answer(response)
            if text is not None:
                return text
            if not waits or not (response.is_success or response.status_code in RETRY_STATUS):
                raise RuntimeError(failure(self.model, response))
            time.sleep(retry_after(response, waits.pop(0)))

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


def answer(response: httpx.Response) -> str | None:
    """The reply's text; None for an error, including a 200 reply that carries one instead of choices,
    which is how OpenRouter reports a failure upstream."""

    if not response.is_success:
        return None
    choices = response.json().get("choices")
    return (choices[0]["message"].get("content") or "") if choices else None


def failure(model: str, response: httpx.Response) -> str:
    """What went wrong, in the API's own words when it gives a reason, and what to check."""

    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    if isinstance(payload, dict):
        error = payload.get("error")
        payload = (error.get("message") if isinstance(error, dict) else error) or payload.get("detail") \
            or payload.get("message")
    reason = " ".join(str(payload or "").split())[:300]
    hint = HINTS.get(response.status_code)
    return (f"chat model {model} returned {response.status_code}" + (f": {reason}" if reason else "")
            + (f" ({hint})" if hint else ""))


def retry_after(response: httpx.Response, default: float) -> float:
    """Seconds before the next try: the reply's Retry-After when it is a number, at most MAX_WAIT."""

    try:
        return min(MAX_WAIT, max(0.0, float(response.headers.get("retry-after", default))))
    except ValueError:  # an HTTP date
        return default


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
