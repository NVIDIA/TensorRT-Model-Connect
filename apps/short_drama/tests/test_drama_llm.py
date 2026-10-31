# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from trtmc_short_drama import llm
from trtmc_short_drama.llm import FREE_APIS, ChatClient

OK = {"choices": [{"message": {"content": "好"}}]}


def replies(monkeypatch, *responses):
    """Answer each chat request with the next response; return the requests and the waits."""

    queue, requests, waits = list(responses), [], []

    def post(url, **kwargs):
        requests.append((url, kwargs))
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(llm.httpx, "post", post)
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    return requests, waits


def api_client():
    return ChatClient("https://api.example/v1/", "free-model", "secret-key", local=False)


def ask(client):
    return client.complete("system", "user", max_tokens=10, temperature=0.1)


def test_a_rate_limited_api_is_asked_again_after_the_wait_it_names(monkeypatch):
    requests, waits = replies(monkeypatch, httpx.Response(429, headers={"Retry-After": "2"}),
                              httpx.Response(503, text="busy"), httpx.ConnectError("reset"),
                              httpx.Response(200, json=OK))
    assert ask(api_client()) == "好"
    assert waits == [2.0, 20.0, 60.0]
    url, kwargs = requests[0]
    assert url == "https://api.example/v1/chat/completions"
    assert kwargs["headers"] == {"Authorization": "Bearer secret-key"} and "enable_thinking" not in kwargs["json"]


def test_retries_stop_after_three_waits_and_say_why(monkeypatch):
    limited = httpx.Response(429, headers={"Retry-After": "3600"},
                             json={"error": {"message": "Rate limit exceeded: free-models-per-day", "code": 429}})
    _, waits = replies(monkeypatch, *[limited] * 4)
    with pytest.raises(RuntimeError) as failure:
        ask(api_client())
    assert waits == [llm.MAX_WAIT] * 3
    assert str(failure.value) == ("chat model free-model returned 429: Rate limit exceeded: free-models-per-day "
                                  "(the free requests of this minute or day are used up; wait, or pick another model)")


def test_a_rejected_key_or_model_fails_at_once_with_the_api_reason(monkeypatch):
    _, waits = replies(monkeypatch, httpx.Response(401, text="Header of type `authorization` was missing"),
                       httpx.Response(404, json={"error": {"message": "No endpoints found matching your data policy"}}))
    with pytest.raises(RuntimeError, match=r"returned 401: Header of type `authorization` was missing "
                                           r"\(check the API key in Settings\)"):
        ask(api_client())
    with pytest.raises(RuntimeError, match=r"404: No endpoints found .* \(check the model name in Settings\)"):
        ask(api_client())
    assert waits == []


def test_an_error_inside_a_200_reply_is_retried(monkeypatch):
    _, waits = replies(monkeypatch, httpx.Response(200, json={"error": {"message": "upstream overloaded"}}),
                       httpx.Response(200, json={"choices": [{"message": {"content": None}}]}))
    assert ask(api_client()) == "" and waits == [5.0]


def test_the_local_server_is_asked_once(monkeypatch):
    requests, waits = replies(monkeypatch, httpx.Response(500, json={"detail": "bad shape"}))
    with pytest.raises(RuntimeError, match="returned 500: bad shape$"):
        ask(ChatClient("http://127.0.0.1:8701", "qwen"))
    assert waits == [] and requests[0][0] == "http://127.0.0.1:8701/v1/chat/completions"
    assert requests[0][1]["json"]["enable_thinking"] is False and requests[0][1]["headers"] == {}


def test_free_apis_are_https_and_offer_a_default_model():
    for api in FREE_APIS.values():
        assert api.base_url.startswith("https://") and not api.base_url.endswith("/")
        assert api.models and api.key_url.startswith("https://") and api.models_url.startswith("https://")
