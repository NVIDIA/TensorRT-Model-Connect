# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import stat
import time

from fastapi.testclient import TestClient

from trtmc_short_drama.engine import Served
from trtmc_short_drama.web import Services, create_app

PLAN_STEPS = [{"location": "temple", "characters": ["aq"], "speaker": "aq", "line": "滚出去！", "seconds": 6}]
BIBLE = {"title": "阿Q正传", "characters": [{"id": "aq", "name": "阿Q", "look": "thin"}],
         "locations": [{"id": "temple", "name": "土谷祠", "look": "a shrine"}],
         "episodes": [{"number": 1, "title": "一", "summary": "s", "chapters": [1]}]}


class FakeEngine:
    url = "http://127.0.0.1:1"
    loaded: list[str] = []

    def __init__(self):
        self.ensured = []

    def ensure(self, models):
        self.ensured.append(sorted(m.name for m in models))

    def stop(self):
        pass


class FakeChat:
    def complete_json(self, system, user, **_):
        if user.startswith("Summarize"):
            return {"summary": "x"}
        return BIBLE if user.startswith("Plan") else PLAN_STEPS


def make_client(tmp_path: Path, monkeypatch, allowed_hosts=None):
    services = Services(FakeEngine(), {"q": ("llm", Served("q", tmp_path / "q.bundle")),
                                       "v": ("video", Served("v", tmp_path / "v.bundle")),
                                       "w": ("asr", Served("w", tmp_path / "w.bundle"))}, tmp_path)
    monkeypatch.setattr(Services, "chat", lambda self, name: FakeChat())
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    app = create_app(services, plugins=plugins, info={"gpu": "test", "tier": "a"}, allowed_hosts=allowed_hosts)
    return TestClient(app, headers={"X-Short-Drama": "1"}), services


def wait(client):
    for _ in range(300):
        jobs = client.get("/api/state").json()["jobs"]
        if jobs and all(j["status"] not in {"queued", "running"} for j in jobs):
            return jobs
        time.sleep(0.02)
    raise AssertionError("jobs did not finish")


def test_create_plan_edit_and_select_flow(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        state = client.get("/api/state").json()
        assert state["flows"] == {"default": [1]} and state["sample"] is True
        created = client.post("/api/projects", json={"title": "AQ", "sample": True, "episodes": 1,
                                                     "max_seconds": 30}).json()
        project = client.get(f"/api/projects/{created['id']}").json()
        assert project["novel_chars"] > 8000 and project["meta"]["flow"] == "default@1"

        assert client.post(f"/api/projects/{created['id']}/run", json={"step": "plan"}).status_code == 200
        jobs = wait(client)
        assert jobs[0]["status"] == "done", jobs
        plan = client.get(f"/api/projects/{created['id']}").json()["plan"]
        assert plan["episodes"][0]["shots"][0]["line"] == "滚出去！"

        assert "<d>[Chinese] 滚出去！</d>" in client.get(f"/api/projects/{created['id']}").json()["previews"]["e1s1"]
        plan["episodes"][0]["shots"][0]["line"] = "你还不配！"
        assert client.put(f"/api/projects/{created['id']}/plan", json=plan).status_code == 200
        assert client.get(f"/api/projects/{created['id']}").json()["plan"]["episodes"][0]["shots"][0]["line"] == "你还不配！"

        flow = client.get("/api/flows/default").json()["flow"]
        saved = client.post("/api/flows/default", json={"flow": flow, "note": "same"}).json()
        assert saved == {"reference": "default@2"}
        assert client.put(f"/api/projects/{created['id']}/flow", json={"flow": "default@2"}).json()["flow"] == "default@2"


def test_episode_steps_need_an_episode_and_shots_need_approved_stills(tmp_path, monkeypatch):
    client, services = make_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/projects", json={"title": "x", "novel": "第一章\n文本", "episodes": 1}).json()
        assert client.post(f"/api/projects/{created['id']}/run", json={"step": "shots"}).status_code == 400
        client.post(f"/api/projects/{created['id']}/run", json={"step": "plan"})
        wait(client)
        client.post(f"/api/projects/{created['id']}/run", json={"step": "shots", "episode": 1})
        jobs = wait(client)
        assert jobs[0]["status"] == "failed" and "approve these reference stills" in jobs[0]["error"]
        assert services.engine.ensured[-1] == ["v", "w"]  # speech recognition stays with the video model


def test_files_stay_inside_the_project_and_keys_stay_private(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/projects", json={"title": "x", "novel": "文本"}).json()
        assert client.get(f"/api/projects/{created['id']}/files/novel.txt").status_code == 200
        assert client.get(f"/api/projects/{created['id']}/files/../../settings.json").status_code == 404
        assert client.get("/api/projects/..%2f..%2fetc/files/passwd").status_code == 404
        client.put("/api/settings/external-chat", json={"base_url": "https://example.invalid/v1",
                                                        "model": "m", "api_key": "secret-key"})
        state = client.get("/api/state").json()
        assert state["external_chat"] == {"base_url": "https://example.invalid/v1", "model": "m", "has_key": True}
        assert "secret-key" not in json.dumps(state)
        assert stat.S_IMODE((tmp_path / "secrets.json").stat().st_mode) == 0o600


def test_choosing_a_take_switches_the_clip_and_drops_its_timing(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/projects", json={"title": "x", "novel": "文本"}).json()
        state_file = tmp_path / "projects" / created["id"] / "episodes" / "01" / "shots.json"
        state_file.parent.mkdir(parents=True)
        takes = [{"take": 1, "file": "c1.mp4", "sheet": "s1.jpg"}, {"take": 2, "file": "c2.mp4", "sheet": "s2.jpg"}]
        state_file.write_text(json.dumps({"assembled": {}, "e1s1": {**takes[1], "takes": takes, "speech": {}}}))
        state = client.put(f"/api/projects/{created['id']}/episodes/1", json={"use_take": {"e1s1": 1}}).json()
        assert state["e1s1"]["file"] == "c1.mp4" and "speech" not in state["e1s1"] and "assembled" not in state
        assert client.put(f"/api/projects/{created['id']}/episodes/1", json={"use_take": {"e1s1": 3}}).status_code == 404


def test_previous_frame_is_used_only_when_the_place_continues():
    from trtmc_short_drama.media import continues

    plan = {"episodes": [{"number": 1, "shots": [{"location": "a"}, {"location": "b"}]},
                         {"number": 2, "shots": [{"location": "b"}]}, {"number": 3, "shots": [{"location": "c"}]}]}
    assert continues(plan, 2) and not continues(plan, 3) and not continues(plan, 1)


def test_writes_need_the_app_header_and_a_local_host(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch, allowed_hosts=["testserver", "127.0.0.1"])
    with client:
        assert client.post("/api/projects", json={"title": "x", "novel": "文"},
                           headers={"X-Short-Drama": ""}).status_code == 403
        assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 400
        assert client.post("/api/projects", json={"title": "x", "novel": "文"}).status_code == 200


def test_edited_plans_cannot_use_ids_that_escape_the_project(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/projects", json={"title": "x", "novel": "文"}).json()
        bad = {"characters": [{"id": "../../evil"}], "episodes": []}
        assert client.put(f"/api/projects/{created['id']}/plan", json=bad).status_code == 400
        shot = {"characters": [{"id": "aq"}], "episodes": [{"number": 1, "shots": [{"id": "../x"}]}]}
        assert client.put(f"/api/projects/{created['id']}/plan", json=shot).status_code == 400


def test_a_saved_key_never_follows_a_new_endpoint(tmp_path, monkeypatch):
    client, services = make_client(tmp_path, monkeypatch)
    with client:
        client.put("/api/settings/external-chat", json={"base_url": "https://a.example/v1", "model": "m",
                                                        "api_key": "secret-key"})
        assert services.api_key() == "secret-key"
        client.put("/api/settings/external-chat", json={"base_url": "https://b.example/v1", "model": "m"})
        assert services.api_key() == "" and client.get("/api/state").json()["external_chat"]["has_key"] is False
        assert client.put("/api/settings/external-chat", json={"base_url": "file:///etc", "model": "m"}).status_code == 400



def test_a_failed_episode_step_does_not_cancel_other_episodes():
    from trtmc_short_drama.jobs import Jobs

    def execute(job):
        if job.step == "subtitles":
            raise RuntimeError("boom")

    jobs = Jobs(execute)
    try:
        submitted = [jobs.submit("p", step, episode, {}) for step, episode in
                     (("subtitles", 1), ("assemble", 1), ("shots", 2), ("assemble", 2))]
        for _ in range(200):
            if all(j.status not in {"queued", "running"} for j in submitted):
                break
            time.sleep(0.01)
        assert [j.status for j in submitted] == ["failed", "cancelled", "done", "done"]
    finally:
        jobs.close()


def test_a_non_loopback_app_needs_its_token(tmp_path, monkeypatch):
    services = Services(FakeEngine(), {}, tmp_path)
    app = create_app(services, plugins=None, info={"gpu": "test", "tier": "a"}, token="s3cret")
    with TestClient(app, headers={"X-Short-Drama": "1"}) as client:
        assert client.get("/api/state").status_code == 401
        assert client.get("/?token=wrong").status_code == 401
        assert client.get("/?token=s3cret").status_code == 200  # sets the cookie
        assert client.get("/api/state").status_code == 200
    with TestClient(app) as other:
        assert other.get("/api/state", headers={"X-Short-Drama-Token": "s3cret"}).status_code == 200
        assert other.get("/static/app.css").status_code == 200

